import gc
import json
import os
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
# https://huggingface.co/docs/accelerate/v0.11.0/en/memory#accelerate.find_executable_batch_size
from accelerate.utils import find_executable_batch_size
from ..baseline import SingleBehaviorRedTeamingMethod
from ..model_utils import get_template
from ..check_refusal_utils import check_refusal_completions
from .gcg_posinit_random_utils import sample_control, get_nonascii_toks, insert_optim_embed_pos, interleave_behavior_and_controls
import time
import random
# ============================== GCG CLASS DEFINITION ============================== #
class GCG_posinit_random(SingleBehaviorRedTeamingMethod):
    def __init__(self,
                 target_model,
                 targets_path,
                 num_steps=50,
                 adv_string_init="!",
                 allow_non_ascii=False,
                 search_width=512,
                 num_adv_string=20,
                 use_prefix_cache=True,
                 eval_steps=10,
                 eval_with_check_refusal=False,
                 check_refusal_min_loss=0.1,
                 early_stopping=False,
                 early_stopping_min_loss=0.1,
                 starting_search_batch_size=None,
                 **model_kwargs):
        """
        :param target_model: a dictionary specifying the target model (kwargs to load_model_and_tokenizer)
        :param targets_path: the path to the targets JSON file
        :param num_steps: the number of optimization steps to use
        :param adv_string_init: the initial adversarial string to start with
        :param allow_non_ascii: whether to allow non-ascii characters when sampling candidate updates
        :param search_width: the number of candidates to sample at each step
        :param use_prefix_cache: whether to use KV cache for static prefix tokens (WARNING: Not all models support this)
        :param eval_steps: the number of steps between each evaluation
        :param eval_with_check_refusal: whether to generate and check for refusal every eval_steps; used for early stopping
        :param check_refusal_min_loss: a minimum loss before checking for refusal
        :param early_stopping: an alternate form of early stopping, based solely on early_stopping_min_loss
        :param early_stopping_min_loss: the loss at which to stop optimization if early_stopping is True
        :param starting_search_batch_size: the initial search_batch_size; will auto reduce later in case of OOM
        """
        super().__init__(target_model=target_model, **model_kwargs)  # assigns self.model, self.tokenizer, and self.model_name_or_path
        self.num_steps = num_steps
        self.adv_string_init = adv_string_init
        self.allow_non_ascii = allow_non_ascii
        self.search_width = search_width
        self.num_adv_string = num_adv_string
        self.use_prefix_cache = use_prefix_cache
        if use_prefix_cache and self.model.config.use_cache != use_prefix_cache:
            print(f"WARNING: setting model.config.use_cache={use_prefix_cache}")
            self.model.config.use_cache = use_prefix_cache
        
        with open(targets_path, 'r', encoding='utf-8') as file:
            self.behavior_id_to_target = json.load(file)
        
        ### Eval Vars ###
        self.eval_steps = eval_steps
        self.eval_with_check_refusal = eval_with_check_refusal
        self.check_refusal_min_loss = check_refusal_min_loss
        self.early_stopping = early_stopping
        self.early_stopping_min_loss = early_stopping_min_loss
        self.starting_search_batch_size = starting_search_batch_size

        ## Instruction template
        template = get_template(**target_model)['prompt']
        self.template = template
        self.before_tc, self.after_tc = template.split("{instruction}")
        
    def generate_test_cases_single_behavior(self, behavior_dict, verbose=False, **kwargs):
        """
        Generates test cases for a single behavior

        :param behavior: a dictionary specifying the behavior to generate test cases for
        :param verbose: whether to print progress
        :return: a test case and logs
        """

        # starting search_batch_size, will auto reduce batch_size later if go OOM (resets for every new behavior)
        self.search_batch_size = self.starting_search_batch_size if self.starting_search_batch_size else self.search_width

        # ========== Behavior and Target str ==========
        behavior = behavior_dict['Behavior']
        context_str = behavior_dict['ContextString']
        behavior_id = behavior_dict['BehaviorID']
        
        target = self.behavior_id_to_target[behavior_id]
        if context_str:
            behavior = f"{context_str}\n\n---\n\n{behavior}"

        print(f'Behavior: {behavior_id} || Target: {target}')
        ### Targeted Model and Tokenier ###
        model = self.model
        device = model.device
        tokenizer = self.tokenizer

        ### GCG hyperparams ###
        num_steps = self.num_steps
        adv_string_init = self.adv_string_init
        allow_non_ascii = self.allow_non_ascii
        search_width = self.search_width
        num_adv_string = self.num_adv_string
        
        before_tc = self.before_tc
        after_tc = self.after_tc

        ### Eval vars ###
        eval_steps = self.eval_steps
        eval_with_check_refusal = self.eval_with_check_refusal
        check_refusal_min_loss = self.check_refusal_min_loss
        early_stopping = self.early_stopping
        early_stopping_min_loss = self.early_stopping_min_loss

        embed_layer = model.get_input_embeddings()
        vocab_size = embed_layer.weight.shape[0]  # can be larger than tokenizer.vocab_size for some models
        vocab_embeds = embed_layer(torch.arange(0, vocab_size).long().to(model.device)).detach()

        not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)
        # remove redundant tokens
        not_allowed_tokens = torch.unique(not_allowed_tokens)

        # ========== setup optim_ids and cached components ========== #
        pos_init_start = time.time()

        init_optim_id = tokenizer(" " + adv_string_init, return_tensors="pt", add_special_tokens=False).to(device)['input_ids']
        if init_optim_id.shape[1] != 1:
            print(f'[init_optim_id] fallback to original: "{adv_string_init}"')
            init_optim_id = tokenizer(adv_string_init, return_tensors="pt", add_special_tokens=False).to(device)['input_ids']
        else:
            print(f'[init_optim_id] using with space: " {adv_string_init}"')

        num_optim_tokens = num_adv_string

        cache_input_ids = tokenizer([before_tc], padding=False)['input_ids'] # some tokenizer have <s> for before_tc
        cache_input_ids += tokenizer([behavior, after_tc, target], padding=False, add_special_tokens=False)['input_ids']
        cache_input_ids = [torch.tensor(input_ids, device=device).unsqueeze(0) for input_ids in cache_input_ids] # make tensor separately because can't return_tensors='pt' in tokenizer
        before_ids, behavior_ids, after_ids, target_ids = cache_input_ids
        before_embeds, behavior_embeds, after_embeds, target_embeds = [embed_layer(input_ids) for input_ids in cache_input_ids]
        
        if self.use_prefix_cache:
            # precompute KV cache for everything before the optimized tokens
            input_embeds = torch.cat([before_embeds], dim=1)
            with torch.no_grad():
                outputs = model(inputs_embeds=input_embeds.to(model.dtype), use_cache=True)
                self.prefix_cache = outputs.past_key_values

        pos_init_start = time.time()

        behavior_length = len(behavior_ids[0])
        possible_positions = list(range(behavior_length + 1))

        optim_pos = torch.tensor(random.choices(possible_positions, k=num_adv_string), dtype=torch.long, device=device)
        optim_ids = init_optim_id.repeat(1, num_optim_tokens)

        pos_init_time = time.time() - pos_init_start
        print(f"Optimized positions: {optim_pos.tolist()}, Time taken: {pos_init_time:.2f} seconds")

        # ========== run optimization ========== #
        all_losses = []
        all_test_cases = []
        all_optim_ids = []

        chosen_positions = []       # 선택된 optim 슬롯 인덱스 (0..num_optim_tokens-1)
        chosen_token_ids = []       # 그 슬롯에 새로 들어간 토큰 id
        chosen_token_texts = []  
        # all_samples = []
        # all_sample_controls = []
        # all_sample_token_vals = []
        # all_sample_token_poses = []
        # all_sample_losses = []
        step_times = []

        vocab_embeds = embed_layer(torch.arange(0, vocab_size).long().to(model.device))
        before_embeds, behavior_embeds, after_embeds, target_embeds = [embed_layer(input_ids).to(model.dtype) for input_ids in cache_input_ids]

        for i in tqdm(range(num_steps)):
            start_time = time.time()
            # ========== compute coordinate token_gradient ========== #
            # create input
            optim_ids_onehot = torch.zeros((1, num_optim_tokens, vocab_size), device=device, dtype=model.dtype)
            optim_ids_onehot.scatter_(2, optim_ids.unsqueeze(2), 1.0).requires_grad_()
            optim_embeds = torch.matmul(optim_ids_onehot.squeeze(0), vocab_embeds).unsqueeze(0)

            behavior_optim_embeds = insert_optim_embed_pos(behavior_embeds, optim_embeds, optim_pos)

            # forward pass
            if self.use_prefix_cache:
                input_embeds = torch.cat([behavior_optim_embeds, after_embeds, target_embeds], dim=1)
                outputs = model(inputs_embeds=input_embeds, past_key_values=self.prefix_cache)
            else:
                input_embeds = torch.cat([before_embeds, behavior_optim_embeds, after_embeds, target_embeds], dim=1)
                outputs = model(inputs_embeds=input_embeds)
            
            logits = outputs.logits

            # compute loss
            # Shift so that tokens < n predict n
            tmp = input_embeds.shape[1] - target_embeds.shape[1]
            shift_logits = logits[..., tmp-1:-1, :].contiguous()
            shift_labels = target_ids
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            token_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]
            token_grad = token_grad / token_grad.norm(dim=-1, keepdim=True)

            # ========== Sample a batch of new tokens based on the coordinate gradient. ========== #
            sampled_top_indices, token_val, token_pos = sample_control(optim_ids.squeeze(0), token_grad.squeeze(0), search_width, topk=256, temp=1, not_allowed_tokens=not_allowed_tokens)

            # ========= Filter sampled_top_indices that aren't the same after detokenize->retokenize ========= #
            sampled_top_indices_inserted = interleave_behavior_and_controls(behavior_ids.repeat(search_width, 1), sampled_top_indices, optim_pos.repeat(search_width, 1))

            sampled_top_indices_text = tokenizer.batch_decode(sampled_top_indices_inserted)
            new_sampled_top_indices, new_sampled_top_control_indices = [], []
            new_token_val, new_token_pos = [], []

            count = 0
            for j in range(len(sampled_top_indices_text)):
                # tokenize again
                tmp = tokenizer(sampled_top_indices_text[j], return_tensors="pt", add_special_tokens=False).to(device)['input_ids'][0]
                # if the tokenized text is different, then set the sampled_top_indices to padding_top_indices
                if not torch.equal(tmp, sampled_top_indices_inserted[j]):
                    count += 1
                    continue
                else:
                    new_sampled_top_indices.append(sampled_top_indices[j])
                    new_sampled_top_control_indices.append(sampled_top_indices_inserted[j])
                    new_token_val.append(token_val[j])
                    new_token_pos.append(token_pos[j])
            
            if len(new_sampled_top_indices) == 0:
                print('All removals; defaulting to keeping all')
                count = 0
            else:
                sampled_top_indices = torch.stack(new_sampled_top_indices)
                sampled_top_indices_inserted = torch.stack(new_sampled_top_control_indices)
                token_val = torch.stack(new_token_val)
                token_pos = torch.stack(new_token_pos)
            
            if count >= search_width // 2:
                print('\nLots of removals:', count)
            
            new_search_width = search_width - count

            # ========== Compute loss on these candidates and take the argmin. ========== #
            # create input
            sampled_top_embeds = embed_layer(sampled_top_indices_inserted)

            if self.use_prefix_cache:
                input_embeds = torch.cat([sampled_top_embeds,
                                        after_embeds.repeat(new_search_width, 1, 1),
                                        target_embeds.repeat(new_search_width, 1, 1)], dim=1)
            else:
                input_embeds = torch.cat([before_embeds.repeat(new_search_width, 1, 1),
                                        sampled_top_embeds,
                                        after_embeds.repeat(new_search_width, 1, 1),
                                        target_embeds.repeat(new_search_width, 1, 1)], dim=1)
            
            # Auto Find Batch Size for foward candidates (each time go OOM will decay search_batch_size // 2)
            loss = find_executable_batch_size(self.compute_candidates_loss, self.search_batch_size)(input_embeds, target_ids)

            # ========== Update the optim_ids with the best candidate ========== #
            optim_ids = sampled_top_indices[loss.argmin()].unsqueeze(0)
            test_case_ids = sampled_top_indices_inserted[loss.argmin()].unsqueeze(0)
            
            best_idx = loss.argmin()

            chosen_slot = token_pos[best_idx].item()         # 어떤 optim 슬롯이 바뀌었는지
            chosen_tok_id = optim_ids[0, chosen_slot].item() # 그 슬롯에 들어간 토큰 id
            chosen_tok_txt = tokenizer.decode([chosen_tok_id])

            chosen_positions.append(chosen_slot)
            chosen_token_ids.append(chosen_tok_id)
            chosen_token_texts.append(chosen_tok_txt)

            test_case = tokenizer.decode(test_case_ids[0])
            all_test_cases.append(test_case)
            current_loss = loss.min().item()
            all_losses.append(current_loss)
            all_optim_ids.append(optim_ids.tolist())

            step_times.append(time.time() - start_time)

            # all_samples.append(tokenizer.batch_decode(sampled_top_indices_inserted))
            # all_sample_controls.append(tokenizer.batch_decode(sampled_top_indices))
            # all_sample_token_vals.append(token_val.tolist())
            # all_sample_token_poses.append(token_pos.tolist())

            # all_sample_losses.append(loss.tolist())

            # ========== Eval and Early Stopping ========== #
            if (i % eval_steps == 0) or (i == num_steps - 1):
                p_output = f'\n===>Step {i}\n===>Test Case: {test_case}\n===>Loss: {current_loss}, Time: {sum(step_times):.2f}s'
                if eval_with_check_refusal and current_loss < check_refusal_min_loss:
                    input_str = self.template.format(instruction=test_case)
                    is_refusal, completions, _ = check_refusal_completions(model, tokenizer, inputs=[input_str])
                    p_output += f"\n\n===>Completion: {completions[0]}"
                    if not is_refusal[0]:
                        break

                if verbose:
                    print(p_output)

            if early_stopping and current_loss < early_stopping_min_loss:
                print(f'Early stopping at step {i} with loss {current_loss}')
                print()
                break

            del input_embeds, sampled_top_embeds, logits, shift_logits, loss
            torch.cuda.empty_cache()
            gc.collect()

        logs = {'final_loss': current_loss, 'all_losses': all_losses, 'all_test_cases': all_test_cases, 'all_optim_ids' : all_optim_ids, 'optim_pos': optim_pos.tolist(), 'step_times': step_times, 'position_init_time': pos_init_time,
                'chosen_positions': chosen_positions, 'chosen_token_ids': chosen_token_ids, 'chosen_token_texts': chosen_token_texts}
                #'all_samples': all_samples, 'all_sample_controls': all_sample_controls,'all_sample_token_vals': all_sample_token_vals, 'all_sample_token_poses': all_sample_token_poses, 'all_sample_losses': all_sample_losses}

        return test_case, logs

    def compute_candidates_loss(self, search_batch_size, input_embeds, target_ids):
        if self.search_batch_size != search_batch_size:
            print(f"INFO: Setting candidates search_batch_size to {search_batch_size})")
            self.search_batch_size = search_batch_size
            torch.cuda.empty_cache()
            gc.collect()

        all_loss = []
        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i+search_batch_size]

                if self.use_prefix_cache:
                    # Expand prefix cache to match batch size
                    prefix_cache_batch = self.prefix_cache
                    current_batch_size = input_embeds_batch.shape[0]
                    prefix_cache_batch = []
                    for i in range(len(self.prefix_cache)):
                        prefix_cache_batch.append([])
                        for j in range(len(self.prefix_cache[i])):
                            prefix_cache_batch[i].append(self.prefix_cache[i][j].expand(current_batch_size, -1, -1, -1))
                    outputs = self.model(inputs_embeds=input_embeds_batch, past_key_values=prefix_cache_batch)
                else:
                    outputs = self.model(inputs_embeds=input_embeds_batch)
            logits = outputs.logits

            # compute loss
            # Shift so that tokens < n predict n
            tmp = input_embeds_batch.shape[1] - target_ids.shape[1]
            shift_logits = logits[..., tmp-1:-1, :].contiguous()
            shift_labels = target_ids.repeat(input_embeds_batch.shape[0], 1)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(input_embeds_batch.shape[0], -1).mean(dim=1)
            all_loss.append(loss)
        
            del outputs, logits, loss
            torch.cuda.empty_cache()
            gc.collect()
            
        return torch.cat(all_loss, dim=0)
        