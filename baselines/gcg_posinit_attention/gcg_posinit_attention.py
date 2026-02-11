import gc
import json
import os
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
from accelerate.utils import find_executable_batch_size
from ..baseline import SingleBehaviorRedTeamingMethod
from ..model_utils import get_template
from ..check_refusal_utils import check_refusal_completions
from .gcg_posinit_attention_utils import sample_control, get_nonascii_toks, insert_optim_embed_pos, interleave_behavior_and_controls, generate_positions
import time

class GCG_posinit_attention(SingleBehaviorRedTeamingMethod):
    def __init__(self,
                 target_model,
                 targets_path,
                 num_steps=50,
                 adv_string_init="!",
                 allow_non_ascii=False,
                 search_width=512,
                 num_adv_string=20,
                 attention_temp=1.0,
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
        super().__init__(target_model=target_model, **model_kwargs)  
        self.num_steps = num_steps
        self.adv_string_init = adv_string_init
        self.allow_non_ascii = allow_non_ascii
        self.search_width = search_width
        self.num_adv_string = num_adv_string
        self.attention_temp = attention_temp
        self.use_prefix_cache = use_prefix_cache
        if use_prefix_cache and self.model.config.use_cache != use_prefix_cache:
            print(f"WARNING: setting model.config.use_cache={use_prefix_cache}")
            self.model.config.use_cache = use_prefix_cache
        
        with open(targets_path, 'r', encoding='utf-8') as file:
            self.behavior_id_to_target = json.load(file)
        
        self.eval_steps = eval_steps
        self.eval_with_check_refusal = eval_with_check_refusal
        self.check_refusal_min_loss = check_refusal_min_loss
        self.early_stopping = early_stopping
        self.early_stopping_min_loss = early_stopping_min_loss
        self.starting_search_batch_size = starting_search_batch_size

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

        self.search_batch_size = self.starting_search_batch_size if self.starting_search_batch_size else self.search_width

        behavior = behavior_dict['Behavior']
        context_str = behavior_dict['ContextString']
        behavior_id = behavior_dict['BehaviorID']
        
        target = self.behavior_id_to_target[behavior_id]
        if context_str:
            behavior = f"{context_str}\n\n---\n\n{behavior}"

        print(f'Behavior: {behavior_id} || Target: {target}')
        model = self.model
        device = model.device
        tokenizer = self.tokenizer

        num_steps = self.num_steps
        adv_string_init = self.adv_string_init
        allow_non_ascii = self.allow_non_ascii
        search_width = self.search_width
        num_adv_string = self.num_adv_string
        attention_temp = self.attention_temp
        
        before_tc = self.before_tc
        after_tc = self.after_tc

        eval_steps = self.eval_steps
        eval_with_check_refusal = self.eval_with_check_refusal
        check_refusal_min_loss = self.check_refusal_min_loss
        early_stopping = self.early_stopping
        early_stopping_min_loss = self.early_stopping_min_loss

        embed_layer = model.get_input_embeddings()
        vocab_size = embed_layer.weight.shape[0]  
        vocab_embeds = embed_layer(torch.arange(0, vocab_size).long().to(model.device)).detach()

        not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)
        not_allowed_tokens = torch.unique(not_allowed_tokens)

        pos_init_start = time.time()

        init_optim_id = tokenizer(" " + adv_string_init, return_tensors="pt", add_special_tokens=False).to(device)['input_ids']
        if init_optim_id.shape[1] != 1:
            print(f'[init_optim_id] fallback to original: "{adv_string_init}"')
            init_optim_id = tokenizer(adv_string_init, return_tensors="pt", add_special_tokens=False).to(device)['input_ids']
        else:
            print(f'[init_optim_id] using with space: " {adv_string_init}"')

        num_optim_tokens = num_adv_string

        cache_input_ids = tokenizer([before_tc], padding=False)['input_ids'] 
        cache_input_ids += tokenizer([behavior, after_tc, target], padding=False, add_special_tokens=False)['input_ids']
        cache_input_ids = [torch.tensor(input_ids, device=device).unsqueeze(0) for input_ids in cache_input_ids] 
        before_ids, behavior_ids, after_ids, target_ids = cache_input_ids
        before_embeds, behavior_embeds, after_embeds, target_embeds = [embed_layer(input_ids) for input_ids in cache_input_ids]
        
        if self.use_prefix_cache:
           
            input_embeds = torch.cat([before_embeds], dim=1)
            with torch.no_grad():
                outputs = model(inputs_embeds=input_embeds.to(model.dtype), use_cache=True)
                self.prefix_cache = outputs.past_key_values

        pos_init_start = time.time()

        num_positions = len(behavior_ids[0]) + 1
        insert_positions = list(range(num_positions))

        print(f"Inserting tokens at all {num_positions} positions: {insert_positions}")

        optim_ids = init_optim_id.expand(-1, num_positions)  
        optim_spread_pos = torch.tensor(insert_positions, dtype=torch.long, device=device)

        optim_ids_onehot = torch.zeros((1, num_positions, vocab_size), device=device, dtype=model.dtype)
        optim_ids_onehot.scatter_(2, optim_ids.unsqueeze(2), 1.0)
        optim_embeds = torch.matmul(optim_ids_onehot.squeeze(0), vocab_embeds).unsqueeze(0) 

        inserted = insert_optim_embed_pos(behavior_embeds, optim_embeds, optim_spread_pos)

        if self.use_prefix_cache:
            input_embeds = torch.cat([inserted, after_embeds, target_embeds], dim=1).to(model.dtype)
            outputs = model(inputs_embeds=input_embeds, past_key_values=self.prefix_cache, output_attentions=True)
        else:
            input_embeds = torch.cat([before_embeds, inserted, after_embeds, target_embeds], dim=1).to(model.dtype)
            outputs = model(inputs_embeds=input_embeds, output_attentions=True)

        total_layers = len(outputs.attentions)

        total_layers = len(outputs.attentions)
        upper_half_start = total_layers // 2 

        print(f"Using upper half layers {upper_half_start}:{total_layers} (total layers: {total_layers})")

        attentions = torch.stack(outputs.attentions[upper_half_start:]).sum(dim=(0, 2))

        actual_positions = []
        for i, pos in enumerate(insert_positions):
            actual_pos = pos + i  
            actual_positions.append(actual_pos)

        batch_size, query_len, key_len = attentions.shape
        cached_len = key_len - query_len  
        before_len = before_embeds.shape[1]
        inserted_len = inserted.shape[1]
        after_len = after_embeds.shape[1]
        target_len = target_embeds.shape[1]

        if self.use_prefix_cache:
            query_after_start = inserted_len
            query_after_end = inserted_len + after_len
        else:
            query_after_start = before_len + inserted_len
            query_after_end = before_len + inserted_len + after_len
        chat_positions = list(range(query_after_start, query_after_end))

        if self.use_prefix_cache:
            key_inserted_start = cached_len
            key_inserted_end = cached_len + inserted_len
        else:
            key_inserted_start = before_len
            key_inserted_end = before_len + inserted_len

        attention_weight = attentions[0, chat_positions, key_inserted_start:key_inserted_end].sum(dim=0)[actual_positions]

        print(f"Raw attention scores: {attention_weight.tolist()}")
        attention_probs = torch.softmax(attention_weight / attention_temp, dim=0)
        print(f"Attention probs: {attention_probs}")
        
        optim_pos = generate_positions(attention_probs, num_adv_string)
        optim_ids = init_optim_id.expand(-1, num_adv_string)
        print(f"Initial optim_ids: {optim_ids.shape}")

        pos_init_time = time.time() - pos_init_start

        print(f"Optimized positions: {optim_pos.tolist()}, Time taken: {pos_init_time:.2f} seconds")

        all_losses = []
        all_test_cases = []
        all_optim_ids = []

        chosen_positions = []       
        chosen_token_ids = []      
        chosen_token_texts = []  
        step_times = []

        vocab_embeds = embed_layer(torch.arange(0, vocab_size).long().to(model.device))
        before_embeds, behavior_embeds, after_embeds, target_embeds = [embed_layer(input_ids).to(model.dtype) for input_ids in cache_input_ids]

        for i in tqdm(range(num_steps)):
            start_time = time.time()
            optim_ids_onehot = torch.zeros((1, num_optim_tokens, vocab_size), device=device, dtype=model.dtype)
            optim_ids_onehot.scatter_(2, optim_ids.unsqueeze(2), 1.0).requires_grad_()
            optim_embeds = torch.matmul(optim_ids_onehot.squeeze(0), vocab_embeds).unsqueeze(0)

            behavior_optim_embeds = insert_optim_embed_pos(behavior_embeds, optim_embeds, optim_pos)

            if self.use_prefix_cache:
                input_embeds = torch.cat([behavior_optim_embeds, after_embeds, target_embeds], dim=1)
                outputs = model(inputs_embeds=input_embeds, past_key_values=self.prefix_cache)
            else:
                input_embeds = torch.cat([before_embeds, behavior_optim_embeds, after_embeds, target_embeds], dim=1)
                outputs = model(inputs_embeds=input_embeds)
            
            logits = outputs.logits

            tmp = input_embeds.shape[1] - target_embeds.shape[1]
            shift_logits = logits[..., tmp-1:-1, :].contiguous()
            shift_labels = target_ids
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            token_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]
            token_grad = token_grad / token_grad.norm(dim=-1, keepdim=True)

            sampled_top_indices, token_val, token_pos = sample_control(optim_ids.squeeze(0), token_grad.squeeze(0), search_width, topk=256, temp=1, not_allowed_tokens=not_allowed_tokens)

            sampled_top_indices_inserted = interleave_behavior_and_controls(behavior_ids.repeat(search_width, 1), sampled_top_indices, optim_pos.repeat(search_width, 1))

            sampled_top_indices_text = tokenizer.batch_decode(sampled_top_indices_inserted)
            new_sampled_top_indices, new_sampled_top_control_indices = [], []
            new_token_val, new_token_pos = [], []

            count = 0
            for j in range(len(sampled_top_indices_text)):
                tmp = tokenizer(sampled_top_indices_text[j], return_tensors="pt", add_special_tokens=False).to(device)['input_ids'][0]
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
            
            loss = find_executable_batch_size(self.compute_candidates_loss, self.search_batch_size)(input_embeds, target_ids)

            optim_ids = sampled_top_indices[loss.argmin()].unsqueeze(0)
            test_case_ids = sampled_top_indices_inserted[loss.argmin()].unsqueeze(0)
            
            best_idx = loss.argmin()

            chosen_slot = token_pos[best_idx].item()         
            chosen_tok_id = optim_ids[0, chosen_slot].item()
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

            tmp = input_embeds_batch.shape[1] - target_ids.shape[1]
            shift_logits = logits[..., tmp-1:-1, :].contiguous()
            shift_labels = target_ids.repeat(input_embeds_batch.shape[0], 1)
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(input_embeds_batch.shape[0], -1).mean(dim=1)
            all_loss.append(loss)
        
            del outputs, logits, loss
            torch.cuda.empty_cache()
            gc.collect()
            
        return torch.cat(all_loss, dim=0)
        