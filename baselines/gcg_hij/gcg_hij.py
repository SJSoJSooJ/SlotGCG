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
from .gcg_hij_utils import sample_control, get_nonascii_toks

class GCG_hij(SingleBehaviorRedTeamingMethod):
    def __init__(self,
                 target_model,
                 targets_path,
                 num_steps=50,
                 adv_string_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
                 allow_non_ascii=False,
                 search_width=512,
                 use_prefix_cache=True,
                 attention_weight=100,
                 eval_steps=10,
                 eval_with_check_refusal=False,
                 check_refusal_min_loss=0.1,
                 early_stopping=False,
                 early_stopping_min_loss=0.1,
                 starting_search_batch_size=None,
                 **model_kwargs):
        
        super().__init__(target_model=target_model, **model_kwargs)  
        self.num_steps = num_steps
        self.adv_string_init = adv_string_init
        self.allow_non_ascii = allow_non_ascii
        self.search_width = search_width
        self.use_prefix_cache = use_prefix_cache
        if use_prefix_cache and self.model.config.use_cache != use_prefix_cache:
            print(f"WARNING: setting model.config.use_cache={use_prefix_cache}")
            self.model.config.use_cache = use_prefix_cache
        self.attention_weight = attention_weight

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
        

        self.search_batch_size = self.starting_search_batch_size if self.starting_search_batch_size else self.search_width

        behavior = behavior_dict['Behavior']
        context_str = behavior_dict['ContextString']
        behavior_id = behavior_dict['BehaviorID']
        
        target = self.behavior_id_to_target[behavior_id]
        behavior += ' '
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
        before_tc = self.before_tc
        after_tc = self.after_tc

        eval_steps = self.eval_steps
        eval_with_check_refusal = self.eval_with_check_refusal
        check_refusal_min_loss = self.check_refusal_min_loss
        early_stopping = self.early_stopping
        early_stopping_min_loss = self.early_stopping_min_loss

        embed_layer = model.get_input_embeddings()
        vocab_size = embed_layer.weight.shape[0] 
        vocab_embeds = embed_layer(torch.arange(0, vocab_size).long().to(model.device))
        not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)
        not_allowed_tokens = torch.unique(not_allowed_tokens)

        optim_ids = tokenizer(adv_string_init, return_tensors="pt", add_special_tokens=False).to(device)['input_ids']
        num_optim_tokens = len(optim_ids[0])

        cache_input_ids = tokenizer([before_tc], padding=False)['input_ids'] 
        cache_input_ids += tokenizer([behavior, after_tc, target], padding=False, add_special_tokens=False)['input_ids']
        cache_input_ids = [torch.tensor(input_ids, device=device).unsqueeze(0) for input_ids in cache_input_ids] 
        before_ids, behavior_ids, after_ids, target_ids = cache_input_ids
        before_embeds, behavior_embeds, after_embeds, target_embeds = [embed_layer(input_ids) for input_ids in cache_input_ids]
        
        if self.use_prefix_cache:
            input_embeds = torch.cat([before_embeds], dim=1)
            with torch.no_grad():
                outputs = model(inputs_embeds=input_embeds, output_attentions=True, use_cache=True)
                self.prefix_cache = outputs.past_key_values

        all_losses = []
        all_test_cases = []
        all_logit_losses = []
        all_attn_losses = []
        for i in tqdm(range(num_steps)):
            optim_ids_onehot = torch.zeros((1, num_optim_tokens, vocab_size), device=device, dtype=model.dtype)
            optim_ids_onehot.scatter_(2, optim_ids.unsqueeze(2), 1.0).requires_grad_()
            optim_embeds = torch.matmul(optim_ids_onehot.squeeze(0), vocab_embeds).unsqueeze(0)
            if self.use_prefix_cache:
                input_embeds = torch.cat([behavior_embeds, optim_embeds, after_embeds, target_embeds], dim=1)
                outputs = model(inputs_embeds=input_embeds, output_attentions=True, past_key_values=self.prefix_cache)
            else:
                input_embeds = torch.cat([before_embeds, behavior_embeds, optim_embeds, after_embeds, target_embeds], dim=1)
                outputs = model(inputs_embeds=input_embeds, output_attentions=True)

            total_layers = len(outputs.attentions)
            l1 = max(0, int(0.1 * total_layers))
            l2 = min(total_layers, int(0.9 * total_layers) + 1)

            logits = outputs.logits
            attentions = torch.stack(outputs.attentions[l1:l2]).mean(dim=(0, 2))
            tmp = input_embeds.shape[1] - target_embeds.shape[1] 

            shift_logits = logits[..., tmp-1:-1, :].contiguous()
            shift_labels = target_ids
            loss_fct = CrossEntropyLoss()
            logit_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            batch_size, query_len, key_len = attentions.shape
            cached_len = key_len - query_len 
            before_len = before_embeds.shape[1]
            behavior_len = behavior_embeds.shape[1]
            control_len = optim_embeds.shape[1]
            after_len = after_embeds.shape[1]
            target_len = target_embeds.shape[1]

            if self.use_prefix_cache:
                query_after_start = behavior_len + control_len
                query_after_end = query_after_start + after_len
            else:
                query_after_start = before_len + behavior_len + control_len
                query_after_end = query_after_start + after_len

            chat_positions = list(range(query_after_start, query_after_end))

            if self.use_prefix_cache:
                key_control_start = cached_len + behavior_len
                key_control_end = cached_len + behavior_len + control_len
            else:  
                key_control_start = before_len + behavior_len
                key_control_end = before_len + behavior_len + control_len

            attn_loss = attentions[0, chat_positions, key_control_start:key_control_end].mean()

            loss = logit_loss - self.attention_weight * attn_loss

            token_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

            
            sampled_top_indices = sample_control(optim_ids.squeeze(0), token_grad.squeeze(0), search_width, topk=256, temp=1, not_allowed_tokens=not_allowed_tokens)

            
            sampled_top_indices_text = tokenizer.batch_decode(sampled_top_indices)
            new_sampled_top_indices = []
            count = 0
            for j in range(len(sampled_top_indices_text)):
                tmp = tokenizer(sampled_top_indices_text[j], return_tensors="pt", add_special_tokens=False).to(device)['input_ids'][0]
                if not torch.equal(tmp, sampled_top_indices[j]):
                    count += 1
                    continue
                else:
                    new_sampled_top_indices.append(sampled_top_indices[j])
            
            if len(new_sampled_top_indices) == 0:
                print('All removals; defaulting to keeping all')
                count = 0
            else:
                sampled_top_indices = torch.stack(new_sampled_top_indices)
            
            if count >= search_width // 2:
                print('\nLots of removals:', count)
            
            new_search_width = search_width - count

            sampled_top_embeds = embed_layer(sampled_top_indices)
            if self.use_prefix_cache:
                input_embeds = torch.cat([behavior_embeds.repeat(new_search_width, 1, 1),
                                        sampled_top_embeds,
                                        after_embeds.repeat(new_search_width, 1, 1),
                                        target_embeds.repeat(new_search_width, 1, 1)], dim=1)

            else:
                input_embeds = torch.cat([before_embeds.repeat(new_search_width, 1, 1),
                                        behavior_embeds.repeat(new_search_width, 1, 1),
                                        sampled_top_embeds,
                                        after_embeds.repeat(new_search_width, 1, 1),
                                        target_embeds.repeat(new_search_width, 1, 1)], dim=1)
            
            embeds = [before_embeds, behavior_embeds, optim_embeds, after_embeds, target_embeds]
            loss, logit_losses, attn_losses = find_executable_batch_size(self.compute_candidates_loss, self.search_batch_size)(input_embeds, target_ids, embeds)

            optim_ids = sampled_top_indices[loss.argmin()].unsqueeze(0)
            
            test_case_ids = torch.cat([behavior_ids, optim_ids], dim=1)
            test_case = tokenizer.decode(test_case_ids[0])
            all_test_cases.append(test_case)
            current_loss, current_loss_idx = torch.min(loss, dim=0)

            all_losses.append(current_loss)
            all_logit_losses.append(logit_losses[current_loss_idx])
            all_attn_losses.append(attn_losses[current_loss_idx])

            if (i % eval_steps == 0) or (i == num_steps - 1):
                p_output = f'\n===>Step {i}\n===>Test Case: {test_case}\n===>Loss: {current_loss}'
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

            del input_embeds, sampled_top_embeds, logits, shift_logits, logit_loss, attn_loss, attentions, logit_losses, attn_losses
            torch.cuda.empty_cache()
            gc.collect()

        logs = {'final_loss': float(current_loss.item() if isinstance(current_loss, torch.Tensor) else current_loss), 
                'all_losses': [x.item() if x.dim() == 0 else x.tolist() for x in all_losses], 
                'all_test_cases': all_test_cases, 
                'all_logit_losses': [float(x.item()) if isinstance(x, torch.Tensor) else float(x) for x in all_logit_losses], 
                'all_attn_losses': [float(x.item()) if isinstance(x, torch.Tensor) else float(x) for x in all_attn_losses]}

        return test_case, logs

    def compute_candidates_loss(self, search_batch_size, input_embeds, target_ids, embeds):
        if self.search_batch_size != search_batch_size:
            print(f"INFO: Setting candidates search_batch_size to {search_batch_size})")
            self.search_batch_size = search_batch_size
            torch.cuda.empty_cache()
            gc.collect()


        all_logit_loss = []
        all_attn_loss = []
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
                    outputs = self.model(inputs_embeds=input_embeds_batch, output_attentions=True, past_key_values=prefix_cache_batch)
                else:
                    outputs = self.model(inputs_embeds=input_embeds_batch, output_attentions=True)
            logits = outputs.logits
            attentions = outputs.attentions

            tmp = input_embeds_batch.shape[1] - target_ids.shape[1]
            shift_logits = logits[..., tmp-1:-1, :].contiguous()
            shift_labels = target_ids.repeat(input_embeds_batch.shape[0], 1)
            loss_fct = CrossEntropyLoss(reduction='none')
            loss_logit = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss_logit = loss_logit.view(input_embeds_batch.shape[0], -1).mean(dim=1)

            total_layers = len(outputs.attentions)
            l1 = max(0, int(0.1 * total_layers))
            l2 = min(total_layers, int(0.9 * total_layers) + 1)

            attentions = torch.stack(outputs.attentions[l1:l2]).mean(dim=(0, 2))
            before_embeds, behavior_embeds, optim_embeds, after_embeds, target_embeds = embeds
            batch_size, query_len, key_len = attentions.shape
            cached_len = key_len - query_len
            before_len = before_embeds.shape[1]
            behavior_len = behavior_embeds.shape[1]
            control_len = optim_embeds.shape[1]
            after_len = after_embeds.shape[1]
            target_len = target_embeds.shape[1]

            if self.use_prefix_cache:
                query_after_start = behavior_len + control_len
                query_after_end = query_after_start + after_len
            else:
                query_after_start = before_len + behavior_len + control_len
                query_after_end = query_after_start + after_len

            chat_positions = list(range(query_after_start, query_after_end))

            if self.use_prefix_cache:
                key_control_start = cached_len + behavior_len
                key_control_end = cached_len + behavior_len + control_len
            else:  
                key_control_start = before_len + behavior_len
                key_control_end = before_len + behavior_len + control_len
            
            batch_attn_loss = []
            for b_idx in range(attentions.size(0)):
                single_attention = attentions[b_idx:b_idx+1, chat_positions, key_control_start:key_control_end].mean() 
                batch_attn_loss.append(single_attention.unsqueeze(0))

            all_attn_loss.append(torch.cat(batch_attn_loss, dim=0))
            all_logit_loss.append(loss_logit)

            del outputs, logits, loss_logit, attentions, single_attention
            torch.cuda.empty_cache()
            gc.collect()
            
        return torch.cat(all_logit_loss, dim=0) - self.attention_weight * torch.cat(all_attn_loss, dim=0), torch.cat(all_logit_loss, dim=0), - self.attention_weight * torch.cat(all_attn_loss, dim=0)
        