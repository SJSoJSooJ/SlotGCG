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
from .gcg_onlyone_utils import sample_control, get_nonascii_toks
import time

# ============================== GCG Five Tokens CLASS DEFINITION ============================== #
class GCG_onlyone(SingleBehaviorRedTeamingMethod):
    def __init__(self,
                 target_model,
                 targets_path,
                 num_steps=50,
                 adv_string_init="! ! ! ! !",  # Five tokens initialization
                 allow_non_ascii=False,
                 search_width=512,
                 use_prefix_cache=False,  # Disabled for multi-position
                 eval_steps=10,
                 eval_with_check_refusal=False,
                 check_refusal_min_loss=0.1,
                 early_stopping=False,
                 early_stopping_min_loss=0.1,
                 starting_search_batch_size=None,
                 attention_weight=1.0,  # Added attention weight parameter
                 **model_kwargs):
        """
        :param target_model: a dictionary specifying the target model (kwargs to load_model_and_tokenizer)
        :param targets_path: the path to the targets JSON file
        :param num_steps: the number of optimization steps to use
        :param adv_string_init: the initial adversarial string to start with (five tokens)
        :param allow_non_ascii: whether to allow non-ascii characters when sampling candidate updates
        :param search_width: the number of candidates to sample at each step
        :param use_prefix_cache: whether to use KV cache (disabled for multi-position)
        :param eval_steps: the number of steps between each evaluation
        :param eval_with_check_refusal: whether to generate and check for refusal every eval_steps
        :param check_refusal_min_loss: a minimum loss before checking for refusal
        :param early_stopping: early stopping based on loss threshold
        :param early_stopping_min_loss: the loss at which to stop optimization
        :param starting_search_batch_size: the initial search_batch_size
        :param attention_weight: weight for attention loss component
        """
        super().__init__(target_model=target_model, **model_kwargs)
        self.num_steps = num_steps
        self.adv_string_init = adv_string_init
        self.allow_non_ascii = allow_non_ascii
        self.search_width = search_width
        self.use_prefix_cache = False  # Disabled for multi-position attacks
        self.attention_weight = attention_weight  # Added attention weight
        
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
        Generates test cases for a single behavior with independent position optimization of 5-token sequences

        :param behavior: a dictionary specifying the behavior to generate test cases for
        :param verbose: whether to print progress
        :return: a test case and logs
        """

        # starting search_batch_size
        self.search_batch_size = self.starting_search_batch_size if self.starting_search_batch_size else self.search_width

        # ========== Behavior and Target str ==========
        behavior = behavior_dict['Behavior']
        context_str = behavior_dict['ContextString']
        behavior_id = behavior_dict['BehaviorID']
        
        target = self.behavior_id_to_target[behavior_id]
        behavior += ' '
        if context_str:
            behavior = f"{context_str}\n\n---\n\n{behavior}"

        print(f'Behavior: {behavior_id} || Target: {target}')
        
        ### Model and Tokenizer ###
        model = self.model
        device = model.device
        tokenizer = self.tokenizer

        ### GCG hyperparams ###
        num_steps = self.num_steps
        adv_string_init = self.adv_string_init
        allow_non_ascii = self.allow_non_ascii
        search_width = self.search_width
        before_tc = self.before_tc
        after_tc = self.after_tc

        ### Eval vars ###
        eval_steps = self.eval_steps
        eval_with_check_refusal = self.eval_with_check_refusal
        check_refusal_min_loss = self.check_refusal_min_loss
        early_stopping = self.early_stopping
        early_stopping_min_loss = self.early_stopping_min_loss

        embed_layer = model.get_input_embeddings()
        vocab_size = embed_layer.weight.shape[0]
        vocab_embeds = embed_layer(torch.arange(0, vocab_size).long().to(model.device))

        not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)
        if not_allowed_tokens is not None:
            not_allowed_tokens = torch.unique(not_allowed_tokens)

        # ========== Setup cached components ========== #
        cache_input_ids = tokenizer([before_tc], padding=False)['input_ids']
        cache_input_ids += tokenizer([behavior, after_tc, target], padding=False, add_special_tokens=False)['input_ids']
        cache_input_ids = [torch.tensor(input_ids, device=device).unsqueeze(0) for input_ids in cache_input_ids]
        before_ids, behavior_ids, after_ids, target_ids = cache_input_ids
        before_embeds, behavior_embeds, after_embeds, target_embeds = [embed_layer(input_ids) for input_ids in cache_input_ids]

        # ========== Setup optim_ids (5 tokens) ========== #
        optim_ids = tokenizer(adv_string_init, return_tensors="pt", add_special_tokens=False).to(device)['input_ids']
        num_optim_tokens = len(optim_ids[0])
        print(f'Using {num_optim_tokens} adversarial tokens')

        # ========== Get behavior tokens and setup all positions ========== #
        behavior_tokens = behavior_ids[0]  # Remove batch dimension
        num_behavior_tokens = len(behavior_tokens)
        num_positions = num_behavior_tokens + 1  # Can insert before each token + after last token
        
        print(f'Behavior has {num_behavior_tokens} tokens, optimizing {num_positions} positions independently')
        
        # ========== Run independent optimization for each position ========== #
        position_results = {}
        
        for pos in range(num_positions):
            print(f'\n=== Optimizing Position {pos} (inserting {num_optim_tokens} tokens) ===')
            
            # Initialize adversarial tokens for this position (reset for each position)
            current_optim_ids = optim_ids.clone()
            
            # Logs for this position
            position_losses = []
            position_logit_losses = []  # Added for tracking logit losses
            position_attn_losses = []   # Added for tracking attention losses
            position_test_cases = []
            position_chosen_positions = []
            position_chosen_token_ids = []
            position_chosen_token_texts = []
            position_step_times = []
            
            # Optimize this position for num_steps
            for step in tqdm(range(num_steps), desc=f'Position {pos}'):
                start_time = time.time()
                
                # ========== Compute coordinate token_gradient ========== #
                # Create input based on position
                if pos == 0:
                    # Insert before first behavior token
                    behavior_input_ids = torch.cat([current_optim_ids, behavior_ids], dim=1)
                elif pos == num_behavior_tokens:
                    # Insert after last behavior token  
                    behavior_input_ids = torch.cat([behavior_ids, current_optim_ids], dim=1)
                else:
                    # Insert between tokens
                    behavior_before = behavior_tokens[:pos].unsqueeze(0)
                    behavior_after = behavior_tokens[pos:].unsqueeze(0)
                    behavior_input_ids = torch.cat([behavior_before, current_optim_ids, behavior_after], dim=1)
                
                # Create one-hot encoding for optimization
                optim_ids_onehot = torch.zeros((1, num_optim_tokens, vocab_size), device=device, dtype=model.dtype)
                optim_ids_onehot.scatter_(2, current_optim_ids.unsqueeze(2), 1.0).requires_grad_()
                optim_embeds = torch.matmul(optim_ids_onehot.squeeze(0), vocab_embeds).unsqueeze(0)

                # Create complete input based on position
                if pos == 0:
                    behavior_input_embeds = torch.cat([optim_embeds, behavior_embeds], dim=1)
                elif pos == num_behavior_tokens:
                    behavior_input_embeds = torch.cat([behavior_embeds, optim_embeds], dim=1)
                else:
                    behavior_before_embeds = embed_layer(behavior_tokens[:pos].unsqueeze(0))
                    behavior_after_embeds = embed_layer(behavior_tokens[pos:].unsqueeze(0))
                    behavior_input_embeds = torch.cat([behavior_before_embeds, optim_embeds, behavior_after_embeds], dim=1)

                # Complete input sequence
                input_embeds = torch.cat([before_embeds, behavior_input_embeds, after_embeds, target_embeds], dim=1)
                outputs = model(inputs_embeds=input_embeds)
                logits = outputs.logits

                # Compute loss
                tmp = input_embeds.shape[1] - target_embeds.shape[1]
                shift_logits = logits[..., tmp-1:-1, :].contiguous()
                shift_labels = target_ids
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                token_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]
                token_grad = token_grad / token_grad.norm(dim=-1, keepdim=True)

                # ========== Sample a batch of new tokens based on the coordinate gradient (like gcg.py) ========== #
                sampled_top_indices, token_val, token_pos = sample_control(
                    current_optim_ids.squeeze(0), 
                    token_grad.squeeze(0), 
                    search_width, 
                    topk=256, 
                    temp=1, 
                    not_allowed_tokens=not_allowed_tokens
                )

                # ========= Filter sampled_top_indices that aren't the same after detokenize->retokenize ========= #
                sampled_top_indices_text = tokenizer.batch_decode(sampled_top_indices)
                new_sampled_top_indices = []
                new_token_val, new_token_pos = [], []
                count = 0
                for j in range(len(sampled_top_indices_text)):
                    # tokenize again
                    tmp = tokenizer(sampled_top_indices_text[j], return_tensors="pt", add_special_tokens=False).to(device)['input_ids'][0]
                    # if the tokenized text is different, then skip
                    if not torch.equal(tmp, sampled_top_indices[j]):
                        count += 1
                        continue
                    else:
                        new_sampled_top_indices.append(sampled_top_indices[j])
                        new_token_val.append(token_val[j])
                        new_token_pos.append(token_pos[j])
                
                if len(new_sampled_top_indices) == 0:
                    print('All removals; defaulting to keeping all')
                    count = 0
                else:
                    sampled_top_indices = torch.stack(new_sampled_top_indices)
                    token_val = torch.stack(new_token_val)
                    token_pos = torch.stack(new_token_pos)
                
                if count >= search_width // 2:
                    print('\nLots of removals:', count)
                
                new_search_width = search_width - count

                # ========== Compute loss on these candidates and take the argmin ========== #
                # Create input embeds for all candidates based on current position
                sampled_top_embeds = embed_layer(sampled_top_indices)
                
                # Prepare embeddings based on position for all candidates
                if pos == 0:
                    behavior_input_embeds_candidates = torch.cat([
                        sampled_top_embeds,
                        behavior_embeds.repeat(new_search_width, 1, 1)
                    ], dim=1)
                elif pos == num_behavior_tokens:
                    behavior_input_embeds_candidates = torch.cat([
                        behavior_embeds.repeat(new_search_width, 1, 1),
                        sampled_top_embeds
                    ], dim=1)
                else:
                    behavior_before_embeds = embed_layer(behavior_tokens[:pos].unsqueeze(0))
                    behavior_after_embeds = embed_layer(behavior_tokens[pos:].unsqueeze(0))
                    behavior_input_embeds_candidates = torch.cat([
                        behavior_before_embeds.repeat(new_search_width, 1, 1),
                        sampled_top_embeds,
                        behavior_after_embeds.repeat(new_search_width, 1, 1)
                    ], dim=1)

                # Complete input sequences for all candidates
                input_embeds_candidates = torch.cat([
                    before_embeds.repeat(new_search_width, 1, 1),
                    behavior_input_embeds_candidates,
                    after_embeds.repeat(new_search_width, 1, 1),
                    target_embeds.repeat(new_search_width, 1, 1)
                ], dim=1)
                
                # Auto Find Batch Size for forward candidates (following gcg.py approach)
                candidate_loss, candidate_logit_loss, candidate_attn_loss = find_executable_batch_size(self.compute_candidates_loss, self.search_batch_size)(
                                                                                input_embeds_candidates, 
                                                                                target_ids, 
                                                                                pos, 
                                                                                before_embeds.shape[1], 
                                                                                behavior_embeds.shape[1], 
                                                                                after_embeds.shape[1], 
                                                                                num_optim_tokens
                                                                            )

                # ========== Update the optim_ids with the best candidate ========== #
                best_idx = candidate_loss.argmin()
                current_optim_ids = sampled_top_indices[best_idx].unsqueeze(0)
                
                # Create test case for this position
                if pos == 0:
                    test_case_ids = torch.cat([current_optim_ids, behavior_ids], dim=1)
                elif pos == num_behavior_tokens:
                    test_case_ids = torch.cat([behavior_ids, current_optim_ids], dim=1)
                else:
                    behavior_before = behavior_tokens[:pos].unsqueeze(0)
                    behavior_after = behavior_tokens[pos:].unsqueeze(0)
                    test_case_ids = torch.cat([behavior_before, current_optim_ids, behavior_after], dim=1)
                
                test_case = tokenizer.decode(test_case_ids[0])
                position_test_cases.append(test_case)
                
                # Store loss information using the correct indices
                current_loss = candidate_loss.min().item()
                current_logit_loss = candidate_logit_loss[best_idx].item()
                current_attn_loss = candidate_attn_loss[best_idx].item()
                
                position_losses.append(current_loss)
                position_logit_losses.append(current_logit_loss)
                position_attn_losses.append(current_attn_loss)
                
                chosen_slot = token_pos[best_idx].item()
                chosen_tok_id = current_optim_ids[0, chosen_slot].item()
                chosen_tok_txt = tokenizer.decode([chosen_tok_id])

                position_chosen_positions.append(chosen_slot)
                position_chosen_token_ids.append(chosen_tok_id)
                position_chosen_token_texts.append(chosen_tok_txt)
                position_step_times.append(time.time() - start_time)

                # Clean up
                del input_embeds, input_embeds_candidates, outputs, logits, shift_logits, candidate_loss, candidate_logit_loss, candidate_attn_loss
                torch.cuda.empty_cache()
                gc.collect()

                # ========== Evaluation and Early Stopping ========== #
                if (step % eval_steps == 0) or (step == num_steps - 1):
                    p_output = f'\n===>Position {pos}, Step {step}\n===>Test Case: {test_case}\n===>Total Loss: {current_loss:.4f}, Logit Loss: {current_logit_loss:.4f}, Attn Loss: {current_attn_loss:.4f}, Time: {sum(position_step_times):.2f}s'
                    
                    if eval_with_check_refusal and current_loss < check_refusal_min_loss:
                        input_str = self.template.format(instruction=test_case)
                        is_refusal, completions, _ = check_refusal_completions(model, tokenizer, inputs=[input_str])
                        p_output += f"\n\n===>Completion: {completions[0]}"
                        if not is_refusal[0]:
                            break

                    if verbose:
                        print(p_output)

                if early_stopping and current_loss < early_stopping_min_loss:
                    print(f'Early stopping at position {pos}, step {step} with loss {current_loss}')
                    break
            
            # Store results for this position
            position_results[pos] = {
                'final_loss': float(position_losses[-1]) if position_losses else float('inf'),
                'final_logit_loss': float(position_logit_losses[-1]) if position_logit_losses else float('inf'),
                'final_attn_loss': float(position_attn_losses[-1]) if position_attn_losses else float('inf'),
                'final_test_case': position_test_cases[-1] if position_test_cases else behavior,
                'all_losses': [float(x) for x in position_losses],
                'all_logit_losses': [float(x) for x in position_logit_losses],
                'all_attn_losses': [float(x) for x in position_attn_losses],
                'all_test_cases': position_test_cases,
                'chosen_positions': position_chosen_positions,
                'chosen_token_ids': position_chosen_token_ids,
                'chosen_token_texts': position_chosen_token_texts,
                'step_times': [float(t) for t in position_step_times],
            }
            
            print(f'Position {pos} final - Total: {position_results[pos]["final_loss"]:.4f}, Logit: {position_results[pos]["final_logit_loss"]:.4f}, Attn: {position_results[pos]["final_attn_loss"]:.4f}')

        # ========== Find overall best position ========== #
        best_pos = min(position_results.keys(), key=lambda p: position_results[p]['final_loss'])
        best_result = position_results[best_pos]
        
        print(f'\nBest overall position: {best_pos} with Total Loss: {best_result["final_loss"]:.4f}, Logit Loss: {best_result["final_logit_loss"]:.4f}, Attn Loss: {best_result["final_attn_loss"]:.4f}')
        
        # ========== Prepare logs ========== #
        logs = {
            'best_position': best_pos,
            'best_final_loss': best_result['final_loss'],
            'best_final_logit_loss': best_result['final_logit_loss'],
            'best_final_attn_loss': best_result['final_attn_loss'],
            'best_final_test_case': best_result['final_test_case'],
            'num_positions': num_positions,
            'position_results': position_results,
            'num_behavior_tokens': num_behavior_tokens,
            'num_optim_tokens': num_optim_tokens
        }

        return best_result['final_test_case'], logs

    def compute_candidates_loss(self, search_batch_size, input_embeds, target_ids, pos, before_len, behavior_len, after_len, num_optim_tokens):
        """Compute loss for candidate sequences with attention loss (following gcg_onlyone.py approach)"""
        if self.search_batch_size != search_batch_size:
            print(f"INFO: Setting candidates search_batch_size to {search_batch_size})")
            self.search_batch_size = search_batch_size
            torch.cuda.empty_cache()
            gc.collect()

        all_loss = []
        all_logit_loss = []
        all_attn_loss = []
        
        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i+search_batch_size]
                outputs = self.model(inputs_embeds=input_embeds_batch, output_attentions=True)
                
                # Extract attention information
                total_layers = len(outputs.attentions)
                upper_half_start = total_layers // 2  # 상위 절반 시작점
                
                logits = outputs.logits
                attentions = torch.stack(outputs.attentions[upper_half_start:])

                # Compute logit loss
                tmp = input_embeds_batch.shape[1] - target_ids.shape[1]
                shift_logits = logits[..., tmp-1:-1, :].contiguous()
                shift_labels = target_ids.repeat(input_embeds_batch.shape[0], 1)
                loss_fct = CrossEntropyLoss(reduction='none')
                logit_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                logit_loss = logit_loss.view(input_embeds_batch.shape[0], -1).mean(dim=1)
                
                # Compute attention loss for each sample in batch
                n_targeted_layers, batch_size_current, n_heads, query_len, key_len = attentions.shape
                device = attentions.device
                
                # Calculate positions based on insertion position
                if pos == 0:
                    # [before] [control] [behavior] [after] [target]
                    query_after_start = before_len + num_optim_tokens + behavior_len
                    key_control_start = before_len
                elif pos >= behavior_len:  # Insert at end
                    # [before] [behavior] [control] [after] [target]  
                    query_after_start = before_len + behavior_len + num_optim_tokens
                    key_control_start = before_len + behavior_len
                else:
                    # [before] [behavior_before] [control] [behavior_after] [after] [target]
                    behavior_before_len = pos
                    query_after_start = before_len + behavior_before_len + num_optim_tokens + (behavior_len - behavior_before_len)
                    key_control_start = before_len + behavior_before_len

                query_after_end = query_after_start + after_len
                chat_positions = list(range(query_after_start, min(query_after_end, query_len)))
                key_control_end = key_control_start + num_optim_tokens

                # Calculate attention loss for each sample
                attn_losses = []
                for b in range(batch_size_current):
                    if len(chat_positions) > 0 and key_control_end <= key_len:
                        # Extract relevant attention: (n_targeted_layers, n_heads, len(chat_positions), key_control_tokens)
                        relevant_attn = attentions[:, b, :, chat_positions, key_control_start:key_control_end]
                        
                        # Apply sequential mean pooling:
                        # 1. key_control dimension (axis=-1) 
                        relevant_attn = relevant_attn.mean(dim=-1)  # (n_targeted_layers, n_heads, len(chat_positions))
                        
                        # 2. n_heads dimension (axis=-2)
                        relevant_attn = relevant_attn.mean(dim=-2)  # (n_targeted_layers, len(chat_positions))
                        
                        # 3. chat_positions dimension (axis=-1) 
                        relevant_attn = relevant_attn.mean(dim=-1)  # (n_targeted_layers,)
                        
                        # 4. n_targeted_layers dimension (axis=-1)
                        attn_loss_sample = relevant_attn.mean(dim=-1)  # scalar
                    else:
                        attn_loss_sample = torch.tensor(0.0, device=device)
                    attn_losses.append(attn_loss_sample)
                
                attn_loss = torch.stack(attn_losses)
                
                # Combined loss
                total_loss = logit_loss
                
                all_loss.append(total_loss)
                all_logit_loss.append(logit_loss) 
                all_attn_loss.append(attn_loss)
        
                del outputs, logits, attentions, total_loss, logit_loss, attn_loss
                torch.cuda.empty_cache()
                gc.collect()
            
        return torch.cat(all_loss, dim=0), torch.cat(all_logit_loss, dim=0), torch.cat(all_attn_loss, dim=0)