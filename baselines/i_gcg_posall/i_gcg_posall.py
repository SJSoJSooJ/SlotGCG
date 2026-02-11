import gc
import json
import os
import random
import copy
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
# https://huggingface.co/docs/accelerate/v0.11.0/en/memory#accelerate.find_executable_batch_size
from accelerate.utils import find_executable_batch_size
from ..baseline import SingleBehaviorRedTeamingMethod
from ..model_utils import get_template
from ..check_refusal_utils import check_refusal_completions
from .i_gcg_posall_utils import sample_control, get_nonascii_toks, apply_manual_multicoord_update
import time

# ============================== I-GCG CLASS DEFINITION ============================== #
class I_GCG_posall(SingleBehaviorRedTeamingMethod):
    def __init__(self,
                 target_model,
                 targets_path,
                 num_steps=50,
                 adv_string_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
                 allow_non_ascii=False,
                 search_width=512,
                 topk=256,
                 K=7,
                 use_prefix_cache=True,
                 eval_steps=10,
                 eval_with_check_refusal=False,
                 check_refusal_min_loss=0.1,
                 early_stopping=False,
                 early_stopping_min_loss=0.1,
                 starting_search_batch_size=None,
                 num_adv_string=None,  # 새로운 파라미터: adversarial string 개수
                 **model_kwargs):
        """
        :param target_model: a dictionary specifying the target model (kwargs to load_model_and_tokenizer)
        :param targets_path: the path to the targets JSON file
        :param num_steps: the number of optimization steps to use
        :param adv_string_init: the initial adversarial string to start with
        :param allow_non_ascii: whether to allow non-ascii characters when sampling candidate updates
        :param search_width: the number of candidates to sample at each step
        :param topk: top-k tokens to sample from
        :param K: number of top candidates for multi-coordinate update
        :param use_prefix_cache: whether to use KV cache for static prefix tokens (WARNING: Not all models support this)
        :param eval_steps: the number of steps between each evaluation
        :param eval_with_check_refusal: whether to generate and check for refusal every eval_steps; used for early stopping
        :param check_refusal_min_loss: a minimum loss before checking for refusal
        :param early_stopping: an alternate form of early stopping, based solely on early_stopping_min_loss
        :param early_stopping_min_loss: the loss at which to stop optimization if early_stopping is True
        :param starting_search_batch_size: the initial search_batch_size; will auto reduce later in case of OOM
        :param num_adv_string: number of adversarial strings to optimize at different positions
        """
        super().__init__(target_model=target_model, **model_kwargs)
        self.num_steps = num_steps
        self.adv_string_init = adv_string_init
        self.allow_non_ascii = allow_non_ascii
        self.search_width = search_width
        self.topk = topk
        self.K = K
        self.use_prefix_cache = use_prefix_cache
        self.num_adv_string = num_adv_string  # 새로운 속성
        
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
        각 위치별로 독립적으로 adversarial suffix를 최적화하여 테스트 케이스를 생성합니다.

        :param behavior: a dictionary specifying the behavior to generate test cases for
        :param verbose: whether to print progress
        :return: 각 위치별 최적화된 test case들과 logs의 딕셔너리
        """
        
        # ========== Behavior and Target str ==========
        behavior = behavior_dict['Behavior']
        context_str = behavior_dict['ContextString']
        behavior_id = behavior_dict['BehaviorID']
        
        target = self.behavior_id_to_target[behavior_id]
        user_prompt = behavior + ' '
        if context_str:
            user_prompt = f"{context_str}\n\n---\n\n{user_prompt}"

        print(f'Behavior: {behavior_id} || Target: {target}')
        
        ### Targeted Model and Tokenizer ###
        model = self.model
        device = model.device
        tokenizer = self.tokenizer

        # num_adv_string이 지정되지 않았다면 기본값 설정
        if self.num_adv_string is None:
            # user_prompt의 토큰 길이를 기준으로 설정
            user_prompt_tokens = tokenizer(user_prompt, return_tensors="pt", add_special_tokens=False)['input_ids']
            self.num_adv_string = len(user_prompt_tokens[0])
        
        # 각 위치별로 최적화를 수행할 결과 저장소
        position_results = {}
        
        # 각 위치에서 adversarial string을 삽입하여 최적화
        for position in range(self.num_adv_string):
            print(f"\n========== Optimizing at position {position} ==========")
            
            # 현재 위치에서 최적화 수행
            test_case, logs = self._optimize_at_position(
                user_prompt, target, position, verbose
            )
            
            position_results[position] = {
                'test_case': test_case,
                'logs': logs,
                'position': position
            }
            
            if verbose:
                print(f"Position {position} - Final test case: {test_case}")
                print(f"Position {position} - Final loss: {logs['final_loss']}")
        
        return position_results

    def _optimize_at_position(self, user_prompt, target, position, verbose=False):
        """
        특정 위치에서 adversarial suffix를 최적화합니다.
        """
        # starting search_batch_size, will auto reduce batch_size later if go OOM
        self.search_batch_size = self.starting_search_batch_size if self.starting_search_batch_size else self.search_width

        ### Targeted Model and Tokenizer ###
        model = self.model
        device = model.device
        tokenizer = self.tokenizer

        ### I-GCG hyperparams ###
        num_steps = self.num_steps
        adv_string_init = self.adv_string_init
        allow_non_ascii = self.allow_non_ascii
        search_width = self.search_width
        topk = self.topk
        K = self.K
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
        not_allowed_tokens = torch.unique(not_allowed_tokens)

        # ========== 위치별 입력 구성 ========== #
        # user_prompt를 토큰화
        user_prompt_tokens = tokenizer(user_prompt, return_tensors="pt", add_special_tokens=False).to(device)['input_ids'][0]
        
        # 현재 위치에서 user_prompt를 분할
        if position == 0:
            # 맨 앞에 삽입
            prefix_tokens = torch.tensor([], device=device, dtype=torch.long)
            suffix_tokens = user_prompt_tokens
        elif position >= len(user_prompt_tokens):
            # 맨 뒤에 삽입
            prefix_tokens = user_prompt_tokens
            suffix_tokens = torch.tensor([], device=device, dtype=torch.long)
        else:
            # 중간에 삽입
            prefix_tokens = user_prompt_tokens[:position]
            suffix_tokens = user_prompt_tokens[position:]

        # ========== setup optim_ids and cached components ========== #
        optim_ids = tokenizer(adv_string_init, return_tensors="pt", add_special_tokens=False).to(device)['input_ids']
        num_optim_tokens = len(optim_ids[0])

        # 캐시할 구성 요소들 준비
        cache_input_ids = []
        
        # before_tc
        before_tc_ids = tokenizer([before_tc], padding=False)['input_ids'][0]
        cache_input_ids.append(torch.tensor(before_tc_ids, device=device).unsqueeze(0))
        
        # prefix part of user_prompt
        if len(prefix_tokens) > 0:
            cache_input_ids.append(prefix_tokens.unsqueeze(0))
        
        # suffix part of user_prompt (will be added after optim_ids)
        suffix_ids = suffix_tokens.unsqueeze(0) if len(suffix_tokens) > 0 else None
        
        # after_tc and target
        after_tc_ids = tokenizer([after_tc], padding=False, add_special_tokens=False)['input_ids'][0]
        target_ids = tokenizer([target], padding=False, add_special_tokens=False)['input_ids'][0]
        cache_input_ids.append(torch.tensor(after_tc_ids, device=device).unsqueeze(0))
        cache_input_ids.append(torch.tensor(target_ids, device=device).unsqueeze(0))
        
        # 임베딩 변환
        cache_embeds = [embed_layer(input_ids) for input_ids in cache_input_ids]
        
        if len(prefix_tokens) > 0:
            before_tc_embeds, prefix_embeds, after_tc_embeds, target_embeds = cache_embeds
        else:
            before_tc_embeds, after_tc_embeds, target_embeds = cache_embeds
            prefix_embeds = None
        
        suffix_embeds = embed_layer(suffix_ids) if suffix_ids is not None else None
        target_ids_tensor = torch.tensor(target_ids, device=device).unsqueeze(0)
        
        if self.use_prefix_cache:
            # precompute KV cache for everything before the optimized tokens
            if prefix_embeds is not None:
                input_embeds = torch.cat([before_tc_embeds, prefix_embeds], dim=1)
            else:
                input_embeds = before_tc_embeds
            
            with torch.no_grad():
                outputs = model(inputs_embeds=input_embeds, use_cache=True)
                self.prefix_cache = outputs.past_key_values

        # ========== run optimization ========== #
        all_losses = []
        all_test_cases = []
        chosen_positions = []
        chosen_token_ids = []
        chosen_token_texts = []  
        step_times = []

        for i in tqdm(range(num_steps), desc=f"Position {position}"):
            start_time = time.time()
            
            # ========== compute coordinate token_gradient ========== #
            optim_ids_onehot = torch.zeros((1, num_optim_tokens, vocab_size), device=device, dtype=model.dtype)
            optim_ids_onehot.scatter_(2, optim_ids.unsqueeze(2), 1.0).requires_grad_()
            optim_embeds = torch.matmul(optim_ids_onehot.squeeze(0), vocab_embeds).unsqueeze(0)

            # forward pass with position-aware input construction
            if self.use_prefix_cache:
                # optim_embeds + suffix + after_tc + target
                input_parts = [optim_embeds]
                if suffix_embeds is not None:
                    input_parts.append(suffix_embeds)
                input_parts.extend([after_tc_embeds, target_embeds])
                input_embeds = torch.cat(input_parts, dim=1)
                outputs = model(inputs_embeds=input_embeds, past_key_values=self.prefix_cache)
            else:
                # before_tc + prefix + optim_embeds + suffix + after_tc + target
                input_parts = [before_tc_embeds]
                if prefix_embeds is not None:
                    input_parts.append(prefix_embeds)
                input_parts.append(optim_embeds)
                if suffix_embeds is not None:
                    input_parts.append(suffix_embeds)
                input_parts.extend([after_tc_embeds, target_embeds])
                input_embeds = torch.cat(input_parts, dim=1)
                outputs = model(inputs_embeds=input_embeds)
            
            logits = outputs.logits

            # compute loss
            tmp = input_embeds.shape[1] - target_embeds.shape[1]
            shift_logits = logits[..., tmp-1:-1, :].contiguous()
            shift_labels = target_ids_tensor
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            token_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]
            token_grad = token_grad / token_grad.norm(dim=-1, keepdim=True)

            # ========== Sample a batch of new tokens based on the coordinate gradient ========== #
            sampled_top_indices, token_val, token_pos = sample_control(
                optim_ids.squeeze(0), token_grad.squeeze(0), search_width, 
                topk=topk, temp=1, not_allowed_tokens=not_allowed_tokens
            )

            # ========= Filter sampled_top_indices that aren't the same after detokenize->retokenize ========= #
            sampled_top_indices_text = tokenizer.batch_decode(sampled_top_indices)
            new_sampled_top_indices = []
            new_token_val, new_token_pos = [], []
            count = 0
            for j in range(len(sampled_top_indices_text)):
                tmp = tokenizer(sampled_top_indices_text[j], return_tensors="pt", add_special_tokens=False).to(device)['input_ids'][0]
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

            # ========== Compute loss on candidates with position-aware construction ========== #
            sampled_top_embeds = embed_layer(sampled_top_indices)
            
            if self.use_prefix_cache:
                # 후보들을 배치로 구성
                candidate_input_parts = [sampled_top_embeds]
                if suffix_embeds is not None:
                    candidate_input_parts.append(suffix_embeds.repeat(new_search_width, 1, 1))
                candidate_input_parts.extend([
                    after_tc_embeds.repeat(new_search_width, 1, 1),
                    target_embeds.repeat(new_search_width, 1, 1)
                ])
                candidate_input_embeds = torch.cat(candidate_input_parts, dim=1)
            else:
                candidate_input_parts = [before_tc_embeds.repeat(new_search_width, 1, 1)]
                if prefix_embeds is not None:
                    candidate_input_parts.append(prefix_embeds.repeat(new_search_width, 1, 1))
                candidate_input_parts.append(sampled_top_embeds)
                if suffix_embeds is not None:
                    candidate_input_parts.append(suffix_embeds.repeat(new_search_width, 1, 1))
                candidate_input_parts.extend([
                    after_tc_embeds.repeat(new_search_width, 1, 1),
                    target_embeds.repeat(new_search_width, 1, 1)
                ])
                candidate_input_embeds = torch.cat(candidate_input_parts, dim=1)
            
            # Auto Find Batch Size for forward candidates
            single_losses = find_executable_batch_size(
                self.compute_candidates_loss, self.search_batch_size
            )(candidate_input_embeds, target_ids_tensor)

            # Apply manual multi-coordinate update
            k = K
            losses_temp, idx1 = torch.sort(single_losses, descending=False)
            
            new_adv_suffix = tokenizer.batch_decode(sampled_top_indices)
            adv_suffix = tokenizer.decode(optim_ids.squeeze(0), skip_special_tokens=True)
            
            all_new_adv_suffix = apply_manual_multicoord_update(tokenizer, new_adv_suffix, adv_suffix, idx1[:k])
            
            # Choose best multi-coordinate candidate
            if all_new_adv_suffix:
                multi_candidates_ids = []
                for suffix in all_new_adv_suffix:
                    suffix_ids = tokenizer(suffix, return_tensors="pt", add_special_tokens=False).to(device)['input_ids']
                    if suffix_ids.shape[1] == num_optim_tokens:
                        multi_candidates_ids.append(suffix_ids)
                
                if multi_candidates_ids:
                    multi_candidates_ids = torch.cat(multi_candidates_ids, dim=0)
                    multi_embeds = embed_layer(multi_candidates_ids)
                    b = multi_candidates_ids.shape[0]
                    
                    # Position-aware multi-candidate evaluation
                    if self.use_prefix_cache:
                        mc_input_parts = [multi_embeds]
                        if suffix_embeds is not None:
                            mc_input_parts.append(suffix_embeds.repeat(b, 1, 1))
                        mc_input_parts.extend([
                            after_tc_embeds.repeat(b, 1, 1),
                            target_embeds.repeat(b, 1, 1)
                        ])
                        mc_input_embeds = torch.cat(mc_input_parts, dim=1)
                    else:
                        mc_input_parts = [before_tc_embeds.repeat(b, 1, 1)]
                        if prefix_embeds is not None:
                            mc_input_parts.append(prefix_embeds.repeat(b, 1, 1))
                        mc_input_parts.append(multi_embeds)
                        if suffix_embeds is not None:
                            mc_input_parts.append(suffix_embeds.repeat(b, 1, 1))
                        mc_input_parts.extend([
                            after_tc_embeds.repeat(b, 1, 1),
                            target_embeds.repeat(b, 1, 1)
                        ])
                        mc_input_embeds = torch.cat(mc_input_parts, dim=1)

                    multi_losses = find_executable_batch_size(
                        self.compute_candidates_loss, self.search_batch_size
                    )(mc_input_embeds, target_ids_tensor)

                    best_multi_idx = multi_losses.argmin()
                    base_ids = optim_ids[0].detach().clone()
                    optim_ids = multi_candidates_ids[best_multi_idx].unsqueeze(0)
                    current_loss = multi_losses.min().item()
                    
                    # Track changes for logging
                    changed = torch.nonzero(optim_ids[0].ne(base_ids), as_tuple=True)[0].tolist()
                    chosen_positions.append(changed)
                    chosen_token_ids.append([optim_ids[0, c].item() for c in changed])
                    chosen_token_texts.append([tokenizer.decode([optim_ids[0, c].item()]) for c in changed])
                else:
                    # Fallback to single best candidate
                    best_idx = single_losses.argmin()
                    base_ids = optim_ids[0].detach().clone()
                    optim_ids = sampled_top_indices[best_idx].unsqueeze(0)
                    current_loss = single_losses.min().item()
                    
                    changed = torch.nonzero(optim_ids[0].ne(base_ids), as_tuple=True)[0].tolist()
                    chosen_positions.append(changed)
                    chosen_token_ids.append([optim_ids[0, c].item() for c in changed])
                    chosen_token_texts.append([tokenizer.decode([optim_ids[0, c].item()]) for c in changed])
            else:
                # Fallback to single best candidate
                best_idx = single_losses.argmin()
                base_ids = optim_ids[0].detach().clone()
                optim_ids = sampled_top_indices[best_idx].unsqueeze(0)
                current_loss = single_losses.min().item()
                
                changed = torch.nonzero(optim_ids[0].ne(base_ids), as_tuple=True)[0].tolist()
                chosen_positions.append(changed)
                chosen_token_ids.append([optim_ids[0, c].item() for c in changed])
                chosen_token_texts.append([tokenizer.decode([optim_ids[0, c].item()]) for c in changed])
            
            # 최종 test case 구성 (위치에 맞게)
            optimized_suffix = tokenizer.decode(optim_ids[0], skip_special_tokens=True)
            if position == 0:
                # 맨 앞에 삽입된 경우
                test_case = optimized_suffix + tokenizer.decode(suffix_tokens, skip_special_tokens=True)
            elif len(suffix_tokens) == 0:
                # 맨 뒤에 삽입된 경우
                test_case = tokenizer.decode(prefix_tokens, skip_special_tokens=True) + optimized_suffix
            else:
                # 중간에 삽입된 경우
                test_case = (tokenizer.decode(prefix_tokens, skip_special_tokens=True) + 
                           optimized_suffix + 
                           tokenizer.decode(suffix_tokens, skip_special_tokens=True))
            
            all_test_cases.append(test_case)
            all_losses.append(current_loss)
            step_times.append(time.time() - start_time)

            # ========== Eval and Early Stopping ========== #
            if (i % eval_steps == 0) or (i == num_steps - 1):
                p_output = f'\n===>Position {position}, Step {i}\n===>Test Case: {test_case}\n===>Loss: {current_loss}, Time: {sum(step_times):.2f}s'
                if eval_with_check_refusal and current_loss < check_refusal_min_loss:
                    input_str = self.template.format(instruction=test_case)
                    is_refusal, completions, _ = check_refusal_completions(model, tokenizer, inputs=[input_str])
                    p_output += f"\n\n===>Completion: {completions[0]}"
                    if not is_refusal[0]:
                        break

                if verbose:
                    print(p_output)

            if early_stopping and current_loss < early_stopping_min_loss:
                print(f'Early stopping at position {position}, step {i} with loss {current_loss}')
                break

            # 메모리 정리
            del candidate_input_embeds, sampled_top_embeds, logits, shift_logits, loss
            torch.cuda.empty_cache()
            gc.collect()

        logs = {
            'final_loss': current_loss, 
            'all_losses': all_losses, 
            'all_test_cases': all_test_cases, 
            'step_times': step_times,
            'chosen_positions': chosen_positions, 
            'chosen_token_ids': chosen_token_ids, 
            'chosen_token_texts': chosen_token_texts,
            'position': position
        }

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