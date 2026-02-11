# TODO: support multi-node; current implementation with Ray only work for 1 node
import os
import json
from typing import List
import random
import torch
import numpy as np
from tqdm import tqdm
from ..baseline import RedTeamingMethod
from ..check_refusal_utils import check_refusal_completions
# from ..eval_utils import jailbroken
from .attngcg_posinit_attention_ray_actors import GCGModelWorkerPosinit, sample_control, get_nonascii_toks, interleave_behavior_and_controls, generate_positions, scale_positions_for_behavior
import ray
import time

# ============================== EnsembleGCGPosinit CLASS DEFINITION ============================== #
class EnsembleAttnGCG_posinit_attention(RedTeamingMethod):
    use_ray = True

    def __init__(self,
                 target_models,
                 targets_path=None,
                 num_steps=50,
                 adv_string_init="!",
                 num_adv_string=20,
                 attention_temp=8,
                 attention_weight=100,
                 allow_non_ascii=False,
                 search_width=512,
                 use_prefix_cache=True,
                 eval_steps=10,
                 progressive_behaviors=True,
                 **kwargs):
        """
        :param target_models: a list of target model configurations (dictionaries containing num_gpus and kwargs for GCGModelWorker)
        :param targets_path: the path to the targets JSON file
        :param num_steps: the number of optimization steps to use
        :param adv_string_init: the initial adversarial string to start with (single token)
        :param num_adv_string: number of adversarial tokens to insert
        :param allow_non_ascii: whether to allow non-ascii characters when sampling candidate updates
        :param search_width: the number of candidates to sample at each step
        :param use_prefix_cache: whether to use KV cache for static prefix tokens (WARNING: Not all models support this)
        :param eval_steps: the number of steps between each evaluation
        :param progressive_behaviors: whether to progressively add behaviors to optimize or optimize all at once
        """
        self.num_steps = num_steps
        self.adv_string_init = adv_string_init
        self.num_adv_string = num_adv_string
        self.allow_non_ascii = allow_non_ascii
        self.search_width = search_width
        self.use_prefix_cache = use_prefix_cache
        self.attention_temp = attention_temp
        self.attention_weight = attention_weight

        if targets_path is None:
            raise ValueError("targets_path must be specified")
        with open(targets_path, 'r', encoding='utf-8') as file:
            self.behavior_id_to_target = json.load(file)
        self.targets_path = targets_path

        self.eval_steps = eval_steps
        self.progressive_behaviors = progressive_behaviors

        ## Init workers accross gpus with ray
        num_gpus_list = [m.get('num_gpus', 1) for m in target_models]
        target_models = [m['target_model'] for m in target_models]
        self.models = target_models

        self.gcg_models = []
        # Allocate GPUs to workers based on their requirements of gpu
        for model_config, required_gpus in zip(target_models, num_gpus_list):            
            # Create a GCGModelWorkerPosinit instance with allocated GPUs
            worker_actor = GCGModelWorkerPosinit.options(num_gpus=required_gpus).remote(
                search_width=search_width,
                use_prefix_cache=use_prefix_cache, 
                attention_weight=attention_weight,
                **model_config
            )
            self.gcg_models.append(worker_actor)

        initialization_models = [w.is_initialized.remote() for w in self.gcg_models]
        print("\n".join(ray.get(initialization_models)))

        self.tokenizers = ray.get([w.get_attribute_.remote('tokenizer') for w in self.gcg_models])
        # Check that all tokenizers are the same type
        assert len(set([type(t) for t in self.tokenizers])) == 1, "All tokenizers must be the same type"
        self.sample_tokenizer = self.tokenizers[0]  # Used for initializing optimized tokens and decoding result; assumes all tokenizers are the same
        self.not_allowed_tokens = torch.unique(None if allow_non_ascii else get_nonascii_toks(self.sample_tokenizer))

    def initialize_positions(self, behavior_list, targets, current_behavior_indices):
        """
        Initialize optimal positions for adversarial tokens using attention-based approach.
        Computes attention scores for each behavior, performs linear interpolation to max length,
        and averages to get final position distribution.
        """
        print("Starting position initialization with attention-based approach...")
        
        # Get initial token ID
        init_optim_id = self.sample_tokenizer(" " + self.adv_string_init, return_tensors="pt", add_special_tokens=False)['input_ids']
        if init_optim_id.shape[1] != 1:
            print(f'[init_optim_id] fallback to original: "{self.adv_string_init}"')
            init_optim_id = self.sample_tokenizer(self.adv_string_init, return_tensors="pt", add_special_tokens=False)['input_ids']
        else:
            print(f'[init_optim_id] using with space: " {self.adv_string_init}"')

        # Get behavior lengths for the selected behaviors
        selected_behavior_lengths = []
        for behavior_idx in current_behavior_indices:
            behavior_length = len(self.sample_tokenizer(behavior_list[behavior_idx], add_special_tokens=False)['input_ids'])
            selected_behavior_lengths.append(behavior_length)
        
        # Use maximum length among selected behaviors
        max_behavior_length = max(selected_behavior_lengths)
        print(f"Selected behavior lengths: {selected_behavior_lengths}")
        print(f"Max behavior length: {max_behavior_length}")
        
        # Store for later scaling
        self.max_behavior_length = max_behavior_length
        
        pos_init_start = time.time()
        
        # Collect attention scores from all workers and behaviors
        all_interpolated_scores = []
        
        for worker_idx, worker in enumerate(self.gcg_models):
            print(f"\nProcessing worker {worker_idx+1}/{len(self.gcg_models)}")
            
            worker_interpolated_scores = []
            
            # Get attention scores for each selected behavior
            for i, behavior_idx in enumerate(current_behavior_indices):
                behavior_length = selected_behavior_lengths[i]
                
                try:
                    # Compute attention scores for this specific behavior
                    attention_scores = ray.get(worker.compute_attention_scores_single.remote(
                        init_optim_id, behavior_idx, behavior_length
                    ))
                    
                    print(f"  Behavior {behavior_idx} (len={behavior_length}): scores shape {len(attention_scores)}")
                    
                    # Linear interpolation to max_behavior_length + 1
                    if behavior_length == max_behavior_length:
                        # No interpolation needed
                        interpolated_scores = attention_scores
                    else:
                        # Create interpolation indices
                        original_indices = torch.linspace(0, behavior_length, behavior_length + 1)
                        target_indices = torch.linspace(0, behavior_length, max_behavior_length + 1)
                        
                        # Linear interpolation
                        interpolated_scores = torch.tensor(
                            np.interp(
                                target_indices.cpu().numpy(), 
                                original_indices.cpu().numpy(), 
                                attention_scores.detach().cpu().numpy()
                            ), 
                            device=attention_scores.device,
                            dtype=attention_scores.dtype
                        )
                    
                    worker_interpolated_scores.append(interpolated_scores)
                    print(f"  Interpolated to length {len(interpolated_scores)}")
                    
                except Exception as e:
                    print(f"  Error computing attention scores for behavior {behavior_idx}: {e}")
                    # Fallback to uniform distribution for this behavior
                    uniform_scores = torch.ones(max_behavior_length + 1) / (max_behavior_length + 1)
                    worker_interpolated_scores.append(uniform_scores)
            
            if worker_interpolated_scores:
                # Average across behaviors for this worker
                worker_avg_scores = torch.stack(worker_interpolated_scores).mean(dim=0)
                all_interpolated_scores.append(worker_avg_scores)
                print(f"  Worker {worker_idx+1} avg scores: {worker_avg_scores[:5].tolist()}...{worker_avg_scores[-5:].tolist()}")
        
        if not all_interpolated_scores:
            print("Warning: No valid attention scores computed. Using uniform distribution.")
            # Fallback to uniform distribution
            attention_probs = torch.ones(max_behavior_length + 1) / (max_behavior_length + 1)
        else:
            # Average attention scores across all workers
            avg_attention_scores = torch.stack(all_interpolated_scores).mean(dim=0)
            print(f"Final averaged scores: {avg_attention_scores[:5].tolist()}...{avg_attention_scores[-5:].tolist()}")
            
            # Apply temperature scaling and softmax
            attention_temp = self.attention_temp
            attention_probs = torch.softmax(avg_attention_scores / attention_temp, dim=0)
            print(f"Attention probabilities: {attention_probs[:5].tolist()}...{attention_probs[-5:].tolist()}")
        
        # Generate positions using the attention-based approach on max_behavior_length scale
        optim_pos = generate_positions(attention_probs, self.num_adv_string)
        optim_ids = init_optim_id.expand(-1, self.num_adv_string)
        
        print(f"Generated positions (on max length {max_behavior_length} scale): {optim_pos.tolist()}")

        pos_init_time = time.time() - pos_init_start

        print(f"Position initialization time: {pos_init_time:.2f} seconds")
        print(f"Note: During usage, positions will be scaled as: actual_pos = round(pos * behavior_len / {max_behavior_length})")
        
        return optim_ids, optim_pos, [pos_init_time]
    
    def update_positions_only(self, behavior_list, current_behavior_indices, existing_optim_ids, existing_optim_pos):
        """
        Update only the positions using attention-based approach for expanded behavior set.
        Keep the existing optim_ids unchanged but find optimal positions for the expanded behavior set.
        """
        print("Updating positions for expanded behavior set using attention-based approach...")
        
        # Get behavior lengths for all selected behaviors
        selected_behavior_lengths = []
        for behavior_idx in current_behavior_indices:
            behavior_length = len(self.sample_tokenizer(behavior_list[behavior_idx], add_special_tokens=False)['input_ids'])
            selected_behavior_lengths.append(behavior_length)
        
        # Use maximum length among selected behaviors
        new_max_behavior_length = max(selected_behavior_lengths)
        print(f"Selected behavior lengths: {selected_behavior_lengths}")
        print(f"New max behavior length: {new_max_behavior_length}")
        print(f"Previous max behavior length: {self.max_behavior_length}")
        print(f"Keeping existing tokens: {self.sample_tokenizer.decode(existing_optim_ids[0])}")
        
        # Update max_behavior_length
        self.max_behavior_length = new_max_behavior_length
        print(f"Updated all workers with new max_behavior_length: {self.max_behavior_length}")
        
        pos_update_start = time.time()
        
        # Use attention-based approach similar to initialization
        # Get initial token ID
        init_optim_id = self.sample_tokenizer(" " + self.adv_string_init, return_tensors="pt", add_special_tokens=False)['input_ids']
        if init_optim_id.shape[1] != 1:
            print(f'[init_optim_id] fallback to original: "{self.adv_string_init}"')
            init_optim_id = self.sample_tokenizer(self.adv_string_init, return_tensors="pt", add_special_tokens=False)['input_ids']
        else:
            print(f'[init_optim_id] using with space: " {self.adv_string_init}"')
        
        # Collect attention scores from all workers and behaviors
        all_interpolated_scores = []
        
        for worker_idx, worker in enumerate(self.gcg_models):
            print(f"\nProcessing worker {worker_idx+1}/{len(self.gcg_models)} for position update")
            
            worker_interpolated_scores = []
            
            # Get attention scores for each selected behavior
            for i, behavior_idx in enumerate(current_behavior_indices):
                behavior_length = selected_behavior_lengths[i]
                
                try:
                    # Compute attention scores for this specific behavior
                    attention_scores = ray.get(worker.compute_attention_scores_single.remote(
                        init_optim_id, behavior_idx, behavior_length
                    ))
                    
                    print(f"  Behavior {behavior_idx} (len={behavior_length}): scores shape {len(attention_scores)}")
                    
                    # Linear interpolation to new_max_behavior_length + 1
                    if behavior_length == new_max_behavior_length:
                        # No interpolation needed
                        interpolated_scores = attention_scores
                    else:
                        # Create interpolation indices
                        original_indices = torch.linspace(0, behavior_length, behavior_length + 1)
                        target_indices = torch.linspace(0, behavior_length, new_max_behavior_length + 1)
                        
                        # Linear interpolation
                        interpolated_scores = torch.tensor(
                            np.interp(
                                target_indices.cpu().numpy(), 
                                original_indices.cpu().numpy(), 
                                attention_scores.detach().cpu().numpy()
                            ), 
                            device=attention_scores.device,
                            dtype=attention_scores.dtype
                        )
                    
                    worker_interpolated_scores.append(interpolated_scores)
                    print(f"  Interpolated to length {len(interpolated_scores)}")
                    
                except Exception as e:
                    print(f"  Error computing attention scores for behavior {behavior_idx}: {e}")
                    # Fallback to uniform distribution for this behavior
                    uniform_scores = torch.ones(new_max_behavior_length + 1) / (new_max_behavior_length + 1)
                    worker_interpolated_scores.append(uniform_scores)
            
            if worker_interpolated_scores:
                # Average across behaviors for this worker
                worker_avg_scores = torch.stack(worker_interpolated_scores).mean(dim=0)
                all_interpolated_scores.append(worker_avg_scores)
        
        if not all_interpolated_scores:
            print("Warning: No valid attention scores computed. Using uniform distribution.")
            # Fallback to uniform distribution
            attention_probs = torch.ones(new_max_behavior_length + 1) / (new_max_behavior_length + 1)
        else:
            # Average attention scores across all workers
            avg_attention_scores = torch.stack(all_interpolated_scores).mean(dim=0)
            print(f"Final averaged scores: {avg_attention_scores[:5].tolist()}...{avg_attention_scores[-5:].tolist()}")
            
            # Apply temperature scaling and softmax
            attention_temp = self.attention_temp
            attention_probs = torch.softmax(avg_attention_scores / attention_temp, dim=0)
            print(f"Attention probabilities: {attention_probs[:5].tolist()}...{attention_probs[-5:].tolist()}")
        
        # Generate new positions using the attention-based approach
        num_existing_tokens = existing_optim_ids.shape[1]
        updated_optim_pos = generate_positions(attention_probs, num_existing_tokens)
        
        pos_update_time = time.time() - pos_update_start
        
        print(f"Updated positions: {updated_optim_pos.tolist()}")
        print(f"Position update time: {pos_update_time:.2f} seconds")
        
        return updated_optim_pos, [pos_update_time]

    def generate_test_cases(self, behaviors, verbose=False):
        """
        Generates test cases by jointly optimizing over the provided behaviors using position-based GCG.
        A test case for each behavior is returned at the end.

        :param behaviors: a list of behavior dictionaries specifying the behaviors to generate test cases for
        :param verbose: whether to print progress
        :return: a dictionary of test cases, where the keys are the behavior IDs and the values are lists of test cases
        """
        # ===================== generate test cases ===================== #
        behavior_list = [b['Behavior'] for b in behaviors]
        context_str_list = [b['ContextString'] for b in behaviors]
        behavior_id_list = [b['BehaviorID'] for b in behaviors]

        targets = [self.behavior_id_to_target[behavior_id] for behavior_id in behavior_id_list]
        
        # Adding context string to behavior string if available
        tmp = []
        for behavior, context_str in zip(behavior_list, context_str_list):
            if context_str:
                behavior = f"{context_str}\n\n---\n\n{behavior}"
            tmp.append(behavior)
        behavior_list = tmp

        total_behaviors = len(behavior_list)
        num_behaviors = 1 if self.progressive_behaviors else total_behaviors
        all_behavior_indices = list(range(total_behaviors))

        gcg_models = self.gcg_models
        # Init behaviors, targets, embeds
        ray.get([w.init_behaviors_and_targets.remote(behavior_list, targets) for w in gcg_models])        

        # Initialize positions for adversarial tokens
        current_behavior_indices = all_behavior_indices[:num_behaviors]
        optim_ids, optim_pos, pos_init_times = self.initialize_positions(behavior_list, targets, current_behavior_indices)
        num_optim_tokens = optim_ids.shape[1]

        logs = []
        all_losses = []
        all_logit_losses = []  # Add this
        all_attn_losses = []   # Add this
        all_test_cases = []
        all_optim_ids = []
        all_optim_strs = []
        all_optim_poses = []   
        chosen_positions = []      
        chosen_token_ids = []    
        chosen_token_texts = []
        steps_since_last_added_behavior = 0
        
        # ========== run optimization ========== #
        for i in tqdm(range(self.num_steps)):
            # select behaviors to use in search step
            current_behavior_indices = all_behavior_indices[:num_behaviors]

            # ========= Get normalized token gradients for all models and average across models ========= #
            futures = []
            for worker in gcg_models:
                futures.append(worker.get_token_gradient_posinit.remote(optim_ids, optim_pos, current_behavior_indices, self.max_behavior_length))
            grads = ray.get(futures)
            
            # clipping extra tokens
            min_embedding_value = min(grad.shape[2] for grad in grads)
            grads = [grad[:, :, :min_embedding_value] for grad in grads]
            token_grad = sum(grads) / len(grads)

            # ========= Select candidate replacements ========= #
            sampled_top_indices = sample_control(optim_ids.squeeze(0), token_grad.squeeze(0), self.search_width,
                                                 topk=256, temp=1, not_allowed_tokens=self.not_allowed_tokens)
            
            # ========= Filter sampled_top_indices that aren't the same after detokenize->retokenize ========= #
            sampled_top_indices_text = self.sample_tokenizer.batch_decode(sampled_top_indices)
            new_sampled_top_indices = []
            count = 0
            for j in range(len(sampled_top_indices_text)):
                # tokenize again
                tmp = self.sample_tokenizer(sampled_top_indices_text[j], return_tensors="pt",
                                            add_special_tokens=False).to(sampled_top_indices.device)['input_ids'][0]
                # if the tokenized text is different, then set the sampled_top_indices to padding_top_indices
                if not torch.equal(tmp, sampled_top_indices[j]):
                    count += 1
                    continue
                else:
                    new_sampled_top_indices.append(sampled_top_indices[j])
            
            sampled_top_indices = torch.stack(new_sampled_top_indices)
            
            if count >= self.search_width // 2:
                print('\nLots of removals:', count)

            # ========= Evaluate candidate optim ========= #
            futures = []
            for worker in gcg_models:
                future = worker.compute_loss_on_candidates_batch_posinit.remote(
                    sampled_top_indices, optim_pos, current_behavior_indices, self.max_behavior_length
                )
                futures.append(future)

            results = ray.get(futures)
            losses = torch.stack([result[0] for result in results])  # combined losses
            logit_losses = torch.stack([result[1] for result in results])  # logit losses
            attn_losses = torch.stack([result[2] for result in results])  # attention losses

            # mean across all behaviors
            model_losses = losses.mean(1)
            model_logit_losses = logit_losses.mean(1)
            model_attn_losses = attn_losses.mean(1)

            # mean across all models
            behavior_losses = losses.mean(0)
            behavior_logit_losses = logit_losses.mean(0)
            behavior_attn_losses = attn_losses.mean(0)

            # mean across all behaviors and models 
            avg_losses = losses.mean(1).mean(0)
            avg_logit_losses = logit_losses.mean(1).mean(0)
            avg_attn_losses = attn_losses.mean(1).mean(0)

            selected_id = avg_losses.argmin().item()

            # ========== Update the optim_ids with the best candidate ========== #
            base_ids = optim_ids[0].detach().clone()  # Store before update
            optim_ids = sampled_top_indices[selected_id].unsqueeze(0)

            # Track changes for logging - ensure tensors are on same device
            base_ids = base_ids.to(optim_ids.device)  # Move to same device
            changed = torch.nonzero(optim_ids[0].ne(base_ids), as_tuple=True)[0].tolist()
            chosen_positions.append(changed)
            chosen_token_ids.append([optim_ids[0, c].item() for c in changed])
            chosen_token_texts.append([self.sample_tokenizer.decode([optim_ids[0, c].item()]) for c in changed])

            # ========== Get test cases and append to lists for logging ========== #
            # Create test cases by inserting optimized tokens at specified positions
            test_cases = {}
            for b_idx, (behavior, b_id) in enumerate(zip(behavior_list, behavior_id_list)):
                # Tokenize behavior
                behavior_ids = self.sample_tokenizer(behavior, return_tensors="pt", add_special_tokens=False)['input_ids']
                behavior_length = behavior_ids.shape[1]
                
                # Clamp positions to valid range for this behavior
                scaled_optim_pos = scale_positions_for_behavior(optim_pos, behavior_length, self.max_behavior_length)
                
                # Ensure all tensors are on the same device (GPU)
                device = optim_ids.device  # Use the device of optim_ids
                behavior_ids = behavior_ids.to(device)
                scaled_optim_pos = scaled_optim_pos.to(device)
                
                test_case_ids = interleave_behavior_and_controls(
                    behavior_ids, optim_ids, scaled_optim_pos.unsqueeze(0)
                )[0]
                
                test_case = self.sample_tokenizer.decode(test_case_ids.cpu())  # Move to CPU only for decode
                test_cases[b_id] = [test_case]

            all_test_cases.append(test_cases)
            current_loss = avg_losses.min().item()
            current_logit_loss = avg_logit_losses[selected_id].item()
            current_attn_loss = avg_attn_losses[selected_id].item()

            all_losses.append(current_loss)
            all_logit_losses.append(current_logit_loss)
            all_attn_losses.append(current_attn_loss)
            
            all_optim_ids.append(optim_ids.tolist())
            all_optim_poses.append(optim_pos.tolist()) 

            # Store individual token strings for this step
            optim_token_strs = [self.sample_tokenizer.decode([token_id]) for token_id in optim_ids[0]]
            all_optim_strs.append(optim_token_strs)

            optim_str = self.sample_tokenizer.decode(optim_ids[0])

            if verbose and ((i % self.eval_steps == 0) or (i == self.num_steps - 1)):
                print("\n" + "="*50)
                print(f'\nStep {i} | Optimized Suffix: {optim_str}')
                print(f'Combined Loss: {current_loss} | Logit Loss: {current_logit_loss} | Attention Loss: {current_attn_loss}')
                print(f'Positions: {optim_pos.tolist()}')
                
                model_losses_info = []
                for model_idx in range(len(losses)):
                    model_name = self.models[model_idx]['model_name_or_path']
                    model_loss = model_losses[model_idx].min().item()
                    model_logit_loss = model_logit_losses[model_idx][selected_id].item()
                    model_attn_loss = model_attn_losses[model_idx][selected_id].item()
                    model_losses_info.append(f"->Model: {model_name} | Combined: {model_loss:.4f} | Logit: {model_logit_loss:.4f} | Attn: {model_attn_loss:.4f}")

                print("\n".join(model_losses_info))
                print()
            
            steps_since_last_added_behavior += 1
            if self.progressive_behaviors:
                # Test current optimized tokens and positions on all current behaviors
                # Get responses from all models for all current behaviors
                futures = []
                for behavior_idx in current_behavior_indices:
                    for worker in gcg_models:
                        future = worker.generate.remote(optim_ids, optim_pos, behavior_idx, self.max_behavior_length)
                        futures.append(future)
                
                responses = ray.get(futures)
                
                # Use the provided check_refusal_completions function
                refusal_flags, _, _ = check_refusal_completions(completions=responses)
                all_responses_non_refusal = not any(refusal_flags)
                
                if all_responses_non_refusal and len(current_behavior_indices) < total_behaviors:
                    print(f'\nAdding behavior to list of behaviors to optimize (case 3 - all {len(responses)} responses are non-refusal)\n')
                    num_behaviors += 1
                    steps_since_last_added_behavior = 0
                    
                    # Update positions only (keep existing optim_ids)
                    current_behavior_indices = all_behavior_indices[:num_behaviors]
                    print(f"Updating positions for {num_behaviors} behaviors (keeping existing tokens)...")
                    optim_pos, new_pos_update_times = self.update_positions_only(behavior_list, current_behavior_indices, optim_ids, optim_pos)
                    pos_init_times.extend(new_pos_update_times)

        
        logs = {
            'final_loss': current_loss, 
            'final_logit_loss': current_logit_loss,      # Add this
            'final_attn_loss': current_attn_loss,        # Add this
            'all_losses': all_losses, 
            'all_logit_losses': all_logit_losses,        # Add this
            'all_attn_losses': all_attn_losses,          # Add this
            'all_test_cases': all_test_cases,
            'all_optim_ids': all_optim_ids,
            'all_optim_strs': all_optim_strs,
            'all_optim_poses': all_optim_poses,
            'optim_pos': optim_pos.tolist(),
            'position_init_time': pos_init_times,
            'chosen_positions': chosen_positions,
            'chosen_token_ids': chosen_token_ids,
            'chosen_token_texts': chosen_token_texts,
            'num_behaviors': num_behaviors
        }
        return test_cases, logs

    # ===== Defining saving methods =====
    # Saves test cases for all behaviors into one test_cases.json file
    @staticmethod
    def get_output_file_path(save_dir, behavior_id, file_type, run_id=None):
        run_id = f"_{run_id}" if run_id else ""
        return os.path.join(save_dir, f"{file_type}{run_id}.json")
    
    def save_test_cases(self, save_dir, test_cases, logs=None, method_config=None, run_id=None):
        test_cases_save_path = self.get_output_file_path(save_dir, None, 'test_cases', run_id)
        logs_save_path = self.get_output_file_path(save_dir, None, 'logs', run_id)
        method_config_save_path = self.get_output_file_path(save_dir, None, 'method_config', run_id)

        os.makedirs(os.path.dirname(test_cases_save_path), exist_ok=True)
        if test_cases is not None:
            with open(test_cases_save_path, 'w', encoding='utf-8') as f:
                json.dump(test_cases, f, indent=4)

        if logs is not None:
            with open(logs_save_path, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=4)
        
        if method_config is not None:
            # mask token or api_key
            self._replace_tokens(method_config)
            with open(method_config_save_path, 'w', encoding='utf-8') as f:
                method_config["dependencies"] = {l.__name__: l.__version__ for l in self.default_dependencies}
                json.dump(method_config, f, indent=4)
    
    @staticmethod
    def merge_test_cases(save_dir):
        """
        Merges {save_dir}/test_cases_{run_id}.json into {save_dir}/test_cases.json
        """
        test_cases = {}
        logs = {}
        assert os.path.exists(save_dir), f"{save_dir} does not exist"

        # Find all subset directories
        run_ids = [d.split('_')[-1].split('.')[0] for d in os.listdir(save_dir) if d.startswith('test_cases_')]

        # Load test cases from each test_cases_{run_id}.json file and merge
        for run_id in run_ids:
            test_cases_path = os.path.join(save_dir, f'test_cases_{run_id}.json')
            with open(test_cases_path, 'r') as f:
                test_cases_part = json.load(f)
            for behavior_id in test_cases_part:
                if behavior_id in test_cases:
                    test_cases[behavior_id].extend(test_cases_part[behavior_id])
                else:
                    test_cases[behavior_id] = test_cases_part[behavior_id]
        
        # Save merged test cases and logs
        test_cases_save_path = os.path.join(save_dir, 'test_cases.json')
        with open(test_cases_save_path, 'w') as f:
            json.dump(test_cases, f, indent=4)
        
        print(f"Saved test_cases.json to {test_cases_save_path}")