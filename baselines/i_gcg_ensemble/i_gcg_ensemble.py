# i_gcg_ensemble.py
import os
import json
import copy
from typing import List
import random
import torch
from tqdm import tqdm
from ..baseline import RedTeamingMethod
from ..check_refusal_utils import check_refusal_completions
from .i_gcg_ray_actors import GCGModelWorker, sample_control, get_nonascii_toks, apply_manual_multicoord_update
import ray

class Ensemble_I_GCG(RedTeamingMethod):
    use_ray = True

    def __init__(self,
                 target_models,
                 targets_path=None,
                 num_steps=50,
                 adv_string_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
                 allow_non_ascii=False,
                 search_width=512,
                 K=7,  # Multi-coordinate update candidates
                 use_prefix_cache=True,
                 eval_steps=10,
                 progressive_behaviors=True,
                 **kwargs):
        """
        I-GCG Ensemble: Combines multi-model ensemble with multi-coordinate update
        
        :param target_models: a list of target model configurations
        :param targets_path: the path to the targets JSON file
        :param num_steps: the number of optimization steps to use
        :param adv_string_init: the initial adversarial string to start with
        :param allow_non_ascii: whether to allow non-ascii characters
        :param search_width: the number of candidates to sample at each step
        :param K: number of top candidates for multi-coordinate update
        :param use_prefix_cache: whether to use KV cache for static prefix tokens
        :param eval_steps: the number of steps between each evaluation
        :param progressive_behaviors: whether to progressively add behaviors
        """
        self.num_steps = num_steps
        self.adv_string_init = adv_string_init
        self.allow_non_ascii = allow_non_ascii
        self.search_width = search_width
        self.K = K  # Multi-coordinate update parameter
        self.use_prefix_cache = use_prefix_cache

        if targets_path is None:
            raise ValueError("targets_path must be specified")
        with open(targets_path, 'r', encoding='utf-8') as file:
            self.behavior_id_to_target = json.load(file)
        self.targets_path = targets_path

        self.eval_steps = eval_steps
        self.progressive_behaviors = progressive_behaviors

        # Init workers across gpus with ray
        num_gpus_list = [m.get('num_gpus', 1) for m in target_models]
        target_models = [m['target_model'] for m in target_models]
        self.models = target_models

        self.gcg_models = []
        for model_config, required_gpus in zip(target_models, num_gpus_list):            
            worker_actor = GCGModelWorker.options(num_gpus=required_gpus).remote(
                search_width=search_width,
                use_prefix_cache=use_prefix_cache, 
                **model_config
            )
            self.gcg_models.append(worker_actor)

        initialization_models = [w.is_initialized.remote() for w in self.gcg_models]
        print("\n".join(ray.get(initialization_models)))

        self.tokenizers = ray.get([w.get_attribute_.remote('tokenizer') for w in self.gcg_models])
        assert len(set([type(t) for t in self.tokenizers])) == 1, "All tokenizers must be the same type"
        self.sample_tokenizer = self.tokenizers[0]
        self.not_allowed_tokens = torch.unique(None if allow_non_ascii else get_nonascii_toks(self.sample_tokenizer))

    def generate_test_cases(self, behaviors, verbose=False):
        """
        Generates test cases with I-GCG multi-coordinate update strategy
        """
        # Prepare behaviors and targets
        behavior_list = [b['Behavior'] for b in behaviors]
        context_str_list = [b['ContextString'] for b in behaviors]
        behavior_id_list = [b['BehaviorID'] for b in behaviors]

        targets = [self.behavior_id_to_target[behavior_id] for behavior_id in behavior_id_list]
        behavior_list = [b + ' ' for b in behavior_list]  # add a space before the optimized string

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

        optim_ids = self.sample_tokenizer(self.adv_string_init, return_tensors="pt", add_special_tokens=False)['input_ids']
        num_optim_tokens = optim_ids.shape[1]

        logs = []
        all_losses = []
        all_test_cases = []
        steps_since_last_added_behavior = 0
        
        # Track changes for logging (I-GCG specific)
        chosen_positions = []
        chosen_token_ids = []
        chosen_token_texts = []
        
        # ========== run optimization with I-GCG ========== #
        for i in tqdm(range(self.num_steps)):
            # select behaviors to use in search step
            current_behavior_indices = all_behavior_indices[:num_behaviors]

            # ========= Get normalized token gradients for all models ========= #
            futures = []
            for worker in gcg_models:
                futures.append(worker.get_token_gradient.remote(optim_ids, num_optim_tokens, current_behavior_indices))
            grads = ray.get(futures)

            # clipping extra tokens
            min_embedding_value = min(grad.shape[2] for grad in grads)
            grads = [grad[:, :, :min_embedding_value] for grad in grads]
            token_grad = sum(grads) / len(grads)

            # ========= Select candidate replacements ========= #
            sampled_top_indices = sample_control(
                optim_ids.squeeze(0), 
                token_grad.squeeze(0), 
                self.search_width,
                topk=256, 
                temp=1, 
                not_allowed_tokens=self.not_allowed_tokens
            )
            
            # ========= Filter sampled_top_indices ========= #
            sampled_top_indices_text = self.sample_tokenizer.batch_decode(sampled_top_indices)
            new_sampled_top_indices = []
            count = 0
            for j in range(len(sampled_top_indices_text)):
                tmp = self.sample_tokenizer(
                    sampled_top_indices_text[j], 
                    return_tensors="pt",
                    add_special_tokens=False
                ).to(sampled_top_indices.device)['input_ids'][0]
                
                if not torch.equal(tmp, sampled_top_indices[j]):
                    count += 1
                    continue
                else:
                    new_sampled_top_indices.append(sampled_top_indices[j])
            
            if len(new_sampled_top_indices) == 0:
                print('All removals; defaulting to keeping all')
                sampled_top_indices = sampled_top_indices
            else:
                sampled_top_indices = torch.stack(new_sampled_top_indices)
            
            if count >= self.search_width // 2:
                print('\nLots of removals:', count)

            # ========= Evaluate single-coordinate candidates ========= #
            futures = []
            for worker in gcg_models:
                future = worker.compute_loss_on_candidates_batch.remote(sampled_top_indices, current_behavior_indices)
                futures.append(future)

            losses = ray.get(futures)
            losses = torch.stack(losses)
            
            # mean across all behaviors and models for single candidates
            single_losses = losses.mean(1).mean(0)
            
            # ========= Apply I-GCG Multi-coordinate Update ========= #
            losses_sorted, idx_sorted = torch.sort(single_losses, descending=False)

            # Convert sampled candidates to text for multi-coordinate processing
            new_adv_suffix = self.sample_tokenizer.batch_decode(sampled_top_indices)
            adv_suffix = self.sample_tokenizer.decode(optim_ids.squeeze(0), skip_special_tokens=True)

            # Apply manual multi-coordinate merging
            all_new_adv_suffix = apply_manual_multicoord_update(
                self.sample_tokenizer, 
                new_adv_suffix, 
                adv_suffix, 
                idx_sorted[:self.K]
            )

            # Track original ids for change detection
            base_ids = optim_ids[0].detach().clone()

            # Initialize variables for progressive behavior logic
            selected_id = None
            behavior_losses_for_progress = None

            # Evaluate multi-coordinate candidates if they exist
            if all_new_adv_suffix:
                multi_candidates_ids = []
                for suffix in all_new_adv_suffix:
                    suffix_ids = self.sample_tokenizer(
                        suffix, 
                        return_tensors="pt", 
                        add_special_tokens=False
                    )['input_ids']
                    if suffix_ids.shape[1] == num_optim_tokens:
                        multi_candidates_ids.append(suffix_ids)
                
                if multi_candidates_ids:
                    multi_candidates_ids = torch.cat(multi_candidates_ids, dim=0)
                    
                    # Evaluate multi-coordinate candidates across all models
                    futures = []
                    for worker in gcg_models:
                        future = worker.compute_loss_on_candidates_batch.remote(
                            multi_candidates_ids, 
                            current_behavior_indices
                        )
                        futures.append(future)
                    
                    multi_losses = ray.get(futures)
                    multi_losses = torch.stack(multi_losses)
                    
                    # mean across all behaviors and models
                    avg_multi_losses = multi_losses.mean(1).mean(0)
                    
                    # Choose best multi-coordinate candidate
                    best_multi_idx = avg_multi_losses.argmin()
                    optim_ids = multi_candidates_ids[best_multi_idx].unsqueeze(0)
                    current_loss = avg_multi_losses.min().item()
                    
                    # Track changes for logging
                    changed = torch.nonzero(optim_ids[0].ne(base_ids), as_tuple=True)[0].tolist()
                    chosen_positions.append(changed)
                    chosen_token_ids.append([optim_ids[0, c].item() for c in changed])
                    chosen_token_texts.append([self.sample_tokenizer.decode([optim_ids[0, c].item()]) for c in changed])
                    
                    # For progressive behavior logic - use multi_losses
                    model_losses = multi_losses.mean(1)  # [num_models]
                    behavior_losses_for_progress = multi_losses.mean(0)  # [num_behaviors, num_multi_candidates]
                    selected_id = best_multi_idx  # Use the selected multi-coordinate candidate index
                    
                else:
                    # Fallback to single best candidate
                    best_idx = single_losses.argmin()
                    optim_ids = sampled_top_indices[best_idx].unsqueeze(0)
                    current_loss = single_losses.min().item()
                    
                    # Track changes
                    changed = torch.nonzero(optim_ids[0].ne(base_ids), as_tuple=True)[0].tolist()
                    chosen_positions.append(changed)
                    chosen_token_ids.append([optim_ids[0, c].item() for c in changed])
                    chosen_token_texts.append([self.sample_tokenizer.decode([optim_ids[0, c].item()]) for c in changed])
                    
                    # For progressive behavior logic - use original losses
                    model_losses = losses.mean(1)
                    behavior_losses_for_progress = losses.mean(0)  # [num_behaviors, num_single_candidates]
                    selected_id = best_idx  # Use the selected single candidate index
                    
            else:
                # Fallback to single best candidate
                best_idx = single_losses.argmin()
                optim_ids = sampled_top_indices[best_idx].unsqueeze(0)
                current_loss = single_losses.min().item()
                
                # Track changes
                changed = torch.nonzero(optim_ids[0].ne(base_ids), as_tuple=True)[0].tolist()
                chosen_positions.append(changed)
                chosen_token_ids.append([optim_ids[0, c].item() for c in changed])
                chosen_token_texts.append([self.sample_tokenizer.decode([optim_ids[0, c].item()]) for c in changed])
                
                # For progressive behavior logic - use original losses
                model_losses = losses.mean(1)
                behavior_losses_for_progress = losses.mean(0)  # [num_behaviors, num_single_candidates]
                selected_id = best_idx  # Use the selected single candidate index

            # ========== Get test cases and append to lists for logging ========== #
            optim_str = self.sample_tokenizer.decode(optim_ids[0])
            test_cases = {b_id: [b + optim_str] for b, b_id in zip(behavior_list, behavior_id_list)}
            all_test_cases.append(test_cases)
            all_losses.append(current_loss)

            if verbose and ((i % self.eval_steps == 0) or (i == self.num_steps - 1)):
                print("\n" + "="*50)
                print(f'\nStep {i} | Optimized Suffix: {optim_str} | Avg Loss: {current_loss}')
                print(f'Changed positions: {changed if changed else "None"}')
                model_loss_strs = [
                    f"->Model: {self.models[j]['model_name_or_path']} | Loss {model_losses[j].min().item()}" 
                    for j in range(len(model_losses))
                ]
                print("\n".join(model_loss_strs))
                print()
            
            steps_since_last_added_behavior += 1
            if self.progressive_behaviors:

                # Test current optimized string on all current behaviors
                optim_str = self.sample_tokenizer.decode(optim_ids[0])
                
                # Get responses from all models for all current behaviors
                futures = []
                for behavior_idx in current_behavior_indices:
                    for worker in gcg_models:
                        future = worker.generate.remote(optim_str, behavior_idx)
                        futures.append(future)
                
                responses = ray.get(futures)
                
                # Use the provided check_refusal_completions function
                refusal_flags, _, _ = check_refusal_completions(completions=responses)
                all_responses_non_refusal = not any(refusal_flags)
                
                if all_responses_non_refusal and len(current_behavior_indices) < total_behaviors:
                    print(f'\nAdding behavior to list of behaviors to optimize (case 3 - all {len(responses)} responses are non-refusal)\n')
                    num_behaviors += 1
                    steps_since_last_added_behavior = 0
            
        
        logs = {
            'final_loss': current_loss, 
            'all_losses': all_losses, 
            'all_test_cases': all_test_cases, 
            'num_behaviors': num_behaviors,
            'chosen_positions': chosen_positions,
            'chosen_token_ids': chosen_token_ids,
            'chosen_token_texts': chosen_token_texts
        }
        return test_cases, logs

    # ===== Defining saving methods (same as EnsembleGCG) =====
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

        run_ids = [d.split('_')[-1].split('.')[0] for d in os.listdir(save_dir) if d.startswith('test_cases_')]

        for run_id in run_ids:
            test_cases_path = os.path.join(save_dir, f'test_cases_{run_id}.json')
            with open(test_cases_path, 'r') as f:
                test_cases_part = json.load(f)
            for behavior_id in test_cases_part:
                if behavior_id in test_cases:
                    test_cases[behavior_id].extend(test_cases_part[behavior_id])
                else:
                    test_cases[behavior_id] = test_cases_part[behavior_id]
        
        test_cases_save_path = os.path.join(save_dir, 'test_cases.json')
        with open(test_cases_save_path, 'w') as f:
            json.dump(test_cases, f, indent=4)
        
        print(f"Saved test_cases.json to {test_cases_save_path}")