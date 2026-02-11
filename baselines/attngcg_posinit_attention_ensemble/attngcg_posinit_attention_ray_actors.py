from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import ray
from torch.nn import CrossEntropyLoss
import numpy as np
from typing import List, Union
import random
from ..model_utils import get_template, load_model_and_tokenizer
# https://huggingface.co/docs/accelerate/v0.11.0/en/memory#accelerate.find_executable_batch_size
from accelerate.utils import find_executable_batch_size
import transformers
import gc
import time

def scale_positions_for_behavior(optim_pos, behavior_length, max_behavior_length):
    """
    Scale positions from max_behavior_length scale to actual behavior length.
    
    Args:
        optim_pos: positions on max_behavior_length scale
        behavior_length: actual length of current behavior
        max_behavior_length: max length used during initialization
    
    Returns:
        scaled positions clamped to valid range [0, behavior_length]
    """
    
    # Scale positions: actual_pos = round(pos * behavior_len / max_behavior_len)
    scaled_pos = torch.round(optim_pos.float() * behavior_length / max_behavior_length).long()
    
    # Clamp to valid range
    scaled_pos = torch.clamp(scaled_pos, 0, behavior_length)
    
    return scaled_pos

def generate_positions(attention_probs, num_adv_string):
    raw = attention_probs * num_adv_string
    tokens = raw.floor().int()
    remaining = num_adv_string - tokens.sum().item()
    
    # ì†Œìˆ˜ì  ë¶€ë¶„ì´ 1ì— ê°€ê¹Œìš´ ìˆœì„œëŒ€ë¡œ ë‚¨ì€ í† í° ë¶„ë°°
    fractional_parts = raw - tokens.float()  # ì†Œìˆ˜ì  ë¶€ë¶„ ê³„ì‚°

    _, top_indices = torch.topk(fractional_parts, remaining)  # ì†Œìˆ˜ì ì´ í° ìˆœì„œ
    tokens[top_indices] += 1
    print(f"raw distribution: {raw.tolist()}")
    print(f"fractional_parts: {fractional_parts.tolist()}")
    print(f"Token distribution: {tokens.tolist()}, Total: {tokens.sum().item()}")
    
    # í…ì„œë¡œ actual_positions ìƒì„±
    actual_positions = torch.repeat_interleave(
        torch.arange(len(tokens), device=tokens.device),
        tokens
    )
    
    return actual_positions

def insert_optim_embed_pos(behavior_embeds, optim_embeds, insert_pos):
    """
    Insert each token in `optim_embeds` into `behavior_embeds` at corresponding position in `insert_pos`.
    Positions are relative to the original behavior sequence and must be adjusted as we insert.
    
    Args:
        behavior_embeds: (1, B, D)
        optim_embeds: (1, N, D)
        insert_pos: list[int] of length N

    Returns:
        tuple: (inserted_embeds, optim_indices)
            - inserted_embeds: Tensor (1, B + N, D)
            - optim_indices: Tensor of shape (N,), actual indices of optim tokens in the result
    """
    assert behavior_embeds.shape[0] == 1
    assert optim_embeds.shape[0] == 1
    assert optim_embeds.shape[1] == len(insert_pos)

    if isinstance(insert_pos, torch.Tensor):
        insert_pos = insert_pos.tolist()
    
    behavior_seq = behavior_embeds[0]   # (B, D)
    optim_seq = optim_embeds[0]         # (N, D)

    new_tokens = []
    
    # insert_posëŠ" behaviorì˜ ì›ëž˜ index ê¸°ì¤€
    # 같은 위치에 여러 토큰이 삽입되는 경우, 원래 순서 유지
    insert_map = {}
    for i, p in enumerate(insert_pos):
        insert_map.setdefault(p, []).append(i)

    # Track indices as we build the sequence
    optim_indices_ordered = []
    current_idx = 0

    for i in range(behavior_seq.shape[0] + 1):
        # ë¨¼ì € insert_pos[i]ì— í•´ë‹¹í•˜ëŠ" adversarial token ì‚½ìž…
        if i in insert_map:
            for optim_i in insert_map[i]:
                new_tokens.append(optim_seq[optim_i:optim_i+1])
                optim_indices_ordered.append((optim_i, current_idx))
                current_idx += 1

        # ê·¸ ë‹¤ìŒì— behavior[i] ì‚½ìž… (i < Bì¼ ë•Œë§Œ)
        if i < behavior_seq.shape[0]:
            new_tokens.append(behavior_seq[i:i+1])
            current_idx += 1
    
    # Sort by original optim index to get indices in original order
    optim_indices_ordered.sort(key=lambda x: x[0])
    actual_optim_indices = [idx for _, idx in optim_indices_ordered]
    
    result_embeds = torch.cat([t.unsqueeze(0) for t in new_tokens], dim=1)  # (1, B+N, D)
    
    # Convert to tensor
    optim_indices_tensor = torch.tensor(actual_optim_indices, dtype=torch.long, device=behavior_embeds.device)

    return result_embeds, optim_indices_tensor

def interleave_behavior_and_controls(behavior_ids, optim_ids, insert_pos_batch):
    """
    Insert each token in `optim_ids` into `behavior_ids` at corresponding position in `insert_pos_batch`.
    Supports batch processing.

    Args:
        behavior_ids: (B, L_b) Tensor
        optim_ids: (B, L_o) Tensor
        insert_pos_batch: List[List[int]] or (B, L_o) Tensor

    Returns:
        Tensor: (B, L_b + L_o)
    """
    B = behavior_ids.size(0)
    interleaved = []

    # Ensure insert_pos_batch is list of lists
    if isinstance(insert_pos_batch, torch.Tensor):
        insert_pos_batch = insert_pos_batch.tolist()

    for b in range(B):
        behavior_seq = behavior_ids[b]   # (L_b,)
        optim_seq = optim_ids[b]         # (L_o,)
        insert_pos = insert_pos_batch[b] # list[int]

        new_tokens = []

        # insert_map: index â†' list of optim_idx
        insert_map = {}
        for i, p in enumerate(insert_pos):
            insert_map.setdefault(p, []).append(i)

        for i in range(behavior_seq.size(0) + 1):
            if i in insert_map:
                for optim_i in insert_map[i]:
                    new_tokens.append(optim_seq[optim_i:optim_i+1])  # (1,)
            if i < behavior_seq.size(0):
                new_tokens.append(behavior_seq[i:i+1])  # (1,)

        # Concatenate tokens: (L_b + L_o,)
        interleaved.append(torch.cat(new_tokens, dim=0).unsqueeze(0))  # (1, L)

    # Pad to maximum length
    max_len = max(x.size(1) for x in interleaved)
    padded = torch.zeros(B, max_len, dtype=behavior_ids.dtype, device=behavior_ids.device)

    for i, row in enumerate(interleaved):
        padded[i, :row.size(1)] = row
        if row.size(1) < max_len:
            print(f"[PAD] Sample {i} padded from length {row.size(1)} to {max_len}")

    return padded  # (B, max_len)

@ray.remote
class GCGModelWorkerPosinit:
    def __init__(self, 
                 model_name_or_path,
                 allow_non_ascii=False, 
                 return_tensors="pt",
                 seed=0,
                 num_gpus=1,
                 search_width=512,
                 use_prefix_cache=True,
                 attention_weight=100,
                 **model_kwargs):
        random.seed(seed)

        self.search_width = search_width
        # starting search_batch_size, will auto reduce batch_size later if go OOM (distinct for each worker, but not each behavior for simplicity)
        self.search_batch_size = search_width

        self.use_prefix_cache = use_prefix_cache
        self.attention_weight = attention_weight
        
        self.model_name = model_name_or_path.split("/")[-1]
        self.model_name_or_path = model_name_or_path
        
        self.num_gpus = num_gpus
        self.model, self.tokenizer = load_model_and_tokenizer(model_name_or_path, **model_kwargs)
        self.device = self.model.device
        self.return_tensors = return_tensors

        self.embed_layer = self.model.get_input_embeddings()
        self.vocab_size = self.embed_layer.weight.shape[0]  # can be larger than tokenizer.vocab_size for some models
        self.vocab_embeds = self.embed_layer(torch.arange(0, self.vocab_size).long().to(self.device))

        # Init instruction template, embeds and cache
        template = get_template(self.model_name_or_path, **model_kwargs)['prompt'].split("{instruction}")
        self.before_tc, self.after_tc = template[0], template[1]
        before_ids = self.tokenizer(self.before_tc, return_tensors=self.return_tensors, add_special_tokens=True).to(self.device)['input_ids']
        self.before_embeds = self.embed_layer(before_ids)
        after_ids = self.tokenizer(self.after_tc, return_tensors=self.return_tensors, add_special_tokens=False).to(self.device)['input_ids']
        self.after_embeds = self.embed_layer(after_ids)

        if self.use_prefix_cache:
            with torch.no_grad():
                outputs = self.model(inputs_embeds=self.before_embeds, use_cache=True)
                self.prefix_cache = outputs.past_key_values
        
        self.behaviors, self.targets = None, None
        self.all_target_ids = []
        self.all_target_embeds = []
        self.all_behavior_embeds = []
        self.all_behavior_ids = []

    def init_behaviors_and_targets(self, behaviors: List[str], targets: List[str]):
        # NOTE: self.targets is ONLY used here, self.behaviors only used here and in generate
        self.behaviors, self.targets = behaviors, targets

        for behavior, target in zip(behaviors, targets):
            behavior_ids = self.tokenizer(behavior, return_tensors=self.return_tensors, add_special_tokens=False).to(self.device)['input_ids']
            target_ids = self.tokenizer(target, return_tensors=self.return_tensors, add_special_tokens=False).to(self.device)['input_ids']
            behavior_embeds, target_embeds = self.embed_layer(behavior_ids), self.embed_layer(target_ids)

            # Move to CPUs here as number of behaviors can be large thus go OOM
            self.all_target_ids.append(target_ids.cpu())
            self.all_target_embeds.append(target_embeds.cpu())
            self.all_behavior_embeds.append(behavior_embeds.cpu())
            self.all_behavior_ids.append(behavior_ids.cpu())
            

    def compute_attention_scores_single(self, init_token_id, behavior_idx, behavior_length):
        """
        Compute attention scores for inserting tokens at all possible positions for a single behavior.
        Returns attention scores of length (behavior_length + 1).
        """
        behavior_embeds = self.all_behavior_embeds[behavior_idx].to(self.device)
        behavior_ids = self.all_behavior_ids[behavior_idx].to(self.device)
        target_ids = self.all_target_ids[behavior_idx].to(self.device)
        target_embeds = self.all_target_embeds[behavior_idx].to(self.device)
        
        # Ensure init_token_id is on the correct device
        init_token_id = init_token_id.to(self.device)

        num_positions = behavior_length + 1
        insert_positions = list(range(num_positions))

        print(f"Computing attention for {num_positions} positions: {insert_positions}")

        optim_ids = init_token_id.expand(-1, num_positions)  # (1, num_positions)
        optim_spread_pos = torch.tensor(insert_positions, dtype=torch.long, device=self.device)

        optim_ids_onehot = torch.zeros((1, num_positions, self.vocab_size), device=self.device, dtype=self.model.dtype)
        optim_ids_onehot.scatter_(2, optim_ids.unsqueeze(2), 1.0)
        optim_embeds = torch.matmul(optim_ids_onehot.squeeze(0), self.vocab_embeds).unsqueeze(0)  # (1, num_positions, D)

        inserted, optim_indices = insert_optim_embed_pos(behavior_embeds, optim_embeds, optim_spread_pos)

        # Forward pass
        if self.use_prefix_cache:
            input_embeds = torch.cat([inserted, self.after_embeds, target_embeds], dim=1).to(self.model.dtype)
            outputs = self.model(inputs_embeds=input_embeds, past_key_values=self.prefix_cache, output_attentions=True)
        else:
            input_embeds = torch.cat([self.before_embeds, inserted, self.after_embeds, target_embeds], dim=1).to(self.model.dtype)
            outputs = self.model(inputs_embeds=input_embeds, output_attentions=True)

        total_layers = len(outputs.attentions)
        upper_half_start = total_layers // 2  

        print(f"Using upper half layers {upper_half_start}:{total_layers} (total layers: {total_layers})")

        attentions = torch.stack(outputs.attentions[upper_half_start:]).sum(dim=(0, 2))

        actual_positions = []
        for i, pos in enumerate(insert_positions):
            actual_pos = pos + i  
            actual_positions.append(actual_pos)

        # í¬ê¸° ì •ë³´
        batch_size, query_len, key_len = attentions.shape
        cached_len = key_len - query_len if self.use_prefix_cache else 0 
        before_len = self.before_embeds.shape[1]
        inserted_len = inserted.shape[1]
        after_len = self.after_embeds.shape[1]
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

        # Attention scores 
        attention_weight = attentions[0, chat_positions, key_inserted_start:key_inserted_end].sum(dim=0)[actual_positions]

        print(f"Raw attention scores: {attention_weight.tolist()}")
    
        return attention_weight.cpu()  # Return to CPU for further processing

    def get_token_gradient_posinit(self, worker_optim_ids, worker_optim_pos, selected_behavior_indices, max_behavior_length=None):
        """
        Calculate the token gradient for position-based optimization across selected behaviors
        """
        token_grads = []
        for selected_behavior_idx in selected_behavior_indices:
            token_grads.append(self.compute_token_gradient_posinit(worker_optim_ids, worker_optim_pos, selected_behavior_idx, max_behavior_length))
        token_grad = sum(token_grads) / len(token_grads)
        return token_grad / token_grad.norm(dim=-1, keepdim=True)
        
    def compute_token_gradient_posinit(self, worker_optim_ids, worker_optim_pos, selected_behavior_idx, max_behavior_length=None):
        """
        Compute token gradient for position-based optimization for a specific behavior.
        Scale positions to valid range for this behavior.
        """
        behavior_embeds = self.all_behavior_embeds[selected_behavior_idx].to(self.device)
        behavior_ids = self.all_behavior_ids[selected_behavior_idx].to(self.device)
        target_ids = self.all_target_ids[selected_behavior_idx].to(self.device)
        target_embeds = self.all_target_embeds[selected_behavior_idx].to(self.device)
        
        # Ensure tensors are on the correct device
        worker_optim_ids = worker_optim_ids.to(self.device)
        worker_optim_pos = worker_optim_pos.to(self.device)
        
        num_optim_tokens = worker_optim_ids.shape[1]
        
        # Scale positions to valid range for this behavior
        behavior_length = behavior_ids.shape[1]
        if max_behavior_length is None:
            max_behavior_length = behavior_length
        scaled_optim_pos = scale_positions_for_behavior(worker_optim_pos, behavior_length, max_behavior_length)
        
        optim_ids_onehot = torch.zeros((1, num_optim_tokens, self.vocab_size), device=self.device).to(behavior_embeds.dtype)
        optim_ids_onehot.scatter_(2, worker_optim_ids.unsqueeze(2), 1.0).requires_grad_()
        optim_embeds = torch.matmul(optim_ids_onehot.squeeze(0), self.vocab_embeds).unsqueeze(0)

        # Insert optimized tokens at scaled positions
        behavior_with_optim, optim_indices = insert_optim_embed_pos(behavior_embeds, optim_embeds, scaled_optim_pos)
        
        if self.use_prefix_cache:
            input_embeds = torch.cat([behavior_with_optim, self.after_embeds, target_embeds], dim=1)
            outputs = self.model(inputs_embeds=input_embeds, past_key_values=self.prefix_cache, output_attentions=True)
        else:
            input_embeds = torch.cat([self.before_embeds, behavior_with_optim, self.after_embeds, target_embeds], dim=1)
            outputs = self.model(inputs_embeds=input_embeds, output_attentions=True)

        logits = outputs.logits
        
        # Compute logit loss
        tmp = input_embeds.shape[1] - target_embeds.shape[1]
        shift_logits = logits[..., tmp-1:-1, :].contiguous()
        shift_labels = target_ids
        loss_fct = CrossEntropyLoss()
        logit_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Compute attention loss using last layer
        attention = outputs.attentions[-1]  # Last layer: (batch, heads, query_len, key_len)

        # Extract attention: target tokens (query) → all previous tokens (key)
        attn = attention[:, :, tmp-1:].mean(2)  # Average over target tokens, (batch, heads, key_len)
        tmp_attn = attn.mean(1)  # Average over heads, (batch, key_len)
        tmp_input = tmp_attn[:, :tmp]  # Only look at input portion (before target)

        # Calculate control positions
        if self.use_prefix_cache:
            # In tmp_input: [behavior_optim] [after]
            key_control = optim_indices  # Already relative to start
        else:
            # In tmp_input: [before] [behavior_optim] [after]
            key_control = optim_indices + self.before_embeds.shape[1]

        attn_loss = tmp_input[0, key_control].mean()

        # Combined loss with attention weight
        loss = logit_loss - self.attention_weight * attn_loss

        token_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

        return token_grad
    
    def compute_loss_on_candidates_batch_posinit(self, sampled_top_indices, worker_optim_pos, selected_behavior_indices, max_behavior_length=None):
        """
        Compute loss for position-based candidates across multiple behaviors
        """
        all_losses = []
        all_logit_losses = []
        all_attn_losses = []
        for behavior_idx in selected_behavior_indices:
            losses, logit_losses, attn_losses = self.compute_loss_on_candidates_posinit(sampled_top_indices, worker_optim_pos, behavior_idx, max_behavior_length)
            all_losses.append(losses)
            all_logit_losses.append(logit_losses)
            all_attn_losses.append(attn_losses)
        behavior_losses = torch.stack(all_losses, dim=0)
        behavior_logit_losses = torch.stack(all_logit_losses, dim=0)
        behavior_attn_losses = torch.stack(all_attn_losses, dim=0)
        return behavior_losses, behavior_logit_losses, behavior_attn_losses
        
    def compute_loss_on_candidates_posinit(self, sampled_top_indices, worker_optim_pos, selected_behavior_idx, max_behavior_length=None):
        """
        Compute loss for position-based candidates for a specific behavior.
        Scale positions to valid range for this behavior.
        """
        behavior_ids = self.all_behavior_ids[selected_behavior_idx].to(self.device)
        target_ids = self.all_target_ids[selected_behavior_idx].to(self.device)
        target_embeds = self.all_target_embeds[selected_behavior_idx].to(self.device)
        
        # Ensure tensors are on the correct device
        sampled_top_indices = sampled_top_indices.to(self.device)
        worker_optim_pos = worker_optim_pos.to(self.device)
        
        search_width = sampled_top_indices.shape[0]

        # Scale positions to valid range for this behavior
        behavior_length = behavior_ids.shape[1]
        if max_behavior_length is None:
            max_behavior_length = behavior_length
        scaled_optim_pos = scale_positions_for_behavior(worker_optim_pos, behavior_length, max_behavior_length)

        # Create interleaved sequences with optimized tokens at scaled positions
        sampled_top_indices_inserted = interleave_behavior_and_controls(
            behavior_ids.repeat(search_width, 1), 
            sampled_top_indices, 
            scaled_optim_pos.repeat(search_width, 1)
        )

        sampled_top_embeds = self.embed_layer(sampled_top_indices_inserted)
        
        if self.use_prefix_cache:
            input_embeds = torch.cat([sampled_top_embeds,
                                    self.after_embeds.repeat(search_width, 1, 1),
                                    target_embeds.repeat(search_width, 1, 1)], dim=1)
        else:
            input_embeds = torch.cat([self.before_embeds.repeat(search_width, 1, 1),
                                    sampled_top_embeds,
                                    self.after_embeds.repeat(search_width, 1, 1),
                                    target_embeds.repeat(search_width, 1, 1)], dim=1)
        
        # Need to compute optim_indices for attention loss calculation
        behavior_embeds = self.all_behavior_embeds[selected_behavior_idx].to(self.device)
        num_optim_tokens = worker_optim_pos.shape[0]
        optim_embeds_dummy = torch.zeros((1, num_optim_tokens, behavior_embeds.shape[2]), device=self.device, dtype=behavior_embeds.dtype)
        _, optim_indices = insert_optim_embed_pos(behavior_embeds, optim_embeds_dummy, scaled_optim_pos)
        
        total_loss, logit_loss, attn_loss = find_executable_batch_size(self.compute_candidates_logits_with_attention, self.search_batch_size)(input_embeds, target_ids, optim_indices)

        return total_loss, logit_loss, attn_loss
 
    def compute_candidates_logits(self, search_batch_size, input_embeds):
        if self.search_batch_size != search_batch_size:
            print(f"INFO: Setting candidates search_batch_size to {search_batch_size})")
            self.search_batch_size = search_batch_size
            gc.collect() 
            torch.cuda.empty_cache() 

        logits = []
        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i+search_batch_size]

                if self.use_prefix_cache:
                    prefix_cache = self.prefix_cache
                    current_batch_size = input_embeds_batch.shape[0]
                    prefix_cache_batch = []
                    for i in range(len(prefix_cache)):
                        prefix_cache_batch.append([])
                        for j in range(len(prefix_cache[i])):
                            prefix_cache_batch[i].append(prefix_cache[i][j].expand(current_batch_size, -1, -1, -1))
                    outputs = self.model(inputs_embeds=input_embeds_batch, past_key_values=prefix_cache_batch)
                else:
                    outputs = self.model(inputs_embeds=input_embeds_batch)
            logits.append(outputs.logits)
        logits = torch.cat(logits, dim=0)
        return logits

    def compute_candidates_logits_with_attention(self, search_batch_size, input_embeds, target_ids, optim_indices):
        """
        Compute both logit loss and attention loss for candidate sequences
        """
        if self.search_batch_size != search_batch_size:
            print(f"INFO: Setting candidates search_batch_size to {search_batch_size})")
            self.search_batch_size = search_batch_size
            gc.collect() 
            torch.cuda.empty_cache() 

        all_logit_loss = []
        all_attn_loss = []
        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i+search_batch_size]

                if self.use_prefix_cache:
                    prefix_cache = self.prefix_cache
                    current_batch_size = input_embeds_batch.shape[0]
                    prefix_cache_batch = []
                    for j in range(len(prefix_cache)):
                        prefix_cache_batch.append([])
                        for k in range(len(prefix_cache[j])):
                            prefix_cache_batch[j].append(prefix_cache[j][k].expand(current_batch_size, -1, -1, -1))
                    outputs = self.model(inputs_embeds=input_embeds_batch, past_key_values=prefix_cache_batch, output_attentions=True)
                else:
                    outputs = self.model(inputs_embeds=input_embeds_batch, output_attentions=True)
            
            logits = outputs.logits
            attentions = outputs.attentions

            # Compute logit loss
            tmp = input_embeds_batch.shape[1] - target_ids.shape[1]
            shift_logits = logits[..., tmp-1:-1, :].contiguous()
            shift_labels = target_ids.repeat(input_embeds_batch.shape[0], 1)
            loss_fct = CrossEntropyLoss(reduction='none')
            loss_logit = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss_logit = loss_logit.view(input_embeds_batch.shape[0], -1).mean(dim=1)


            # Compute attention loss using last layer
            attention = outputs.attentions[-1]  # Last layer: (batch, heads, query_len, key_len)

            # Extract attention: target tokens (query) → all previous tokens (key)
            attn = attention[:, :, tmp-1:].mean(2)  # Average over target tokens, (batch, heads, key_len)
            tmp_attn = attn.mean(1)  # Average over heads, (batch, key_len)
            tmp_input = tmp_attn[:, :tmp]  # Only look at input portion (before target)

            # Calculate control positions
            if self.use_prefix_cache:
                # In tmp_input: [behavior_optim] [after]
                key_control = optim_indices  # Already relative to start
            else:
                # In tmp_input: [before] [behavior_optim] [after]
                key_control = optim_indices + self.before_embeds.shape[1]

            # Attention loss for each sample in batch
            batch_attn_loss = []
            for b_idx in range(tmp_input.size(0)):
                single_attn_loss = tmp_input[b_idx, key_control].mean()
                batch_attn_loss.append(single_attn_loss.unsqueeze(0))

            all_attn_loss.append(torch.cat(batch_attn_loss, dim=0))
            all_logit_loss.append(loss_logit)

            del outputs, logits, loss_logit, attention, single_attn_loss
            torch.cuda.empty_cache()
            gc.collect()
            
        logit_losses = torch.cat(all_logit_loss, dim=0)
        attn_losses = torch.cat(all_attn_loss, dim=0)
        total_losses = logit_losses - self.attention_weight * attn_losses
        
        return total_losses, logit_losses, attn_losses

    def generate(self, optim_ids, optim_pos, selected_behavior_idx, max_behavior_length=None):
        """
        Generate completion using position-based adversarial tokens.
        Scale positions to valid range for this behavior.
        
        Args:
            optim_ids: (1, N) tensor of optimized token IDs
            optim_pos: (N,) tensor of positions where tokens should be inserted
            selected_behavior_idx: index of the behavior to generate for
            max_behavior_length: max behavior length for scaling (optional)
        """
        behavior_ids = self.all_behavior_ids[selected_behavior_idx].to(self.device)
        behavior_length = behavior_ids.shape[1]
        
        # Scale positions to valid range for this behavior
        if max_behavior_length is None:
            max_behavior_length = behavior_length
        scaled_optim_pos = scale_positions_for_behavior(optim_pos.to(self.device), behavior_length, max_behavior_length)
        optim_ids = optim_ids.to(self.device)
        
        # Create interleaved sequence with optimized tokens at specified positions
        test_case_ids = interleave_behavior_and_controls(
            behavior_ids, optim_ids, scaled_optim_pos.unsqueeze(0)
        )[0]
        
        # Create full input with template
        before_ids = self.tokenizer(self.before_tc, return_tensors=self.return_tensors, add_special_tokens=True).to(self.device)['input_ids']
        after_ids = self.tokenizer(self.after_tc, return_tensors=self.return_tensors, add_special_tokens=False).to(self.device)['input_ids']
        
        full_input_ids = torch.cat([before_ids, test_case_ids.unsqueeze(0), after_ids], dim=1)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=full_input_ids,
                do_sample=False, 
                max_new_tokens=16, 
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        # Decode only the generated part
        generated_ids = outputs[0][full_input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text

    def is_initialized(self):
        return f"==cuda:{ray.get_gpu_ids()}. {self.model_name_or_path}"
    
    def get_attribute_(self, attr_name):
        return getattr(self, attr_name, None)

#### Helper
def sample_control(control_toks, grad, search_width, topk=256, temp=1, not_allowed_tokens=None):
    if not_allowed_tokens is not None:
        # grad[:, not_allowed_tokens.to(grad.device)] = np.infty
        grad = grad.clone()
        grad[:, not_allowed_tokens.to(grad.device)] = grad.max() + 1

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(search_width, 1)
    new_token_pos = torch.arange(
        0, 
        len(control_toks), 
        len(control_toks) / search_width,
        device=grad.device
    ).type(torch.int64)
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk, (search_width, 1),
        device=grad.device)
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    return new_control_toks


def get_nonascii_toks(tokenizer, device='cpu'):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)
    
    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)

    if "Baichuan2" in tokenizer.name_or_path:
        ascii_toks += [i for i in range(101, 1000)]

    if "Llama-3.1" in tokenizer.name_or_path:
        ascii_toks += [i for i in range(128000, 128256)]
    
    if "Qwen2.5" in tokenizer.name_or_path:
        ascii_toks += [i for i in range(151643, 151665)]
        
    return torch.tensor(ascii_toks, device=device)