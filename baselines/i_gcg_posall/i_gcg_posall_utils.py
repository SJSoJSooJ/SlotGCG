import gc
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    return new_control_toks, new_token_val.reshape(-1), new_token_pos

def get_nonascii_toks(tokenizer, device='cpu'):
    """
    Get non-ASCII tokens to filter out - following original implementation.
    """
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

    # Model-specific token ranges
    if "Baichuan2" in tokenizer.name_or_path:
        ascii_toks += [i for i in range(101, 1000)]

    if "Llama-3.1" in tokenizer.name_or_path:
        ascii_toks += [i for i in range(128000, 128256)]
    
    if "Qwen2.5" in tokenizer.name_or_path:
        ascii_toks += [i for i in range(151643, 151665)]
        
    return torch.tensor(ascii_toks, device=device)

def apply_manual_multicoord_update(tokenizer, new_adv_suffix, adv_suffix, k_best_indices):
    """
    Apply manual multi-coordinate update following attack_llm_core logic exactly.
    """
    ori_adv_suffix_ids = tokenizer(adv_suffix, add_special_tokens=False).input_ids
    adv_suffix_ids = tokenizer(adv_suffix, add_special_tokens=False).input_ids
    best_new_adv_suffix_ids = copy.copy(adv_suffix_ids)
    all_new_adv_suffix = []
    
    for idx_i in range(len(k_best_indices)):
        idx = k_best_indices[idx_i]
        if idx < len(new_adv_suffix):
            temp_new_adv_suffix = new_adv_suffix[idx]
            temp_new_adv_suffix_ids = tokenizer(temp_new_adv_suffix, add_special_tokens=False).input_ids

            # Multi-coordinate merging: update different positions progressively
            for suffix_num in range(min(len(adv_suffix_ids), len(temp_new_adv_suffix_ids))):
                if adv_suffix_ids[suffix_num] != temp_new_adv_suffix_ids[suffix_num]:
                    best_new_adv_suffix_ids[suffix_num] = temp_new_adv_suffix_ids[suffix_num]

            all_new_adv_suffix.append(tokenizer.decode(best_new_adv_suffix_ids, skip_special_tokens=True))
    
    return all_new_adv_suffix
