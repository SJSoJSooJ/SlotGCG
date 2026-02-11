import torch

def insert_optim_embed_pos(behavior_embeds, optim_embeds, insert_pos):
    """
    Insert each token in `optim_embeds` into `behavior_embeds` at corresponding position in `insert_pos`.
    Positions are relative to the original behavior sequence and must be adjusted as we insert.
    
    Args:
        behavior_embeds: (1, B, D)
        optim_embeds: (1, N, D)
        insert_pos: list[int] of length N

    Returns:
        Tensor: (1, B + N, D)
    """
    assert behavior_embeds.shape[0] == 1
    assert optim_embeds.shape[0] == 1
    assert optim_embeds.shape[1] == len(insert_pos)

    if isinstance(insert_pos, torch.Tensor):
        insert_pos = insert_pos.tolist()
    
    behavior_seq = behavior_embeds[0]   # (B, D)
    optim_seq = optim_embeds[0]         # (N, D)

    new_tokens = []

    # insert_pos는 behavior의 원래 index 기준
    insert_map = {}
    for i, p in enumerate(insert_pos):
        insert_map.setdefault(p, []).append(i)

    for i in range(behavior_seq.shape[0] + 1):
        # 먼저 insert_pos[i]에 해당하는 adversarial token 삽입
        if i in insert_map:
            for optim_i in insert_map[i]:
                new_tokens.append(optim_seq[optim_i:optim_i+1])

        # 그 다음에 behavior[i] 삽입 (i < B일 때만)
        if i < behavior_seq.shape[0]:
            new_tokens.append(behavior_seq[i:i+1])

    return torch.cat([t.unsqueeze(0) for t in new_tokens], dim=1)  # (1, B+N, D)

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

        # insert_map: index → list of optim_idx
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



########## GCG Utils ##########
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