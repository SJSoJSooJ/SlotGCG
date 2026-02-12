import torch

def generate_positions(attention_probs, num_adv_string):
    """
    Generate token positions based on attention probabilities.

    :param attention_probs: Softmax probabilities over positions
    :param num_adv_string: Total number of adversarial tokens to distribute
    :return: Tensor of positions where each position appears according to allocation
    """
    raw = attention_probs * num_adv_string
    tokens = raw.floor().int()
    remaining = num_adv_string - tokens.sum().item()

    fractional_parts = raw - tokens.float()

    _, top_indices = torch.topk(fractional_parts, remaining)
    tokens[top_indices] += 1
    print(f"raw distribution: {raw.tolist()}")
    print(f"fractional_parts: {fractional_parts.tolist()}")
    print(f"Token distribution: {tokens.tolist()}, Total: {tokens.sum().item()}")

    actual_positions = torch.repeat_interleave(
        torch.arange(len(tokens), device=tokens.device),
        tokens
    )

    return actual_positions


def insert_optim_embed_pos(behavior_embeds, optim_embeds, insert_pos):
    """
    Insert optimization embeddings at specified positions within behavior embeddings.

    :param behavior_embeds: [1, seq_len, embed_dim] behavior embeddings
    :param optim_embeds: [1, num_optim_tokens, embed_dim] adversarial token embeddings
    :param insert_pos: List or tensor of positions where to insert each optim token
    :return: [1, new_seq_len, embed_dim] interleaved embeddings
    """
    assert behavior_embeds.shape[0] == 1
    assert optim_embeds.shape[0] == 1
    assert optim_embeds.shape[1] == len(insert_pos)

    if isinstance(insert_pos, torch.Tensor):
        insert_pos = insert_pos.tolist()

    behavior_seq = behavior_embeds[0]
    optim_seq = optim_embeds[0]

    new_tokens = []

    insert_map = {}
    for i, p in enumerate(insert_pos):
        insert_map.setdefault(p, []).append(i)

    for i in range(behavior_seq.shape[0] + 1):
        if i in insert_map:
            for optim_i in insert_map[i]:
                new_tokens.append(optim_seq[optim_i:optim_i+1])

        if i < behavior_seq.shape[0]:
            new_tokens.append(behavior_seq[i:i+1])

    return torch.cat([t.unsqueeze(0) for t in new_tokens], dim=1)


def interleave_behavior_and_controls(behavior_ids, optim_ids, insert_pos_batch):
    """
    Interleave behavior token IDs with optimization token IDs at specified positions.
    Used for creating final test cases.

    :param behavior_ids: [B, behavior_seq_len] behavior token IDs
    :param optim_ids: [B, num_optim_tokens] adversarial token IDs
    :param insert_pos_batch: [B, num_optim_tokens] positions for each batch
    :return: [B, max_len] interleaved token IDs (padded)
    """
    B = behavior_ids.size(0)
    interleaved = []

    if isinstance(insert_pos_batch, torch.Tensor):
        insert_pos_batch = insert_pos_batch.tolist()

    for b in range(B):
        behavior_seq = behavior_ids[b]
        optim_seq = optim_ids[b]
        insert_pos = insert_pos_batch[b]

        new_tokens = []

        insert_map = {}
        for i, p in enumerate(insert_pos):
            insert_map.setdefault(p, []).append(i)

        for i in range(behavior_seq.size(0) + 1):
            if i in insert_map:
                for optim_i in insert_map[i]:
                    new_tokens.append(optim_seq[optim_i:optim_i+1])
            if i < behavior_seq.size(0):
                new_tokens.append(behavior_seq[i:i+1])


        interleaved.append(torch.cat(new_tokens, dim=0).unsqueeze(0))


    max_len = max(x.size(1) for x in interleaved)
    padded = torch.zeros(B, max_len, dtype=behavior_ids.dtype, device=behavior_ids.device)

    for i, row in enumerate(interleaved):
        padded[i, :row.size(1)] = row
        if row.size(1) < max_len:
            print(f"[PAD] Sample {i} padded from length {row.size(1)} to {max_len}")

    return padded
