import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from tqdm import tqdm
from ..baseline import SingleBehaviorRedTeamingMethod
from ..model_utils import get_template
from .gbda_posinit_attention_utils import insert_optim_embed_pos, interleave_behavior_and_controls, generate_positions
import json
import time

# ============================== GBDA CLASS DEFINITION ============================== #
class GBDA_posinit_attention(SingleBehaviorRedTeamingMethod):
    def __init__(self,
                target_model,
                num_optim_tokens=20,
                num_steps=500,
                lr=0.2,
                noise_scale=0.2,
                attention_temp=1.0,
                targets_path=None,
                **model_kwargs):
        """
        :param target_model: a dictionary specifying the target model (kwargs to load_model_and_tokenizer)
        :param num_optim_tokens: the number of tokens in each test case
        :param num_steps: the number of optimization steps to use
        :param lr: the learning rate to use
        :param noise_scale: amount of noise to use for random initialization
        :param attention_temp: temperature for attention-based position allocation
        :param targets_path: the path to the targets JSON file
        """
        super().__init__(target_model, **model_kwargs)
        self.num_optim_tokens = num_optim_tokens
        self.num_steps = num_steps
        self.lr = lr
        self.noise_scale = noise_scale
        self.attention_temp = attention_temp

        if targets_path is None:
            raise ValueError("targets_path must be specified")
        with open(targets_path, 'r', encoding='utf-8') as file:
            self.behavior_id_to_target = json.load(file)
        self.targets_path = targets_path

        template = get_template(**target_model)['prompt']
        self.template = template
        self.before_tc, self.after_tc = template.split("{instruction}")

    def generate_test_cases_single_behavior(self, behavior, num_generate, verbose=False):
        """
        Generates test cases for a single behavior

        :param behavior: a dictionary specifying the behavior to generate test cases for
        :param num_generate: the number of test cases to generate in parallel
        :param verbose: whether to print progress
        :return: a list of test case and a list of logs
        """
        # ========== Behavior and Target str ==========
        # get required variables from behavior dictionary
        behavior_dict = behavior
        behavior = behavior_dict['Behavior']
        context_str = behavior_dict['ContextString']
        behavior_id = behavior_dict['BehaviorID']
    
        target = self.behavior_id_to_target[behavior_id]
        behavior += ' '
        # handle setup for behaviors with context strings
        if context_str:
            behavior = f"{context_str}\n\n---\n\n{behavior}"

        # GBDA hyperparams
        num_optim_tokens = self.num_optim_tokens
        num_steps = self.num_steps
        lr = self.lr
        noise_scale = self.noise_scale

        # Vars
        model = self.model
        tokenizer = self.tokenizer
        device = model.device
        before_tc = self.before_tc
        after_tc = self.after_tc
        embed_layer = self.model.get_input_embeddings()

        # ========== Init Cache Embeds ========
        cache_input_ids = tokenizer([before_tc], padding=False)['input_ids'] # some tokenizer have <s> for before_tc
        cache_input_ids += tokenizer([behavior, after_tc, target], padding=False, add_special_tokens=False)['input_ids']
        cache_input_ids = [torch.tensor(input_ids, device=device).unsqueeze(0) for input_ids in cache_input_ids] # make tensor separately because can't return_tensors='pt' in tokenizer
        before_ids, behavior_ids, after_ids, target_ids = cache_input_ids
        before_embeds, behavior_embeds, after_embeds, target_embeds = [embed_layer(input_ids) for input_ids in cache_input_ids]

        # ========== Position Initialization using Attention ========== #
        print("\n========== Computing Attention-based Positions ==========")
        pos_init_start = time.time()

        num_positions = len(behavior_ids[0]) + 1
        insert_positions = list(range(num_positions))

        # Insert initial tokens at all positions to compute attention
        init_optim_ids = tokenizer("!", return_tensors="pt", add_special_tokens=False).to(device)['input_ids']
        optim_ids_temp = init_optim_ids.expand(-1, num_positions)

        # Create one-hot and get embeddings
        vocab_size = embed_layer.weight.shape[0]
        optim_ids_onehot = torch.zeros((1, num_positions, vocab_size), device=device, dtype=model.dtype)
        optim_ids_onehot.scatter_(2, optim_ids_temp.unsqueeze(2), 1.0)
        optim_embeds_temp = torch.matmul(optim_ids_onehot.squeeze(0), embed_layer.weight).unsqueeze(0)

        # Insert at all positions
        inserted = insert_optim_embed_pos(behavior_embeds, optim_embeds_temp, insert_positions)

        # Run forward pass with attention
        input_embeds = torch.cat([before_embeds, inserted, after_embeds, target_embeds], dim=1).to(model.dtype)
        outputs = model(inputs_embeds=input_embeds, output_attentions=True)

        # Extract attention weights from upper half layers
        total_layers = len(outputs.attentions)
        upper_half_start = total_layers // 2
        print(f"Using upper half layers {upper_half_start}:{total_layers} (total layers: {total_layers})")

        attentions = torch.stack(outputs.attentions[upper_half_start:]).sum(dim=(0, 2))

        # Calculate actual positions after insertion
        actual_positions = []
        for i, pos in enumerate(insert_positions):
            actual_pos = pos + i
            actual_positions.append(actual_pos)

        # Extract attention weights for inserted tokens
        batch_size, query_len, key_len = attentions.shape
        before_len = before_embeds.shape[1]
        inserted_len = inserted.shape[1]
        after_len = after_embeds.shape[1]
        target_len = target_embeds.shape[1]

        query_after_start = before_len + inserted_len
        query_after_end = before_len + inserted_len + after_len
        key_inserted_start = before_len
        key_inserted_end = before_len + inserted_len

        chat_positions = list(range(query_after_start, query_after_end))
        attention_weight = attentions[0, chat_positions, key_inserted_start:key_inserted_end].sum(dim=0)[actual_positions]

        print(f"Raw attention scores: {attention_weight.tolist()}")
        attention_probs = torch.softmax(attention_weight / self.attention_temp, dim=0)
        print(f"Attention probs: {attention_probs.tolist()}")

        # Generate optimized positions based on attention
        optim_pos = generate_positions(attention_probs, num_optim_tokens)

        pos_init_time = time.time() - pos_init_start
        print(f"Optimized positions: {optim_pos.tolist()}, Time taken: {pos_init_time:.2f} seconds\n")

        # ========== setup log_coeffs (the optimizable variables) ========== #
        with torch.no_grad():
            embeddings = embed_layer(torch.arange(0, self.tokenizer.vocab_size, device=device).long())

        # ========== setup log_coeffs (the optimizable variables) ========== #
        log_coeffs = torch.zeros(num_generate, num_optim_tokens, embeddings.size(0))
        log_coeffs += torch.randn_like(log_coeffs) * noise_scale  # add noise to initialization
        log_coeffs = log_coeffs.cuda()
        log_coeffs.requires_grad_()

        # ========== setup optimizer and scheduler ========== #
        optimizer = torch.optim.Adam([log_coeffs], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)
        taus = np.linspace(1, 0.1, num_steps)

        all_losses = []
        all_test_cases = []
        # ========== run optimization ========== #
        for i in range(num_steps):
            coeffs = torch.nn.functional.gumbel_softmax(log_coeffs, hard=False, tau=taus[i]).to(embeddings.dtype) # B x T x V
            optim_embeds = (coeffs @ embeddings[None, :, :]) # B x T x D

            # Insert optimization embeddings at attention-based positions
            behavior_optim_embeds_list = []
            for b in range(num_generate):
                inserted = insert_optim_embed_pos(
                    behavior_embeds,
                    optim_embeds[b:b+1],
                    optim_pos
                )
                behavior_optim_embeds_list.append(inserted)
            behavior_optim_embeds = torch.cat(behavior_optim_embeds_list, dim=0)

            input_embeds = torch.cat([before_embeds.repeat(num_generate, 1, 1),
                                    behavior_optim_embeds,
                                    after_embeds.repeat(num_generate, 1, 1),
                                    target_embeds.repeat(num_generate, 1, 1)], dim=1)
            outputs = model(inputs_embeds=input_embeds)

            logits = outputs.logits
            
            # ========== compute loss ========== #
            # Shift so that tokens < n predict n
            tmp = input_embeds.shape[1] - target_embeds.shape[1]
            shift_logits = logits[..., tmp-1:-1, :].contiguous()
            shift_labels = target_ids.repeat(num_generate, 1)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(num_generate, -1).mean(dim=1)
        
            # ========== update log_coeffs ========== #
            optimizer.zero_grad()
            loss.mean().backward(inputs=[log_coeffs])
            optimizer.step()
            scheduler.step()

            # ========== retrieve optim_tokens and test_cases========== #
            optim_tokens = torch.argmax(log_coeffs, dim=2)
            # Interleave behavior and optimized tokens at computed positions
            test_cases_tokens = interleave_behavior_and_controls(
                behavior_ids.repeat(num_generate, 1),
                optim_tokens,
                optim_pos.unsqueeze(0).repeat(num_generate, 1)
            )
            test_cases = tokenizer.batch_decode(test_cases_tokens, skip_special_tokens=True)

            # ========= Saving and logging test cases ========= #
            current_loss = loss.detach().cpu().tolist()
            all_test_cases.append(test_cases)
            all_losses.append(current_loss)

        logs = [{
                'final_loss': current_loss[i],
                'all_losses': [loss[i] for loss in all_losses],
                'all_test_cases': [test_cases[i] for test_cases in all_test_cases],
                'optim_pos': optim_pos.tolist(),
                'position_init_time': pos_init_time
            } for i in range(num_generate)]
        return test_cases, logs
