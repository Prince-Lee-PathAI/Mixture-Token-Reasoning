# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import DynamicCache

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8


def top_p_sampling(probs, top_p=0.9):
    """
    Top-p (nucleus) sampling.
    
    Args:
        probs: Probability distribution, shape (vocab_size,)
        top_p: Cumulative probability threshold (default 0.9)
    
    Returns:
        sampled_token_id: The sampled token ID
    """
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)

    # Determine cutoff index: the first index where cumulative sum exceeds top_p
    cutoff = torch.searchsorted(cumulative_probs, top_p)

    # Mask everything after cutoff
    sorted_probs[cutoff + 1:] = 0.0

    # Normalize remaining
    sorted_probs /= sorted_probs.sum()

    # Sample
    sampled_idx = torch.multinomial(sorted_probs, 1).item()
    return sorted_indices[sampled_idx].item()

class Coconut_shared(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        state_combination_method = "continuous_only",  # "continuous_only" , "add", "adapter", "cross_attention"
        combination_use_gating=False,
    ):

        super(Coconut_shared, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.state_combination_method = state_combination_method
        self.combination_use_gating = combination_use_gating

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

        hidden_dim = self.embedding.embedding_dim

        # ---- modules for different combination methods ----
        if self.state_combination_method == "adapter":
            # 2-layer MLP (adapter)
            self.adapter = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        if self.state_combination_method == "projection":
            # simple linear projection of [cont || disc]
            self.proj = nn.Linear(2 * hidden_dim, hidden_dim)

        if self.state_combination_method == "cross_attention":
            # Q from continuous, K/V from discrete
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=4, batch_first=True
            )

        # optional gating: mixes continuous_hidden with candidate_state
        if self.combination_use_gating and self.state_combination_method != "continuous_only":
            self.gate_layer = nn.Linear(2 * hidden_dim, hidden_dim)

    def _combine_states(self, continuous_hidden, discrete_hidden):
        method = self.state_combination_method

        if method == "continuous_only":
            candidate = continuous_hidden

        elif method == "add":
            candidate = continuous_hidden + discrete_hidden

        elif method == "adapter":
            cat = torch.cat([continuous_hidden, discrete_hidden], dim=-1)
            candidate = self.adapter(cat)

        elif method == "projection":
            cat = torch.cat([continuous_hidden, discrete_hidden], dim=-1)
            candidate = self.proj(cat)

        elif method == "cross_attention":
            # Q = continuous, K/V = discrete
            q = continuous_hidden.unsqueeze(0).unsqueeze(0)  # (1,1,H)
            k = discrete_hidden.unsqueeze(0).unsqueeze(0)
            v = discrete_hidden.unsqueeze(0).unsqueeze(0)
            attn_out, _ = self.cross_attn(q, k, v)
            candidate = attn_out.squeeze(0).squeeze(0)  # (H,)

        else:
            raise ValueError(f"Unknown state_combination_method: {method}")

        # optional gating: mix candidate with original continuous state
        if getattr(self, "combination_use_gating", False) and method != "continuous_only":
            gate_input = torch.cat([continuous_hidden, candidate], dim=-1)
            gate = torch.sigmoid(self.gate_layer(gate_input))
            # element-wise interpolation between continuous and candidate
            candidate = gate * candidate + (1.0 - gate) * continuous_hidden

        return candidate
    

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):

        logits = []

        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # before the earliest latent token position

        kv_cache = None
        # print("max_n_latents:", max_n_latents)
        for pass_idx in range(max_n_latents):

            if kv_cache == None:
                # first forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

            else:
                # extract kv cache to reuse
                past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                past_key_values = tuple(past_key_values)
                # past_key_values=DynamicCache.from_legacy_cache(past_key_values)
                # print(past_key_values.get_seq_length())
                outputs = self.base_causallm(use_cache=True,
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True
                )

                hidden_states_offset = next_compute_range[0]
                # when we use kv_cache for the first k tokens
                # in `outputs.hidden_states`, [0, k) will be skipped
                # so we need to keep this offset to correctly use the last hidden states

            logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            continuous_hidden_states = outputs.hidden_states[
                -1
            ]  # Get the last layer hidden states
            kv_cache = outputs.past_key_values

            # feedback the continuous thoughts to the input_embeds

            # first decide the positions to feedback
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # to avoid in-place operations
            # break down inputs_embeds (bs, len, hidden_size) into a list of list of 1-d tensors
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # replace some of them with continuous thoughts and discrete ideas !!!!!!!!!!!!!!!!!!!!!
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair

                # print(continuous_hidden_states.shape)
                
                # positional index in the continuous hidden states
                pos_idx = token_idx - 1 - hidden_states_offset
                
                # continuous_hidden: get the last hidden state at the position
                continuous_hidden = continuous_hidden_states[
                    batch_idx, pos_idx, :
                ]

                # default: no discrete state
                discrete_hidden = None

                if self.state_combination_method != "continuous_only":
                    # sample a discrete token from the continuous hidden
                    logits_for_position = self.base_causallm.lm_head(
                        continuous_hidden
                    )  # (vocab_size,)
                    probs = torch.nn.functional.softmax(logits_for_position, dim=-1)
                    sampled_token_id = top_p_sampling(probs, top_p=0.9)
                    discrete_hidden = self.embedding(
                        torch.tensor(
                            sampled_token_id, device=continuous_hidden.device
                        )
                    )

                replacement_state = self._combine_states(
                    continuous_hidden, discrete_hidden
                )

                # replace it with the substitution hidden states
                tensor_list[batch_idx][token_idx] = replacement_state

            # assemble the new inputs_embeds
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # final pass
        outputs = self.base_causallm(use_cache=True,
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=tuple(
                [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
        )

        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()  # placeholder. not used.
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds

        # get the first token using the current hidden state
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # get other tokens
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            # in FSDP, the number of forward pass need to be the same across devices
            while (
                self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
            ):  # leave some room for latent tokens
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            # for analysis purpose
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds

        else:
            return torch.tensor(tokens).view(1, -1)

class Coconut_no_shared(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        state_combination_method = "continuous_only",  # "continuous_only" , "add", "adapter", "cross_attention"
        combination_use_gating=False,
    ):

        super(Coconut_no_shared, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.state_combination_method = state_combination_method
        self.combination_use_gating = combination_use_gating

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

        hidden_dim = self.embedding.embedding_dim

        # ---- modules for different combination methods ----
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(MAX_N_LATENT)
        ])

        if self.state_combination_method == "projection":
            self.projs = nn.ModuleList([
                nn.Linear(2 * hidden_dim, hidden_dim)
                for _ in range(MAX_N_LATENT)
            ])

        if self.state_combination_method == "cross_attention":
            # Q from continuous, K/V from discrete
            self.cross_attns = nn.ModuleList([
                nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
                for _ in range(MAX_N_LATENT)
            ])

        # optional gating: mixes continuous_hidden with candidate_state
        if self.combination_use_gating and self.state_combination_method != "continuous_only":
            self.gates = nn.ModuleList([
                nn.Linear(2 * hidden_dim, hidden_dim)
                for _ in range(MAX_N_LATENT)
            ])

    def _combine_states(self, continuous_hidden, discrete_hidden, pass_idx):

        method = self.state_combination_method

        # continuous only
        if method == "continuous_only":
            candidate = continuous_hidden

        elif method == "add":
            candidate = continuous_hidden + discrete_hidden

        elif method == "adapter":
            cat = torch.cat([continuous_hidden, discrete_hidden], dim=-1)
            candidate = self.adapters[pass_idx](cat)

        elif method == "projection":
            cat = torch.cat([continuous_hidden, discrete_hidden], dim=-1)
            candidate = self.projs[pass_idx](cat)

        elif method == "cross_attention":
            q = continuous_hidden.unsqueeze(0).unsqueeze(0)
            k = discrete_hidden.unsqueeze(0).unsqueeze(0)
            v = discrete_hidden.unsqueeze(0).unsqueeze(0)
            attn_out, _ = self.cross_attns[pass_idx](q, k, v)
            candidate = attn_out.squeeze(0).squeeze(0)

        # optional gating (per-step)
        if self.combination_use_gating and method != "continuous_only":
            gate_input = torch.cat([continuous_hidden, candidate], dim=-1)
            gate = torch.sigmoid(self.gates[pass_idx](gate_input))
            candidate = gate * candidate + (1 - gate) * continuous_hidden

        return candidate
    

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):

        logits = []

        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # before the earliest latent token position

        kv_cache = None
        # print("max_n_latents:", max_n_latents)
        for pass_idx in range(max_n_latents):

            if kv_cache == None:
                # first forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

            else:
                # extract kv cache to reuse
                past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                past_key_values = tuple(past_key_values)
                # past_key_values=DynamicCache.from_legacy_cache(past_key_values)
                # print(past_key_values.get_seq_length())
                outputs = self.base_causallm(use_cache=True,
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True
                )

                hidden_states_offset = next_compute_range[0]
                # when we use kv_cache for the first k tokens
                # in `outputs.hidden_states`, [0, k) will be skipped
                # so we need to keep this offset to correctly use the last hidden states

            logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            continuous_hidden_states = outputs.hidden_states[
                -1
            ]  # Get the last layer hidden states
            kv_cache = outputs.past_key_values

            # feedback the continuous thoughts to the input_embeds

            # first decide the positions to feedback
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # to avoid in-place operations
            # break down inputs_embeds (bs, len, hidden_size) into a list of list of 1-d tensors
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # replace some of them with continuous thoughts and discrete ideas !!!!!!!!!!!!!!!!!!!!!
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair

                # print(continuous_hidden_states.shape)
                
                # positional index in the continuous hidden states
                pos_idx = token_idx - 1 - hidden_states_offset
                
                # continuous_hidden: get the last hidden state at the position
                continuous_hidden = continuous_hidden_states[
                    batch_idx, pos_idx, :
                ]

                # default: no discrete state
                discrete_hidden = None

                if self.state_combination_method != "continuous_only":
                    # sample a discrete token from the continuous hidden
                    logits_for_position = self.base_causallm.lm_head(
                        continuous_hidden
                    )  # (vocab_size,)
                    probs = torch.nn.functional.softmax(logits_for_position, dim=-1)
                    sampled_token_id = top_p_sampling(probs, top_p=0.9)
                    discrete_hidden = self.embedding(
                        torch.tensor(
                            sampled_token_id, device=continuous_hidden.device
                        )
                    )

                replacement_state = self._combine_states(
                    continuous_hidden, discrete_hidden, pass_idx
                )

                # replace it with the substitution hidden states
                tensor_list[batch_idx][token_idx] = replacement_state

            # assemble the new inputs_embeds
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # final pass
        outputs = self.base_causallm(use_cache=True,
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=tuple(
                [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
        )

        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()  # placeholder. not used.
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds

        # get the first token using the current hidden state
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # get other tokens
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            # in FSDP, the number of forward pass need to be the same across devices
            while (
                self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
            ):  # leave some room for latent tokens
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            # for analysis purpose
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds

        else:
            return torch.tensor(tokens).view(1, -1)
        
class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):

        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):

        logits = []

        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # before the earliest latent token position

        kv_cache = None

        for pass_idx in range(max_n_latents):

            if kv_cache == None:
                # first forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

            else:
                # extract kv cache to reuse
                past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                past_key_values = tuple(past_key_values)
                # past_key_values=DynamicCache.from_legacy_cache(past_key_values)
                # print(past_key_values.get_seq_length())
                outputs = self.base_causallm(use_cache=True,
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True
                )

                hidden_states_offset = next_compute_range[0]
                # when we use kv_cache for the first k tokens
                # in `outputs.hidden_states`, [0, k) will be skipped
                # so we need to keep this offset to correctly use the last hidden states

            logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            continuous_hidden_states = outputs.hidden_states[
                -1
            ]  # Get the last layer hidden states
            kv_cache = outputs.past_key_values

            # feedback the continuous thoughts to the input_embeds

            # first decide the positions to feedback
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # to avoid in-place operations
            # break down inputs_embeds (bs, len, hidden_size) into a list of list of 1-d tensors
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # replace some of them with continuous thoughts and discrete ideas !!!!!!!!!!!!!!!!!!!!!
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair

                # print(continuous_hidden_states.shape)
                
                # positional index in the continuous hidden states
                pos_idx = token_idx - 1 - hidden_states_offset
                
                # print(f"batch_idx: {batch_idx}, token_idx: {token_idx}, pos_idx: {pos_idx}")
                # print(f"hidden_states_offset: {hidden_states_offset}")
                
                # continuous_hidden: get the last hidden state at the position
                continuous_hidden = continuous_hidden_states[
                    batch_idx, pos_idx, :
                ]
                # discrete_hidden: sample a token and get its embedding
                logits_for_position = self.base_causallm.lm_head(continuous_hidden)  # (vocab_size,)
                probs = torch.nn.functional.softmax(logits_for_position, dim=-1)
                # Top-p sampling
                sampled_token_id = top_p_sampling(probs, top_p=0.9)
                discrete_hidden = self.embedding(torch.tensor(sampled_token_id, device=continuous_hidden.device))
 

                # replace it with the preceding last hidden states
                tensor_list[batch_idx][token_idx] = continuous_hidden + discrete_hidden

            # assemble the new inputs_embeds
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # final pass
        outputs = self.base_causallm(use_cache=True,
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=tuple(
                [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
        )

        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()  # placeholder. not used.
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds

        # get the first token using the current hidden state
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # get other tokens
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            # in FSDP, the number of forward pass need to be the same across devices
            while (
                self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
            ):  # leave some room for latent tokens
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            # for analysis purpose
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds

        else:
            return torch.tensor(tokens).view(1, -1)
