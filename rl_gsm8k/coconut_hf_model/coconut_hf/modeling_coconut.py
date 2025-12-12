import copy
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_coconut import CoconutConfig

MAX_N_LATENT = 8


def top_p_sample_1d(probs: torch.Tensor, top_p: float = 0.9) -> int:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)
    cutoff = torch.searchsorted(cumulative, torch.tensor(top_p, device=probs.device))
    if cutoff < sorted_probs.numel() - 1:
        sorted_probs[cutoff + 1 :] = 0.0
    sorted_probs = sorted_probs / (sorted_probs.sum() + 1e-12)
    sampled = torch.multinomial(sorted_probs, 1).item()
    return sorted_indices[sampled].item()


class CoconutNoSharedForCausalLM(PreTrainedModel):
    config_class = CoconutConfig
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    def __init__(self, config: CoconutConfig):
        super().__init__(config)

        base_cfg_dict = copy.deepcopy(config.base_model_config)
        base_model_type = base_cfg_dict.pop("model_type", None)
        base_cfg_dict.pop("auto_map", None)
        base_cfg_dict.pop("_name_or_path", None)

        if base_model_type is None:
            raise ValueError("base_model_config must contain 'model_type'")

        base_cfg = AutoConfig.for_model(base_model_type, **base_cfg_dict)
        self.base_causallm = AutoModelForCausalLM.from_config(base_cfg)

        self.latent_token_id = config.latent_token_id
        self.start_latent_id = config.start_latent_id
        self.end_latent_id = config.end_latent_id
        self.eos_token_id = config.eos_token_id

        self.state_combination_method = config.state_combination_method
        self.combination_use_gating = bool(config.combination_use_gating)

        self.embedding = self.base_causallm.get_input_embeddings()
        hidden_dim = self.embedding.embedding_dim

        # ✅ 与训练版 Coconut_no_shared 对齐：adapters 永远存在（checkpoint 里也永远有）
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(MAX_N_LATENT)
            ]
        )

        # 其余模块按训练版逻辑：只有对应 method 才创建
        self.projs = None
        if self.state_combination_method == "projection":
            self.projs = nn.ModuleList(
                [nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(MAX_N_LATENT)]
            )

        self.cross_attns = None
        if self.state_combination_method == "cross_attention":
            self.cross_attns = nn.ModuleList(
                [
                    nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
                    for _ in range(MAX_N_LATENT)
                ]
            )

        self.gates = None
        if self.combination_use_gating and self.state_combination_method != "continuous_only":
            self.gates = nn.ModuleList(
                [nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(MAX_N_LATENT)]
            )

        self.post_init()

    def get_input_embeddings(self):
        return self.base_causallm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.base_causallm.set_input_embeddings(value)
        self.embedding = self.base_causallm.get_input_embeddings()

    def resize_token_embeddings(self, new_num_tokens=None, pad_to_multiple_of=None):
        out = self.base_causallm.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of=pad_to_multiple_of
        )
        self.embedding = self.base_causallm.get_input_embeddings()
        return out

    def _build_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        bs, seqlen = input_ids.shape
        return torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bs, -1)

    def _combine_states(
        self,
        continuous_hidden: torch.Tensor,
        discrete_hidden: Optional[torch.Tensor],
        pass_idx: int,
    ) -> torch.Tensor:
        method = self.state_combination_method

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
            q = continuous_hidden.view(1, 1, -1)
            k = discrete_hidden.view(1, 1, -1)
            v = discrete_hidden.view(1, 1, -1)
            attn_out, _ = self.cross_attns[pass_idx](q, k, v)
            candidate = attn_out.view(-1)

        else:
            raise ValueError(f"Unknown state_combination_method: {method}")

        if self.gates is not None and method != "continuous_only":
            gate_in = torch.cat([continuous_hidden, candidate], dim=-1)
            gate = torch.sigmoid(self.gates[pass_idx](gate_in))
            candidate = gate * candidate + (1.0 - gate) * continuous_hidden

        return candidate

    def _apply_latent_passes(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        top_p: float = 0.9,
        sample_discrete: bool = True,
    ) -> torch.Tensor:
        device = input_ids.device
        bs, seqlen = input_ids.shape
        inputs_embeds = self.embedding(input_ids)

        latent_pos = (input_ids == self.latent_token_id).nonzero(as_tuple=False)
        latent_lists = [[] for _ in range(bs)]
        for b, p in latent_pos.tolist():
            latent_lists[b].append(p)

        max_latents = min(max((len(x) for x in latent_lists), default=0), MAX_N_LATENT)
        if max_latents == 0:
            return inputs_embeds

        # 稳妥起见：每个 pass 全序列重算（兼容 transformers 新 Cache）
        for pass_idx in range(max_latents):
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]  # (bs, seqlen, hidden)

            new_embeds = inputs_embeds.clone()

            for b in range(bs):
                if len(latent_lists[b]) <= pass_idx:
                    continue
                token_idx = latent_lists[b][pass_idx]
                if token_idx - 1 < 0:
                    continue

                continuous_hidden = hidden[b, token_idx - 1, :]

                if self.state_combination_method == "continuous_only":
                    discrete_hidden = None
                else:
                    logits_for_pos = self.base_causallm.lm_head(continuous_hidden)
                    probs = torch.softmax(logits_for_pos, dim=-1)

                    if sample_discrete:
                        sampled_id = top_p_sample_1d(probs, top_p=top_p)
                    else:
                        sampled_id = int(torch.argmax(probs).item())

                    discrete_hidden = self.embedding(
                        torch.tensor(sampled_id, device=device, dtype=torch.long)
                    )

                replacement = self._combine_states(continuous_hidden, discrete_hidden, pass_idx)
                new_embeds[b, token_idx, :] = replacement

            inputs_embeds = new_embeds

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids is required")

        device = input_ids.device
        bs, seqlen = input_ids.shape

        if attention_mask is None:
            attention_mask = torch.ones((bs, seqlen), device=device, dtype=torch.long)
        if position_ids is None:
            position_ids = self._build_position_ids(input_ids)

        inputs_embeds = self._apply_latent_passes(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            top_p=0.9,
            sample_discrete=True,
        )

        out = self.base_causallm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        logits = out.logits

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 16,
        do_sample: bool = False,
        top_p: float = 0.9,
        temperature: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        device = input_ids.device
        bs, seqlen = input_ids.shape

        if attention_mask is None:
            attention_mask = torch.ones((bs, seqlen), device=device, dtype=torch.long)

        sequences = []
        for b in range(bs):
            cur_ids = input_ids[b : b + 1].clone()
            cur_mask = attention_mask[b : b + 1].clone()

            for _ in range(max_new_tokens):
                out = self.forward(cur_ids, attention_mask=cur_mask)
                logits = out.logits[0, -1, :]

                if temperature is not None and temperature != 1.0:
                    logits = logits / max(float(temperature), 1e-6)

                if do_sample:
                    probs = torch.softmax(logits, dim=-1)
                    next_id = top_p_sample_1d(probs, top_p=top_p)
                else:
                    next_id = int(torch.argmax(logits).item())

                next_tok = torch.tensor([[next_id]], device=device, dtype=torch.long)
                cur_ids = torch.cat([cur_ids, next_tok], dim=1)
                cur_mask = torch.cat(
                    [cur_mask, torch.ones((1, 1), device=device, dtype=torch.long)], dim=1
                )

                if self.eos_token_id is not None and next_id == int(self.eos_token_id):
                    break

            sequences.append(cur_ids[0])

        max_len = max(x.numel() for x in sequences)
        pad_id = int(self.eos_token_id) if self.eos_token_id is not None else 0
        padded = []
        for x in sequences:
            if x.numel() < max_len:
                pad = torch.full((max_len - x.numel(),), pad_id, device=device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)
            padded.append(x)

        return torch.stack(padded, dim=0)
    
    def get_correct_attn_implementation(self, *args, **kwargs):
        return "eager"
