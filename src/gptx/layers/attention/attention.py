"""TODO @liutong zhou implement the MultiHeadAttention class"""
import math
from typing import Optional

import torch
from torch import nn

from gptx.configs.llama_config import Config
from gptx.layers.attention.sdpa import scaled_dot_product_attention
from gptx.layers.positional_encoding import apply_rope, build_rope_cache


class MultiHeadAttention(nn.Module):
    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        self.config = config
        self.head_dim = self.n_embd // self.n_head if not config.head_dim
        self.config.rope_n_elem = int(self.rotary_percentage * self.head_size) if not self.config.rope_n_elem

        shape = (config.n_head + 2 * config.n_query_groups) * self.head_dim
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        # if `head_size` is explicitly specified in the config, `n_emd` might not be equal to `head_size * n_head`
        self.proj = nn.Linear(self.head_size * config.n_head, config.n_embd, bias=config.bias)
        # disabled by default
        self.kv_cache: Optional[KVCache] = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.head_dim)
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if self.config.n_query_groups != self.config.n_head and (input_pos is None or self.config.n_query_groups != 1):
            k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.head_dim)
            v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.head_dim)

        q = q.reshape(B, -1, T, self.head_dim)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.head_dim)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.head_dim)  # (B, nh_v, T, hs)

        q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

        y = self.scaled_dot_product_attention(q, k, v, mask)
        y = y.reshape(B, T, self.head_dim * self.config.n_head)  # re-assemble all head outputs side by side

        # output projection
        return self.proj(y)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.head_dim)

        # NOTE: with softcapping we cannot use SDPA
        y = scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=0.0,
            scale=scale,
            is_causal=mask is None,
            enable_gqa=self.config.n_query_groups != self.config.n_head
        )
        return y.transpose(1, 2)