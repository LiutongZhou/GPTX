"""Functional implementation of the scaled dot-product attention"""

from __future__ import annotations

import math

import torch
from torch.nn.functional import (
    scaled_dot_product_attention as _scaled_dot_product_attention,
)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention.

    Parameters
    ----------
    query : torch.Tensor
        Query tensor; shape (N, ..., Hq, L, E).
    key : torch.Tensor
        Key tensor; shape (N, ..., H, S, E).
    value : torch.Tensor
        Value tensor; shape (N, ..., H, S, Ev).
    attn_mask : torch.Tensor, optional
        Attention mask; shape must be broadcastable to the shape of attention weights,
        which is (N, ..., L, S). Two types of masks are supported.
        A boolean mask where a value of True indicates that the element *should* take part in attention.
        A float mask of the same type as query, key, value that is added to the attention score.
    dropout_p : float
        Dropout probability; if greater than 0.0, dropout is applied.
    is_causal : bool
        If set to true, the attention masking is a lower triangular matrix when the mask is a
        square matrix. The attention masking has the form of the upper left causal bias due to the
        alignment (see torch.nn.attention.bias.CausalBias) when the mask is a non-square matrix.
        An error is thrown if both attn_mask and is_causal are set.
    scale : float, optional
        Scaling factor applied prior to softmax. If None, the default value is set
        to 1 / sqrt(E).
    enable_gqa : bool
        If set to True, Grouped Query Attention (GQA) is enabled, by default it is set to False.

    Returns
    -------
    torch.Tensor
        Attention output; shape (N, ..., Hq, L, Ev).

    Shape legend
    ------------
    - N: Batch size ... : Any number of other batch dimensions (optional)
    - S: Source sequence length
    - L: Target sequence length
    - E: Embedding dimension of the query and key
    - Ev: Embedding dimension of the value
    - Hq: Number of heads of query
    - H: Number of heads of key and value
    """
    try:
        return _scaled_dot_product_attention(
            query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
        )
    except:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        scale_factor = math.sqrt(scale_factor)
        attn_bias = torch.zeros((L, S), dtype=query.dtype)
        if is_causal:
            assert (
                attn_mask is None
            ), f"{is_causal=} and attn_mask cannot be set simultaneously"
            temp_mask = torch.ones((L, S), dtype=torch.bool).triu(diagonal=1)
            attn_bias.masked_fill_(temp_mask, float("-inf"))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        if enable_gqa:
            num_heads_query, num_heads_key, num_heads_value = (
                query.size(-3),
                key.size(-3),
                value.size(-3),
            )
            assert (
                num_heads_query % num_heads_key == 0
            ), f"{num_heads_query=} and {num_heads_key=} must be divisible"
            assert (
                num_heads_query % num_heads_value == 0
            ), f"{num_heads_query=} and {num_heads_value=} must be divisible"
            key = key.repeat_interleave(num_heads_query // num_heads_key, dim=-3)
            value = value.repeat_interleave(num_heads_query // num_heads_value, dim=-3)

        attn_weight = (scale_factor * query) @ (scale_factor * key.transpose(-2, -1))
        attn_weight += attn_bias.to(dtype=attn_weight.dtype, device=attn_weight.device)
        attn_score = torch.softmax(attn_weight, dim=-1)
        attn_score = torch.dropout(attn_score, dropout_p, train=True)
        return attn_score @ value
