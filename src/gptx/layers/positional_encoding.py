"""Positional Embedding layers"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from einops import repeat

from .._base import TensorCache

__all__ = ["RotaryPositionalEmbedding", "index_select"]


def index_select(
    tensor: torch.Tensor, dim: int, index: torch.IntTensor | torch.LongTensor
) -> torch.Tensor:
    """Generalizing torch.index_select to support multi-dimensional index.

    Parameters
    ----------
    tensor : torch.Tensor
        the input tensor to select from along the specified dimension
    dim : int
        the dimension along which we select by index
    index : IntTensor | LongTensor
        shape = (..., index_size) with the last dimension being the indices
        and all the previous dimensions as batch dimensions
        containing the indices to select from the tensor

    Examples
    --------
    >>> cos_cache = torch.arange(10).reshape(5, 2)
    >>> index = torch.as_tensor([[0, 1, 2], [2, 3, 4]])
    >>> index_select(cos_cache, dim=0, index=index)
    tensor([[[0, 1],
            [2, 3],
            [4, 5]],
           [[4, 5],
            [6, 7],
            [8, 9]]])
    >>> index_select(cos_cache, dim=0, index=[4, 3, 2, 1, 0])
    tensor([[8, 9],
        [6, 7],
        [4, 5],
        [2, 3],
        [0, 1]])
    """
    index = torch.as_tensor(index, dtype=torch.long)
    while dim < 0:
        dim += tensor.ndim

    if index.ndim == 1:
        return tensor.index_select(dim, index)

    *batch_shape, index_size = index.shape
    index_flattened = index.view(-1)  # shape = (prod(*batch_shape, index_size), )
    tensor_select_flattened = tensor.index_select(
        dim, index_flattened
    )  # shape = (..., dim=prod(*batch_shape, index_size), ...)
    unflatten_shape = tensor.shape[:dim] + (-1, index_size) + tensor.shape[dim + 1 :]
    tensor_select_unflatten = tensor_select_flattened.view(unflatten_shape)
    tensor_select = tensor_select_unflatten.movedim(dim, 0)
    output_shape = tuple(batch_shape) + tensor_select.shape[1:]
    return tensor_select.view(*output_shape)


class CosSinCache(TensorCache):
    _allowed_keys = ("cos", "sin")


class RotaryPositionalEmbedding[T: torch.Tensor](nn.Module):
    """A layer that apply Rotary Positional Embedding (RoPE) to the input tensor.

    Examples
    --------
    >>> rope = RotaryPositionalEmbedding(dim=8, max_seq_len=128)
    >>> x = torch.randn(1, 2, 1, 8)  # (batch_size, num_heads, seq_len, hidden_dim)
    >>> x_roped = rope(x)
    >>> x_roped.shape
    torch.Size([1, 2, 1, 8])
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int | None = None,
        base=500_000,
        condense_ratio=1,
        interleaved=False,
        device: torch.device | None = None,
        dtype=torch.float32,
    ):
        """Initialize the Rotary Positional Embedding (RoPE) layer.

        Parameters
        ----------
        dim : int
            rope hidden dimension. (usually same as the hidden_dim of each attention head)
        max_seq_len : int | None
            max sequence length. Default is None.
            If None, it will be dynamically set to the max len of observed sequences.
        base : int
            The base for computing inverse frequencies \Theta in the rotary matrix
        condense_ratio : int
            Ratio to condense the position indices. Default is 1.
            If condense_ratio > 1, the position indices will be divided by condense_ratio.
        interleaved : bool
            If False, rotate the second half of the hidden dimension.
            If True, construct interleaved even and odd pairs of the hidden dimension for rotation.
        device : torch.device | None
        dtype : torch.dtype
            For numerical stability, it is recommended to use torch.float32.
        """
        super().__init__()
        self.dim = dim
        self.base = base
        self.condense_ratio = condense_ratio
        self.interleaved = interleaved
        self.device = device
        self.dtype = dtype
        self.cos_sin_cache = CosSinCache()
        self._max_seq_len = None
        self.max_seq_len = max_seq_len

    @property
    def max_seq_len(self) -> int | None:
        return self._max_seq_len

    @max_seq_len.setter
    def max_seq_len(self, value: int | None):
        match value:
            case None:
                self.cos_sin_cache.clear()
            case int() if value > (self._max_seq_len or 0):
                self._max_seq_len = value
                self._prefill_cos_sin_cache()
        self._max_seq_len = value

    def _prefill_cos_sin_cache(self) -> CosSinCache:
        """Precompute the cos and sin parts of the corresponding elements in the rotary matrix.

        Each element in the rotary matrix corresponds to a (position, dimension) pair.
        The cache will be reused in the forward pass.

        Returns
        -------
        cos_sin_cache : CosSinCache
            cos_sin_cache["cos"], cos_sin_cache["sin"] are precomputed cosine and sine parts
            of the rotary matrix. Both have shape = (seq_len, hidden_dim // 2)
        """
        base, dim, max_seq_len = self.base, self.dim, self.max_seq_len
        # Compute the inverse frequencies theta
        theta = base ** (
            torch.arange(0, dim, 2, dtype=self.dtype, device=self.device) / -dim
        )
        # Create position indices `[0, 1, ..., seq_len - 1]`
        position_idx = (
            torch.arange(max_seq_len, dtype=self.dtype, device=self.device)
            / self.condense_ratio
        )
        # shape (seq_len, hidden_dim // 2)
        idx_cross_theta = torch.outer(position_idx, theta)
        self.cos_sin_cache["cos"] = idx_cross_theta.cos()
        self.cos_sin_cache["sin"] = idx_cross_theta.sin()
        return self.cos_sin_cache

    def forward(
        self, x: T, position_idx: torch.IntTensor | torch.LongTensor | None = None
    ) -> T:
        """Apply the Rotary Positional Embedding (RoPE) to the input tensor.

        Parameters
        ----------
        x : T
            Input tensor. Usually the projected query or key tensor.
            shape (batch_size, ..., seq_len, head_dim)
        position_idx : torch.IntTensor | torch.LongTensor | None
            shape (seq_len, ) or (batch_size, seq_len).
            If None, the position indices will be set to [0, ..., seq_len - 1] for each sequence.

        Returns
        -------
        x_roped : T
            shape (batch_size, num_heads, seq_len, head_dim)
        """
        assert (
            x.ndim >= 2
        ), f"x must have at least 2 dimensions (seq_len, hidden_dim), but got {x.ndim=}"
        if position_idx is None:
            seq_len = x.shape[-2]
            if seq_len > (self.max_seq_len or 0):
                warnings.warn(
                    f"max position index of the input tensor x {seq_len}"
                    f" exceeded the preset max_seq_len {self.max_seq_len}. "
                    f"Overwriting the max_seq_len of the RoPE layer {self}",
                    UserWarning,
                )

                self.max_seq_len = seq_len
            cos_cached, sin_cached = (
                self.cos_sin_cache["cos"],
                self.cos_sin_cache["sin"],
            )
            cos = cos_cached[:seq_len]
            sin = sin_cached[:seq_len]
        else:  # position_idx is not None
            cur_max_seq_len = position_idx[..., -1].max().item()
            if cur_max_seq_len > (self.max_seq_len or 0):
                warnings.warn(
                    f"max position index of the input tensor x {cur_max_seq_len}"
                    f" exceeded the preset max_seq_len {self.max_seq_len}. "
                    f"Overwriting the max_seq_len of the RoPE layer {self}",
                    UserWarning,
                )

                self.max_seq_len = cur_max_seq_len
            cos_cached, sin_cached = (
                self.cos_sin_cache["cos"],
                self.cos_sin_cache["sin"],
            )
            cos = index_select(cos_cached, dim=0, index=position_idx)
            sin = index_select(sin_cached, dim=0, index=position_idx)
        return self.apply_rope(x, cos, sin, interleaved=self.interleaved)

    @staticmethod
    def apply_rope(x: T, cos: torch.Tensor, sin: torch.Tensor, interleaved=False) -> T:
        """Apply the Rotary Positional Embedding (RoPE) to the projected query or key (x).

        Parameters
        ----------
        x : T
            shape (batch_size, ..., seq_len, head_dim)
        cos, sin : torch.Tensor
            shape (seq_len, rotary_dim // 2) if batch_size == 1.
            (batch_size, seq_len, rotary_dim // 2) for multiple sequences.
            In most cases, rotary_dim = hidden_dim.

        Returns
        -------
        x_roped : T
            shape (batch_size, num_heads, seq_len, head_dim)
        """
        rotary_dim = cos.shape[-1] * 2
        assert (
            x.ndim >= 2
        ), f"x must have at least 2 dimensions (seq_len, hidden_dim), but got {x.ndim=}"
        assert rotary_dim <= x.shape[-1]
        x_to_rope, x_no_rope = x[..., :rotary_dim], x[..., rotary_dim:]
        cos = repeat(
            cos, "... d -> ... (2 d)" if not interleaved else "... d -> ... (d 2)"
        )  # shape (seq_len, rotary_dim)
        sin = repeat(
            sin, "... d -> ... (2 d)" if not interleaved else "... d -> ... (d 2)"
        )  # shape (seq_len, rotary_dim)

        while cos.ndim < x_to_rope.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        def _to_sin_correspondents(x: T, interleave: bool = False) -> T:
            if not interleave:
                x1, x2 = x.chunk(chunks=2, dim=-1)
                return torch.cat((-x2, x1), dim=-1)
            else:
                x1, x2 = x[..., 0::2], x[..., 1::2]
                x_interleaved = torch.stack(
                    (-x2, x1), dim=-1
                )  # shape (..., hidden_dim // 2, 2)
                # "... half_dim 2 -> ... (half_dim 2)"  # shape (..., hidden_dim)
                return x_interleaved.flatten(
                    start_dim=-2, end_dim=-1
                )  # shape (..., hidden_dim)

        x_roped = x_to_rope * cos + _to_sin_correspondents(x_to_rope, interleaved) * sin
        return torch.cat((x_roped, x_no_rope), dim=-1)
