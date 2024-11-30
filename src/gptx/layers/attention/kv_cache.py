"""Key-value cache for caching keys and values in attention layers at inference time."""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Annotated, Any, NamedTuple, override

import torch

from ..._base import TensorCache


class KeyValue[
    StatesProjected: Annotated[
        torch.Tensor, {"shape": ("batch_size", "num_heads", "seq_len", "head_dim")}
    ]
](NamedTuple):
    """Named tuple for projected key and value tensors.

    Attributes
    ----------
    key : StatesProjected
        Projected key tensor
    value : StatesProjected
        Projected value tensor
    """

    key: StatesProjected
    value: StatesProjected


class KVCache(ABC, TensorCache):
    """Abstract base class for key-value cache corresponding to an attention layer"""

    _allowed_keys = ("keys", "values")

    def __init__(self):
        super(ABC, self).__init__()

    @override
    def __setitem__(self, name: str, tensor: torch.Tensor):
        """Override the __setitem__ method to enforce the name of the tensor to be cached."""
        if name not in self._allowed_keys:
            raise KeyError(
                f"Only {self._allowed_keys} are allowed as keys for the cache."
            )
        super().__setitem__(name, tensor)

    @abstractmethod
    @override
    def __len__(self) -> int:
        """Return the sequence length of the cached (keys, values)"""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def forward(self, key_value: KeyValue, **kwargs: dict[str, Any]) -> KeyValue:
        """Cache the current key_value and return all the cached keys and values.

        Parameters
        ----------
        key_value : KeyValue
            The current projected key and value tensors to cache.
        kwargs : dict[str, Any]
            Additional arguments for the cache subclass.

        Returns
        -------
        key_value_all : KeyValue
            All the cached keys and values including the current key value pair.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _validate_key_value_shape(self, key_value: KeyValue) -> None:
        """Validate the shape of the key and value tensors.

        Raises
        ------
        ValueError
            If the key and value tensors have different shapes or are not 4D tensors.
        """
        key, value = key_value
        if key.ndim != value.ndim != 4 or key.shape != value.shape:
            raise ValueError(
                f"Expected key and value to have identical shape: "
                f"(batch_size, num_heads, seq_len, head_dim) but got "
                f"{key.shape=} and {value.shape=}"
            )


class DynamicKVCache(KVCache):
    """Dynamic key-value cache can grow dynamically while decoding.

    As more tokens are generated, the cache grows dynamically. No sequence length limit.

    However, the model might not be compiled by torch.compile if DynamicKVCache is used due
    to dynamically changed shapes at inference time.
    """

    def __len__(self) -> int:
        """Return the sequence length of the cached (keys, values)"""
        if "values" not in self:
            return 0
        return self["values"].size(2)

    def _raise_if_empty(self):
        """Raise an AttributeError if the cache is empty."""
        if not self:
            raise AttributeError("The cache is empty.")

    @property
    def batch_size(self) -> int:
        """Return the max supported batch size"""
        self._raise_if_empty()
        return self["values"].size(0)

    @property
    def num_heads(self) -> int:
        """Return the number of heads in the cached keys and values."""
        self._raise_if_empty()
        return self["values"].size(1)

    @property
    def head_dim(self) -> int:
        """Return the head dimension of the cached keys and values."""
        self._raise_if_empty()
        return self["values"].size(3)

    def forward(self, key_value: KeyValue, **kwargs: dict[str, Any]) -> KeyValue:
        """Cache the current key_value and return all the cached keys and values.

        Parameters
        ----------
        key_value : KeyValue
            The current projected key and value tensors to cache.

        Returns
        -------
        key_value_all : KeyValue
            All the cached keys and values including the current key value pair.

        Raises
        ------
        ValueError
            If the key and value tensors have different shapes or are not 4D tensors.
        """
        self._validate_key_value_shape(key_value)
        key, value = key_value
        if not self:
            self["keys"] = key
            self["values"] = value
        else:
            self["keys"] = torch.cat((self["keys"], key), dim=-2)
            self["values"] = torch.cat((self["values"], value), dim=-2)
        return KeyValue(self["keys"], self["values"])


class StaticKVCache(KVCache):
    """Static key-value cache with pre-allocated memory for caching keys and values.

    The cache pre-allocates a tensor of shape (max_batch_size, num_heads, max_seq_len, head_dim)
    for caching keys and values. This is `torch.compile` friendly but has a sequence length limit.
    """

    def __init__(
        self,
        max_batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        **kwargs,
    ):
        """Initialize the static key-value cache with keys and values as pre-allocated empty tensors."""
        super().__init__()
        self.max_batch_size = max_batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.shape = shape = (max_batch_size, num_heads, max_seq_len, head_dim)
        for name in self._allowed_keys:
            self[name] = torch.empty(shape, dtype=dtype, device=device, **kwargs)
        self.used_seq_len = 0

    def __len__(self) -> int:
        """Return the used sequence length of the cached (keys, values)"""
        return self.used_seq_len

    @override
    def clear(self):
        """Reset all cached keys and values to empty"""
        for tensor_cached in self.values():
            tensor_cached.zero_()
        self.used_seq_len = 0

    @override
    def _validate_key_value_shape(self, key_value: KeyValue) -> None:
        """Validate the shape of the key and value tensors.

        Raises
        ------
        ValueError
            If the key and value tensors have different shapes or are not 4D tensors or
            have incompatible shapes with the cached keys and values.
        """
        super()._validate_key_value_shape(key_value)
        key, value = key_value
        batch_size, num_heads, num_new_tokens, head_dim = value.shape
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} of current key value exceeds the maximum batch size "
                f"{self.max_batch_size} set in {self}"
            )
        if self.used_seq_len + num_new_tokens > self.max_seq_len:
            raise ValueError(
                f"Sequence length {self.used_seq_len + num_new_tokens} exceeds the maximum "
                f"sequence length {self.max_seq_len} set in {self}"
            )
        if head_dim != self.head_dim:
            raise ValueError(
                f"Head dimension {head_dim} of current key value does not match the head dimension "
                f"{self.head_dim} set in {self}"
            )
        if num_heads != self.num_heads:
            raise ValueError(
                f"Number of heads {num_heads} of current key value does not match the number of heads "
                f"{self.num_heads} set in {self}"
            )

    def forward(self, key_value: KeyValue, **kwargs) -> KeyValue:
        """Cache the current key_value and return all the cached keys and values.

        Parameters
        ----------
        key_value : KeyValue
            The current projected key and value tensors to cache.
            shape (batch_size, num_heads, num_new_tokens, head_dim)

        Returns
        -------
        key_value_all : KeyValue
            All the cached keys and values including the current key value pair.

        Raises
        ------
        ValueError
            If the key and value tensors have different shapes or are not 4D tensors or
            have incompatible shapes with the cached keys and values.
        """
        self._validate_key_value_shape(key_value)
        key_new, value_new = key_value
        batch_size, _, num_new_tokens, _ = key_new.shape
        keys, values = self["keys"], self["values"]
        if key_new.device != keys.device or value_new.device != values.device:
            warnings.warn(
                "Current (key,value) and cached (keys,values) are not on the same device.\n"
                "Moving (key,value) to the device of cached (keys,values)",
                UserWarning,
            )
            key_new = key_new.to(device=keys.device)
            value_new = value_new.to(device=values.device)
        if keys.dtype != key_new.dtype or values.dtype != value_new.dtype:
            warnings.warn(
                "Current (key,value) are not of the same dtype with cached (keys,values).\n"
                "Casting cached (keys,values) to the dtype of current (key,value)",
                UserWarning,
            )
            self["keys"] = keys = keys.to(dtype=key_new.dtype)
            self["values"] = values = values.to(dtype=value_new.dtype)
        keys[
            :batch_size, :, self.used_seq_len : self.used_seq_len + num_new_tokens, :
        ] = key_new
        values[
            :batch_size, :, self.used_seq_len : self.used_seq_len + num_new_tokens, :
        ] = value_new
        self.used_seq_len += num_new_tokens
        return KeyValue(
            keys[:batch_size, :, : self.used_seq_len, :],
            values[:batch_size, :, : self.used_seq_len, :],
        )
