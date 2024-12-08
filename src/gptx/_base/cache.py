"""Base cache classes"""

from __future__ import annotations

from typing import ItemsView, Iterable, Iterator, KeysView, Mapping, ValuesView

import torch
from torch import nn


class TensorCache(nn.Module):
    """A TensorCache object can be attached to an arbitrary Module for caching tensors.

    Cached tensors are registered as non-persistent buffers and won't be saved in the state_dict.

    Attributes
    ----------
    _allowed_keys : None | tuple[str, ...]
        If set in subclass, only tensor named with allowed keys can enter the cache.
        If None, any key is allowed. Default is None.
    """

    _allowed_keys : None | tuple[str, ...] = None

    def __init__(
        self,
        from_: (
            TensorCache
            | Mapping[str, torch.Tensor]
            | Iterable[tuple[str, torch.Tensor]]
            | None
        ) = None,
    ):
        super().__init__()
        self._keys: set[str] = set()
        if from_ is not None:
            self.update(from_)

    def update(
        self,
        other: (
            TensorCache
            | Mapping[str, torch.Tensor]
            | Iterable[tuple[str, torch.Tensor]]
        ),
    ):
        """Update the cache with the tensors from the other cache / dict / iterable."""
        name_tensor_pairs: Iterable[tuple[str, torch.Tensor]]
        if isinstance(other, (Mapping, TensorCache)):
            name_tensor_pairs = other.items()
        elif isinstance(other, Iterable):
            name_tensor_pairs = other
        else:
            raise TypeError(f"Unsupported type {type(other)=} for updating the cache.")
        for name, tensor in name_tensor_pairs:
            self[name] = tensor

    def __setitem__(self, name: str, tensor: torch.Tensor):
        """Add/update a tensor to the cache."""
        if self._allowed_keys and name not in self._allowed_keys:
            raise KeyError(
                f"Only {self._allowed_keys} are allowed as keys for the cache."
            )
        self._keys.add(name)
        self.register_buffer(name, tensor, persistent=False)

    def __contains__(self, name: str) -> bool:
        """Check if a tensor with the given name is cached."""
        return name in self._keys

    def __getitem__(self, name: str) -> torch.Tensor:
        """Return the cached tensor with the given name.

        Raises
        ------
        KeyError
            If the tensor with the given name is not found in the cache.
        """
        if name not in self:
            raise KeyError(f"Tensor with name '{name}' not found in cache.")
        return self.get_buffer(name)

    def __delitem__(self, name: str):
        """Delete the cached tensor with the given name."""
        self._keys.discard(name)
        delattr(self, name)

    def __len__(self) -> int:
        """Return the number of cached tensors."""
        return len(self._keys)

    def __bool__(self) -> bool:
        """Return whether the cache is empty."""
        return len(self) > 0

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over the cached tensor names."""
        return iter(self._keys)

    @property
    def cached_tensors(self) -> dict[str, torch.Tensor]:
        """Return a dictionary of the cached tensors."""
        return dict(self.named_buffers())

    def clear(self):
        """Clear all the cached tensors and free the GPU memory."""
        is_on_cuda = False
        for name, tensor in self.cached_tensors.items():
            is_on_cuda &= tensor.is_cuda
            del self[name]
        if is_on_cuda:
            torch.cuda.empty_cache()

    def keys(self) -> KeysView[str]:
        """Return an iterable over the cached tensor names."""
        return self.cached_tensors.keys()

    def values(self) -> ValuesView[torch.Tensor]:
        """Return an iterable over the cached tensors."""
        return self.cached_tensors.values()

    def items(self) -> ItemsView[str, torch.Tensor]:
        """Return an iterable over the cached tensor names and tensors."""
        return self.cached_tensors.items()

    def pop(self, name: str) -> torch.Tensor:
        """Remove and return the cached tensor with the given name.

        Raises
        ------
        KeyError
            If the tensor with the given name is not found in the cache.
        """
        tensor = self[name]
        del self[name]
        return tensor
