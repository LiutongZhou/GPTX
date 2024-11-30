"""Base cache classes"""

from __future__ import annotations

from typing import ItemsView, Iterable, Iterator, KeysView, Literal, Mapping, ValuesView

import torch
from torch import nn


class TensorCache(nn.Module):
    """A TensorCache object can be attached to an arbitrary Module for caching tensors.

    Cached tensors are registered as non-persistent buffers and won't be saved in the state_dict.
    """

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
        self._keys: dict[str, Literal[None]] = {}
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
        if hasattr(other, "items"):
            for name, tensor in other.items():
                self[name] = tensor
        elif isinstance(other, Iterable):
            for name, tensor in other:
                self[name] = tensor

    def __setitem__(self, name: str, tensor: torch.Tensor):
        """Add/update a tensor to the cache."""
        self._keys[name] = None
        self.register_buffer(name, tensor, persistent=False)

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
        del self._keys[name]
        delattr(self, name)

    def __contains__(self, name: str) -> bool:
        """Check if a tensor with the given name is cached."""
        return name in self._keys

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
        for name, tensor in self.cached_tensors:
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
        """Remove and return the cached tensor with the given name."""
        tensor = self[name]
        del self[name]
        return tensor
