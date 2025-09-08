"""Test cases for the TensorCache class."""

import pytest
import torch

from gptx._base import TensorCache


class TestTensorCache:

    def test_cache_initialization_with_none(self):
        cache = TensorCache()
        assert not cache
        assert len(cache) == 0

    def test_cache_initialization_with_mapping(self):
        cache = TensorCache({"a": torch.tensor([1, 2, 3])})
        assert "a" in cache
        assert torch.equal(cache["a"], torch.tensor([1, 2, 3]))
        assert "b" not in cache

    def test_cache_initialization_with_iterable(self):
        cache = TensorCache([("b", torch.tensor([4, 5, 6]))])
        assert "b" in cache
        assert torch.equal(cache["b"], torch.tensor([4, 5, 6]))

    def test_cache_initialization_with_tensor_cache(self):
        old_cache = TensorCache({"a tensor": torch.tensor([7.0, 8.0, 9.0])})
        cache = TensorCache(old_cache)
        assert "a tensor" in cache
        assert torch.allclose(cache["a tensor"], torch.tensor([7.0, 8.0, 9.0]))

    def test_cache_update(self):
        cache = TensorCache()
        cache.update({"a": torch.tensor([1, 2, 3])})
        cache.update([("b", torch.tensor([4, 5, 6]))])

        assert len(cache) == 2
        assert "a" in cache
        assert torch.equal(cache["a"], torch.tensor([1, 2, 3]))

        assert "b" in cache
        assert torch.equal(cache["b"], torch.tensor([4, 5, 6]))

        cache.update(TensorCache({"a tensor": torch.tensor([7.0, 8.0, 9.0, 10.0])}))
        assert len(cache) == 3
        assert "a tensor" in cache
        assert torch.allclose(cache["a tensor"], torch.tensor([7.0, 8.0, 9.0, 10.0]))

    def test_cache_set_get_delete_clear(self):
        cache = TensorCache()
        cache["a b c"] = torch.tensor([1, 2, 3])
        cache["a_b_c"] = torch.tensor([4, 5, 6])
        assert len(cache) == 2
        assert torch.equal(cache["a b c"], torch.tensor([1, 2, 3]))
        assert torch.equal(cache["a_b_c"], torch.tensor([4, 5, 6]))

        cache["d"] = torch.arange(7, 11, dtype=torch.float)
        assert "d" in cache
        assert len(cache) == 3
        assert torch.allclose(cache["d"], torch.tensor([7, 8, 9, 10]) * 1.0)

        del cache["d"]
        assert "d" not in cache
        with pytest.raises(KeyError):
            cache["d"]

        assert cache
        assert len(cache) == 2
        assert "a b c" in cache
        assert "a_b_c" in cache
        assert torch.equal(cache["a_b_c"], torch.tensor([4, 5, 6]))

        cache.clear()
        assert not cache
        assert len(cache) == 0
        with pytest.raises(KeyError):
            cache["a b c"]

    def test_cache_pop(self):
        cache = TensorCache(
            {"a": torch.tensor([1, 2, 3]), "b": torch.tensor([4, 5, 6])}
        )
        assert len(cache) == 2

        a = cache.pop("a")
        assert torch.equal(a, torch.tensor([1, 2, 3]))
        assert "a" not in cache

        assert "b" in cache
        assert len(cache) == 1
        assert torch.equal(cache["b"], torch.tensor([4, 5, 6]))

        b = cache.pop("b")
        assert "b" not in cache
        assert not cache
        assert torch.equal(b, torch.tensor([4, 5, 6]))

    def test_cache_iter(self):
        cache = TensorCache(
            {"a": torch.tensor([1, 2, 3]), "b": torch.tensor([4, 5, 6])}
        )
        keys = set(cache)
        assert keys == {"a", "b"}

    def test_cache_keys_values_items(self):
        cache = TensorCache(
            {"a": torch.tensor([1, 2, 3]), "b": torch.tensor([4, 5, 6])}
        )
        keys = list(cache.keys())
        values = list(cache.values())
        items = list(cache.items())
        assert keys == ["a", "b"]
        assert torch.equal(values[0], torch.tensor([1, 2, 3]))
        assert torch.equal(values[1], torch.tensor([4, 5, 6]))
        assert items[0][0] == "a"
        assert torch.equal(items[0][1], torch.tensor([1, 2, 3]))
        assert items[1][0] == "b"
        assert torch.equal(items[1][1], torch.tensor([4, 5, 6]))
