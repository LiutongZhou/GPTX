import torch

from gptx.layers.positional_encoding import RotaryPositionalEmbedding


def test_rotary_positional_embedding():
    rope = RotaryPositionalEmbedding(dim=8, max_seq_len=128)
    x = torch.randn(1, 2, 1, 8)  # (batch_size, num_heads, seq_len, dim)
    x_roped = rope(x)
    assert x_roped.shape == x.shape

    x = torch.randn(4, 2, 4, 8)  # (batch_size, num_heads, seq_len, dim)
    x_roped = rope(x)
    assert x_roped.shape == x.shape

    x = torch.randn(2, 4, 8)  # (batch_size, num_heads, seq_len, dim)
    x_roped = rope(x)
    assert x_roped.shape == x.shape

    x = torch.randn(4, 8)  # (batch_size, num_heads, seq_len, dim)
    x_roped = rope(x)
    assert x_roped.shape == x.shape


def test_rotary_positional_embedding_interleaved():
    rope_interleaved = RotaryPositionalEmbedding(
        dim=128, max_seq_len=1024, interleaved=True
    )
    rope_rotate_half = RotaryPositionalEmbedding(
        dim=128, max_seq_len=1024, interleaved=False
    )
    q = torch.randn(1, 128)  # single token query
    k = torch.randn(10, 128)  # keys for 10 prefix tokens
    q_roped_interleaved = rope_interleaved(
        q,
        position_idx=torch.as_tensor(
            [
                9,
            ]
        ),
    )
    k_roped_interleaved = rope_interleaved(k, position_idx=torch.arange(10))
    q_dot_k_roped_interleaved = q_roped_interleaved @ k_roped_interleaved.T
    assert q_dot_k_roped_interleaved.shape == (1, 10)
    q_roped_rotate_half = rope_rotate_half(
        q,
        position_idx=torch.as_tensor(
            [
                9,
            ]
        ),
    )
    k_roped_rotate_half = rope_rotate_half(k, position_idx=torch.arange(10))
    q_dot_k_roped_rotate_half = q_roped_rotate_half @ k_roped_rotate_half.T
    assert q_dot_k_roped_rotate_half.shape == (1, 10)
