""" """

from dataclasses import dataclass


@dataclass
class RoPEAdjustments:
    factor: float = 32.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_seq_len: int = 8192


"""
class BaseLlama32Config:
    name: str
    block_size: int = 131072
    vocab_size: int = 128000
    padded_vocab_size: int = 128256
    rotary_percentage: float = 1.0
    parallel_residual: bool = False
    bias: bool = False
    norm_class_name: str = "RMSNorm"
    mlp_class_name: Type[nn.Module] = LLaMAMLP
    rope_base: int = 500000
    rope_adjustments: RoPEAdjustments = RoPEAdjustments()
    n_layer: int = 0
    n_embd: int = 0
    n_head: int = 0
    n_query_groups: int = 8
    intermediate_size: int = 0

@dataclass
class Llama32_1B(BaseLlama32Config):
    name: str = "Llama-3.2-1B"
    n_layer: int = 16
    n_embd: int = 2048
    n_head: int = 32
    intermediate_size: int = 8192

@dataclass
class Llama32_7B(BaseLlama32Config):
   name: str = "Llama-3.2-7B"
    n_layer: int = 32
    n_embd: int = 4096
    n_head: int = 32
    intermediate_size: int = 11008
"""


@dataclass
class Config:
    """Model Architecture Configuration.

    Parameters
    ----------
    n_embd : int
        Vocabulary's embedding dimension.
    intermediate_size : int
        Intermediate hidden layer's dimension.
        Usually 4 times the embedding dimension.
    bias : bool
        Whether to use bias in the linear layers.
    """

    name: str = "Llama-3.2-1B"
    block_size: int = 131072
    vocab_size: int = 128000
    n_layer: int = 16
    n_embd: int = 2048
    n_head: int = 32
    n_query_groups: int = 8
    rotary_percentage: float = 1.0
    parallel_residual: bool = False
    bias: bool = False
    norm_class_name: str = "RMSNorm"
    intermediate_size: int = 8192
    rope_base: int = 500000
    rope_adjustments: RoPEAdjustments = RoPEAdjustments()
