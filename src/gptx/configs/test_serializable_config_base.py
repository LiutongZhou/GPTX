"""Test cases for the .serializable_config_base.py module."""

import statistics
import tempfile
from collections import namedtuple
from dataclasses import dataclass
from itertools import product
from typing import Annotated, Any, Callable, Literal

import pytest
from pydantic import AfterValidator, Field, ValidationError

from .serializable_config_base import SerializableConfigBase


class FooBlockConfig(SerializableConfigBase):
    name: str
    """Name of the block"""


FooAttentionShape = namedtuple(
    "FooAttentionShape", ["embed_dim", "hidden_dim", "kv_compressed_rank", "q_compressed_rank", "rope_rank"]
)


@dataclass
class FooAttentionConfig:
    """For testing dataclass serialization"""

    attention_weight_shape: FooAttentionShape = FooAttentionShape(32, 32, 32, 32, 32)
    type: Literal["MLA"] = "MLA"
    num_heads: int = 8
    use_bias: bool = True
    use_flash_attention: bool = True

    def __post_init__(self):
        if self.type != "MLA":
            raise NotImplementedError(f"{self.type} is not supported")


def divisible_by_8(attention: FooAttentionConfig) -> FooAttentionConfig:
    """Validator to ensure that attention_weight_shape are divisible by 8."""
    for shape in attention.attention_weight_shape:
        if shape % 8 != 0:
            raise ValueError("shape must be divisible by 8")
    return attention


class FooTransformerBlockConfig(FooBlockConfig):
    name: Literal["Multi-head Latent Attention Transformer"] = "Multi-head Latent Attention Transformer"
    attention: Annotated[FooAttentionConfig, AfterValidator(divisible_by_8)] = Field(default_factory=FooAttentionConfig)


class FooMambaBlockConfig(FooBlockConfig):
    activation: Callable[[list[float]], float] = max
    """Activation function for the hidden states. For testing built-in functions"""


class FooModelConfig(SerializableConfigBase):
    name: str
    """Name of the model"""
    version: tuple[int, ...] | str
    """Version of the model"""


class FooGPTModelConfig(FooModelConfig):
    blocks: list[FooBlockConfig] | None = None
    """List of blocks"""


def mock_muon_clip_optimizer(*args, **kwargs):
    """Mock custom optimizer for testing udf serialization"""


class FooLinearWarmupWithCosineDecay:
    """Mock custom scheduler for testing user defined class serialization"""


class FooTrainerConfig(SerializableConfigBase):
    model: FooModelConfig
    """Model to be trained"""
    optimizer: Callable = mock_muon_clip_optimizer
    """Optimizer to be used for training. To test user defined function serialization"""
    scheduler: type = FooLinearWarmupWithCosineDecay
    """Scheduler to be used for training. To test user defined class serialization"""
    checkpointing_steps: list[int] | range = range(100, 1000, 100)
    """Checkpointing steps for the training process"""


class FooExperimentConfig(SerializableConfigBase):
    trainer: FooTrainerConfig
    """Trainer configuration"""
    logger: dict[str, Any] = {"wandb": "cohere", "tensorboard": "./logs"}
    """Experiment tracking"""


@pytest.fixture
def all_in_one_experiment_config() -> FooExperimentConfig:
    """Instantiate a complex nested config for testing."""
    return FooExperimentConfig(
        trainer=FooTrainerConfig(
            model=FooGPTModelConfig(
                name="gpt",
                version=(5, 0),
                blocks=[
                    FooTransformerBlockConfig(
                        attention=FooAttentionConfig(
                            attention_weight_shape=FooAttentionShape(
                                embed_dim=1024,
                                hidden_dim=1024,
                                kv_compressed_rank=32,
                                q_compressed_rank=32,
                                rope_rank=64,
                            )
                        )
                    ),
                    FooMambaBlockConfig(name="SSM"),
                ]
                * 3,
            )
        )
    )


@pytest.mark.parametrize("format_, flatten", product(("yaml", "json", "yml"), (False, True)))
def test_complex_config_serialization(
    all_in_one_experiment_config: FooExperimentConfig, format_: Literal["yaml", "json"], flatten: bool
):
    """Test serialization and deserialization of a complex nested config."""
    with tempfile.NamedTemporaryFile(suffix=f".{format_}") as temp_file:
        # Test serialization
        all_in_one_experiment_config.to_file(temp_file.name, flatten=flatten)

        # Test deserialization
        loaded_config = FooExperimentConfig.from_file(temp_file.name)

        # Verify the loaded config matches the original
        assert loaded_config.logger == {"wandb": "cohere", "tensorboard": "./logs"}
        assert loaded_config.trainer.optimizer is mock_muon_clip_optimizer
        assert loaded_config.trainer.scheduler is FooLinearWarmupWithCosineDecay
        assert loaded_config.trainer.checkpointing_steps == list(range(100, 1000, 100))
        assert isinstance(loaded_config.trainer.model, FooGPTModelConfig)
        assert loaded_config.trainer.model.name == "gpt"
        assert loaded_config.trainer.model.version == (5, 0)
        assert len(loaded_config.trainer.model.blocks) == 6  # 3 * 2 blocks

        # Verify block types and properties
        blocks = loaded_config.trainer.model.blocks
        assert isinstance(blocks[0], FooTransformerBlockConfig)
        assert isinstance(blocks[1], FooMambaBlockConfig)
        # check the transformer block config values
        assert isinstance(blocks[0].attention, FooAttentionConfig)
        assert blocks[0].attention.num_heads == 8
        assert blocks[0].attention.type == "MLA"
        assert isinstance(blocks[0].attention.attention_weight_shape, FooAttentionShape)
        assert blocks[0].attention.attention_weight_shape.hidden_dim == 1024
        assert blocks[0].attention.attention_weight_shape.kv_compressed_rank == 32
        assert blocks[0].attention.attention_weight_shape.rope_rank == 64
        # check the mamba block config values
        assert blocks[1].name == "SSM"
        assert blocks[1].activation is max


@pytest.mark.parametrize("format_, flatten", product(("yaml", "json", "yml"), (False, True)))
def test_patching_config_on_loading(
    all_in_one_experiment_config: FooExperimentConfig, format_: Literal["yaml", "json", "yml"], flatten: bool
):
    """Test patching a config on loading."""
    with tempfile.NamedTemporaryFile(suffix=f".{format_}") as temp_file:
        # Test serialization
        all_in_one_experiment_config.to_file(temp_file.name, flatten=flatten)

        # Test deserialization with patching
        blocks = [  # changing the blocks of model architecture
            {
                "type": {
                    "module": "tests.test_config.test_serializable_config_base",
                    "qualified_name": "FooMambaBlockConfig",
                },
                "args": {"name": "SSM", "activation": {"type": {"module": "statistics", "qualified_name": "mean"}}},
            },
            {
                "type": {
                    "module": "tests.test_config.test_serializable_config_base",
                    "qualified_name": "FooTransformerBlockConfig",
                },
                "args": {
                    "attention": {
                        "type": {
                            "module": "tests.test_config.test_serializable_config_base",
                            "qualified_name": "FooAttentionConfig",
                        },
                        "args": {
                            "attention_weight_shape": {
                                "type": {
                                    "module": "tests.test_config.test_serializable_config_base",
                                    "qualified_name": "FooAttentionShape",
                                },
                                "args": {
                                    "hidden_dim": 16,
                                    "embed_dim": 16,
                                    "kv_compressed_rank": 8,
                                    "q_compressed_rank": 8,
                                    "rope_rank": 8,
                                },
                            }
                        },
                    }
                },
            },
        ]
        # apply patch
        loaded_config = FooExperimentConfig.from_file(
            temp_file.name,
            patch_dict={
                "args.logger.wandb": "cohere-ai",
                "args.trainer.args.checkpointing_steps": [50, 100],
                "args.trainer.args.model.args.blocks": blocks,
            },
        )

        # check overwriting string values
        assert all_in_one_experiment_config != loaded_config
        assert all_in_one_experiment_config.logger["wandb"] == "cohere"
        assert loaded_config.logger["wandb"] == "cohere-ai"
        # check overwriting list values
        assert loaded_config.trainer.checkpointing_steps == [50, 100]
        # check overwriting model blocks
        assert len(loaded_config.trainer.model.blocks) == 2
        # check overwriting classes and callable functions
        assert isinstance(all_in_one_experiment_config.trainer.model.blocks[0], FooTransformerBlockConfig)
        assert isinstance(loaded_config.trainer.model.blocks[0], FooMambaBlockConfig)
        assert loaded_config.trainer.model.blocks[0].activation is statistics.mean
        # check overwriting nested field values
        assert isinstance(loaded_config.trainer.model.blocks[1], FooTransformerBlockConfig)
        assert loaded_config.trainer.model.blocks[1].attention.attention_weight_shape.hidden_dim == 16
        assert all_in_one_experiment_config.trainer.model.blocks[0].attention.attention_weight_shape.hidden_dim == 1024


def test_config_attribute_validation(all_in_one_experiment_config: FooExperimentConfig):
    transformer_block_config: FooTransformerBlockConfig = all_in_one_experiment_config.trainer.model.blocks[0]
    with pytest.raises(ValidationError):
        transformer_block_config.name = "SSM"


def test_config_deserialization_attribute_validation():
    """Test that deserialization of a config raises an error if the field is not valid."""

    _ = FooMambaBlockConfig.from_serializable_dict({"name": "SSM", "activation": max})

    with pytest.raises(ValidationError, match="type=missing"):
        _ = FooMambaBlockConfig()

    with pytest.raises(ValidationError, match="type=missing"):
        FooMambaBlockConfig.from_serializable_dict({"activation": max})

    with pytest.raises(ValidationError, match="Input should be callable"):
        FooMambaBlockConfig.from_serializable_dict({"name": "SSM", "activation": "max"})

    _ = FooTransformerBlockConfig()

    with pytest.raises(ValidationError, match="shape must be divisible by 8"):
        _ = FooTransformerBlockConfig.from_serializable_dict({"attention": {"attention_weight_shape": (1, 2, 3, 4, 5)}})
