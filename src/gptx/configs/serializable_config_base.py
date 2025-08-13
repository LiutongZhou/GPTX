"""Base Class for creating composable, serializable Configs.

Featuring streamlined serialization and deserialization to/from easy-to-edit config files.

Intended for use in large-scale, modular configuration management
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
import types
from pathlib import Path
from typing import Any, ClassVar, TypeAlias, TypedDict

import yaml
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

__all__ = ["FullyQualifiedName", "SerializableConfigBase"]

_FunctionOrClassType: TypeAlias = type | types.FunctionType | types.BuiltinFunctionType


def _is_dataclass_instance(obj: Any) -> bool:
    """Check if the object is a dataclass instance.

    Examples
    --------
    >>> @dataclasses.dataclass
    ... class Config:
    ...     field: int
    >>> config = Config(field=42)
    >>> _is_dataclass_instance(config)
    True
    >>> _is_dataclass_instance(Config)
    False
    """
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def _is_namedtuple_instance(obj: Any) -> bool:
    """Check if the object is a namedtuple instance.

    Examples
    --------
    >>> from collections import namedtuple
    >>> Point = namedtuple('Point', ['x', 'y'])
    >>> point = Point(x=1, y=2)
    >>> _is_namedtuple_instance(point)
    True
    >>> try:  # this will not work because namedtuple is a function
    ...     isinstance(point, namedtuple)
    ... except TypeError as exception:
    ...     print(exception)
    isinstance() arg 2 must be a type, a tuple of types, or a union
    """
    return isinstance(obj, tuple) and hasattr(obj, "_fields") and hasattr(obj, "_asdict")


def _merge_dict(base_dict: dict[str, Any], *others: dict[str, Any]) -> None:
    """Recursively update the base dict in place with (key, value) from other dicts.

    (key, value) from the right-most dict will overwrite its left dicts.
    A deep merge is performed - the merge will be done recursively for nested dictionaries.

    Parameters
    ----------
    base_dict : dict
        The dictionary to be updated (modified in place)
    others : dict
        The dictionaries containing updates. Will overwrite values in base_dict

    Examples
    --------
    >>> base_dict = {'a': 1, 'b': {'c': 2, 'd': [1, 2]}}
    >>> other_dict1 = {'b': {'d': 3, 'e': 4}}
    >>> other_dict2 = {'a': 5, 'b': {'c': 6, 'f': 7}}
    >>> _merge_dict(base_dict, other_dict1, other_dict2)
    >>> base_dict
    {'a': 5, 'b': {'c': 6, 'd': 3, 'e': 4, 'f': 7}}
    """
    for other_dict in others:
        for key, value in other_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                _merge_dict(base_dict[key], value)
            else:
                base_dict[key] = value


def _flatten_dict(dict_: dict[str, Any]) -> dict[str, Any]:
    """Given a nested dictionary, return a flattened dictionary with dot notation keys.

    Examples
    --------
    >>> nested_dict = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}, "f": 4}
    >>> _flatten_dict(nested_dict)
    {'a': 1, 'b.c': 2, 'b.d.e': 3, 'f': 4}
    """
    flat_dict = {}

    def _flatten(current_dict: dict[str, Any], _path: list[str] | None = None) -> None:
        if _path is None:
            _path = []
        for key, value in current_dict.items():
            _path.append(key)
            if isinstance(value, dict):
                _flatten(value, _path)
            else:
                flat_dict[".".join(_path)] = value
            _path.pop()

    _flatten(dict_)
    return flat_dict


def _unflatten_dict(flat_dict: dict[str, Any]) -> dict[str, Any]:
    """Given a flattened dictionary with dot notation keys, return a nested dictionary.

    If the input dictionary is not flattened, it will return a copy of the same.

    Examples
    --------
    >>> flat_dict = {'a': 1, 'b.c': 2, 'b.d.e': 3, 'f': 4}
    >>> _unflatten_dict(flat_dict)
    {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}, 'f': 4}
    """
    nested_dict = {}

    for key, value in flat_dict.items():
        if not isinstance(key, str):
            raise ValueError(f"Invalid key type {type(key)=}. Expected str for key in flat dict.")
        current = nested_dict
        parts = key.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return nested_dict


class NotSerializableError(ValueError):
    """Raised when trying to serialize a class or function that cannot be serialized."""


def _check_is_serializable(obj: _FunctionOrClassType) -> None:
    """Check if the function or class is serializable. Raises NotSerializableError if not serializable."""
    if obj.__name__ == "<lambda>":
        raise NotSerializableError(f"Lambda function {obj} is not serializable by reference-based serialization")
    if ".<locals>." in obj.__qualname__:
        raise NotSerializableError(
            f"Nested function or class {obj} defined inside another function "
            f"is not serializable by reference-based serialization"
        )


class _FullyQualifiedNameDict(TypedDict):
    """For type hinting and type checking only. Not used at runtime."""

    module: str
    qualified_name: str


@dataclasses.dataclass(frozen=True)
class FullyQualifiedName:
    """For storing the fully qualified name of a class or a function.

    Examples
    --------
    Constructing a fully qualified name from a class or function

    >>> from concurrent.futures import ProcessPoolExecutor
    >>> fully_qualified_name = FullyQualifiedName.from_class_or_function(ProcessPoolExecutor)
    >>> print(fully_qualified_name)
    concurrent.futures.process.ProcessPoolExecutor

    To dict representation of the fully qualified name

    >>> dict_ = fully_qualified_name.to_dict()
    >>> dict_
    {'module': 'concurrent.futures.process', 'qualified_name': 'ProcessPoolExecutor'}

    Load a class or function from its fully qualified name dict representation

    >>> loaded_ = FullyQualifiedName.load_from_dict(dict_)
    >>> loaded_ is ProcessPoolExecutor
    True
    """

    module: str
    qualified_name: str

    def __str__(self) -> str:
        return f"{self.module}.{self.qualified_name}"

    @classmethod
    def from_class_or_function(cls, obj: _FunctionOrClassType) -> Self:
        """Construct the fully qualified name of a class or function."""
        return cls(obj.__module__, obj.__qualname__)

    def load(self) -> _FunctionOrClassType:
        """Return a class or function defined in a module given its fully qualified name."""
        module, qualified_name = self.module, self.qualified_name
        if not module:
            raise AttributeError(f"Empty module name {module=}")
        if not qualified_name:
            raise AttributeError(f"Empty qualified name {qualified_name=}")

        import_error = ImportError(
            f"Could not import {module}.{qualified_name}\n"
            f"Ensure the module {module} and the class or function {qualified_name} exist.\n"
            f"Also make sure the class or function is defined in the module's top level, "
            f"not nested inside another function"
        )

        try:
            module_imported = importlib.import_module(module)
        except ModuleNotFoundError as error:
            raise import_error from error

        cur = module_imported
        for part in qualified_name.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                raise import_error
        function_or_class = cur
        return function_or_class

    def to_dict(self) -> _FullyQualifiedNameDict:
        """Return the dict representation of the fully qualified name

        A way of reference-based serialization of classes and functions.
        """
        return {"module": self.module, "qualified_name": self.qualified_name}

    @classmethod
    def from_dict(cls, dict_: _FullyQualifiedNameDict) -> Self:
        """Given the dict representation, return a fully qualified name instance"""
        return cls(**dict_)

    @classmethod
    def load_from_dict(cls, dict_: _FullyQualifiedNameDict) -> _FunctionOrClassType:
        """Return a class or function defined in a module given the dict representation of its fully qualified name

        This is the deserialization counterpart to `to_dict`.
        """
        return cls.from_dict(dict_).load()


def set_config_or_dict_attr(target: BaseModel | dict, path: str, value: Any) -> None:
    """Set a value to a nested attribute (referenced by dot path) on a BaseModel or dict."""
    if not isinstance(target, (BaseModel, dict)):
        raise TypeError("target must be a BaseModel or dict")

    *parents, last = path.split(".")
    cur: BaseModel | dict = target

    for key in parents:
        if isinstance(cur, BaseModel):
            cur = getattr(cur, key)
        else:
            if key in cur and cur[key] is None:
                cur[key] = {}
            cur = cur.setdefault(key, {})

    if isinstance(cur, BaseModel):
        setattr(cur, last, value)
    else:
        cur[last] = value


def apply_patch_dict(target: BaseModel | dict, patch_dict: dict[str, Any] | None = None) -> None:
    """Apply a patch dictionary to a BaseModel or dict, setting values at specified dot paths."""
    if patch_dict:
        for path, value in patch_dict.items():
            set_config_or_dict_attr(target, path, value)


class SerializableConfigBase(BaseModel):
    """Base class for creating composable configs with streamlined serialization/deserialization to/from files.

    Examples
    --------
    >>> from typing import Literal
    >>> from pydantic import Field
    >>> from tempfile import NamedTemporaryFile
    >>> class ModelCheckpointConfig(SerializableConfigBase):
    ...    checkpoint_path: str
    ...
    >>> class ServerConfig(SerializableConfigBase):
    ...    incremental_sampling_n : int | list[int] = 100
    ...    "Number of decode steps for batch inference before checking stop criteria."
    ...    concurrency: int = 8
    ...    "Max number of concurrent requests to the server."
    ...    batch_size: int = 64
    ...
    >>> class GenerateConfig(SerializableConfigBase):
    ...    use_cpu: bool = False
    ...    "Use CPU for inference"
    ...    use_flash_attention: bool = True
    ...    "Use flash attention for inference"
    ...
    >>> class BenchmarkConfig(SerializableConfigBase):
    ...    model : ModelCheckpointConfig | list[ModelCheckpointConfig]
    ...    "Model to benchmark. Can be a single model or a list of models."
    ...    num_requests : list[int] = [10, 100, 1000]
    ...    "Benchmark the inference efficiency of the model with different number of requests."
    ...    n_tokens : list[int] = [100, 1000]
    ...    "Try different number of tokens to generate for each request."
    ...    inference_engine : ServerConfig|GenerateConfig = Field(default_factory=GenerateConfig)
    ...
    >>> benchmark_config = BenchmarkConfig(model = ModelCheckpointConfig(checkpoint_path="model"))

    Streamlined serialization and deserialization to/from files

    >>> for format_ in [".json", ".yaml"]:
    ...     with NamedTemporaryFile(suffix=format_, mode='w') as tmp_file:
    ...         benchmark_config.to_file(tmp_file.name)  # Save the config to a yaml, json, gin file
    ...         loaded_ = BenchmarkConfig.load_from_file(
    ...                 tmp_file.name,
    ...                 patch_dict={"args.model.args.checkpoint_path": "./model_v2"},
    ...             )   # load the config from a yaml, json, gin file with config value overwrite on-the-fly
    ...         loaded_.n_tokens = [1024]   # Modify the loaded config
    ...         assert loaded_.model.checkpoint_path == "./model_v2", f"{loaded_.model.checkpoint_path=}"
    ...         assert benchmark_config.model.checkpoint_path == "model"
    """

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
        validate_assignment=True,
        use_enum_values=True,
    )

    # Registry for all subclasses, keyed by its fully qualified name.
    _subclass_registry: ClassVar[dict[FullyQualifiedName, type[SerializableConfigBase]]] = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses of _SerializableConfig in _subclass_registry."""
        super().__init_subclass__(**kwargs)
        fully_qualified_name: FullyQualifiedName = FullyQualifiedName.from_class_or_function(cls)
        SerializableConfigBase._subclass_registry.setdefault(fully_qualified_name, cls)

    def to_serializable_dict(self, flatten: bool = False) -> dict[str, Any]:
        """Return a serializable dict representation of the config.

        The dict is constructed recursively for every field in the config.

        Parameters
        ----------
        flatten : bool, optional
            If True, flattens the nested dictionary structure into a single level dictionary
            with dot notation in keys (default is False).

        See Also
        --------
        SerializableConfig.from_serializable_dict : Reconstructs the config from a serialized dict.
        """

        def _serialize(obj: Any) -> Any:
            if isinstance(obj, SerializableConfigBase):
                return {
                    "type": FullyQualifiedName.from_class_or_function(type(obj)).to_dict(),
                    "args": {key: _serialize(getattr(obj, key)) for key in type(obj).model_fields.keys()},
                }
            elif _is_dataclass_instance(obj):
                return {
                    "type": FullyQualifiedName.from_class_or_function(type(obj)).to_dict(),
                    "args": {field: _serialize(value) for field, value in dataclasses.asdict(obj).items()},
                }
            elif isinstance(obj, dict):
                return {key: _serialize(value) for key, value in obj.items()}
            elif _is_namedtuple_instance(obj):
                return {
                    "type": FullyQualifiedName.from_class_or_function(type(obj)).to_dict(),
                    "args": {key: _serialize(value) for key, value in obj._asdict().items()},
                }
            elif isinstance(obj, (list, tuple, set, frozenset)):
                return type(obj)(_serialize(item) for item in obj)
            elif isinstance(obj, range):
                return [_serialize(item) for item in obj]
            elif isinstance(obj, (type, types.FunctionType, types.BuiltinFunctionType)):
                _check_is_serializable(obj)
                return {"type": FullyQualifiedName.from_class_or_function(obj).to_dict()}
            else:
                return obj

        serializable_dict = _serialize(self)
        return _flatten_dict(serializable_dict) if flatten else serializable_dict

    @classmethod
    def from_serializable_dict(
        cls, serialized_dict: dict[str, Any], patch_dict: dict[str, Any] | None = None, **kwargs: Any
    ) -> Self:
        """Reconstruct a config from a serialized dictionary.

        Supports patching the config with user provided attribute_path -> value mapping

        Examples
        --------
        >>> class WandbConfig(SerializableConfigBase):
        ...    project: str = "fax_benchmarking"
        ...    entity: str = "cohere"
        ...    log_code: bool = True
        ...
        >>> class RunConfig(SerializableConfigBase):
        ...    checkpoint_to: str = "./checkpoints"
        ...    wandb: WandbConfig = WandbConfig()
        ...
        >>> run_config_default = RunConfig()
        >>> run_config_default_dict = run_config_default.to_serializable_dict()

        Overwrite the default value of `wandb.log_code` to False at loading time

        >>> run_config_loaded = RunConfig.from_serializable_dict(
        ...     run_config_default_dict, patch_dict={"wandb.log_code": False}
        ... )
        ...
        >>> run_config_loaded.wandb.log_code
        False

        See Also
        --------
        SerializableConfig.to_serializable_dict : Converts the config to a serializable dict.
        """

        def _deserialize(obj: Any) -> Any:
            match obj:
                case {"type": {"module": str(module), "qualified_name": str(qualified_name)}, "args": dict(kwargs)}:
                    fully_qualified_name = FullyQualifiedName.from_dict(
                        {"module": module, "qualified_name": qualified_name}
                    )
                    if fully_qualified_name in SerializableConfigBase._subclass_registry:
                        config_class: type[SerializableConfigBase] = SerializableConfigBase._subclass_registry[
                            fully_qualified_name
                        ]
                        return config_class.model_validate({key: _deserialize(value) for key, value in kwargs.items()})
                    else:  # namedtuple or custom dataclass
                        custom_class: type = fully_qualified_name.load()
                        return custom_class(**{key: _deserialize(value) for key, value in kwargs.items()})
                case {
                    "type": {"module": str(module), "qualified_name": str(qualified_name)}
                }:  # function or class reference
                    return FullyQualifiedName.load_from_dict({"module": module, "qualified_name": qualified_name})
                case dict():
                    return {key: _deserialize(value) for key, value in obj.items()}
                case list() | tuple() | set() | frozenset():
                    return type(obj)(_deserialize(item) for item in obj)
                case _:
                    return obj

        serialized_dict = _unflatten_dict(serialized_dict)
        _merge_dict(serialized_dict, kwargs)
        if isinstance(obj := _deserialize(serialized_dict), dict):
            config = cls.model_validate(obj)
        else:
            config = obj
        if patch_dict:
            apply_patch_dict(config, patch_dict)
        return config

    def to_file(self, file_path: str | os.PathLike[str], flatten: bool = False) -> None:
        """Save the config to a json or yaml file."""
        serializable_dict = self.to_serializable_dict(flatten=flatten)
        match Path(file_path).suffix.lower():
            case ".json":
                with Path(file_path).open(mode="w") as file:
                    json.dump(serializable_dict, file, indent=4)
            case ".yaml" | ".yml":
                with Path(file_path).open(mode="w") as file:
                    yaml.safe_dump(serializable_dict, file, sort_keys=False)
            case _:
                raise ValueError("Unsupported file format. Use .json, .yaml, .yml")

    @classmethod
    def from_file(
        cls, *file_paths: str | os.PathLike[str], patch_dict: dict[str, Any] | None = None, **kwargs: Any
    ) -> Self:
        """Creat the config instance from json or yaml files"""
        config_dict_merged: dict[str, Any] = {}
        for file_path in file_paths:
            match Path(file_path).suffix.lower():
                case ".json":
                    with Path(file_path).open(mode="r") as file:
                        config_dict = json.load(file)
                case ".yaml" | ".yml":
                    with Path(file_path).open(mode="r") as file:
                        config_dict = yaml.safe_load(file)
                case _:
                    raise ValueError("Unsupported file format. Use .json, .yaml, .yml")
            _merge_dict(config_dict_merged, config_dict)
        # apply the patch dict before deserialization
        config_dict_merged = _unflatten_dict(config_dict_merged)
        apply_patch_dict(config_dict_merged, patch_dict)
        return cls.from_serializable_dict(config_dict_merged, **kwargs)
