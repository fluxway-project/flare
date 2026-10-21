from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import tyro

T = TypeVar("T")


def dict_to_cli_args(config_dict: Mapping[str, Any], prefix: str = "") -> list[str]:
    args: list[str] = []
    for key, value in config_dict.items():
        if value is None:
            continue
        arg_name = f"{prefix}{key}".replace("_", "-")
        if isinstance(value, Mapping):
            args.extend(dict_to_cli_args(value, prefix=f"{prefix}{key}."))
        elif isinstance(value, bool):
            if value:
                args.append(f"--{arg_name}")
            else:
                args.append(f"--no-{arg_name}")
        elif isinstance(value, (list, tuple)):
            for item in value:
                args.extend([f"--{arg_name}", str(item)])
        else:
            args.extend([f"--{arg_name}", str(value)])
    return args


class LayeredConfigLoader:
    def __init__(
        self,
        *,
        config_arg_names: Sequence[str] = (
            "--config",
            "--user-config",
            "--config-file",
        ),
        auto_config_resolver: Callable[[Sequence[str]], Path | None] | None = None,
    ) -> None:
        self._config_arg_names = tuple(config_arg_names)
        self._auto_config_resolver = auto_config_resolver

    def load(
        self,
        config_type: type[T],
        args: Sequence[str] | None = None,
    ) -> T:
        pre_parser = argparse.ArgumentParser(add_help=False)
        for flag in self._config_arg_names:
            pre_parser.add_argument(flag, type=Path, default=None)
        known_args, remaining_argv = pre_parser.parse_known_args(args)

        config_path: Path | None = None
        for flag in self._config_arg_names:
            dest = flag.lstrip("-").replace("-", "_")
            val = getattr(known_args, dest, None)
            if val is not None:
                config_path = val
                break

        if config_path is None and self._auto_config_resolver is not None:
            config_path = self._auto_config_resolver(remaining_argv)

        file_args: list[str] = []
        if config_path is not None:
            if not config_path.is_file():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            with config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise TypeError(
                    f"Configuration file content must be a JSON dictionary, got {type(data).__name__}"
                )
            file_args = dict_to_cli_args(data)

        final_argv = file_args + remaining_argv
        return tyro.cli(config_type, args=final_argv)


__all__ = ["LayeredConfigLoader", "dict_to_cli_args"]
