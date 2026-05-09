from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

DEFAULT_RUN_INFO = Path("run_info.yaml")


def load_run_info_arg() -> tuple[argparse.ArgumentParser, dict[str, Any]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--run-info",
        type=Path,
        default=DEFAULT_RUN_INFO,
        help="YAML file containing run metadata and defaults.",
    )
    args, _ = parser.parse_known_args()
    return parser, load_run_info(args.run_info)


def load_run_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required run info file: {path}")
    yaml = YAML(typ="safe")
    data = yaml.load(path)
    if data is None:
        raise ValueError(f"{path} must contain run metadata.")
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level.")
    return data


def config_get(config: dict[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing required run_info key: {dotted_path}")
        value = value[key]
    return value


def config_path(config: dict[str, Any], dotted_path: str) -> Path:
    return Path(config_get(config, dotted_path))
