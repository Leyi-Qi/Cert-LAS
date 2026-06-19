import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def load_json_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return data


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: Optional[Iterable[str]] = None,
) -> argparse.Namespace:
    """Parse a parser that supports JSON defaults via --config_path.

    Command-line flags override values loaded from the JSON file.
    """
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config_path", type=str, default=None)
    config_args, remaining = config_parser.parse_known_args(argv)
    config = load_json_config(config_args.config_path)
    if config:
        for action in parser._actions:
            if action.dest in config:
                action.required = False
        parser.set_defaults(**config)
    if not any(action.dest == "config_path" for action in parser._actions):
        parser.add_argument("--config_path", type=str, default=config_args.config_path)
    args = parser.parse_args(argv)
    args.config_path = config_args.config_path
    return args


def require_existing_path(path: str, label: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")
