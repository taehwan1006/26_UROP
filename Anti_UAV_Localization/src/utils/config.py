"""YAML config 로딩."""

from pathlib import Path
from typing import Union

import yaml


def load_config(config_path: Union[str, Path]) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
