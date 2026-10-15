"""Load named system prompts from configuration."""

import yaml

from .io import PROJECT_ROOT


PROMPTS_PATH = PROJECT_ROOT / "config" / "prompts.yaml"


def load_system_prompt(name: str) -> str:
    with PROMPTS_PATH.open(encoding="utf-8") as handle:
        prompts = yaml.safe_load(handle)
    return prompts[name]
