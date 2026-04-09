"""YAML prompt loader — reads all .yaml files in this directory at import time.

Each YAML file contains a top-level ``prompts`` mapping where keys are prompt
constant names (e.g. ``SEC_AGENT_SYSTEM_PROMPT``) and values have a ``template``
field with the actual prompt text.

The loader strips one trailing newline that YAML block scalars (``|``) append,
so the loaded strings match the original Python constants exactly.
"""

from pathlib import Path
from typing import Dict

import yaml


def load_prompts() -> Dict[str, str]:
    """Load all prompt templates from YAML files in the prompts directory.

    Returns a flat dict mapping prompt constant names to their template strings.
    Raises ``ValueError`` on duplicate keys across files.
    """
    prompts_dir = Path(__file__).parent
    all_prompts: Dict[str, str] = {}

    for yaml_file in sorted(prompts_dir.glob("*.yaml")):
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        if not data or "prompts" not in data:
            continue

        for name, entry in data["prompts"].items():
            if name in all_prompts:
                raise ValueError(
                    f"Duplicate prompt key '{name}' found in {yaml_file.name}. "
                    f"Each prompt name must be unique across all YAML files."
                )
            # YAML block scalars (|) append a trailing newline — strip it
            # so the loaded string matches the original Python constant.
            all_prompts[name] = entry["template"].rstrip("\n")

    return all_prompts
