"""System prompts for the analyst agent.

Prompts are stored as YAML files in this directory and loaded at import time.
All prompt constants (e.g. ``SEC_AGENT_SYSTEM_PROMPT``, ``TOOL_CAPABILITIES``)
are available as module-level attributes, so existing imports like::

    from agents.prompts import SEC_AGENT_SYSTEM_PROMPT

continue to work unchanged.
"""

from agents.prompts.loader import load_prompts

# Load all YAML prompt templates once at import time and inject them
# into the module namespace so they're importable as top-level constants.
_ALL_PROMPTS = load_prompts()
globals().update(_ALL_PROMPTS)

# Explicit __all__ so star-imports and IDE autocompletion work correctly.
__all__ = list(_ALL_PROMPTS.keys())
