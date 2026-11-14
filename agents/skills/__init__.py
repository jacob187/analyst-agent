"""Skill catalogue for the skills-based chat agent.

A skill is a directory ``agents/skills/<name>/SKILL.md``: YAML frontmatter
(``name``, ``description``, ``tools``) followed by the skill's instructions as
the Markdown body. The agent's base system prompt is ``system_prompt.md`` in
this directory. Every skill is parsed and validated at import, so a typo in a
tool name or a missing field fails at startup instead of mid-conversation.

This package is the only source of the skills agent's prompt text; it never
reads ``agents/prompts/*.yaml``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import yaml

SKILLS_DIR = Path(__file__).parent
SYSTEM_PROMPT_FILE = SKILLS_DIR / "system_prompt.md"
SKILL_FILE_NAME = "SKILL.md"

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


class SkillError(ValueError):
    """A skill file is missing, malformed, or references an unknown tool."""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    tools: tuple[str, ...]
    instructions: str
    path: Path


def known_tool_names() -> frozenset[str]:
    """Every tool name the chat agent can be given, read from the real factories.

    Factories only build closures (no I/O), so placeholder arguments are safe.
    Research tools are included even though they need a Tavily key at runtime.
    """
    from agents.tools.briefing_tools import create_briefing_tools
    from agents.tools.market_tools import create_market_tools
    from agents.tools.research_tools import create_research_tools
    from agents.tools.sec_tools import create_sec_tools
    from agents.tools.stock_tools import create_stock_tools

    sec_tools, _ = create_sec_tools("TICKER", None)
    tools = [
        *sec_tools,
        *create_stock_tools("TICKER"),
        *create_market_tools(),
        *create_research_tools("TICKER", "placeholder"),
        *create_briefing_tools("TICKER"),
    ]
    return frozenset(t.name for t in tools)


def parse_skill(path: Path) -> Skill:
    """Parse one SKILL.md file; raise SkillError if it is malformed."""
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise SkillError(f"{path}: missing '---' frontmatter block")
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as e:
        raise SkillError(f"{path}: invalid frontmatter YAML: {e}") from e
    if not isinstance(meta, dict):
        raise SkillError(f"{path}: frontmatter must be a mapping")

    name = meta.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise SkillError(f"{path}: 'name' must be a lowercase kebab-case string")
    description = meta.get("description")
    if not isinstance(description, str) or not description.strip():
        raise SkillError(f"{path}: 'description' is required")
    tools = meta.get("tools")
    if (
        not isinstance(tools, list)
        or not tools
        or not all(isinstance(t, str) and t for t in tools)
    ):
        raise SkillError(f"{path}: 'tools' must be a non-empty list of tool names")
    if len(set(tools)) != len(tools):
        raise SkillError(f"{path}: 'tools' lists a tool more than once")
    instructions = match.group(2).strip()
    if not instructions:
        raise SkillError(f"{path}: instructions (the body after the frontmatter) are required")

    return Skill(
        name=name,
        description=" ".join(description.split()),
        tools=tuple(tools),
        instructions=instructions,
        path=path,
    )


def load_skills(
    skills_dir: Path = SKILLS_DIR, known_tools: Iterable[str] | None = None
) -> dict[str, Skill]:
    """Load and validate every skill under ``skills_dir``, keyed by name.

    Raises SkillError on a skill directory without SKILL.md, a malformed file,
    a duplicate skill name, or a tool name no factory produces.
    """
    known = frozenset(known_tools) if known_tools is not None else known_tool_names()
    skills: dict[str, Skill] = {}
    for child in sorted(skills_dir.iterdir()):
        if not child.is_dir() or child.name.startswith(("_", ".")):
            continue
        skill_file = child / SKILL_FILE_NAME
        if not skill_file.is_file():
            raise SkillError(f"{child}: skill directory has no {SKILL_FILE_NAME}")
        skill = parse_skill(skill_file)
        if skill.name in skills:
            raise SkillError(
                f"{skill_file}: duplicate skill name '{skill.name}' "
                f"(also in {skills[skill.name].path})"
            )
        unknown = [t for t in skill.tools if t not in known]
        if unknown:
            raise SkillError(f"{skill_file}: unknown tool name(s) {unknown}")
        skills[skill.name] = skill
    if not skills:
        raise SkillError(f"{skills_dir}: no skills found")
    return skills


def load_base_prompt(path: Path = SYSTEM_PROMPT_FILE) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise SkillError(f"{path}: base system prompt is empty")
    return text


def render_base_prompt(ticker: str, template: str | None = None) -> str:
    """The base system prompt with ``{ticker}`` filled in.

    Plain replacement rather than ``str.format`` so Markdown braces in the file
    can't break rendering.
    """
    return (template if template is not None else BASE_SYSTEM_PROMPT).replace("{ticker}", ticker)


def render_catalogue(skills: Mapping[str, Skill]) -> str:
    """The name + description list the model picks skills from."""
    lines = [
        "## Skills",
        "",
        "Call `load_skill` with a skill name to get its instructions and unlock its "
        "tools. Load every skill the question needs before answering; you can load "
        "several at once.",
        "",
    ]
    lines.extend(f"- **{s.name}**: {s.description}" for s in skills.values())
    return "\n".join(lines)


SKILLS: dict[str, Skill] = load_skills()
BASE_SYSTEM_PROMPT: str = load_base_prompt()
