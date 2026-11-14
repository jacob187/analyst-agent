"""Skill files on disk: parsing, validation and catalogue coverage (AC2, AC4)."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import agents.skills as skills_pkg
from agents.graph.analyst_graph import _create_tools
from agents.skills import SKILLS, SKILLS_DIR, SkillError, known_tool_names, load_skills

KNOWN = {"get_stock_info", "get_technical_analysis", "analyze_latest_8k"}

VALID = """---
name: {name}
description: A test skill.
tools: [get_stock_info]
---
Do the thing.
"""


def _write_skill(root: Path, dirname: str, text: str) -> Path:
    (root / dirname).mkdir()
    path = root / dirname / "SKILL.md"
    path.write_text(text)
    return path


class TestRealCatalogue:
    def test_every_skill_has_name_description_tools_and_instructions(self):
        assert SKILLS
        for name, skill in SKILLS.items():
            assert skill.name == name
            assert skill.description
            assert skill.tools
            assert skill.instructions
            assert skill.path == SKILLS_DIR / skill.path.parent.name / "SKILL.md"

    def test_import_time_catalogue_is_the_validated_load(self):
        assert skills_pkg.SKILLS == load_skills()

    def test_catalogue_covers_the_six_domains(self):
        assert set(SKILLS) == {
            "filing-analysis",
            "material-events",
            "technicals",
            "market-context",
            "news-research",
            "peer-comparison",
        }

    def test_skills_together_use_every_existing_tool(self):
        # Built exactly as create_planning_workflow builds them, research included.
        tools = _create_tools("AAPL", FakeListChatModel(responses=[]), tavily_api_key="placeholder")
        all_names = {t.name for t in tools}
        used = {t for skill in SKILLS.values() for t in skill.tools}
        assert used == all_names
        assert known_tool_names() == all_names


class TestValidation:
    def test_valid_directory_loads(self, tmp_path):
        _write_skill(tmp_path, "one", VALID.format(name="one"))
        loaded = load_skills(tmp_path, known_tools=KNOWN)
        assert list(loaded) == ["one"]
        assert loaded["one"].tools == ("get_stock_info",)
        assert loaded["one"].instructions == "Do the thing."

    def test_unknown_tool_name_fails(self, tmp_path):
        _write_skill(tmp_path, "one", VALID.format(name="one").replace(
            "[get_stock_info]", "[get_stock_info, get_horoscope]"
        ))
        with pytest.raises(SkillError, match="unknown tool name.*get_horoscope"):
            load_skills(tmp_path, known_tools=KNOWN)

    def test_duplicate_skill_names_fail(self, tmp_path):
        _write_skill(tmp_path, "one", VALID.format(name="same"))
        _write_skill(tmp_path, "two", VALID.format(name="same"))
        with pytest.raises(SkillError, match="duplicate skill name 'same'"):
            load_skills(tmp_path, known_tools=KNOWN)

    @pytest.mark.parametrize(
        "text, message",
        [
            ("no frontmatter here", "frontmatter"),
            ("---\ndescription: d\ntools: [get_stock_info]\n---\nbody\n", "'name'"),
            ("---\nname: Bad Name\ndescription: d\ntools: [get_stock_info]\n---\nbody\n", "'name'"),
            ("---\nname: one\ntools: [get_stock_info]\n---\nbody\n", "'description'"),
            ("---\nname: one\ndescription: d\n---\nbody\n", "'tools'"),
            ("---\nname: one\ndescription: d\ntools: []\n---\nbody\n", "'tools'"),
            ("---\nname: one\ndescription: d\ntools: get_stock_info\n---\nbody\n", "'tools'"),
            (
                "---\nname: one\ndescription: d\ntools: [get_stock_info, get_stock_info]\n---\nbody\n",
                "more than once",
            ),
            ("---\nname: one\ndescription: d\ntools: [get_stock_info]\n---\n   \n", "instructions"),
            ("---\nname: [unclosed\n---\nbody\n", "frontmatter"),
        ],
    )
    def test_malformed_skill_fails(self, tmp_path, text, message):
        _write_skill(tmp_path, "one", text)
        with pytest.raises(SkillError, match=message):
            load_skills(tmp_path, known_tools=KNOWN)

    def test_skill_directory_without_skill_file_fails(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(SkillError, match="no SKILL.md"):
            load_skills(tmp_path, known_tools=KNOWN)

    def test_private_directories_are_ignored(self, tmp_path):
        (tmp_path / "__pycache__").mkdir()
        _write_skill(tmp_path, "one", VALID.format(name="one"))
        assert list(load_skills(tmp_path, known_tools=KNOWN)) == ["one"]

    def test_no_skills_fails(self, tmp_path):
        with pytest.raises(SkillError, match="no skills"):
            load_skills(tmp_path, known_tools=KNOWN)
