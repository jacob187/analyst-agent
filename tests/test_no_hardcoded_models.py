"""Tests must not name real registry models.

Model ids, defaults and lendability live in agents/models.json. A test that
hardcodes an id breaks when that model is renamed, removed or re-flagged, so
tests derive models from the registry (see the selectors in tests/conftest.py)
and use obviously fake ids (`unknown-model`, `custom-model-x`) when they need
an unregistered one.
"""

import ast
import json
import re
from pathlib import Path

_TESTS_DIR = Path(__file__).parent
_MODEL_IDS = [
    entry["id"]
    for entry in json.loads((_TESTS_DIR.parent / "agents" / "models.json").read_text())
]
# Ids are delimited by anything that can't be part of one, so `claude-sonnet-5`
# doesn't match inside `claude-sonnet-5-5` but does inside "anthropic:claude-sonnet-5".
_ID_PATTERN = re.compile(
    r"(?<![\w.-])("
    + "|".join(re.escape(i) for i in sorted(_MODEL_IDS, key=len, reverse=True))
    + r")(?![\w.-])"
)


def _hardcoded_ids(path: Path) -> list[str]:
    hits = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for match in _ID_PATTERN.finditer(node.value):
                line = node.lineno + node.value.count("\n", 0, match.start())
                hits.append(f"{path.relative_to(_TESTS_DIR.parent)}:{line}: {match.group(1)}")
    return hits


def test_no_test_hardcodes_a_registry_model_id():
    test_files = sorted(
        p for p in _TESTS_DIR.rglob("*.py")
        if p != Path(__file__) and "__pycache__" not in p.parts
    )
    hits = [hit for path in test_files for hit in _hardcoded_ids(path)]
    assert not hits, (
        "Registry model ids hardcoded in tests; pick models from the registry "
        "instead (tests/conftest.py selectors):\n" + "\n".join(hits)
    )
