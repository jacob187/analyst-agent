"""Tests for the reconciler node.

Marker: eval_unit — pure-Python logic, no LLM calls.

The reconciler groups step_results by filing_ref and emits a structured
`conflicts` list when two or more workers describe the same filing with
disagreeing values for a tracked field (currently `event_type`). The
synthesizer is then prompted to surface those conflicts rather than emit
two contradictory descriptions of the same filing.
"""

import pytest

from agents.graph.analyst_graph import create_reconciler_node


def _result(tool: str, ref: str | None, **data) -> dict:
    """Build a StepResult-shaped dict for reconciler input."""
    return {
        "tool": tool,
        "data": data,
        "raw": "",
        "filing_ref": ref,
        "error": None,
    }


@pytest.mark.eval_unit
class TestReconciler:
    """The reconciler emits a Conflict per disagreeing filing_ref group."""

    def test_emits_no_conflicts_when_filings_disjoint(self):
        recon = create_reconciler_node()
        state = {
            "step_results": {
                1: _result("a", "acc-1", event_type="earnings"),
                2: _result("b", "acc-2", event_type="guidance"),
            }
        }
        assert recon(state) == {"conflicts": []}

    def test_detects_event_type_disagreement(self):
        """Two steps on the same filing with different event_type → one Conflict.

        This is the original GOOGL bug shape from #30.
        """
        recon = create_reconciler_node()
        state = {
            "step_results": {
                1: _result("get_8k_overview", "acc-1", event_type="earnings"),
                2: _result("get_material_event_summary", "acc-1", event_type="guidance"),
            }
        }
        result = recon(state)

        assert len(result["conflicts"]) == 1
        c = result["conflicts"][0]
        assert c["filing_ref"] == "acc-1"
        assert c["step_ids"] == [1, 2]
        assert c["field"] == "event_type"
        assert set(c["values"]) == {"earnings", "guidance"}

    def test_ignores_step_results_without_filing_ref(self):
        """Stock/market/briefing tools don't emit filing_ref — they're skipped."""
        recon = create_reconciler_node()
        state = {
            "step_results": {
                1: _result("get_stock_info", None),
                2: _result("get_macro_indicator", None),
            }
        }
        assert recon(state) == {"conflicts": []}

    def test_handles_three_way_disagreement(self):
        """Three steps on the same filing → one Conflict with all three values."""
        recon = create_reconciler_node()
        state = {
            "step_results": {
                1: _result("a", "acc-1", event_type="earnings"),
                2: _result("b", "acc-1", event_type="guidance"),
                3: _result("c", "acc-1", event_type="acquisition"),
            }
        }
        result = recon(state)

        assert len(result["conflicts"]) == 1
        c = result["conflicts"][0]
        assert c["step_ids"] == [1, 2, 3]
        assert set(c["values"]) == {"earnings", "guidance", "acquisition"}

    def test_no_conflict_when_values_agree(self):
        """Same filing_ref + same event_type across tools = consistent, not a conflict."""
        recon = create_reconciler_node()
        state = {
            "step_results": {
                1: _result("a", "acc-1", event_type="earnings"),
                2: _result("b", "acc-1", event_type="earnings"),
            }
        }
        assert recon(state) == {"conflicts": []}

    def test_ignores_steps_missing_the_field(self):
        """A step whose data has no event_type doesn't contribute to a conflict."""
        recon = create_reconciler_node()
        state = {
            "step_results": {
                1: _result("a", "acc-1", event_type="earnings"),
                2: _result("b", "acc-1"),  # no event_type field
            }
        }
        assert recon(state) == {"conflicts": []}

    def test_empty_step_results(self):
        recon = create_reconciler_node()
        assert recon({"step_results": {}}) == {"conflicts": []}

    def test_missing_step_results_key(self):
        recon = create_reconciler_node()
        assert recon({}) == {"conflicts": []}
