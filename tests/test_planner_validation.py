"""Pydantic field-limit tests on planner output models.

Caps prevent a single adversarial WS chat query from producing an unbounded
plan that fans out unlimited LLM-tool calls (the WS path bypasses the REST
rate limiter — it counts messages, not LLM calls).
"""

import pytest
from pydantic import ValidationError

from agents.planner import (
    AnalysisStep,
    QueryClassification,
    QueryPlan,
    MAX_PLAN_STEPS,
)


def _step(step_id: int = 1, **overrides) -> dict:
    payload = {
        "id": step_id,
        "action": "fetch risk factors",
        "tool": "get_raw_risk_factors",
        "rationale": "user asked about risks",
        "depends_on": [],
    }
    payload.update(overrides)
    return payload


class TestQueryPlanStepsCap:
    def test_accepts_plan_at_max_steps(self):
        steps = [_step(i) for i in range(1, MAX_PLAN_STEPS + 1)]
        plan = QueryPlan(query_type="complex", requires_planning=True, steps=steps)
        assert len(plan.steps) == MAX_PLAN_STEPS

    def test_rejects_plan_above_max_steps(self):
        steps = [_step(i) for i in range(1, MAX_PLAN_STEPS + 2)]
        with pytest.raises(ValidationError):
            QueryPlan(query_type="complex", requires_planning=True, steps=steps)

    def test_accepts_typical_5_step_plan(self):
        steps = [_step(i) for i in range(1, 6)]
        plan = QueryPlan(query_type="moderate", requires_planning=True, steps=steps)
        assert len(plan.steps) == 5

    def test_accepts_empty_plan(self):
        plan = QueryPlan(query_type="simple", requires_planning=False, steps=[])
        assert plan.steps == []


class TestQueryPlanStringCaps:
    def test_synthesis_approach_capped_at_2000(self):
        with pytest.raises(ValidationError):
            QueryPlan(
                query_type="moderate",
                requires_planning=True,
                steps=[_step()],
                synthesis_approach="x" * 2001,
            )

    def test_query_type_capped(self):
        with pytest.raises(ValidationError):
            QueryPlan(
                query_type="x" * 51,
                requires_planning=False,
                steps=[],
            )


class TestAnalysisStepCaps:
    def test_action_rejected_above_500_chars(self):
        with pytest.raises(ValidationError):
            AnalysisStep(**_step(action="x" * 501))

    def test_rationale_rejected_above_500_chars(self):
        with pytest.raises(ValidationError):
            AnalysisStep(**_step(rationale="x" * 501))

    def test_depends_on_rejected_above_20(self):
        with pytest.raises(ValidationError):
            AnalysisStep(**_step(depends_on=list(range(21))))

    def test_huge_tool_string_rejected(self):
        with pytest.raises(ValidationError):
            AnalysisStep(**_step(tool="x" * 101))

    def test_accepts_500_char_boundary(self):
        step = AnalysisStep(**_step(action="x" * 500, rationale="x" * 500))
        assert len(step.action) == 500


class TestQueryClassificationCaps:
    def test_reasoning_capped_at_1000(self):
        with pytest.raises(ValidationError):
            QueryClassification(
                complexity="simple",
                reasoning="x" * 1001,
                estimated_tools=1,
            )

    def test_estimated_tools_rejected_negative(self):
        with pytest.raises(ValidationError):
            QueryClassification(
                complexity="simple",
                reasoning="ok",
                estimated_tools=-1,
            )

    def test_estimated_tools_rejected_above_20(self):
        with pytest.raises(ValidationError):
            QueryClassification(
                complexity="simple",
                reasoning="ok",
                estimated_tools=21,
            )

    def test_accepts_boundary_values(self):
        QueryClassification(complexity="simple", reasoning="ok", estimated_tools=0)
        QueryClassification(complexity="complex", reasoning="ok", estimated_tools=20)
