"""Tests for the LLM-based query planner (planner.create_plan).

Marker: eval_fast — each test makes a single LLM call to generate a plan.
Uses planner_cases.json as the golden dataset.

The planner returns a QueryPlan with ordered AnalysisSteps, each specifying
a tool name. We validate that:
1. Required tools appear in the plan
2. No hallucinated tool names (all must be in VALID_TOOLS)
3. Step dependencies reference valid step IDs
"""

import pytest

from tests.conftest import VALID_TOOLS


@pytest.mark.eval_fast
class TestQueryPlanner:
    """Validate create_plan against planner_cases.json expectations."""

    def test_required_tools_present(self, planner, planner_cases):
        """Every expected_tool must appear somewhere in the generated plan.

        The plan might include extra tools (that's fine — the LLM may add
        helpful context steps), but the required ones MUST be present.
        """
        for case in planner_cases:
            plan = planner.create_plan(case["query"])
            plan_tools = {step.tool for step in plan.steps}

            for required_tool in case["expected_tools"]:
                assert required_tool in plan_tools, (
                    f"Case '{case['id']}': missing required tool '{required_tool}'. "
                    f"Plan contains: {plan_tools}"
                )

    def test_no_hallucinated_tools(self, planner, planner_cases):
        """Every tool name in the plan must be a real tool or a synthesis sentinel.

        This catches the common LLM failure mode of inventing plausible but
        nonexistent tool names like 'get_earnings_report' or 'analyze_sec_filing'.

        The LLM sometimes generates a final "combine results" step with
        tool='N/A' or 'none' — these aren't hallucinations, they're synthesis
        meta-steps. The step executor handles them gracefully (error message).
        We allow these sentinels but flag anything else.
        """
        synthesis_sentinels = {"N/A", "n/a", "none", "None", "synthesis", "combine"}

        for case in planner_cases:
            plan = planner.create_plan(case["query"])

            for step in plan.steps:
                assert step.tool in VALID_TOOLS or step.tool in synthesis_sentinels, (
                    f"Case '{case['id']}': hallucinated tool '{step.tool}' in step {step.id}. "
                    f"Valid tools: {sorted(VALID_TOOLS)}"
                )

    def test_step_dependencies_valid(self, planner, planner_cases):
        """Step depends_on IDs must reference earlier steps in the plan.

        A step can only depend on steps with a lower ID (already executed).
        Circular or forward dependencies indicate a broken plan.
        """
        for case in planner_cases:
            plan = planner.create_plan(case["query"])
            step_ids = {step.id for step in plan.steps}

            for step in plan.steps:
                for dep_id in step.depends_on:
                    assert dep_id in step_ids, (
                        f"Case '{case['id']}': step {step.id} depends on "
                        f"nonexistent step {dep_id}. Available IDs: {step_ids}"
                    )
                    assert dep_id < step.id, (
                        f"Case '{case['id']}': step {step.id} depends on "
                        f"step {dep_id} which is not earlier in the plan"
                    )
