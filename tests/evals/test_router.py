"""Tests for the LLM-based query classifier (planner.classify_query).

Marker: eval_fast — each test makes a single LLM call to classify a query.
Uses router_cases.json as the golden dataset.

The classifier returns a QueryClassification with:
- complexity: "simple" | "moderate" | "complex"
- reasoning: str
- estimated_tools: int
"""

import pytest


@pytest.mark.eval_fast
class TestQueryClassifier:
    """Validate classify_query against router_cases.json expectations."""

    def test_simple_queries_classified_low(self, planner, router_cases):
        """Simple queries should be classified as simple or moderate (not complex).

        Simple queries need at most 1-2 tools, so the classifier should not
        escalate them to complex. We accept 'moderate' as a pass because the
        LLM might reasonably see a metric lookup as needing context.
        """
        simple_cases = [c for c in router_cases if c["expected_complexity"] == "simple"]

        for case in simple_cases:
            result = planner.classify_query(case["query"])
            assert result.complexity in ("simple", "moderate"), (
                f"Query '{case['id']}' classified as {result.complexity}, "
                f"expected simple/moderate. Reasoning: {result.reasoning}"
            )

    def test_complex_queries_classified_high(self, planner, router_cases):
        """Complex queries should be classified as moderate or complex (not simple).

        These involve multiple data sources and cross-referencing, so the
        classifier should recognize the multi-step nature.
        """
        complex_cases = [c for c in router_cases if c["expected_complexity"] == "complex"]

        for case in complex_cases:
            result = planner.classify_query(case["query"])
            assert result.complexity in ("moderate", "complex"), (
                f"Query '{case['id']}' classified as {result.complexity}, "
                f"expected moderate/complex. Reasoning: {result.reasoning}"
            )

    def test_estimated_tools_reasonable(self, planner, router_cases):
        """estimated_tools should be within sensible bounds.

        Simple queries: 1-2 tools. Complex queries: 2-8 tools (we have 12 SEC
        tools, so requesting more than 8 for a single query is suspicious).
        """
        for case in router_cases:
            result = planner.classify_query(case["query"])

            if case["expected_complexity"] == "simple":
                assert 1 <= result.estimated_tools <= 3, (
                    f"Query '{case['id']}' estimated {result.estimated_tools} tools, "
                    f"expected 1-3 for a simple query"
                )
            else:
                assert 2 <= result.estimated_tools <= 8, (
                    f"Query '{case['id']}' estimated {result.estimated_tools} tools, "
                    f"expected 2-8 for a complex query"
                )
