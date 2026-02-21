"""LLM-as-judge response quality evaluation.

Marker: eval_quality — uses a second LLM call to score agent responses.

Design: We invoke the agent, then ask the same LLM (with structured output)
to score the response on relevance, accuracy, and completeness. This is the
standard LLM-as-judge pattern used in eval frameworks like LangSmith.

QualityScores is a Pydantic model that the judge LLM fills in. Scores are
1-5 (1=terrible, 5=excellent). We set minimum thresholds rather than
expecting perfection, because LLM-as-judge has inherent noise.
"""

import pytest

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage


class QualityScores(BaseModel):
    """Structured scoring model for LLM-as-judge evaluation."""

    relevance: int = Field(
        description="1-5: How relevant is the response to the query?"
    )
    accuracy: int = Field(
        description="1-5: How accurate is the financial data presented?"
    )
    completeness: int = Field(
        description="1-5: How completely does the response address all parts of the query?"
    )
    reasoning: str = Field(
        description="Brief explanation of the scores"
    )


JUDGE_PROMPT = """You are evaluating an AI financial analyst's response.

Query: {query}
Response: {response}

Score the response on three dimensions (1-5 each):
- relevance: Does the response address the query? (1=off-topic, 5=perfectly targeted)
- accuracy: Is the financial data plausible and correctly presented? (1=wrong, 5=accurate)
- completeness: Does it cover all aspects of the query? (1=partial, 5=thorough)

Provide brief reasoning for your scores."""


def _judge_response(llm, query: str, response: str) -> QualityScores:
    """Ask the LLM to score an agent response.

    Uses with_structured_output to get a QualityScores Pydantic model back,
    just like the planner uses it for QueryClassification and QueryPlan.
    """
    judge_llm = llm.with_structured_output(QualityScores)
    prompt = JUDGE_PROMPT.format(query=query, response=response)
    return judge_llm.invoke(prompt)


@pytest.mark.eval_quality
class TestResponseQuality:
    """LLM-as-judge scoring for agent responses."""

    def test_simple_query_quality(self, agent, llm):
        """Simple stock info query should score >= 3 on relevance and accuracy."""
        query = "What is Apple's current stock price and P/E ratio?"
        result = agent.invoke({
            "messages": [HumanMessage(content=query)]
        })
        response = result["messages"][-1].content

        scores = _judge_response(llm, query, response)

        assert scores.relevance >= 3, (
            f"Relevance {scores.relevance}/5 too low. {scores.reasoning}"
        )
        assert scores.accuracy >= 3, (
            f"Accuracy {scores.accuracy}/5 too low. {scores.reasoning}"
        )

    def test_complex_query_quality(self, agent, llm):
        """Complex multi-source query should score >= 3 on relevance and completeness."""
        query = "Analyze Apple's risk factors and financial health from their SEC filings"
        result = agent.invoke({
            "messages": [HumanMessage(content=query)]
        })
        response = result["messages"][-1].content

        scores = _judge_response(llm, query, response)

        assert scores.relevance >= 3, (
            f"Relevance {scores.relevance}/5 too low. {scores.reasoning}"
        )
        assert scores.completeness >= 3, (
            f"Completeness {scores.completeness}/5 too low. {scores.reasoning}"
        )

    def test_golden_query_key_terms(self, agent, llm, golden_queries):
        """Golden queries should mention all must_mention terms in response.

        This is a hard content check — the must_mention terms are fundamental
        to a correct response (e.g., a technical analysis MUST mention RSI).
        We use case-insensitive matching.
        """
        for case in golden_queries:
            result = agent.invoke({
                "messages": [HumanMessage(content=case["query"])]
            })
            response = result["messages"][-1].content.lower()

            for term in case["must_mention"]:
                assert term.lower() in response, (
                    f"Golden query '{case['id']}': response missing '{term}'. "
                    f"First 300 chars: {response[:300]}"
                )
