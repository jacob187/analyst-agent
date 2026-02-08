"""Query planner for decomposing complex financial analysis queries into executable steps."""

from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel

from agents.prompts import (
    QUERY_PLANNER_SYSTEM_PROMPT,
    QUERY_CLASSIFIER_PROMPT,
    TOOL_CAPABILITIES,
)


class AnalysisStep(BaseModel):
    """A single step in an analysis plan."""

    id: int = Field(description="Step number")
    action: str = Field(description="What to do in this step")
    tool: str = Field(description="Which tool to use")
    rationale: str = Field(description="Why this step is needed")
    depends_on: List[int] = Field(
        default_factory=list, description="IDs of steps this depends on"
    )


class QueryClassification(BaseModel):
    """Classification of query complexity."""

    complexity: Literal["simple", "moderate", "complex"] = Field(
        description="Query complexity level"
    )
    reasoning: str = Field(description="Brief explanation of why this complexity level")
    estimated_tools: int = Field(description="Estimated number of tools needed")


class QueryPlan(BaseModel):
    """A complete plan for answering a financial analysis query."""

    query_type: str = Field(
        description="Type of query: simple, moderate, or complex"
    )
    requires_planning: bool = Field(
        description="Whether this query needs multi-step planning"
    )
    steps: List[AnalysisStep] = Field(
        default_factory=list, description="Ordered steps to execute"
    )
    synthesis_approach: str = Field(
        default="",
        description="How to combine results from multiple steps",
    )


class QueryPlanner:
    """Plans and decomposes complex financial analysis queries."""

    def __init__(self, llm: BaseChatModel, ticker: str, has_research_tools: bool = False):
        self.llm = llm
        self.ticker = ticker
        self.has_research_tools = has_research_tools
        self.plan_llm = llm.with_structured_output(QueryPlan)
        self.classify_llm = llm.with_structured_output(QueryClassification)

    def classify_query(self, query: str) -> QueryClassification:
        """
        Use LLM to classify query complexity.

        Returns classification with complexity level, reasoning, and estimated tools.
        """
        research_note = (
            "Research tools ARE available (web search, news, competitors, trends)."
            if self.has_research_tools
            else "Research tools are NOT available."
        )

        prompt = QUERY_CLASSIFIER_PROMPT.format(
            ticker=self.ticker,
            tool_capabilities=TOOL_CAPABILITIES,
            research_note=research_note,
            query=query,
        )

        return self.classify_llm.invoke(prompt)

    def create_plan(self, query: str) -> QueryPlan:
        """
        Analyze a query and create an execution plan.

        For simple queries (single tool needed), returns a minimal plan.
        For complex queries, decomposes into multiple steps.
        """
        research_note = (
            "Research tools (web_search, deep_research, get_company_news, "
            "analyze_competitors, get_industry_trends) ARE available."
            if self.has_research_tools
            else "Research tools are NOT available. Only use SEC and stock market tools."
        )

        prompt = QUERY_PLANNER_SYSTEM_PROMPT.format(
            ticker=self.ticker,
            tool_capabilities=TOOL_CAPABILITIES,
            research_note=research_note,
            query=query,
        )

        plan = self.plan_llm.invoke(prompt)
        return plan

    def should_plan(self, query: str) -> tuple[bool, QueryClassification]:
        """
        Determine if a query needs multi-step planning.

        Returns (should_plan, classification).
        """
        classification = self.classify_query(query)

        # Simple queries with 1 tool don't need planning
        should_plan = (
            classification.complexity != "simple"
            or classification.estimated_tools > 1
        )

        return should_plan, classification


def create_planner(
    llm: BaseChatModel, ticker: str, has_research_tools: bool = False
) -> QueryPlanner:
    """Factory function to create a QueryPlanner instance."""
    return QueryPlanner(llm, ticker, has_research_tools)
