# Research

You are conducting comprehensive research to document the current state of the codebase. This is Phase 1 of the Research → Plan → Implement workflow.

## CRITICAL: Document what EXISTS, not what SHOULD BE
- DO NOT suggest improvements or changes unless explicitly asked
- DO NOT critique the implementation or identify problems
- ONLY describe what exists, where it exists, how it works, and how components interact
- You are creating a technical map of the existing system

## Initial Setup

When this command is invoked, respond with:
```
Starting research (Phase 1: Research → Plan → Implement)

I'll analyze your query and adapt my approach:
- GitHub issues (#NNN) → Issue reproduction and analysis
- LangGraph/tools → Agent architecture exploration
- Frontend → Svelte component and WebSocket research
- General → Codebase exploration

What would you like me to research?
```

## Research Process

### 1. Read Mentioned Files First
If the user mentions specific files, read them FULLY before spawning sub-agents. This ensures full context before decomposing the research.

### 2. Decompose the Research Question
- Break into composable research areas
- Identify components, patterns, or concepts to investigate
- Create a task list to track subtasks

### 3. Spawn Parallel Sub-Agents

Use Task agents to research different aspects concurrently:

**Core Agents:**
- Use **Explore** subagent to find WHERE files and components live
- Use **general-purpose** subagent to understand HOW specific code works

**Domain-Specific Exploration:**
- **LangGraph**: Graph definition, nodes, edges, state in `agents/graph/sec_graph.py`
- **Tools**: SEC tools in `agents/tools/sec_tools.py`, research tools in `agents/tools/research_tools.py`
- **SEC Data**: edgartools integration in `agents/sec_workflow/`
- **Frontend**: Svelte components in `frontend/src/lib/components/`
- **API**: WebSocket handler and REST endpoints in `api/main.py`

### 4. Synthesize Findings

Wait for ALL sub-agents to complete, then:
- Compile results across components
- Connect findings (how systems interact)
- Include specific file paths and line numbers

### 5. Save Research Document

Save to: `data/research/YYYY-MM-DD-{topic}.md`

```markdown
# Research: [Topic]

**Date**: [Current date]
**Phase**: 1 of 3 (Research → Plan → Implement)
**Next Phase**: `/plan`

## Research Question
[Original query]

## Summary
[High-level documentation of what was found]

## Detailed Findings

### [Component/Area 1]
- Description of what exists (`file.py:line`)
- How it connects to other components

### [Component/Area 2]
...

## Code References
- `agents/graph/sec_graph.py:83` - Router node classification
- `agents/tools/sec_tools.py:45` - Tool caching implementation

## Data Flows
[How data moves through the system for this feature/area]

## Open Questions for Planning Phase
[Areas that need consideration during planning]
```

### 6. Present and Suggest Next Steps
- Summarize findings concisely
- Suggest `/plan` to move to Phase 2
- Ask if there are follow-up questions

## Project-Specific Focus Areas
- LangGraph StateGraph: Router → ReAct/Planner → Synthesizer flow
- Tool caching (3-tier: retrievers, LLM analysis, research)
- edgartools integration (SECDataRetrieval wrapper)
- WebSocket auth and message flow
- Pydantic structured output models
- Query complexity classification and planning

## Important Notes
- Use parallel Task agents to maximize efficiency
- Always run fresh research — don't rely on stale documents
- Focus on concrete file paths and line numbers
- Research documents should be self-contained
- Keep the main agent focused on synthesis, not deep file reading
