# Plan

You are creating a detailed implementation plan based on research findings. This is Phase 2 of the Research → Plan → Implement workflow.

## Initial Setup

When this command is invoked, respond with:
```
Starting implementation planning (Phase 2: Research → Plan → Implement)

I'll create a detailed implementation plan. Please provide either:
1. The path to your research document from Phase 1, or
2. A summary of what needs to be implemented
```

## Planning Process

### 1. Context Gathering

If research document exists:
- Read the research document from Phase 1
- Extract key findings and open questions
- Identify components that need modification

If starting fresh:
- Conduct quick codebase exploration using Explore agents
- Identify key files and patterns
- Understand current implementation

### 2. Analyze Requirements
- Clarify the desired outcome
- Identify constraints and dependencies
- Consider existing patterns:
  - LangGraph StateGraph node structure
  - Tool registration pattern (`Tool.from_function`)
  - Pydantic model pattern for structured outputs
  - 3-tier caching strategy
  - WebSocket message protocol (`auth` → `query` → `response`)

### 3. Design Solution

Break implementation into logical phases:
- Each phase should be independently testable
- Phases build on each other
- Include verification steps for each phase
- Include test requirements for each phase

### 4. Create Plan Document

Save to: `data/plans/YYYY-MM-DD-{topic}-plan.md`

```markdown
# Implementation Plan: [Feature/Fix Name]

**Date**: [Current date]
**Phase**: 2 of 3 (Research → Plan → Implement)
**Based on Research**: [Link to research doc if exists]
**Next Phase**: `/implement`

## Overview
[Brief description of what will be implemented and why]

## Current State
[Summary from research — what exists today]

## Desired End State
[Clear description of success]

## Out of Scope
[What will NOT be addressed]

## Implementation

### Phase 1: [Name] [ ]
**Goal**: [What this phase accomplishes]

**Changes**:
1. `path/to/file.py`:
   - [ ] Change description
   - [ ] Another change

2. `tests/test_file.py`:
   - [ ] Test for new functionality

**Verification**:
- [ ] `uv run pytest tests/test_file.py -xvs`
- [ ] Manual check: [description]

### Phase 2: [Name] [ ]
[Same structure]

### Phase 3: [Integration & Tests] [ ]
**Changes**:
1. Test coverage
2. Integration verification
3. Prompt updates if needed

**Verification**:
- [ ] `uv run pytest` — full suite passes
- [ ] Manual end-to-end test via WebSocket

## Testing Strategy
- Unit tests for new functions
- Mocked LLM calls for tool tests
- WebSocket integration tests (pytest-asyncio)

## Risk Mitigation
1. **Risk**: [Potential problem]
   **Mitigation**: [How to handle]

## Success Criteria
- [ ] All functional requirements met
- [ ] Tests pass with coverage for new code
- [ ] No regression in existing features
- [ ] Follows existing patterns (caching, Pydantic models, tool registration)
```

### 5. Review and Refine
- Verify plan follows existing project patterns
- Check phases are properly sized
- Ensure verification steps are comprehensive
- Confirm test requirements are included

### 6. Present Plan
```
Plan created: data/plans/YYYY-MM-DD-{topic}-plan.md

Summary:
- [N phases planned]
- [Key components to modify]

Ready to implement. You can:
1. Review and adjust the plan
2. Proceed: `/implement [plan-file]`
```

## Project-Specific Patterns to Follow

### New Tools
1. Function in `agents/tools/sec_tools.py` or `research_tools.py`
2. Cache strategy (which tier?)
3. Add to `create_sec_tools()` or `create_research_tools()`
4. Update `TOOL_CAPABILITIES` in `agents/prompts.py`

### New Graph Nodes
1. Node function in `agents/graph/sec_graph.py`
2. Update `AnalysisState` TypedDict if new state needed
3. Wire with `add_node()`, `add_edge()`, or `add_conditional_edges()`

### New Pydantic Models
1. Model in `agents/sec_workflow/sec_llm_models.py`
2. Output parser in `SECDocumentProcessor`
3. Analysis method with LLM chain

### New Frontend Components
1. Component in `frontend/src/lib/components/`
2. Wire into `App.svelte` page routing
3. WebSocket message types if needed (update both `api/main.py` and component)

## Planning Best Practices
- **Incremental progress**: Each phase provides value
- **Test alongside code**: Never plan tests as a separate final phase
- **Follow existing patterns**: Match caching, tool registration, Pydantic models
- **Document deviations**: If plan diverges from existing patterns, explain why
