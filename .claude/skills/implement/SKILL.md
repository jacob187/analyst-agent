# Implement

You are implementing a plan step-by-step, following the phases outlined in the planning document. This is Phase 3 of the Research → Plan → Implement workflow.

## Initial Setup

When this command is invoked, respond with:
```
Starting implementation (Phase 3: Research → Plan → Implement)

I'll implement the plan phase by phase, verifying each step before proceeding.

Please provide the path to your implementation plan from Phase 2.
```

## Implementation Process

### 1. Load and Review Plan
- Read the entire plan document
- Identify all phases and their checkboxes
- Note which phases are already complete
- Create a task list matching the plan phases
- Understand success criteria and verification steps

### 2. Pre-Implementation Checks
```bash
git status
git diff
uv run pytest
```

### 3. Phase-by-Phase Implementation

For each incomplete phase in the plan:

#### A. Start Phase
1. Update plan: mark phase as in-progress
2. Update task list: mark as in_progress
3. Announce what you're implementing

#### B. Implement Changes
- Follow the plan's specified changes exactly
- Use Edit tool for modifications
- Create new files with Write only when needed
- Implement in the order specified
- **Write tests alongside each change, not as an afterthought**

#### C. Verify Implementation
Run the verification steps from the plan:
```bash
# Run specific tests
uv run pytest tests/test_module.py -xvs

# Check functionality
uv run python -c "from agents.tools.sec_tools import create_sec_tools; print('OK')"

# Verify no regressions
uv run pytest
```

#### D. Handle Issues
If verification fails:
1. **Stop and analyze** the discrepancy
2. **Document the issue**:
   ```
   Issue in Phase N:
   Expected: [what plan said]
   Found: [actual situation]
   Attempting fix: [your solution]
   ```
3. **Fix and re-verify**
4. If unable to fix, ask user for guidance

#### E. Complete Phase
1. Update plan: mark phase complete with [x]
2. Update task list: mark as completed
3. Brief summary of what was accomplished

### 4. Context Management

After completing each major phase:
1. Update the plan document with progress
2. If context is getting large, suggest `/checkpoint` before continuing

### 5. Final Verification

After all phases complete:
```bash
# Full test suite
uv run pytest --cov=agents --cov=api

# Check imports work
uv run python -c "from agents.graph.sec_graph import create_sec_qa_agent; print('OK')"
```

### 6. Completion Report

```markdown
Implementation Complete

## Summary
- All [N] phases implemented
- [X] files modified
- [Y] tests added/updated

## Verification Results
- All tests passing
- No regressions
- Manual verification done

## Changes Made
[List key files changed with brief descriptions]

## Next Steps
1. Review changes: `git diff`
2. Run final tests: `uv run pytest`

Plan document updated: [path with all phases checked]
```

## Project-Specific Guidelines

### LangGraph Changes
- Test graph compilation: `graph.compile()` should not error
- Test state transitions: verify routing logic
- Test with mocked LLM responses

### Tool Changes
- Verify tool function signature matches `Tool.from_function` expectations
- Test caching behavior (cache hit and miss paths)
- Update `TOOL_CAPABILITIES` prompt string

### Frontend Changes
- Test WebSocket message format compatibility
- Verify Svelte 5 reactivity patterns ($state, $derived)
- Check markdown rendering in ChatMessage

### Database Changes
- Test migration path (init_db creates tables if not exist)
- Verify async operations (aiosqlite patterns)
- Test fire-and-forget saves don't swallow errors

## Recovery from Issues

If implementation gets stuck:
1. Save progress to plan document
2. Document the blocker clearly
3. Suggest options:
   - Skip to next phase
   - Research the issue (back to `/research`)
   - Ask user for guidance
   - Rollback changes

## Following the Plan's Intent
- **Trust the plan** but adapt to reality
- **Document deviations** when necessary
- **Maintain quality** — don't skip verification
- **Keep progress visible** — update plan and task list
