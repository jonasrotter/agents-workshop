# Tasks: AG-UI Interface Refactoring

**Feature**: AG-UI SDK Integration (spec 002)
**Input**: Design documents from `specs/002-agui-refactoring/`
**Documents**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, quickstart.md ✅

---

## Path Conventions

- `src/agents/agui_server.py` - Primary refactoring target (551 → ~250 lines)
- `src/agents/__init__.py` - Module exports
- `notebooks/02_agui_interface.ipynb` - Educational notebook
- `tests/contract/test_agui_schemas.py` - Contract tests
- `tests/integration/test_scenario_02.py` - Integration tests
- `tests/unit/test_agui.py` - New unit tests (to be created)

---

## Phase 1: Setup

**Purpose**: Verify prerequisites and establish baseline

- [X] T201 Verify SDK packages installed in .venv312 environment
- [X] T202 Run baseline tests: `pytest tests/contract/test_agui_schemas.py tests/integration/test_scenario_02.py -v`
- [X] T203 Create git branch `002-agui-refactoring` for refactoring work

**Checkpoint**: Baseline established - all existing tests pass ✅

---

## Phase 2: Foundational (SDK Type Integration)

**Purpose**: Replace custom types with official SDK types - MUST complete before Phase 3

- [X] T204 Replace custom EventType enum with ag_ui.core.EventType in src/agents/agui_server.py
- [X] T205 [P] Replace custom request models (RunAgentInput, Message, Tool, ToolCall) with SDK types in src/agents/agui_server.py
- [X] T206 Refactor AGUIEventEmitter to use EventEncoder from ag_ui.encoder in src/agents/agui_server.py
- [X] T207 Run contract tests to verify SDK type integration in tests/contract/test_agui_schemas.py

**Checkpoint**: SDK types integrated - custom type definitions removed ✅

---

## Phase 3: New SDK Components

**Purpose**: Add production-ready SDK wrappers for Agent Framework integration

- [X] T208 [P] Add create_agui_endpoint() helper function in src/agents/agui_server.py
- [X] T209 [P] Add AGUIClient wrapper class in src/agents/agui_server.py
- [X] T210 [P] Update AGUIServer class to support SDK mode (use_sdk parameter) in src/agents/agui_server.py

**Checkpoint**: New SDK components added ✅

---

## Phase 4: Module Exports

**Purpose**: Update module exports for new components

- [X] T211 Update src/agents/__init__.py with new exports (create_agui_endpoint, AGUIClient, EventType)

**Checkpoint**: All new components exported ✅

---

## Phase 5: Notebook Update

**Purpose**: Update notebook to demonstrate both custom and SDK approaches

- [X] T212 Update notebooks/02_agui_interface.ipynb with SDK integration examples

**Checkpoint**: Notebook updated with SDK examples ✅

---

## Phase 6: Test Updates

**Purpose**: Update and add tests for refactored implementation

- [X] T213 [P] Update tests/contract/test_agui_schemas.py for SDK types
- [X] T214 [P] Create tests/unit/test_agui.py for new components
- [X] T215 Run full test suite and verify coverage ≥80%

**Checkpoint**: All tests pass, coverage ≥80% ✅ (623 tests, 80.35% coverage)

---

## Phase 7: Polish & Validation

**Purpose**: Final cleanup and validation

- [X] T216 Remove unused imports and dead code from src/agents/agui_server.py
- [X] T217 Run linting (ruff) and type checking (mypy) on src/agents/agui_server.py (Note: ruff/mypy not installed, syntax check passed)
- [X] T218 Validate quickstart.md code examples execute correctly
- [X] T219 Commit and push changes to branch 002-agui-refactoring

**Checkpoint**: Refactoring complete, ready for merge ✅

---

## Task Details

### T204: Replace EventType enum
- Remove lines 35-54 (custom EventType enum definition)
- Add import: `from ag_ui.core import EventType`
- Verify `EventType.RUN_STARTED`, `EventType.TEXT_MESSAGE_CONTENT` etc. still accessible
- Update `__all__` exports

### T205: Replace request models
- Remove custom `ToolCall`, `Message`, `Tool`, `RunAgentInput` Pydantic models
- Add imports: `from ag_ui.core import RunAgentInput, Message, Tool`
- Update type hints to use SDK types

### T206: Refactor AGUIEventEmitter
- Add import: `from ag_ui.encoder import EventEncoder`
- Replace `_format_sse()` method with `self._encoder.encode()`
- Simplify all `emit_*` methods to use encoder
- Remove custom event Pydantic models (BaseEvent, RunStartedEvent, etc.)
- Preserve public API: `emit_run_started()`, `emit_text_content()`, etc.

### T208: create_agui_endpoint() helper
- Import: `from agent_framework.ag_ui import add_agent_framework_fastapi_endpoint`
- Signature: `def create_agui_endpoint(app: FastAPI, agent: Any, path: str = "/") -> None`
- Wrap SDK function with type hints, docstring, and error handling

### T209: AGUIClient wrapper
- Import: `from agent_framework.ag_ui import AGUIChatClient`
- Implement: `__init__(endpoint, timeout)`, `send()`, `stream()`, `close()`
- Context manager support: `__aenter__`, `__aexit__`

### T210: AGUIServer SDK mode
- Add `use_sdk: bool = False` parameter to `__init__`
- When `use_sdk=True`: use `add_agent_framework_fastapi_endpoint`
- When `use_sdk=False`: use simplified custom implementation (educational)

### T212: Notebook updates
- Add section: "Part X: Using the Official SDK"
- Show `create_agui_endpoint()` usage
- Show `AGUIClient` usage
- Compare custom vs SDK approaches
- Keep existing custom cells for educational value

### T214: New unit tests
- Test `AGUIEventEmitter` wrapper methods emit correct events
- Test `create_agui_endpoint()` helper registers routes correctly
- Test `AGUIClient` wrapper `send()` and `stream()` methods
- Test `AGUIServer` both SDK and custom modes
- Target: 10+ new unit tests

---

## Dependencies

### Phase Dependencies

```
Phase 1 (Setup)
    ↓
Phase 2 (Foundational) ← CRITICAL GATE
    ↓
Phase 3 (New Components)
    ↓
Phase 4 (Exports)
    ↓
Phase 5 (Notebook)
    ↓
Phase 6 (Tests)
    ↓
Phase 7 (Polish)
```

### Task Dependencies

```
T204 ─┬─→ T206 ─→ T207
T205 ─┘
          ↓
T208 ─┬─→ T211 ─→ T212
T209 ─┤
T210 ─┘
          ↓
T213 ─┬─→ T215
T214 ─┘
          ↓
T216 → T217 → T218 → T219
```

### Parallel Opportunities

```bash
# Phase 2 - after T204:
T205: Request models (different section)

# Phase 3 - all independent files/sections:
T208: create_agui_endpoint()
T209: AGUIClient
T210: AGUIServer SDK mode

# Phase 6 - test files independent:
T213: tests/contract/test_agui_schemas.py
T214: tests/unit/test_agui.py
```

---

## Acceptance Criteria Mapping

| AC | Task(s) | Verified By |
|----|---------|-------------|
| AC-1: Uses SDK endpoint | T208, T210 | `create_agui_endpoint()` uses `add_agent_framework_fastapi_endpoint` |
| AC-2: SDK EventType | T204 | `from ag_ui.core import EventType` in imports |
| AC-3: ChatAgent streaming | T210 | AGUIServer SDK mode with ChatAgent |
| AC-4: Notebook works | T212, T215 | `test_scenario_02.py` passes |
| AC-5: Contract tests pass | T213, T215 | `test_agui_schemas.py` green |
| AC-6: Coverage ≥80% | T215 | pytest-cov report |

---

## Estimated Timeline

| Phase | Tasks | Est. Hours |
|-------|-------|------------|
| Phase 1: Setup | T201-T203 | 0.5h |
| Phase 2: Foundational | T204-T207 | 2h |
| Phase 3: New Components | T208-T210 | 2h |
| Phase 4: Exports | T211 | 0.25h |
| Phase 5: Notebook | T212 | 1h |
| Phase 6: Tests | T213-T215 | 1.5h |
| Phase 7: Polish | T216-T219 | 0.75h |

**Total**: ~8 hours

---

## Implementation Strategy

**MVP**: Phases 1-4 (SDK integration + exports) - ~5 hours
- Delivers core SDK integration
- Tests run against existing contract tests
- Notebook update can be deferred

**Full Delivery**: All phases - ~8 hours
- Complete SDK integration with dual implementation
- Updated notebook with educational content
- Full test coverage

---

## Rollback Plan

If refactoring causes issues:
1. Git revert to pre-refactoring commit on `main`
2. Original custom implementation preserved in git history
3. No breaking changes to public API (backward compatible)
