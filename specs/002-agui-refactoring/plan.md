# Implementation Plan: AG-UI Interface Refactoring

**Branch**: `002-agui-refactoring` | **Date**: 2026-01-07 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-agui-refactoring/spec.md`

## Summary

Refactor the AG-UI protocol implementation to use Microsoft Agent Framework's native AG-UI support:
1. **Server-side**: `add_agent_framework_fastapi_endpoint()` for production use
2. **Client-side**: `AGUIChatClient` for consuming AG-UI servers
3. **Educational**: Keep simplified custom implementation for teaching protocol concepts

This aligns with research.md Section 3 and reduces maintenance burden (~551 lines → ~250 lines).

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: 
- `agent-framework` (1.0.0b251120)
- `agent-framework-ag-ui` (1.0.0b251223) - provides `AGUIChatClient`, `add_agent_framework_fastapi_endpoint`
- `ag-ui-core` (transitive) - provides `EventType`, `EventEncoder`
- `fastapi` - HTTP server

**Storage**: N/A (stateless streaming)
**Testing**: pytest + pytest-cov + pytest-asyncio
**Target Platform**: Local development, Jupyter Notebooks
**Project Type**: Single project (workshop modules)
**Performance Goals**: N/A (educational focus)
**Constraints**: Must maintain backward compatibility with existing notebook cells
**Scale/Scope**: 1 module refactoring (~551 lines → ~250 lines with dual implementation)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Type Safety | ✅ PASS | Will use typed SDK models instead of custom Pydantic |
| II. Test-First | ✅ PASS | Existing tests provide safety net for refactoring |
| III. Clean Code | ✅ PASS | SDK usage reduces code from 551 → ~100 lines |
| IV. Dependencies | ✅ PASS | ag-ui-core already in requirements.txt |
| V. Observability | ✅ PASS | Agent Framework has built-in tracing |

## Project Structure

### Documentation (this feature)

```text
specs/002-agui-refactoring/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (no changes expected)
└── tasks.md             # Phase 2 output
```

### Source Code (affected files)

```text
src/agents/
├── agui_server.py       # PRIMARY: Refactor to use SDK
└── __init__.py          # Update exports

notebooks/
└── 02_agui_interface.ipynb  # Update examples

tests/
├── contract/test_agui_schemas.py    # May need updates
└── integration/test_scenario_02.py  # Verify notebook works
```

**Structure Decision**: Minimal changes - refactoring single module to use SDK patterns.

## Phase 0: Research ✅

**Output**: See research.md for detailed findings

### Key Findings

1. **`ag-ui-core` Package Available**
   - `ag_ui.core.EventType` - Standard event type enum
   - `ag_ui.core.RunAgentInput` - Request schema
   - `ag_ui.encoder.EventEncoder` - SSE formatting

2. **`agent_framework.ag_ui` Module**
   - `add_agent_framework_fastapi_endpoint(app, agent, path)` - One-liner setup
   - Automatic streaming from `ChatAgent.run_stream()`
   - Built-in tool execution events

3. **Migration Strategy**
   - Option A: Use `add_agent_framework_fastapi_endpoint()` (simplest)
   - Option B: Use `ag-ui-core` with custom FastAPI (more control)
   - **Recommendation**: Option B for workshop (educational value)

### Dependencies Verified
- `ag-ui-core` - Already in requirements.txt
- `agent-framework` - Already in requirements.txt

## Phase 1: Design & Contracts ✅

### Design Decision

Use **Option C: Dual Implementation** (see research.md Section 6):
1. **Production**: `add_agent_framework_fastapi_endpoint()` via `create_agui_endpoint()` helper
2. **Educational**: Simplified custom implementation using `ag-ui-core` types
3. **Client**: `AGUIClient` wrapper around `AGUIChatClient`

This approach provides both educational value and production patterns.

### New Components

```python
# src/agents/agui_server.py (refactored ~250 lines)

# 1. Re-export SDK types (backward compat)
from ag_ui.core import EventType  # Replace custom enum
from ag_ui.encoder import EventEncoder

# 2. Simplified AGUIEventEmitter (wraps EventEncoder)
@dataclass
class AGUIEventEmitter:
    """Backward-compatible wrapper around EventEncoder."""
    thread_id: str
    run_id: str
    _encoder: EventEncoder = field(default_factory=EventEncoder, init=False)
    
    def emit_text_content(self, delta: str) -> str:
        return self._encoder.encode({...})

# 3. AGUIServer (simplified, uses SDK internally)
class AGUIServer:
    def __init__(self, agent=None, use_sdk: bool = False): ...
    def create_app(self) -> FastAPI: ...

# 4. NEW: create_agui_endpoint() helper
def create_agui_endpoint(app: FastAPI, agent: ChatAgent, path: str = "/"):
    """Production wrapper for add_agent_framework_fastapi_endpoint."""
    from agent_framework.ag_ui import add_agent_framework_fastapi_endpoint
    add_agent_framework_fastapi_endpoint(app, agent, path)

# 5. NEW: AGUIClient wrapper
class AGUIClient:
    """Simplified client for AG-UI servers."""
    async def send(self, message: str) -> str: ...
    async def stream(self, message: str) -> AsyncGenerator[str, None]: ...
```

### Backward Compatibility

| Component | Old Usage | New Usage | Breaking? |
|-----------|-----------|-----------|-----------|
| `EventType` | Custom enum | Re-export from `ag_ui.core` | No |
| `AGUIEventEmitter` | Custom class | Wrapper around `EventEncoder` | No |
| `AGUIServer` | Custom class | Simplified, SDK option | No |
| `create_agui_server()` | Factory | Updated internally | No |

### Output Files

- `research.md` ✅ - Detailed SDK analysis
- `data-model.md` ✅ - Type mappings and event schemas
- `quickstart.md` ✅ - Usage examples

### Contract Verification

Event schemas unchanged - contracts/agui-events.md compatible.

## Phase 2: Task Breakdown

See [tasks.md](./tasks.md) for detailed implementation tasks.

### Task Summary

| ID | Task | Est. Hours |
|----|------|------------|
| T201 | Replace custom EventType with ag_ui.core.EventType | 0.5h |
| T202 | Replace custom event models with SDK imports | 1h |
| T203 | Refactor AGUIEventEmitter to wrap EventEncoder | 1h |
| T204 | Add create_agui_endpoint() helper | 0.5h |
| T205 | Add AGUIClient wrapper class | 1h |
| T206 | Update AGUIServer to support SDK mode | 1h |
| T207 | Update notebook 02 to show both approaches | 1h |
| T208 | Update contract tests for SDK types | 1h |
| T209 | Run integration tests and verify coverage | 0.5h |

**Total Estimated**: 7.5 hours

## Constitution Re-Check (Post-Design) ✅

| Principle | Status | Post-Design Notes |
|-----------|--------|-------------------|
| I. Type Safety | ✅ PASS | SDK types are fully typed |
| II. Test-First | ✅ PASS | Existing tests as safety net |
| III. Clean Code | ✅ PASS | ~55% code reduction (551 → 250 lines) |
| IV. Dependencies | ✅ PASS | No new deps (ag-ui-core is transitive) |
| V. Observability | ✅ PASS | SDK includes built-in tracing |

## Next Steps

1. Run `/speckit.tasks` to generate detailed implementation tasks
2. Create feature branch `002-agui-refactoring`
3. Execute tasks in order
4. Verify all acceptance criteria

**Branch**: `002-agui-refactoring`
**Plan Path**: `specs/002-agui-refactoring/plan.md`
