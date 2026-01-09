# Implementation Plan: A2A Server SDK Refactoring

**Branch**: `main` | **Date**: 2026-01-06 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/main/spec.md`

## Summary

Refactor the custom A2A server implementation to use the official `a2a-sdk` package, replacing ~400 lines of custom Pydantic models and request handlers with SDK equivalents. This reduces maintenance burden and ensures protocol compliance.

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: `a2a-sdk==0.3.17`, `agent-framework-a2a==1.0.0b251120`, FastAPI  
**Storage**: In-memory (`InMemoryTaskStore` from SDK)  
**Testing**: pytest + pytest-cov + pytest-asyncio  
**Target Platform**: Linux/Windows server  
**Project Type**: Single project (workshop codebase)  
**Performance Goals**: N/A (educational workshop)  
**Constraints**: Backward compatibility with existing notebooks  
**Scale/Scope**: 7 notebooks, ~5000 LOC

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Type Safety | ✅ PASS | SDK uses Pydantic models with full type hints |
| II. Test-First | ✅ PASS | Contract tests updated for SDK types |
| III. Clean Code | ✅ PASS | SDK reduces custom code by >50% |
| IV. Dependencies | ✅ PASS | Using official SDK from requirements.txt |
| V. Observability | ✅ PASS | OpenTelemetry integration preserved |

## Project Structure

### Documentation (this feature)

```text
specs/main/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # SDK analysis and type mappings
├── data-model.md        # Type migration guide
├── quickstart.md        # Quick start guide
├── contracts/           # Interface contracts
│   └── a2a-sdk-interfaces.md
└── tasks.md             # Implementation tasks
```

### Source Code (repository root)

```text
src/
├── agents/
│   ├── __init__.py          # Module exports (updated for SDK types)
│   ├── a2a_server.py        # Refactored to use A2AFastAPIApplication
│   └── ...                  # Other agent implementations
└── ...

tests/
├── contract/
│   └── test_a2a_schemas_sdk.py  # SDK type validation
├── integration/
│   └── test_scenario_03.py      # A2A protocol integration tests
└── unit/
    └── ...                      # Unit tests

notebooks/
└── 03_a2a_protocol.ipynb        # Updated for SDK imports
```

**Structure Decision**: Single project structure retained. SDK integration affects `src/agents/a2a_server.py` and related test files.

## Complexity Tracking

No constitution violations.
