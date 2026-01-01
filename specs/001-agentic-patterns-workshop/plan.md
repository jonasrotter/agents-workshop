# Implementation Plan: Agentic AI Patterns Workshop

**Branch**: `001-agentic-patterns-workshop` | **Date**: 2026-01-01 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-agentic-patterns-workshop/spec.md`

## Summary

Build a hands-on workshop teaching 7 agentic AI patterns using Python, Azure OpenAI, and the Microsoft Agent Framework. Each scenario is delivered as a Jupyter Notebook with executable examples, progressing from simple agents with MCP tools through multi-agent orchestration, AG-UI interfaces, A2A interoperability, declarative workflows, moderated discussions, and evaluation-driven evolution. OpenTelemetry with Azure Monitor provides observability across all scenarios.

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: Microsoft Agent Framework (azure-ai-projects), OpenTelemetry, Azure OpenAI SDK, FastAPI (AG-UI/A2A servers), MCP SDK  
**Storage**: N/A (workshop uses in-memory state; Azure Monitor for telemetry persistence)  
**Testing**: pytest + pytest-cov + pytest-asyncio  
**Target Platform**: Local development (Windows/macOS/Linux), Jupyter Notebooks  
**Project Type**: Single project (workshop modules as notebooks)  
**Performance Goals**: N/A (workshop focuses on learning, not performance)  
**Constraints**: Participants need Azure OpenAI API access and Azure subscription for Monitor  
**Scale/Scope**: 7 scenario modules, ~30-45 min each, ~4-6 hours total workshop

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Type Safety | ✅ PASS | All code uses type hints; mypy strict mode in CI |
| II. Test-First | ⚠️ ADAPTED | Workshop is educational—tests verify examples work, not TDD for each cell |
| III. Clean Code | ✅ PASS | Functions ≤30 lines, docstrings on all public APIs |
| IV. Dependencies | ✅ PASS | pyproject.toml with uv lock file |
| V. Observability | ✅ PASS | OpenTelemetry + Azure Monitor integrated in Scenario 1, used throughout |

**Adaptation Justification (Principle II)**: Workshop notebooks are educational artifacts, not production code. Tests verify that examples execute correctly but don't follow strict TDD cycle. This is acceptable because the primary goal is teaching concepts, not building a production system.

## Project Structure

### Documentation (this feature)

```text
specs/001-agentic-patterns-workshop/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (API schemas)
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
# Workshop structure with Jupyter Notebooks
src/
├── common/                    # Shared utilities across scenarios
│   ├── __init__.py
│   ├── config.py              # Environment and Azure configuration
│   ├── telemetry.py           # OpenTelemetry setup for Azure Monitor
│   └── exceptions.py          # Custom exception hierarchy
├── tools/                     # MCP tool implementations
│   ├── __init__.py
│   ├── search_tool.py         # Web search tool
│   ├── calculator_tool.py     # Calculator tool
│   └── file_tool.py           # File operations tool
├── agents/                    # Reusable agent definitions
│   ├── __init__.py
│   ├── base_agent.py          # Base agent with telemetry
│   ├── research_agent.py      # Research-focused agent
│   └── moderator_agent.py     # Discussion moderator
└── workflows/                 # Workflow definitions
    ├── __init__.py
    └── engine.py              # Deterministic workflow engine

notebooks/
├── 00_setup.ipynb             # Environment setup and verification
├── 01_simple_agent_mcp.ipynb  # Scenario 1: Simple agent + MCP + observability
├── 02_agui_interface.ipynb    # Scenario 2: AG-UI protocol
├── 03_a2a_protocol.ipynb      # Scenario 3: A2A agent exposure
├── 04_deterministic_workflows.ipynb  # Scenario 4: Multi-agent workflows
├── 05_declarative_agents.ipynb       # Scenario 5: YAML-based config
├── 06_agent_discussions.ipynb        # Scenario 6: Moderated debates
└── 07_evaluation_evolution.ipynb     # Scenario 7: Eval + prompt iteration

configs/
├── agents/                    # Declarative agent configs (Scenario 5)
│   ├── research_agent.yaml
│   └── summarizer_agent.yaml
└── workflows/                 # Declarative workflow configs (Scenario 5)
    └── research_pipeline.yaml

tests/
├── conftest.py
├── unit/
│   └── test_common.py
├── integration/
│   └── test_scenarios.py      # Verify each notebook runs
└── contract/
    └── test_protocols.py      # Verify MCP/AG-UI/A2A schemas
```

**Structure Decision**: Single project with Jupyter Notebooks as primary delivery format. Notebooks import from `src/` for reusable components. This enables interactive learning while maintaining code quality in the shared modules.

## Complexity Tracking

> No constitution violations requiring justification. Principle II adaptation is documented above.

---

## Phase Completion Summary

### Phase 0: Research ✅

**Output**: [research.md](./research.md)

Resolved all technical unknowns:
- Microsoft Agent Framework: `agent-framework` Python package with Azure OpenAI
- MCP: `mcp` Python SDK for tool servers  
- AG-UI: `ag-ui-core` with FastAPI for streaming interfaces
- A2A: Google protocol for agent-to-agent interoperability
- OpenTelemetry: `azure-monitor-opentelemetry` for Azure Monitor
- Workflow Engine: Lightweight custom implementation (~100 lines)
- Declarative Config: YAML with Pydantic validation

### Phase 1: Design & Contracts ✅

**Outputs**:
- [data-model.md](./data-model.md) - Workshop module, agent config, workflow schemas
- [contracts/mcp-tools.md](./contracts/mcp-tools.md) - MCP tool JSON schemas
- [contracts/agui-events.md](./contracts/agui-events.md) - AG-UI event schemas
- [contracts/a2a-protocol.md](./contracts/a2a-protocol.md) - A2A protocol schemas
- [quickstart.md](./quickstart.md) - Environment setup guide

**Agent Context Updated**: `.github/agents/copilot-instructions.md`

### Constitution Re-Check (Post-Design) ✅

| Principle | Status | Post-Design Notes |
|-----------|--------|-------------------|
| I. Type Safety | ✅ PASS | All schemas use Pydantic with strict validation |
| II. Test-First | ⚠️ ADAPTED | Tests verify notebook execution, not TDD per cell |
| III. Clean Code | ✅ PASS | Clear separation: common/, tools/, agents/, workflows/ |
| IV. Dependencies | ✅ PASS | All deps defined in pyproject.toml research section |
| V. Observability | ✅ PASS | Trace attributes convention defined in data-model.md |

---

## Next Steps

This plan is complete through Phase 1. To proceed:

1. Run `/speckit.tasks` to generate implementation tasks
2. Tasks will create the actual code in `src/`, `notebooks/`, `configs/`
3. Each task maps to a specific notebook or module

**Branch**: `001-agentic-patterns-workshop`
**Plan Path**: `specs/001-agentic-patterns-workshop/plan.md`
