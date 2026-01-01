# Tasks: Agentic AI Patterns Workshop

**Input**: Design documents from `/specs/001-agentic-patterns-workshop/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

**Tests**: Tests verify notebook execution and protocol schemas. TDD per cell is adapted for educational context.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each scenario.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1-US7)
- Include exact file paths in descriptions

## Path Conventions

Based on plan.md structure:
- `src/common/` - Shared utilities
- `src/tools/` - MCP tool implementations  
- `src/agents/` - Reusable agent definitions
- `src/workflows/` - Workflow definitions
- `notebooks/` - Jupyter notebooks (00-07)
- `configs/` - Declarative YAML configs
- `tests/` - Unit, integration, and contract tests

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and development environment

- [X] T001 Create project structure with pyproject.toml at repository root
- [X] T002 [P] Configure .gitignore for Python, Jupyter, and environment files
- [X] T003 [P] Create .env.example template with Azure OpenAI and Monitor placeholders
- [X] T004 [P] Configure ruff.toml for linting and formatting
- [X] T005 [P] Configure mypy.ini for strict type checking
- [X] T006 [P] Create pytest.ini with asyncio and coverage settings

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No notebook development can begin until this phase is complete

- [X] T007 Create src/common/__init__.py with module exports
- [X] T008 Implement Azure/OpenAI configuration loader in src/common/config.py
- [X] T009 [P] Implement OpenTelemetry + Azure Monitor setup in src/common/telemetry.py
- [X] T010 [P] Implement custom exception hierarchy in src/common/exceptions.py
- [X] T011 [P] Create tests/conftest.py with shared fixtures (mock Azure client, test tracer)
- [X] T012 [P] Create tests/unit/test_common.py for config and telemetry validation
- [X] T013 Implement setup verification notebook in notebooks/00_setup.ipynb

**Checkpoint**: Foundation ready - user story implementation can now begin ✅

---

## Phase 3: User Story 1 - Simple Agent with MCP Tools and Observability (Priority: P1) 🎯 MVP

**Goal**: Build a simple agent, connect tools via MCP, inspect traces & metrics

**Independent Test**: Run standalone agent that connects to MCP server, invokes tools, produces observable traces

### Implementation for User Story 1

- [X] T014 Create src/tools/__init__.py with tool module exports
- [X] T015 [P] [US1] Implement search_web tool in src/tools/search_tool.py per contracts/mcp-tools.md
- [X] T016 [P] [US1] Implement calculate tool in src/tools/calculator_tool.py per contracts/mcp-tools.md
- [X] T017 [P] [US1] Implement file operations tools in src/tools/file_tool.py per contracts/mcp-tools.md
- [X] T018 [US1] Create MCP server wrapper in src/tools/mcp_server.py using FastMCP
- [X] T019 Create src/agents/__init__.py with agent module exports
- [X] T020 [US1] Implement BaseAgent with telemetry integration in src/agents/base_agent.py
- [X] T021 [US1] Implement ResearchAgent extending BaseAgent in src/agents/research_agent.py
- [X] T022 [US1] Create Scenario 1 notebook in notebooks/01_simple_agent_mcp.ipynb with:
  - Agent creation with Azure OpenAI
  - MCP tool discovery and invocation
  - OpenTelemetry trace inspection
  - Hands-on exercise: Add new MCP tool
  - Edge case: MCP server unavailability with graceful degradation
- [X] T023 [P] [US1] Create tests/contract/test_mcp_schemas.py to validate tool schemas
- [X] T024 [P] [US1] Create tests/integration/test_scenario_01.py to verify notebook executes

**Checkpoint**: User Story 1 complete - participants can build agents with MCP tools and see traces ✅

---

## Phase 4: User Story 2 - AG-UI Protocol for Streaming Chat (Priority: P2)

**Goal**: Build user interface for agent using AG-UI protocol with streaming responses

**Independent Test**: Run web-based chat interface communicating with agent via AG-UI, showing streaming responses

### Implementation for User Story 2

- [X] T025 [P] [US2] Create AG-UI FastAPI server module in src/agents/agui_server.py per contracts/agui-events.md
- [X] T026 [US2] Implement streaming event emitter in src/agents/agui_server.py supporting all event types
- [X] T027 [US2] Create Scenario 2 notebook in notebooks/02_agui_interface.ipynb with:
  - AG-UI server setup with FastAPI
  - Streaming token-by-token responses
  - Tool execution status updates
  - Hands-on exercise: Customize UI component rendering
  - Edge case: Connection drop with reconnection status UI
- [X] T028 [P] [US2] Create tests/contract/test_agui_schemas.py to validate event schemas
- [X] T029 [P] [US2] Create tests/integration/test_scenario_02.py to verify notebook executes

**Checkpoint**: User Story 2 complete - participants can build streaming chat interfaces ✅

---

## Phase 5: User Story 3 - A2A Protocol for Agent Interoperability (Priority: P3)

**Goal**: Expose agent as service using A2A protocol for discovery and remote invocation

**Independent Test**: Run agent as A2A server, have client discover capabilities and invoke remotely

### Implementation for User Story 3

- [X] T030 [P] [US3] Create A2A server module in src/agents/a2a_server.py per contracts/a2a-protocol.md
- [X] T031 [US3] Implement Agent Card generation at /.well-known/agent-card.json endpoint
- [X] T032 [US3] Implement A2A JSON-RPC message handler for tasks/send and tasks/get
- [X] T033 [US3] Create Scenario 3 notebook in notebooks/03_a2a_protocol.ipynb with:
  - A2A server setup with Agent Card
  - Capability discovery endpoint
  - Remote task invocation
  - Hands-on exercise: Register agent with discovery service (mock)
  - Edge case: Configurable timeouts with A2A error responses
- [X] T034 [P] [US3] Create tests/contract/test_a2a_schemas.py to validate Agent Card and message schemas
- [X] T035 [P] [US3] Create tests/integration/test_scenario_03.py to verify notebook executes

**Checkpoint**: User Story 3 complete - participants can expose agents via A2A protocol ✅

---

## Phase 6: User Story 4 - Deterministic Multi-Agent Workflows (Priority: P4)

**Goal**: Build deterministic workflows coordinating multiple agents in predictable sequence

**Independent Test**: Run workflow chaining 3+ agents, verify execution order matches definition

### Implementation for User Story 4

- [ ] T036 Create src/workflows/__init__.py with workflow module exports
- [ ] T037 [US4] Implement workflow step definitions in src/workflows/steps.py using data-model.md schemas
- [ ] T038 [US4] Implement deterministic workflow engine in src/workflows/engine.py with:
  - Sequential agent orchestration
  - Data flow between steps
  - Error handling strategies (retry, fallback, abort)
- [ ] T039 [US4] Create Scenario 4 notebook in notebooks/04_deterministic_workflows.ipynb with:
  - Workflow definition and execution
  - Multi-agent pipeline example
  - Error handling demonstration
  - Hands-on exercise: Build custom workflow
- [X] T040 [P] [US4] Create tests/unit/test_workflows.py for workflow engine logic
- [X] T041 [P] [US4] Create tests/integration/test_scenario_04.py to verify notebook executes

**Checkpoint**: User Story 4 complete - participants can build deterministic multi-agent workflows ✅

---

## Phase 7: User Story 5 - Declarative Agents and Workflows (Priority: P5)

**Goal**: Define agents and workflows using YAML configuration without code

**Independent Test**: Modify YAML config file, observe agent behavior change without code changes

### Implementation for User Story 5

- [X] T042 [P] [US5] Create YAML schema definitions in src/common/yaml_loader.py with Pydantic validation
- [X] T043 [P] [US5] Create research_agent.yaml in configs/agents/ per data-model.md AgentConfig schema
- [X] T044 [P] [US5] Create summarizer_agent.yaml in configs/agents/ per data-model.md AgentConfig schema
- [X] T045 [US5] Create research_pipeline.yaml in configs/workflows/ per data-model.md WorkflowConfig schema
- [X] T046 [US5] Implement declarative agent loader in src/agents/declarative.py
- [X] T047 [US5] Create Scenario 5 notebook in notebooks/05_declarative_agents.ipynb with:
  - Loading agents from YAML
  - Loading workflows from YAML
  - Runtime behavior modification
  - Hands-on exercise: Add new agent via config
- [X] T048 [P] [US5] Create tests/unit/test_declarative.py for YAML loading validation
- [X] T049 [P] [US5] Create tests/integration/test_scenario_05.py to verify notebook executes

**Checkpoint**: User Story 5 complete - participants can define agents declaratively ✅

---

## Phase 8: User Story 6 - Moderating Agent Discussions (Priority: P6)

**Goal**: Moderate discussion between multiple agents with different perspectives

**Independent Test**: Run moderated debate between 2-3 agents, observe turn-taking and conclusion synthesis

### Implementation for User Story 6

- [X] T050 [US6] Implement ModeratorAgent in src/agents/moderator_agent.py with:
  - Turn-taking facilitation
  - Conflict resolution strategy
  - Conclusion synthesis
- [X] T051 [US6] Implement discussion protocol in src/agents/discussion.py with:
  - Participant registration
  - Round management
  - Cross-reference detection
- [X] T052 [US6] Create Scenario 6 notebook in notebooks/06_agent_discussions.ipynb with:
  - Multi-agent debate setup
  - Moderator configuration
  - Turn-by-turn observation
  - Hands-on exercise: Add debate participant
- [X] T053 [P] [US6] Create tests/unit/test_discussion.py for discussion protocol
- [X] T054 [P] [US6] Create tests/integration/test_scenario_06.py to verify notebook executes

**Checkpoint**: User Story 6 complete - participants can build moderated agent discussions ✅

---

## Phase 9: User Story 7 - Observability, Evaluation, and Evolution (Priority: P7)

**Goal**: Implement comprehensive observability, evaluate performance, evolve agents

**Independent Test**: Run evaluation suite, review metrics dashboard, apply improvement from feedback

### Implementation for User Story 7

- [X] T055 [P] [US7] Implement evaluation metrics collector in src/common/evaluation.py with:
  - Accuracy metrics
  - Latency tracking
  - Cost estimation
- [X] T056 [P] [US7] Implement prompt iteration framework in src/common/prompt_tuning.py with:
  - Version tracking
  - A/B comparison support
  - Improvement suggestions
- [X] T057 [US7] Create Scenario 7 notebook in notebooks/07_evaluation_evolution.ipynb with:
  - Evaluation suite execution
  - Metrics dashboard visualization
  - Prompt tuning iteration
  - Hands-on exercise: Apply evaluation-driven improvement
- [X] T058 [P] [US7] Create tests/unit/test_evaluation.py for metrics collection
- [X] T059 [P] [US7] Create tests/integration/test_scenario_07.py to verify notebook executes

**Checkpoint**: User Story 7 complete - participants understand evaluation-driven agent evolution ✅

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Final improvements affecting multiple scenarios

- [X] T060 [P] Update README.md with workshop overview, prerequisites, and getting started guide
- [X] T061 [P] Create docs/ARCHITECTURE.md documenting module relationships and design decisions
- [X] T062 [P] Create docs/CONTRIBUTING.md for workshop extension guidelines
- [X] T066 [P] Add estimated completion times to each notebook header (FR-010: 15-45 min per scenario)
- [X] T067 [P] Create docs/COST_GUIDANCE.md with token usage estimates and Azure pricing tier recommendations (FR-011)
- [X] T063 Run quickstart.md validation - verify all setup steps work end-to-end
- [X] T064 Code review pass - ensure all modules follow Constitution (type hints, docstrings, ≤30 line functions)
- [X] T065 Create GitHub Actions workflow in .github/workflows/ci.yml for linting, type checking, and tests

**Checkpoint**: Phase 10 complete - workshop fully polished and ready for delivery ✅

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-9)**: All depend on Foundational phase completion
  - US1 (P1) must complete first (provides base agent pattern)
  - US2-US7 can proceed after US1 (build on foundation)
  - US4 should complete before US5 (workflows needed for declarative)
- **Polish (Phase 10)**: Depends on all user stories being complete

### User Story Dependencies

```text
Foundation (Phase 2)
       │
       ▼
┌──────────────┐
│  US1 (P1)    │ ◄─── MVP: Simple Agent + MCP + Observability
│  MCP + Obs   │
└──────┬───────┘
       │
       ├─────────────────┬─────────────────┬─────────────────┐
       ▼                 ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  US2 (P2)    │  │  US3 (P3)    │  │  US4 (P4)    │  │  US7 (P7)    │
│  AG-UI       │  │  A2A         │  │  Workflows   │  │  Evaluation  │
└──────────────┘  └──────────────┘  └──────┬───────┘  └──────────────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │  US5 (P5)    │
                                   │  Declarative │
                                   └──────┬───────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │  US6 (P6)    │
                                   │  Discussions │
                                   └──────────────┘
```

### Within Each User Story

- Source modules before notebooks
- Notebooks include hands-on exercises
- Tests can run in parallel after implementation

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational [P] tasks can run in parallel (within Phase 2)
- Once US1 completes: US2, US3, US4, US7 can start in parallel
- US5 requires US4 completion (workflow foundation)
- US6 requires US5 completion (declarative agents)
- All test tasks marked [P] within a story can run in parallel

---

## Parallel Example: User Story 1 (Phase 3)

```bash
# After T014 (tools __init__), launch all tool implementations together:
Task T015: "Implement search_web tool in src/tools/search_tool.py"
Task T016: "Implement calculate tool in src/tools/calculator_tool.py"
Task T017: "Implement file operations tools in src/tools/file_tool.py"

# After T022 (notebook complete), launch tests together:
Task T023: "Create tests/contract/test_mcp_schemas.py"
Task T024: "Create tests/integration/test_scenario_01.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Simple Agent + MCP + Observability)
4. **STOP and VALIDATE**: Test scenario independently - participant can:
   - Build an agent with Azure OpenAI
   - Connect MCP tools
   - See execution traces in Azure Monitor
5. Deploy/demo if ready - **this is a complete, valuable workshop module**

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add US1 (MCP + Observability) → Test → **MVP Complete!**
3. Add US2 (AG-UI) → Test → Streaming UI capability
4. Add US3 (A2A) → Test → Agent interoperability
5. Add US4 (Workflows) → Test → Multi-agent orchestration
6. Add US5 (Declarative) → Test → No-code configuration
7. Add US6 (Discussions) → Test → Advanced multi-agent patterns
8. Add US7 (Evaluation) → Test → **Full Workshop Complete!**

Each story adds a complete scenario module without breaking previous ones.

### Parallel Team Strategy

With multiple developers after US1 completion:

| Developer | User Story | Prerequisite |
|-----------|------------|--------------|
| A | US2 (AG-UI) | US1 complete |
| B | US3 (A2A) | US1 complete |
| C | US4 (Workflows) | US1 complete |
| D | US7 (Evaluation) | US1 complete |

Then:
- Developer A → US5 (needs US4)
- Developer B → US6 (needs US5)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story (US1-US7)
- Each scenario module is independently completable and testable
- Notebooks are the primary delivery format - source modules support reuse
- Commit after each task or logical group
- Stop at any checkpoint to validate scenario independently
- Estimated times: Setup 15min, Foundational 30min, each User Story 45-60min
- Total: ~67 tasks across 10 phases
