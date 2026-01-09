# Tasks: Refactor 04_deterministic_workflows.ipynb to Microsoft Agent Framework

**Input**: [refactor-04-workflows.md](./refactor-04-workflows.md)
**Prerequisites**: plan.md, research.md (Microsoft Agent Framework documentation)

**Tests**: NOT requested - focus is on minimal refactoring, tests updated only for compatibility

**Organization**: Tasks grouped by workflow concept to enable incremental migration

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which refactoring phase this task belongs to (SETUP, SEQ, PAR, COND, ERR, BUILD, TEST, DEPR)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Framework Verification)

**Purpose**: Verify Microsoft Agent Framework is installed and WorkflowBuilder API is available

- [X] T001 [SETUP] Verify `agent-framework` package installed in requirements.txt
- [X] T002 [P] [SETUP] Create test script to validate WorkflowBuilder imports in notebooks/test_imports.py
- [X] T003 [SETUP] Run import test to confirm API availability

**Checkpoint**: ✅ Framework imports work - refactoring can begin

---

## Phase 2: Foundational (Notebook Structure Update)

**Purpose**: Update notebook imports and setup cells before refactoring content

**⚠️ CRITICAL**: Imports must work before any content refactoring

- [X] T004 [FOUND] Backup original notebook to notebooks/04_deterministic_workflows_backup.ipynb
- [X] T005 [FOUND] Update Part 1 markdown cell with new learning objectives in notebooks/04_deterministic_workflows.ipynb
- [X] T006 [FOUND] Replace Part 2 imports: remove custom engine imports, add framework imports in notebooks/04_deterministic_workflows.ipynb
- [X] T007 [FOUND] Add Azure OpenAI client setup cell with DefaultAzureCredential in notebooks/04_deterministic_workflows.ipynb

**Checkpoint**: ✅ Notebook loads without import errors - content refactoring can begin

---

## Phase 3: User Story 1 - Sequential Orchestration (Priority: P1) 🎯 MVP

**Goal**: Replace custom `WorkflowEngine.add_step()` with `WorkflowBuilder.add_edge()` pattern

**Independent Test**: Run Part 3-4 cells, verify sequential agent execution produces output

### Implementation for US1

- [X] T008 [SEQ] Refactor Part 3: Replace custom `EchoStep` class with `ChatAgent` instance in notebooks/04_deterministic_workflows.ipynb
- [X] T009 [SEQ] Refactor Part 3: Create workflow using `WorkflowBuilder().set_start_executor().add_edge().build()` in notebooks/04_deterministic_workflows.ipynb
- [X] T010 [SEQ] Refactor Part 4: Replace `SequentialStep` demo with chained `add_edge()` calls in notebooks/04_deterministic_workflows.ipynb
- [X] T011 [SEQ] Update Part 4 markdown to explain `add_edge()` vs custom `SequentialStep` in notebooks/04_deterministic_workflows.ipynb
- [X] T012 [SEQ] Add streaming output demo using `run_stream()` and `WorkflowEvent` in notebooks/04_deterministic_workflows.ipynb

**Checkpoint**: ✅ Sequential workflows work using framework API

---

## Phase 4: User Story 2 - Parallel Execution (Priority: P2)

**Goal**: Replace custom `ParallelStep` with `asyncio.gather()` pattern for concurrent agents

**Independent Test**: Run Part 5 cells, verify all three parallel agents execute concurrently

### Implementation for US2

- [X] T013 [PAR] Refactor Part 5: Create three parallel `ChatAgent` instances (sentiment, entities, topics) in notebooks/04_deterministic_workflows.ipynb
- [X] T014 [PAR] Refactor Part 5: Implement `parallel_analysis()` function using `asyncio.gather()` in notebooks/04_deterministic_workflows.ipynb
- [X] T015 [PAR] Update Part 5 markdown to explain `asyncio.gather()` vs custom `ParallelStep` in notebooks/04_deterministic_workflows.ipynb
- [X] T016 [PAR] Add timing comparison showing concurrent vs sequential execution in notebooks/04_deterministic_workflows.ipynb

**Checkpoint**: ✅ Parallel execution works using asyncio pattern

---

## Phase 5: User Story 3 - Conditional Branching (Priority: P3)

**Goal**: Replace custom `ConditionalStep` with `add_edge(condition=fn)` pattern

**Independent Test**: Run Part 6 cells, verify routing works based on input length

### Implementation for US3

- [X] T017 [COND] Refactor Part 6: Create routing function `route_by_length()` in notebooks/04_deterministic_workflows.ipynb
- [X] T018 [COND] Refactor Part 6: Create conditional workflow with multiple `add_edge(condition=...)` calls in notebooks/04_deterministic_workflows.ipynb
- [X] T019 [COND] Update Part 6 markdown to explain condition functions vs custom `ConditionalStep` in notebooks/04_deterministic_workflows.ipynb
- [X] T020 [COND] Add example showing multiple conditional paths in single workflow in notebooks/04_deterministic_workflows.ipynb

**Checkpoint**: ✅ Conditional routing works using framework API

---

## Phase 6: User Story 4 - Data Transforms (Priority: P4)

**Goal**: Replace custom `DataTransform` with native prompt templating

**Independent Test**: Run Part 7 cells, verify data flows between agents correctly

### Implementation for US4

- [X] T021 [DATA] Refactor Part 7: Replace `DataTransform` with prompt template injection in notebooks/04_deterministic_workflows.ipynb
- [X] T022 [DATA] Show context dict passing between agents via workflow state in notebooks/04_deterministic_workflows.ipynb
- [X] T023 [DATA] Update Part 7 markdown to explain native templating vs custom transform in notebooks/04_deterministic_workflows.ipynb

**Checkpoint**: ✅ Data transformation works using prompt templating

---

## Phase 7: User Story 5 - Error Handling (Priority: P5)

**Goal**: Replace `ErrorStrategy` enum with middleware pattern for retry/fallback

**Independent Test**: Run Part 8-9 cells, verify retry middleware handles transient failures

### Implementation for US5

- [X] T024 [ERR] Refactor Part 8: Create `RetryMiddleware` class with configurable retries in notebooks/04_deterministic_workflows.ipynb
- [X] T025 [ERR] Refactor Part 8: Show agent with middleware: `ChatAgent(..., middleware=[RetryMiddleware()])` in notebooks/04_deterministic_workflows.ipynb
- [X] T026 [ERR] Refactor Part 9: Implement `safe_workflow()` with try/except fallback pattern in notebooks/04_deterministic_workflows.ipynb
- [X] T027 [ERR] Update Part 8-9 markdown to explain middleware vs `ErrorStrategy` enum in notebooks/04_deterministic_workflows.ipynb
- [X] T028 [ERR] Remove `ErrorConfig`, `ErrorStrategy` references from notebook in notebooks/04_deterministic_workflows.ipynb

**Checkpoint**: ✅ Error handling works using middleware pattern

---

## Phase 8: User Story 6 - Builder Pattern (Priority: P6)

**Goal**: Replace custom `WorkflowBuilder` with framework's `WorkflowBuilder`

**Independent Test**: Run Part 10-11 cells, verify complete research pipeline executes

### Implementation for US6

- [X] T029 [BUILD] Refactor Part 10: Replace custom builder with framework `WorkflowBuilder` in notebooks/04_deterministic_workflows.ipynb
- [X] T030 [BUILD] Create multi-agent research pipeline: researcher → analyzer → summarizer in notebooks/04_deterministic_workflows.ipynb
- [X] T031 [BUILD] Show `AgentRunEvent` handling for collecting results in notebooks/04_deterministic_workflows.ipynb
- [X] T032 [BUILD] Update Part 10 markdown to explain framework builder vs custom builder in notebooks/04_deterministic_workflows.ipynb
- [X] T033 [BUILD] Refactor Part 11 hands-on exercise to use framework patterns in notebooks/04_deterministic_workflows.ipynb

**Checkpoint**: ✅ Complete workflows work using framework builder

---

## Phase 9: Test Updates (Compatibility)

**Purpose**: Update tests to work with refactored notebook

- [X] T034 [P] [TEST] Update tests/integration/test_scenario_04.py for new notebook cells
- [X] T035 [P] [TEST] Update tests/unit/test_workflows.py to test any remaining custom code
- [X] T036 [TEST] Run all tests to verify no regressions

**Checkpoint**: ✅ All 63 tests pass with refactored code

---

## Phase 10: Deprecation & Cleanup

**Purpose**: Mark old custom code as deprecated, clean up unused imports

- [X] T037 [P] [DEPR] Add deprecation warning to src/workflows/engine.py docstring
- [X] T038 [P] [DEPR] Add deprecation warning to src/workflows/steps.py docstring
- [X] T039 [DEPR] Remove test_imports.py temporary file from notebooks/
- [X] T040 [DEPR] Update src/workflows/__init__.py exports with deprecation notes
- [X] T041 [DEPR] Delete notebooks/04_deterministic_workflows_backup.ipynb after validation

**Checkpoint**: Codebase is clean, deprecated code is marked

---

## Phase 11: Polish & Documentation

**Purpose**: Final documentation and code quality

- [X] T042 [P] Update Part 12 summary with new learning outcomes in notebooks/04_deterministic_workflows.ipynb
- [X] T043 [P] Add "Before vs After" comparison cell showing code reduction in notebooks/04_deterministic_workflows.ipynb
- [X] T044 Run notebook end-to-end to verify all cells execute
  - ✅ All 12 code cells execute successfully (4-29, skipping markdown)
  - Auth resolved: Uses `ad_token_provider` with `get_bearer_token_provider()` and correct `.openai.azure.com` endpoint
  - Fixed `AgentRunResponse` handling with helper functions in cells 21 and 29
- [X] T045 Update refactor-04-workflows.md with completion status

---

## Dependencies & Execution Order

### Phase Dependencies

```text
Setup (P1) → Foundational (P2) → Sequential (P3) → Parallel (P4) → Conditional (P5)
                                                                           ↓
                                                                   Data Transforms (P6)
                                                                           ↓
                                                                   Error Handling (P7)
                                                                           ↓
                                                                   Builder Pattern (P8)
                                                                           ↓
                                                               Test Updates (P9) → Deprecation (P10) → Polish (P11)
```

### User Story Dependencies

| Story | Can Start After | Dependencies |
|-------|-----------------|--------------|
| US1 (Sequential) | Foundational | Framework imports working |
| US2 (Parallel) | US1 | ChatAgent pattern established |
| US3 (Conditional) | US1 | WorkflowBuilder pattern established |
| US4 (Data) | US1 | Agent-to-agent communication pattern |
| US5 (Error) | US1 | Basic workflow execution working |
| US6 (Builder) | US1-US5 | All patterns available for complete example |

### Parallel Opportunities

**Within Phase 1 (Setup)**:
- T002 can run in parallel with T001

**Within Phase 9 (Tests)**:
- T034 and T035 can run in parallel

**Within Phase 10 (Deprecation)**:
- T037 and T038 can run in parallel

**Within Phase 11 (Polish)**:
- T042 and T043 can run in parallel

---

## Implementation Strategy

### MVP First (US1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: US1 (Sequential)
4. **STOP and VALIDATE**: Test sequential workflows work
5. Continue with remaining phases

### Incremental Delivery

Each phase produces a working notebook:

| After Phase | Notebook State |
|-------------|----------------|
| P3 (US1) | Sequential workflows work with framework |
| P4 (US2) | + Parallel execution works |
| P5 (US3) | + Conditional routing works |
| P6 (US4) | + Data transforms work |
| P7 (US5) | + Error handling works |
| P8 (US6) | + Complete builder pattern works |
| P9-P11 | Tests pass, deprecated code marked, documentation complete |

---

## Task Summary

| Phase | Tasks | Parallel Tasks | Story Label |
|-------|-------|----------------|-------------|
| Setup | 3 | 1 | SETUP |
| Foundational | 4 | 0 | FOUND |
| US1: Sequential | 5 | 0 | SEQ |
| US2: Parallel | 4 | 0 | PAR |
| US3: Conditional | 4 | 0 | COND |
| US4: Data | 3 | 0 | DATA |
| US5: Error | 5 | 0 | ERR |
| US6: Builder | 5 | 0 | BUILD |
| Tests | 3 | 2 | TEST |
| Deprecation | 5 | 2 | DEPR |
| Polish | 4 | 2 | - |
| **Total** | **45** | **7** | - |

---

## Notes

- All notebook edits target: `notebooks/04_deterministic_workflows.ipynb`
- Backup created in T004 before any destructive changes
- Each user story checkpoint allows validation before continuing
- Custom engine code preserved (deprecated, not deleted) for reference
- Framework version: `agent-framework` (Microsoft Agent Framework)
