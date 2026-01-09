# Tasks: Refactor Evaluation to Azure AI Evaluation SDK

**Branch**: `003-evaluation-refactoring` | **Date**: 2026-01-07 | **Plan**: [plan.md](./plan.md)

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1 only - single user story feature)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Environment preparation and SDK verification

- [X] T001 Verify `azure-ai-evaluation>=1.13.7` installation with `pip show azure-ai-evaluation`
- [X] T002 [P] Update `.env.example` with required environment variables:
  - `AZURE_ENDPOINT` - Azure OpenAI endpoint
  - `AZURE_API_KEY` - Azure OpenAI API key  
  - `AZURE_DEPLOYMENT_NAME` - Model deployment (gpt-4o recommended)
  - `AZURE_API_VERSION` - API version (2024-02-15-preview)
- [X] T003 [P] Verify SDK imports work in Python 3.11+ environment

**Checkpoint**: Setup complete - SDK verified and environment configured ✅

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before notebook refactoring

**⚠️ CRITICAL**: Notebook cannot be refactored until evaluation module is updated

- [X] T004 Create `src/common/evaluation_config.py` with:
  - `get_model_config()` function returning Azure OpenAI configuration dict
  - `get_azure_ai_project()` function returning optional Foundry project config
  - Environment variable validation with clear error messages
- [X] T005 [P] Backup `src/common/evaluation.py` to `src/common/evaluation_legacy.py`

**Checkpoint**: Foundation ready - SDK configuration available ✅

---

## Phase 3: User Story 1 - SDK-Based Agent Evaluation (Priority: P1) 🎯 MVP

**Goal**: Participants learn agent evaluation using the official `azure-ai-evaluation` SDK with built-in quality and agent-specific evaluators

**Independent Test**: Run notebook end-to-end, verify all SDK evaluators produce valid scores (1-5 scale), batch evaluation returns aggregated results

### Implementation for User Story 1

#### 3.1 Refactor Evaluation Module

- [X] T006 [US1] Refactor `src/common/evaluation.py` - Add SDK quality evaluator wrappers:
  - `create_relevance_evaluator(model_config)` wrapping `RelevanceEvaluator`
  - `create_coherence_evaluator(model_config)` wrapping `CoherenceEvaluator`
  - `create_fluency_evaluator(model_config)` wrapping `FluencyEvaluator`
  - `create_groundedness_evaluator(model_config)` wrapping `GroundednessEvaluator`
- [X] T007 [US1] Refactor `src/common/evaluation.py` - Add SDK agent evaluator wrappers:
  - `create_intent_resolution_evaluator(model_config)` wrapping `IntentResolutionEvaluator`
  - `create_task_adherence_evaluator(model_config)` wrapping `TaskAdherenceEvaluator`
  - `create_tool_call_accuracy_evaluator(model_config)` wrapping `ToolCallAccuracyEvaluator`
- [X] T008 [US1] Refactor `src/common/evaluation.py` - Add batch evaluation function:
  - `batch_evaluate(data, evaluators, column_mapping, azure_ai_project=None)` wrapping SDK `evaluate()`
- [X] T009 [US1] Retain useful custom components in `src/common/evaluation.py`:
  - Keep `MetricType` enum for categorization
  - Keep `CostMetric` and `OpenAICostCalculator` (SDK doesn't provide cost estimation)
  - Keep `MetricsCollector.record_cost()` functionality
  - Remove or deprecate `ExactMatchEvaluator`, `ContainsEvaluator`, `SemanticSimilarityEvaluator`
- [X] T009.1 [US1] Verify backward compatibility in `tests/unit/test_evaluation.py`:
  - Run existing tests against refactored `evaluation.py`
  - Document any tests that need updating due to removed classes
  - Ensure `CostMetric` and `MetricsCollector.record_cost()` tests still pass

**Checkpoint**: Evaluation Module refactored - All 49 existing tests pass ✅

#### 3.2 Rewrite Notebook

- [X] T010 [US1] Backup `notebooks/07_evaluation_evolution.ipynb` to `notebooks/07_evaluation_evolution_legacy.ipynb`
- [X] T011 [US1] Rewrite notebook Part 1 - Introduction & SDK Setup in `notebooks/07_evaluation_evolution.ipynb`:
  - Update title to "Scenario 07: AI Agent Evaluation with Azure AI Evaluation SDK"
  - Update learning objectives for SDK features
  - Add SDK imports cell with `from azure.ai.evaluation import ...`
  - Add model configuration cell using `src/common/evaluation_config.py`
- [X] T012 [US1] Rewrite notebook Part 2 - Quality Evaluators in `notebooks/07_evaluation_evolution.ipynb`:
  - `RelevanceEvaluator` demonstration with query/response example
  - `CoherenceEvaluator` demonstration  
  - `FluencyEvaluator` demonstration
  - Score interpretation guide (1-5 Likert scale)
- [X] T013 [US1] Rewrite notebook Part 3 - Agent-Specific Evaluators in `notebooks/07_evaluation_evolution.ipynb`:
  - `IntentResolutionEvaluator` with agent response example
  - `TaskAdherenceEvaluator` with task completion example
  - `ToolCallAccuracyEvaluator` with tool call example
  - Pass/fail threshold explanation
- [X] T014 [US1] Rewrite notebook Part 4 - Batch Evaluation in `notebooks/07_evaluation_evolution.ipynb`:
  - Create sample evaluation dataset (inline or JSONL)
  - Use `evaluate()` API with multiple evaluators
  - Column mapping configuration example
  - Display aggregated results
- [X] T015 [P] [US1] Rewrite notebook Part 5 - Custom Evaluators in `notebooks/07_evaluation_evolution.ipynb`:
  - Create simple Python function evaluator (e.g., `answer_length_evaluator`)
  - Combine custom with SDK evaluators in batch
  - Show extensibility pattern
- [X] T016 [P] [US1] Rewrite notebook Part 6 - Azure AI Foundry Integration (Optional) in `notebooks/07_evaluation_evolution.ipynb`:
  - Configure `azure_ai_project` parameter
  - Show result logging to Foundry (mark as optional)
  - Portal viewing explanation
- [X] T017 [US1] Retain notebook Part 7 - Prompt Tuning in `notebooks/07_evaluation_evolution.ipynb`:
  - Keep existing `PromptAnalyzer` content
  - Keep version tracking content
  - Keep A/B testing setup content
  - Add note that this complements SDK
- [X] T018 [US1] Rewrite notebook Part 8 - Exercise in `notebooks/07_evaluation_evolution.ipynb`:
  - Hands-on exercise using SDK evaluators
  - Task: Evaluate agent response with 3+ evaluators
  - Task: Interpret scores and suggest improvements
- [X] T019 [US1] Rewrite notebook Part 9 - Summary in `notebooks/07_evaluation_evolution.ipynb`:
  - SDK components table (evaluator, purpose, output)
  - Key takeaways
  - References to Microsoft Learn docs

#### 3.3 Update Tests

- [X] T020 [P] [US1] Update `tests/unit/test_evaluation.py`:
  - Test `get_model_config()` from `evaluation_config.py`
  - Test SDK evaluator wrapper factory functions
  - Test `batch_evaluate()` function with mocked SDK calls
  - Test retained `CostMetric` functionality
- [X] T021 [US1] Verify `tests/integration/test_scenario_07.py` passes with new notebook

**Checkpoint**: User Story 1 complete - Participants can evaluate agents using official Azure AI Evaluation SDK ✅

---

## Phase 4: Polish & Cross-Cutting Concerns

**Purpose**: Final verification and documentation

- [X] T022 [P] Update `docs/ARCHITECTURE.md` with evaluation module changes
- [X] T023 [P] Update `specs/001-agentic-patterns-workshop/quickstart.md` with new environment variables
- [X] T024 Run full test suite and verify NFR-001: `pytest tests/ -v --tb=short`
  - Confirm all existing tests pass OR document intentional test changes
  - Verify coverage ≥80% on changed files: `pytest --cov=src/common/evaluation --cov-report=term-missing`
- [X] T025 Run type check: `mypy src/common/evaluation.py src/common/evaluation_config.py`
- [X] T026 Run lint: `ruff check src/common/`
- [X] T027 Execute notebook end-to-end and verify all cells pass
- [X] T027.1 Verify NFR-002 execution time: Time notebook execution and confirm <5 minutes
  - Use `%%time` magic in final cell OR external timing
  - Document baseline if exceeds limit with justification

**Checkpoint**: Phase 4 complete - All tests pass, documentation updated

---

## Phase 5: Cleanup

**Purpose**: Remove temporary backup files after verification

- [X] T028 Delete `src/common/evaluation_legacy.py` after verification
- [X] T029 Delete `notebooks/07_evaluation_evolution_legacy.ipynb` after verification

**Checkpoint**: Feature ready for merge ✅

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS notebook refactoring
- **User Story 1 (Phase 3)**: Depends on Foundational completion
- **Polish (Phase 4)**: Depends on User Story 1 completion
- **Cleanup (Phase 5)**: Depends on Polish completion and manual verification

### Within User Story 1

```
T006 → T007 → T008 → T009 (evaluation.py refactoring - sequential)
              ↓
T010 (backup notebook)
  ↓
T011 → T012 → T013 → T014 (notebook core content - sequential)
                      ↓
              T015, T016 (parallel - independent sections)
                      ↓
T017 → T018 → T019 (notebook finalization - sequential)
              ↓
        T020, T021 (tests - parallel)
```

### Parallel Opportunities

Within Phase 3 (User Story 1):
- T015 [Custom Evaluators] and T016 [Foundry Integration] can run in parallel
- T020 [Unit Tests] and T021 [Integration Test] can run in parallel

Within Phase 4:
- T022 [Architecture docs] and T023 [Quickstart docs] can run in parallel

---

## Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| Phase 1: Setup | T001-T003 | SDK verification, environment |
| Phase 2: Foundational | T004-T005 | Config module, backup |
| Phase 3: User Story 1 | T006-T021 | Core refactoring (16 tasks) |
| Phase 4: Polish | T022-T027 | Docs, tests, verification |
| Phase 5: Cleanup | T028-T029 | Remove backups |
| **Total** | **29 tasks** | |
