# Implementation Plan: Refactor Evaluation to Azure AI Evaluation SDK

**Branch**: `003-evaluation-refactoring` | **Date**: 2026-01-07 | **Spec**: [spec.md](./spec.md)

## Summary

Refactor the `07_evaluation_evolution.ipynb` notebook and `src/common/evaluation.py` to use the official `azure-ai-evaluation` SDK instead of the current custom implementation. This aligns the workshop with Microsoft best practices and provides participants with production-ready patterns.

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: `azure-ai-evaluation>=1.13.7` (already in requirements.txt)  
**Affected Files**:
- `notebooks/07_evaluation_evolution.ipynb` - Complete rewrite required
- `src/common/evaluation.py` - Major refactoring to wrap SDK
- `tests/unit/test_evaluation.py` - Update tests for new API
- `tests/integration/test_scenario_07.py` - Verify notebook execution

**SDK Components**:
| Category | SDK Classes |
|----------|-------------|
| Quality Evaluators | `RelevanceEvaluator`, `CoherenceEvaluator`, `FluencyEvaluator`, `GroundednessEvaluator` |
| Agent Evaluators | `IntentResolutionEvaluator`, `TaskAdherenceEvaluator`, `ToolCallAccuracyEvaluator` |
| Batch API | `evaluate()` function |
| Agent Data | `AIAgentConverter` for thread data |

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Type Safety | ✅ PASS | SDK classes are fully typed |
| II. Test-First | ⚠️ EXCEPTION | Workshop notebook context - see formal exception below |
| III. Clean Code | ✅ PASS | SDK provides clean abstractions |
| IV. Dependencies | ✅ PASS | `azure-ai-evaluation` already in requirements.txt |
| V. Observability | ✅ PASS | SDK integrates with Azure Monitor |

### Constitution Exception: Principle II (Test-First)

> **TODO [TECH-DEBT-001]**: This feature adapts Principle II (Test-First Development) for workshop notebook context.
>
> **Justification**: Jupyter notebooks are educational artifacts, not production code. The primary deliverable is participant learning, not code correctness. Integration tests verify notebook execution succeeds, which validates the educational content.
>
> **Mitigation**:
> - Unit tests for `src/common/evaluation.py` wrapper functions ARE written test-first (T020)
> - Integration test `test_scenario_07.py` verifies notebook runs end-to-end (T021)
> - Notebook cells include inline assertions for immediate feedback
>
> **Tracking**: This exception applies only to notebook cells. All `src/` module code follows standard TDD.

## Implementation Phases

### Phase 1: Research & Design

1. **Research SDK API surface** - Document all evaluator classes and their parameters
2. **Design wrapper layer** - Determine what custom code to keep vs. remove
3. **Create data model** - Define evaluation input/output schemas

### Phase 2: Refactor `src/common/evaluation.py`

**Goal**: Keep useful abstractions, replace custom evaluators with SDK wrappers

**Keep (with modifications)**:
- `MetricType` enum - Useful categorization
- `MetricsCollector` - Wrapper around SDK evaluators
- `EvaluationResult` - Unified result container
- Cost estimation (SDK doesn't provide this)

**Remove**:
- `ExactMatchEvaluator` - Use SDK `evaluate()` with custom function
- `ContainsEvaluator` - Use SDK string matching
- `SemanticSimilarityEvaluator` - Use SDK `RelevanceEvaluator` or `CoherenceEvaluator`

**Add**:
- SDK evaluator wrappers
- Model configuration helper
- Azure AI Foundry integration helper

### Phase 3: Rewrite `07_evaluation_evolution.ipynb`

**New Notebook Structure**:

1. **Introduction & Setup** (existing - keep)
2. **Part 1: SDK Overview**
   - Import SDK evaluators
   - Configure model endpoint
3. **Part 2: Quality Evaluators**
   - `RelevanceEvaluator` demo
   - `CoherenceEvaluator` demo
   - `FluencyEvaluator` demo
4. **Part 3: Agent-Specific Evaluators**
   - `IntentResolutionEvaluator` demo
   - `TaskAdherenceEvaluator` demo
   - `ToolCallAccuracyEvaluator` demo
5. **Part 4: Batch Evaluation**
   - Using `evaluate()` API with dataset
   - Column mapping configuration
6. **Part 5: Custom Evaluators**
   - Creating simple custom evaluator
   - Combining with SDK evaluators
7. **Part 6: Azure AI Foundry Integration** (optional)
   - Logging results to Foundry project
   - Viewing results in portal
8. **Part 7: Prompt Tuning** (keep existing content)
   - Version tracking
   - A/B testing setup
9. **Exercise & Summary**

### Phase 4: Update Tests

- Update `tests/unit/test_evaluation.py` for new SDK-based API
- Verify `tests/integration/test_scenario_07.py` passes

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/common/evaluation.py` | MAJOR REFACTOR | Replace custom evaluators with SDK wrappers |
| `notebooks/07_evaluation_evolution.ipynb` | REWRITE | New SDK-focused content |
| `tests/unit/test_evaluation.py` | UPDATE | Test new API |
| `tests/integration/test_scenario_07.py` | VERIFY | Ensure notebook runs |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SDK requires Azure AI Project | Medium | High | Provide fallback for local-only evaluation |
| Model config complexity | Low | Medium | Provide clear environment variable docs |
| Breaking existing tests | Medium | Medium | Run tests after each refactoring step |

## Dependencies

- Azure OpenAI deployment with GPT-4 or GPT-4o for AI-assisted evaluators
- Environment variables: `AZURE_ENDPOINT`, `AZURE_API_KEY`, `AZURE_DEPLOYMENT_NAME`, `AZURE_API_VERSION`

## Success Criteria

1. All notebook cells execute without errors
2. SDK evaluators produce meaningful scores
3. Existing unit tests pass (or are updated appropriately)
4. Integration test confirms notebook runs end-to-end

## Next Steps

After plan approval:
1. Create `tasks.md` with detailed implementation tasks
2. Create feature branch `003-evaluation-refactoring`
3. Begin Phase 2 implementation
