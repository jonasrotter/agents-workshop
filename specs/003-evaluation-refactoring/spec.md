# Feature Specification: Refactor Evaluation to Azure AI Evaluation SDK

**Date**: 2026-01-07 | **Priority**: P1 | **Status**: Draft

## Problem Statement

The current `07_evaluation_evolution.ipynb` notebook and `src/common/evaluation.py` use a completely custom implementation for agent evaluation metrics, despite `azure-ai-evaluation==1.13.7` being listed in `requirements.txt`. This creates:

1. **Inconsistency**: Workshop teaches custom patterns instead of official Microsoft SDK
2. **Maintenance burden**: Custom evaluators require ongoing maintenance vs. SDK updates
3. **Missing features**: SDK offers agent-specific evaluators, safety checks, and Azure AI Foundry integration not available in custom implementation
4. **Workshop credibility**: A Microsoft workshop should showcase Microsoft tools

## User Story

As a workshop participant, I want to learn agent evaluation using the official `azure-ai-evaluation` SDK, so that I can apply industry-standard evaluation patterns to my production AI agents with Azure AI Foundry integration.

### Acceptance Criteria

1. **Given** a participant runs the evaluation notebook, **When** they execute evaluators, **Then** they use `azure.ai.evaluation` SDK classes (not custom implementations)
2. **Given** an agent produces output, **When** the participant evaluates it, **Then** they can use built-in evaluators like `RelevanceEvaluator`, `CoherenceEvaluator`, `FluencyEvaluator`
3. **Given** an agentic workflow, **When** the participant evaluates tool calls, **Then** they can use agent-specific evaluators like `IntentResolutionEvaluator`, `TaskAdherenceEvaluator`, `ToolCallAccuracyEvaluator`
4. **Given** evaluation results, **When** the participant wants to track them, **Then** the notebook demonstrates the Foundry logging pattern (execution is optional, requires Azure AI project)
5. **Given** the participant completes the notebook, **When** they review learnings, **Then** they understand both local evaluation and Azure-integrated continuous evaluation patterns

## Functional Requirements

- **FR-001**: Notebook MUST demonstrate `azure.ai.evaluation` SDK built-in evaluators
- **FR-002**: Notebook MUST show agent-specific evaluators for tool call accuracy and intent resolution
- **FR-003**: Notebook MUST demonstrate batch evaluation using `evaluate()` API
- **FR-004**: Notebook MUST document Azure AI Foundry integration pattern; actual logging is OPTIONAL (requires Azure AI project)
- **FR-005**: `src/common/evaluation.py` MUST be refactored to wrap SDK evaluators (not replace them)
- **FR-006**: Custom prompt tuning features MAY be retained as they complement (not duplicate) SDK

## Non-Functional Requirements

- **NFR-001**: All existing unit tests MUST pass after refactoring
- **NFR-002**: Notebook execution time SHOULD remain under 5 minutes

## Out of Scope

- Continuous evaluation with Azure AI Projects client library (advanced topic)
- AI red teaming agent integration
- Custom evaluator registration to Azure AI Hub

## Technical Notes

### SDK Components to Use

```python
# Quality Evaluators
from azure.ai.evaluation import (
    RelevanceEvaluator,
    CoherenceEvaluator,
    FluencyEvaluator,
    GroundednessEvaluator,
)

# Agent-Specific Evaluators
from azure.ai.evaluation import (
    IntentResolutionEvaluator,
    TaskAdherenceEvaluator,
    ToolCallAccuracyEvaluator,
)

# Batch Evaluation
from azure.ai.evaluation import evaluate
```

### Model Configuration Required

```python
model_config = {
    "azure_deployment": os.getenv("AZURE_DEPLOYMENT_NAME"),
    "api_key": os.getenv("AZURE_API_KEY"),
    "azure_endpoint": os.getenv("AZURE_ENDPOINT"),
    "api_version": os.getenv("AZURE_API_VERSION"),
}
```

## References

- [Azure AI Evaluation SDK Docs](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-evaluation-readme)
- [Evaluate AI Agents Locally](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/agent-evaluate-sdk)
- [Built-in Evaluators](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/evaluate-sdk)
