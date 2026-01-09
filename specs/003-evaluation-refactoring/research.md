# Research: Azure AI Evaluation SDK for Agent Evaluation

**Date**: 2026-01-07 | **Plan**: [plan.md](./plan.md) | **Spec**: [spec.md](./spec.md)

This document consolidates research findings for refactoring to the `azure-ai-evaluation` SDK.

---

## 1. SDK Overview

### Decision
Use `azure-ai-evaluation` SDK (v1.13.7+) for agent evaluation instead of custom implementation.

### Rationale
- **Official Microsoft SDK**: First-party support with Azure AI Foundry integration
- **Comprehensive evaluators**: Built-in quality, safety, and agent-specific evaluators
- **Production-ready**: Battle-tested in Azure AI Foundry production
- **Active development**: Regular updates and new features
- **Workshop alignment**: Microsoft workshop should showcase Microsoft tools

### Package Installation
```bash
pip install azure-ai-evaluation
```

### References
- [PyPI Package](https://pypi.org/project/azure-ai-evaluation/)
- [API Reference](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-evaluation-readme)
- [Agent Evaluation Guide](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/agent-evaluate-sdk)

---

## 2. SDK Evaluator Categories

### 2.1 Quality Evaluators (AI-Assisted)

These evaluators use an LLM to assess response quality. Require `model_config`.

| Evaluator | Purpose | Output | Scale |
|-----------|---------|--------|-------|
| `RelevanceEvaluator` | Measures response relevance to query | `relevance` score | 1-5 |
| `CoherenceEvaluator` | Measures logical flow and structure | `coherence` score | 1-5 |
| `FluencyEvaluator` | Measures linguistic quality | `fluency` score | 1-5 |
| `GroundednessEvaluator` | Measures factual grounding in context | `groundedness` score | 1-5 |

**Usage Pattern:**
```python
from azure.ai.evaluation import RelevanceEvaluator

model_config = {
    "azure_deployment": os.getenv("AZURE_DEPLOYMENT_NAME"),
    "api_key": os.getenv("AZURE_API_KEY"),
    "azure_endpoint": os.getenv("AZURE_ENDPOINT"),
    "api_version": os.getenv("AZURE_API_VERSION"),
}

relevance_eval = RelevanceEvaluator(model_config)
result = relevance_eval(
    query="What is the capital of France?",
    response="The capital of France is Paris."
)
# result: {"relevance": 5.0}
```

### 2.2 Agent-Specific Evaluators (AI-Assisted)

These evaluators are designed for agentic workflows. Require `model_config`.

| Evaluator | Purpose | Output | Scale |
|-----------|---------|--------|-------|
| `IntentResolutionEvaluator` | Did agent resolve user intent? | `intent_resolution` score + `_result` + `_reason` | 1-5 |
| `TaskAdherenceEvaluator` | Did agent complete assigned task? | `task_adherence` score + `_result` + `_reason` | 1-5 |
| `ToolCallAccuracyEvaluator` | Were tool calls appropriate? | `tool_call_accuracy` rate + `_result` + details | 0-1 |

**Output Format:**
```python
# IntentResolutionEvaluator output
{
    "intent_resolution": 5.0,
    "intent_resolution_result": "pass",  # based on threshold (default: 3)
    "intent_resolution_threshold": 3,
    "intent_resolution_reason": "The response fully addresses..."
}
```

**Usage with Reasoning Models:**
```python
from azure.ai.evaluation import IntentResolutionEvaluator

# For reasoning models (o3-mini, etc.)
reasoning_config = {
    "azure_deployment": "o3-mini",
    "api_key": os.getenv("AZURE_API_KEY"),
    "azure_endpoint": os.getenv("AZURE_ENDPOINT"),
    "api_version": os.getenv("AZURE_API_VERSION"),
}

intent_eval = IntentResolutionEvaluator(
    model_config=reasoning_config,
    is_reasoning_model=True
)
```

### 2.3 Safety Evaluators

Require `azure_ai_project` configuration (Azure AI Foundry project).

| Evaluator | Purpose | Requires |
|-----------|---------|----------|
| `ViolenceEvaluator` | Detect violent content | `azure_ai_project` |
| `ContentSafetyEvaluator` | General content safety | `azure_ai_project` |
| `IndirectAttackEvaluator` | Detect prompt injection | `azure_ai_project` |
| `CodeVulnerabilityEvaluator` | Detect code security issues | `azure_ai_project` |

**Azure AI Project Configuration:**
```python
# Option 1: Using project details
azure_ai_project = {
    "subscription_id": "<subscription_id>",
    "resource_group_name": "<resource_group_name>",
    "project_name": "<project_name>",
}

# Option 2: Using project URL
azure_ai_project = "https://{resource_name}.services.ai.azure.com/api/projects/{project_name}"

safety_eval = ContentSafetyEvaluator(azure_ai_project)
```

### 2.4 NLP Evaluators (Non-AI)

These don't require LLM calls - pure algorithmic evaluation.

| Evaluator | Purpose | Output |
|-----------|---------|--------|
| `BleuScoreEvaluator` | BLEU score vs reference | `bleu_score` |
| `RougeScoreEvaluator` | ROUGE score vs reference | `rouge_*` scores |
| `GleuScoreEvaluator` | GLEU score vs reference | `gleu_score` |

**Usage:**
```python
from azure.ai.evaluation import BleuScoreEvaluator

bleu_eval = BleuScoreEvaluator()
result = bleu_eval(
    response="Tokyo is the capital of Japan.",
    ground_truth="The capital of Japan is Tokyo."
)
```

---

## 3. Batch Evaluation API

### `evaluate()` Function

Run multiple evaluators on a dataset in one call.

```python
from azure.ai.evaluation import evaluate

result = evaluate(
    data="data.jsonl",  # Path to evaluation dataset
    evaluators={
        "relevance": relevance_eval,
        "coherence": coherence_eval,
        "fluency": fluency_eval,
    },
    evaluator_config={
        "relevance": {
            "column_mapping": {
                "query": "${data.question}",
                "response": "${data.answer}",
            }
        }
    },
    # Optional: Log to Azure AI Foundry
    azure_ai_project=azure_ai_project,
    # Optional: Save results locally
    output_path="./evaluation_results.json"
)
```

### Data Format

**JSONL Format:**
```json
{"question": "What is Python?", "answer": "Python is a programming language.", "context": "..."}
{"question": "What is AI?", "answer": "AI is artificial intelligence.", "context": "..."}
```

### Column Mapping

Maps dataset columns to evaluator parameters:
- `${data.column_name}` - Reference data column
- `${outputs.column_name}` - Reference target function output (if using `target`)

---

## 4. Agent Data Conversion

### `AIAgentConverter`

Convert Azure AI Agent Service threads to evaluation format.

```python
from azure.ai.evaluation import AIAgentConverter

# Initialize with project client
converter = AIAgentConverter(project_client)

# Convert thread to evaluation data
evaluation_data = converter.prepare_evaluation_data(
    thread_ids=thread_id,
    filename="evaluation_input_data.jsonl"
)
```

### Agent Message Schema

For non-Foundry agents, use this format:

**Simple Format:**
```python
query = "What are the opening hours of the Eiffel Tower?"
response = "The Eiffel Tower is open from 9:30 AM to 11:45 PM."

result = intent_eval(query=query, response=response)
```

**OpenAI Message Format:**
```python
query = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the weather?"}
]
response = [
    {"role": "assistant", "content": "The weather is sunny."}
]

result = intent_eval(query=query, response=response)
```

---

## 5. Custom Evaluators

### Python Function Evaluator

```python
def answer_length_evaluator(response: str) -> dict:
    """Custom evaluator measuring response length."""
    return {"answer_length": len(response.split())}

# Use in batch evaluation
result = evaluate(
    data="data.jsonl",
    evaluators={
        "length": answer_length_evaluator,
        "relevance": relevance_eval,
    }
)
```

### Class-Based Evaluator

```python
class AnswerLengthEvaluator:
    def __init__(self, min_words: int = 10):
        self.min_words = min_words
    
    def __call__(self, response: str) -> dict:
        word_count = len(response.split())
        return {
            "answer_length": word_count,
            "answer_length_result": "pass" if word_count >= self.min_words else "fail"
        }
```

---

## 6. Mapping: Custom → SDK

| Custom Implementation | SDK Replacement | Notes |
|----------------------|-----------------|-------|
| `ExactMatchEvaluator` | Custom function | Simple string comparison |
| `ContainsEvaluator` | Custom function | Check substring |
| `SemanticSimilarityEvaluator` | `RelevanceEvaluator` | AI-assisted semantic matching |
| `MetricsCollector` | Keep + wrap SDK | Aggregation still useful |
| `CostMetric` | Keep | SDK doesn't provide cost estimation |
| `PromptTuner` | Keep | Complementary feature |
| `PromptAnalyzer` | Keep | Complementary feature |

---

## 7. Environment Requirements

### Required Environment Variables

```bash
# Azure OpenAI (required for AI-assisted evaluators)
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_API_KEY=your-api-key
AZURE_DEPLOYMENT_NAME=gpt-4o  # or gpt-4, gpt-4-turbo
AZURE_API_VERSION=2024-02-15-preview

# Azure AI Project (optional, for safety evaluators and Foundry integration)
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group
AZURE_PROJECT_NAME=your-project-name
```

### Model Recommendations

| Evaluator Type | Recommended Model |
|----------------|-------------------|
| Quality (standard) | gpt-4o, gpt-4 |
| Agent-specific | gpt-4o (or o3-mini with `is_reasoning_model=True`) |
| Safety | Handled by Azure AI Service |

---

## 8. Key Differences from Custom Implementation

| Aspect | Custom | SDK |
|--------|--------|-----|
| Evaluation method | Algorithmic + embeddings | LLM-as-judge |
| Score scale | 0.0-1.0 | 1-5 (Likert) |
| Reasoning | None | Chain-of-thought explanation |
| Azure integration | Manual | Native |
| Safety checks | None | Built-in |
| Agent-specific | Generic | Purpose-built |

---

## 9. Sample Notebook Structure

```markdown
1. Introduction & Learning Objectives
2. Environment Setup & SDK Import
3. Quality Evaluators
   - RelevanceEvaluator
   - CoherenceEvaluator
   - FluencyEvaluator
4. Agent-Specific Evaluators
   - IntentResolutionEvaluator
   - TaskAdherenceEvaluator
   - ToolCallAccuracyEvaluator
5. Batch Evaluation with evaluate() API
6. Custom Evaluators
7. (Optional) Azure AI Foundry Integration
8. Prompt Tuning (retained from original)
9. Exercise & Summary
```

---

## 10. Decision Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary SDK | `azure-ai-evaluation` | Official Microsoft SDK |
| Evaluator approach | SDK built-in + custom wrappers | Best of both worlds |
| Prompt tuning | Keep custom | SDK doesn't provide this |
| Cost estimation | Keep custom | SDK doesn't provide this |
| Safety evaluators | Include (optional) | Requires Azure AI Project |
| Integration test | Notebook execution | Verify end-to-end |
