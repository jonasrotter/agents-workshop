"""Evaluation metrics collector for agentic AI systems.

This module provides comprehensive metrics collection for:
- Accuracy measurement
- Latency tracking
- Cost estimation
- Quality scoring

NEW in v2.0: Integration with Azure AI Evaluation SDK
- Quality evaluators: RelevanceEvaluator, CoherenceEvaluator, FluencyEvaluator, GroundednessEvaluator
- Agent evaluators: IntentResolutionEvaluator, TaskAdherenceEvaluator, ToolCallAccuracyEvaluator
- Batch evaluation with evaluate() API
"""

from __future__ import annotations

import json
import statistics
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

# Azure AI Evaluation SDK imports
from azure.ai.evaluation import (
    CoherenceEvaluator,
    FluencyEvaluator,
    GroundednessEvaluator,
    IntentResolutionEvaluator,
    RelevanceEvaluator,
    TaskAdherenceEvaluator,
    ToolCallAccuracyEvaluator,
    evaluate,
)
from azure.identity import DefaultAzureCredential
from opentelemetry import trace

from src.common.config import AzureAIProject, ModelConfig


tracer = trace.get_tracer(__name__)


# =============================================================================
# Enums
# =============================================================================


class MetricType(str, Enum):
    """Types of metrics that can be collected."""

    LATENCY = "latency"
    ACCURACY = "accuracy"
    COST = "cost"
    QUALITY = "quality"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"


class AggregationType(str, Enum):
    """How to aggregate metric values."""

    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE_50 = "p50"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class MetricValue:
    """A single metric measurement."""

    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyMetric:
    """Latency measurement for an operation."""

    operation: str
    duration_ms: float
    start_time: datetime
    end_time: datetime
    success: bool = True
    error: str | None = None


@dataclass
class AccuracyMetric:
    """Accuracy measurement for a response."""

    expected: Any
    actual: Any
    is_correct: bool
    similarity_score: float = 0.0
    evaluation_method: str = "exact_match"


@dataclass
class CostMetric:
    """Cost estimation for an operation."""

    operation: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    model: str = "unknown"


@dataclass
class QualityScore:
    """Quality assessment for a response."""

    dimension: str
    score: float  # 0.0 to 1.0
    explanation: str = ""
    evaluator: str = "manual"


@dataclass
class EvaluationResult:
    """Complete evaluation result for an agent run."""

    run_id: str
    agent_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    latency: LatencyMetric | None = None
    accuracy: AccuracyMetric | None = None
    cost: CostMetric | None = None
    quality_scores: list[QualityScore] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def overall_quality(self) -> float:
        """Calculate overall quality score."""
        if not self.quality_scores:
            return 0.0
        return statistics.mean(s.score for s in self.quality_scores)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over multiple evaluations."""

    metric_name: str
    count: int
    sum_value: float
    min_value: float
    max_value: float
    avg_value: float
    p50_value: float
    p95_value: float
    p99_value: float
    values: list[float] = field(default_factory=list)


# =============================================================================
# Protocols
# =============================================================================


class Evaluator(Protocol):
    """Protocol for custom evaluators."""

    def evaluate(self, expected: Any, actual: Any) -> AccuracyMetric:
        """Evaluate accuracy of actual vs expected."""
        ...


class CostCalculator(Protocol):
    """Protocol for cost calculation."""

    def calculate(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ) -> float:
        """Calculate cost in USD."""
        ...


# =============================================================================
# Default Implementations (DEPRECATED - Use Azure AI Evaluation SDK)
# =============================================================================


class ExactMatchEvaluator:
    """Evaluator using exact string matching.
    
    .. deprecated::
        Use `azure.ai.evaluation` SDK evaluators instead.
        This class is retained for backward compatibility only.
    """

    def __init__(self) -> None:
        warnings.warn(
            "ExactMatchEvaluator is deprecated. Use Azure AI Evaluation SDK evaluators "
            "(RelevanceEvaluator, CoherenceEvaluator, etc.) instead. "
            "See src/common/evaluation.py for create_*_evaluator() factory functions.",
            DeprecationWarning,
            stacklevel=2,
        )

    def evaluate(self, expected: Any, actual: Any) -> AccuracyMetric:
        """Evaluate using exact match."""
        is_correct = str(expected).strip() == str(actual).strip()
        return AccuracyMetric(
            expected=expected,
            actual=actual,
            is_correct=is_correct,
            similarity_score=1.0 if is_correct else 0.0,
            evaluation_method="exact_match",
        )


class ContainsEvaluator:
    """Evaluator checking if expected is contained in actual.
    
    .. deprecated::
        Use `azure.ai.evaluation` SDK evaluators instead.
        This class is retained for backward compatibility only.
    """

    def __init__(self) -> None:
        warnings.warn(
            "ContainsEvaluator is deprecated. Use Azure AI Evaluation SDK evaluators "
            "(RelevanceEvaluator, CoherenceEvaluator, etc.) instead. "
            "See src/common/evaluation.py for create_*_evaluator() factory functions.",
            DeprecationWarning,
            stacklevel=2,
        )

    def evaluate(self, expected: Any, actual: Any) -> AccuracyMetric:
        """Evaluate using contains check."""
        expected_str = str(expected).strip().lower()
        actual_str = str(actual).strip().lower()
        is_correct = expected_str in actual_str

        # Calculate similarity based on overlap
        if is_correct:
            similarity = len(expected_str) / len(actual_str) if actual_str else 0.0
        else:
            similarity = 0.0

        return AccuracyMetric(
            expected=expected,
            actual=actual,
            is_correct=is_correct,
            similarity_score=min(similarity, 1.0),
            evaluation_method="contains",
        )


class SemanticSimilarityEvaluator:
    """Evaluator using semantic similarity (placeholder for embedding-based).
    
    .. deprecated::
        Use `azure.ai.evaluation` SDK evaluators instead.
        Consider using RelevanceEvaluator or CoherenceEvaluator for semantic evaluation.
        This class is retained for backward compatibility only.
    """

    def __init__(self, threshold: float = 0.8):
        warnings.warn(
            "SemanticSimilarityEvaluator is deprecated. Use Azure AI Evaluation SDK "
            "evaluators (RelevanceEvaluator, CoherenceEvaluator) instead. "
            "See src/common/evaluation.py for create_*_evaluator() factory functions.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.threshold = threshold

    def evaluate(self, expected: Any, actual: Any) -> AccuracyMetric:
        """Evaluate using semantic similarity."""
        # Placeholder: In production, use embeddings
        # For now, use simple word overlap
        expected_words = set(str(expected).lower().split())
        actual_words = set(str(actual).lower().split())

        if not expected_words or not actual_words:
            similarity = 0.0
        else:
            intersection = expected_words & actual_words
            union = expected_words | actual_words
            similarity = len(intersection) / len(union)  # Jaccard similarity

        is_correct = similarity >= self.threshold

        return AccuracyMetric(
            expected=expected,
            actual=actual,
            is_correct=is_correct,
            similarity_score=similarity,
            evaluation_method="semantic_similarity",
        )


class OpenAICostCalculator:
    """Cost calculator for OpenAI models."""

    # Prices per 1K tokens (as of 2024)
    PRICES: dict[str, tuple[float, float]] = {
        "gpt-4o": (0.005, 0.015),
        "gpt-4o-mini": (0.00015, 0.0006),
        "gpt-4-turbo": (0.01, 0.03),
        "gpt-4": (0.03, 0.06),
        "gpt-3.5-turbo": (0.0005, 0.0015),
        "text-embedding-3-small": (0.00002, 0.0),
        "text-embedding-3-large": (0.00013, 0.0),
    }

    def calculate(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ) -> float:
        """Calculate cost in USD for OpenAI models."""
        model_key = model.lower()

        # Find exact match first, then substring match
        # Sort keys by length (longest first) to match most specific model first
        for key in sorted(self.PRICES.keys(), key=len, reverse=True):
            if key in model_key:
                input_price, output_price = self.PRICES[key]
                input_cost = (input_tokens / 1000) * input_price
                output_cost = (output_tokens / 1000) * output_price
                return input_cost + output_cost

        # Default to gpt-4o-mini pricing
        input_cost = (input_tokens / 1000) * 0.00015
        output_cost = (output_tokens / 1000) * 0.0006
        return input_cost + output_cost


# =============================================================================
# Azure AI Evaluation SDK Wrappers
# =============================================================================


def _get_credential_if_needed(model_config: ModelConfig) -> DefaultAzureCredential | None:
    """Get Azure credential if API key is not configured.
    
    The Azure AI Evaluation SDK supports two authentication methods:
    1. API key authentication (via api_key in model_config)
    2. Azure credential authentication (via DefaultAzureCredential)
    
    If api_key is empty or not set, returns DefaultAzureCredential.
    """
    if not model_config.get("api_key"):
        return DefaultAzureCredential()
    return None


def create_relevance_evaluator(model_config: ModelConfig) -> RelevanceEvaluator:
    """Create a RelevanceEvaluator from the Azure AI Evaluation SDK.
    
    Measures how relevant the response is to the given query.
    
    Supports both API key and Azure credential authentication.
    If api_key is not set in model_config, uses DefaultAzureCredential.
    
    Args:
        model_config: Azure OpenAI model configuration.
    
    Returns:
        RelevanceEvaluator instance ready to use.
    
    Example:
        >>> config = get_model_config()
        >>> evaluator = create_relevance_evaluator(config)
        >>> result = evaluator(query="What is Python?", response="Python is a programming language.")
        >>> print(result)  # {"relevance": 5.0}
    """
    credential = _get_credential_if_needed(model_config)
    return RelevanceEvaluator(model_config, credential=credential)


def create_coherence_evaluator(model_config: ModelConfig) -> CoherenceEvaluator:
    """Create a CoherenceEvaluator from the Azure AI Evaluation SDK.
    
    Measures the logical flow and structure of the response.
    
    Supports both API key and Azure credential authentication.
    If api_key is not set in model_config, uses DefaultAzureCredential.
    
    Args:
        model_config: Azure OpenAI model configuration.
    
    Returns:
        CoherenceEvaluator instance ready to use.
    
    Example:
        >>> config = get_model_config()
        >>> evaluator = create_coherence_evaluator(config)
        >>> result = evaluator(response="First, we analyze. Then, we implement. Finally, we test.")
        >>> print(result)  # {"coherence": 5.0}
    """
    credential = _get_credential_if_needed(model_config)
    return CoherenceEvaluator(model_config, credential=credential)


def create_fluency_evaluator(model_config: ModelConfig) -> FluencyEvaluator:
    """Create a FluencyEvaluator from the Azure AI Evaluation SDK.
    
    Measures the linguistic quality of the response.
    
    Supports both API key and Azure credential authentication.
    If api_key is not set in model_config, uses DefaultAzureCredential.
    
    Args:
        model_config: Azure OpenAI model configuration.
    
    Returns:
        FluencyEvaluator instance ready to use.
    
    Example:
        >>> config = get_model_config()
        >>> evaluator = create_fluency_evaluator(config)
        >>> result = evaluator(response="The quick brown fox jumps over the lazy dog.")
        >>> print(result)  # {"fluency": 5.0}
    """
    credential = _get_credential_if_needed(model_config)
    return FluencyEvaluator(model_config, credential=credential)


def create_groundedness_evaluator(model_config: ModelConfig) -> GroundednessEvaluator:
    """Create a GroundednessEvaluator from the Azure AI Evaluation SDK.
    
    Measures how well the response is grounded in the provided context.
    
    Supports both API key and Azure credential authentication.
    If api_key is not set in model_config, uses DefaultAzureCredential.
    
    Args:
        model_config: Azure OpenAI model configuration.
    
    Returns:
        GroundednessEvaluator instance ready to use.
    
    Example:
        >>> config = get_model_config()
        >>> evaluator = create_groundedness_evaluator(config)
        >>> result = evaluator(
        ...     context="Paris is the capital of France.",
        ...     response="The capital of France is Paris."
        ... )
        >>> print(result)  # {"groundedness": 5.0}
    """
    credential = _get_credential_if_needed(model_config)
    return GroundednessEvaluator(model_config, credential=credential)


def create_intent_resolution_evaluator(
    model_config: ModelConfig,
    threshold: int = 3,
    is_reasoning_model: bool = False,
) -> IntentResolutionEvaluator:
    """Create an IntentResolutionEvaluator from the Azure AI Evaluation SDK.
    
    Evaluates whether the agent correctly resolved the user's intent.
    
    Supports both API key and Azure credential authentication.
    If api_key is not set in model_config, uses DefaultAzureCredential.
    
    Args:
        model_config: Azure OpenAI model configuration.
        threshold: Pass/fail threshold (1-5). Default is 3.
        is_reasoning_model: Set True for reasoning models like o3-mini.
    
    Returns:
        IntentResolutionEvaluator instance ready to use.
    
    Example:
        >>> config = get_model_config()
        >>> evaluator = create_intent_resolution_evaluator(config)
        >>> result = evaluator(
        ...     query="Book a flight to Paris",
        ...     response="I've booked a flight to Paris for you departing tomorrow."
        ... )
        >>> print(result)  # {"intent_resolution": 5.0, "intent_resolution_result": "pass", ...}
    """
    credential = _get_credential_if_needed(model_config)
    return IntentResolutionEvaluator(
        model_config=model_config,
        credential=credential,
        threshold=threshold,
        is_reasoning_model=is_reasoning_model,
    )


def create_task_adherence_evaluator(
    model_config: ModelConfig,
    threshold: int = 3,
    is_reasoning_model: bool = False,
) -> TaskAdherenceEvaluator:
    """Create a TaskAdherenceEvaluator from the Azure AI Evaluation SDK.
    
    Evaluates whether the agent completed the assigned task.
    
    Supports both API key and Azure credential authentication.
    If api_key is not set in model_config, uses DefaultAzureCredential.
    
    Args:
        model_config: Azure OpenAI model configuration.
        threshold: Pass/fail threshold (1-5). Default is 3.
        is_reasoning_model: Set True for reasoning models like o3-mini.
    
    Returns:
        TaskAdherenceEvaluator instance ready to use.
    
    Example:
        >>> config = get_model_config()
        >>> evaluator = create_task_adherence_evaluator(config)
        >>> result = evaluator(
        ...     query="Calculate 15% tip on $50",
        ...     response="A 15% tip on $50 is $7.50."
        ... )
        >>> print(result)  # {"task_adherence": 5.0, "task_adherence_result": "pass", ...}
    """
    credential = _get_credential_if_needed(model_config)
    return TaskAdherenceEvaluator(
        model_config=model_config,
        credential=credential,
        threshold=threshold,
        is_reasoning_model=is_reasoning_model,
    )


def create_tool_call_accuracy_evaluator(
    model_config: ModelConfig,
    is_reasoning_model: bool = False,
) -> ToolCallAccuracyEvaluator:
    """Create a ToolCallAccuracyEvaluator from the Azure AI Evaluation SDK.
    
    Evaluates whether tool calls made by the agent were appropriate.
    
    Supports both API key and Azure credential authentication.
    If api_key is not set in model_config, uses DefaultAzureCredential.
    
    Args:
        model_config: Azure OpenAI model configuration.
        is_reasoning_model: Set True for reasoning models like o3-mini.
    
    Returns:
        ToolCallAccuracyEvaluator instance ready to use.
    
    Example:
        >>> config = get_model_config()
        >>> evaluator = create_tool_call_accuracy_evaluator(config)
        >>> result = evaluator(
        ...     query="What's the weather in Tokyo?",
        ...     response="The weather in Tokyo is sunny, 25°C.",
        ...     tool_calls=[{"name": "get_weather", "arguments": {"city": "Tokyo"}}],
        ...     tool_definitions=[{"name": "get_weather", "description": "Get weather for a city"}]
        ... )
        >>> print(result)  # {"tool_call_accuracy": 1.0, ...}
    """
    credential = _get_credential_if_needed(model_config)
    return ToolCallAccuracyEvaluator(
        model_config=model_config,
        credential=credential,
        is_reasoning_model=is_reasoning_model,
    )


def batch_evaluate(
    data: str | list[dict[str, Any]],
    evaluators: dict[str, Any],
    evaluator_config: dict[str, dict[str, Any]] | None = None,
    azure_ai_project: AzureAIProject | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Run batch evaluation using the Azure AI Evaluation SDK.
    
    Evaluates multiple samples with multiple evaluators in a single call.
    
    Args:
        data: Path to JSONL file or list of dictionaries with evaluation data.
        evaluators: Dictionary mapping evaluator names to evaluator instances.
        evaluator_config: Optional configuration for column mapping per evaluator.
        azure_ai_project: Optional Azure AI Foundry project for result logging.
        output_path: Optional path to save results locally.
    
    Returns:
        Dictionary with evaluation results including metrics and rows.
    
    Example:
        >>> from src.common.evaluation_config import get_model_config
        >>> config = get_model_config()
        >>> 
        >>> # Create evaluators
        >>> evaluators = {
        ...     "relevance": create_relevance_evaluator(config),
        ...     "coherence": create_coherence_evaluator(config),
        ... }
        >>> 
        >>> # Sample data
        >>> data = [
        ...     {"query": "What is Python?", "response": "Python is a language."},
        ...     {"query": "What is AI?", "response": "AI is artificial intelligence."},
        ... ]
        >>> 
        >>> # Run batch evaluation
        >>> results = batch_evaluate(data, evaluators)
        >>> print(results["metrics"])  # Aggregated scores
    """
    return evaluate(
        data=data,
        evaluators=evaluators,
        evaluator_config=evaluator_config,
        azure_ai_project=azure_ai_project,
        output_path=output_path,
    )


# =============================================================================
# Metrics Collector
# =============================================================================


class MetricsCollector:
    """Collector for evaluation metrics."""

    def __init__(
        self,
        evaluator: Evaluator | None = None,
        cost_calculator: CostCalculator | None = None,
    ):
        self.evaluator = evaluator or ExactMatchEvaluator()
        self.cost_calculator = cost_calculator or OpenAICostCalculator()
        self._metrics: list[MetricValue] = []
        self._evaluations: list[EvaluationResult] = []
        self._latencies: list[LatencyMetric] = []

    @contextmanager
    def measure_latency(self, operation: str):
        """Context manager to measure operation latency."""
        with tracer.start_as_current_span(f"measure_latency:{operation}"):
            start_time = datetime.now()
            start_ns = time.perf_counter_ns()
            error: str | None = None
            success = True

            try:
                yield
            except Exception as e:
                error = str(e)
                success = False
                raise
            finally:
                end_ns = time.perf_counter_ns()
                end_time = datetime.now()
                duration_ms = (end_ns - start_ns) / 1_000_000

                metric = LatencyMetric(
                    operation=operation,
                    duration_ms=duration_ms,
                    start_time=start_time,
                    end_time=end_time,
                    success=success,
                    error=error,
                )
                self._latencies.append(metric)
                self._metrics.append(
                    MetricValue(
                        name=f"latency.{operation}",
                        value=duration_ms,
                        metric_type=MetricType.LATENCY,
                        tags={"operation": operation, "success": str(success)},
                    )
                )

    def record_accuracy(
        self,
        expected: Any,
        actual: Any,
        evaluator: Evaluator | None = None,
    ) -> AccuracyMetric:
        """Record accuracy metric."""
        with tracer.start_as_current_span("record_accuracy"):
            eval_impl = evaluator or self.evaluator
            metric = eval_impl.evaluate(expected, actual)

            self._metrics.append(
                MetricValue(
                    name="accuracy",
                    value=metric.similarity_score,
                    metric_type=MetricType.ACCURACY,
                    tags={
                        "method": metric.evaluation_method,
                        "is_correct": str(metric.is_correct),
                    },
                )
            )

            return metric

    def record_cost(
        self,
        operation: str,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ) -> CostMetric:
        """Record cost metric."""
        with tracer.start_as_current_span("record_cost"):
            cost_usd = self.cost_calculator.calculate(
                input_tokens, output_tokens, model
            )

            metric = CostMetric(
                operation=operation,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=cost_usd,
                model=model,
            )

            self._metrics.append(
                MetricValue(
                    name="cost",
                    value=cost_usd,
                    metric_type=MetricType.COST,
                    tags={"operation": operation, "model": model},
                    metadata={
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                )
            )

            return metric

    def record_quality(
        self,
        dimension: str,
        score: float,
        explanation: str = "",
        evaluator: str = "manual",
    ) -> QualityScore:
        """Record quality score."""
        with tracer.start_as_current_span("record_quality"):
            quality = QualityScore(
                dimension=dimension,
                score=max(0.0, min(1.0, score)),  # Clamp to [0, 1]
                explanation=explanation,
                evaluator=evaluator,
            )

            self._metrics.append(
                MetricValue(
                    name=f"quality.{dimension}",
                    value=quality.score,
                    metric_type=MetricType.QUALITY,
                    tags={"dimension": dimension, "evaluator": evaluator},
                )
            )

            return quality

    def record_custom(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        tags: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MetricValue:
        """Record a custom metric."""
        metric = MetricValue(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags or {},
            metadata=metadata or {},
        )
        self._metrics.append(metric)
        return metric

    def create_evaluation(
        self,
        run_id: str,
        agent_name: str,
        latency: LatencyMetric | None = None,
        accuracy: AccuracyMetric | None = None,
        cost: CostMetric | None = None,
        quality_scores: list[QualityScore] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Create a complete evaluation result."""
        result = EvaluationResult(
            run_id=run_id,
            agent_name=agent_name,
            latency=latency,
            accuracy=accuracy,
            cost=cost,
            quality_scores=quality_scores or [],
            metadata=metadata or {},
        )
        self._evaluations.append(result)
        return result

    def get_metrics(
        self,
        metric_type: MetricType | None = None,
        name_pattern: str | None = None,
    ) -> list[MetricValue]:
        """Get collected metrics with optional filtering."""
        metrics = self._metrics

        if metric_type:
            metrics = [m for m in metrics if m.metric_type == metric_type]

        if name_pattern:
            metrics = [m for m in metrics if name_pattern in m.name]

        return metrics

    def get_evaluations(
        self,
        agent_name: str | None = None,
    ) -> list[EvaluationResult]:
        """Get evaluation results with optional filtering."""
        if agent_name:
            return [e for e in self._evaluations if e.agent_name == agent_name]
        return self._evaluations

    def aggregate(
        self,
        metric_name: str,
    ) -> AggregatedMetrics:
        """Aggregate metrics by name."""
        values = [
            m.value for m in self._metrics
            if m.name == metric_name
        ]

        if not values:
            return AggregatedMetrics(
                metric_name=metric_name,
                count=0,
                sum_value=0.0,
                min_value=0.0,
                max_value=0.0,
                avg_value=0.0,
                p50_value=0.0,
                p95_value=0.0,
                p99_value=0.0,
            )

        sorted_values = sorted(values)
        count = len(values)

        def percentile(data: list[float], p: float) -> float:
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data) - 1)]

        return AggregatedMetrics(
            metric_name=metric_name,
            count=count,
            sum_value=sum(values),
            min_value=min(values),
            max_value=max(values),
            avg_value=statistics.mean(values),
            p50_value=percentile(sorted_values, 50),
            p95_value=percentile(sorted_values, 95),
            p99_value=percentile(sorted_values, 99),
            values=values,
        )

    def summary(self) -> dict[str, Any]:
        """Generate a summary of all collected metrics."""
        metric_names = set(m.name for m in self._metrics)

        aggregations = {}
        for name in metric_names:
            agg = self.aggregate(name)
            aggregations[name] = {
                "count": agg.count,
                "avg": agg.avg_value,
                "min": agg.min_value,
                "max": agg.max_value,
                "p95": agg.p95_value,
            }

        return {
            "total_metrics": len(self._metrics),
            "total_evaluations": len(self._evaluations),
            "metric_types": list(set(m.metric_type.value for m in self._metrics)),
            "aggregations": aggregations,
        }

    def export_json(self) -> str:
        """Export metrics to JSON format."""
        data = {
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "type": m.metric_type.value,
                    "timestamp": m.timestamp.isoformat(),
                    "tags": m.tags,
                }
                for m in self._metrics
            ],
            "evaluations": [
                {
                    "run_id": e.run_id,
                    "agent_name": e.agent_name,
                    "timestamp": e.timestamp.isoformat(),
                    "overall_quality": e.overall_quality,
                }
                for e in self._evaluations
            ],
            "summary": self.summary(),
        }
        return json.dumps(data, indent=2)

    def reset(self) -> None:
        """Reset all collected metrics."""
        self._metrics.clear()
        self._evaluations.clear()
        self._latencies.clear()


# =============================================================================
# Convenience Functions
# =============================================================================


def create_collector(
    evaluator: Evaluator | None = None,
    cost_calculator: CostCalculator | None = None,
) -> MetricsCollector:
    """Create a metrics collector with optional custom components."""
    return MetricsCollector(
        evaluator=evaluator,
        cost_calculator=cost_calculator,
    )


def evaluate_response(
    expected: Any,
    actual: Any,
    method: str = "exact_match",
) -> AccuracyMetric:
    """Evaluate a response using the specified method."""
    evaluators = {
        "exact_match": ExactMatchEvaluator(),
        "contains": ContainsEvaluator(),
        "semantic": SemanticSimilarityEvaluator(),
    }

    evaluator = evaluators.get(method, ExactMatchEvaluator())
    return evaluator.evaluate(expected, actual)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o-mini",
) -> float:
    """Estimate cost for token usage."""
    calculator = OpenAICostCalculator()
    return calculator.calculate(input_tokens, output_tokens, model)
