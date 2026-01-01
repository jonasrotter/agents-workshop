"""Unit tests for evaluation metrics module."""

from __future__ import annotations

import pytest
import time
from datetime import datetime
from typing import Any

from src.common.evaluation import (
    MetricType,
    AggregationType,
    MetricValue,
    LatencyMetric,
    AccuracyMetric,
    CostMetric,
    QualityScore,
    EvaluationResult,
    AggregatedMetrics,
    MetricsCollector,
    ExactMatchEvaluator,
    ContainsEvaluator,
    SemanticSimilarityEvaluator,
    OpenAICostCalculator,
    create_collector,
    evaluate_response,
    estimate_cost,
)


# =============================================================================
# MetricType Tests
# =============================================================================


class TestMetricType:
    """Tests for MetricType enum."""

    def test_all_types_defined(self) -> None:
        """Test that all metric types are defined."""
        assert MetricType.LATENCY is not None
        assert MetricType.ACCURACY is not None
        assert MetricType.COST is not None
        assert MetricType.QUALITY is not None
        assert MetricType.THROUGHPUT is not None
        assert MetricType.ERROR_RATE is not None

    def test_type_values(self) -> None:
        """Test metric type values."""
        assert MetricType.LATENCY.value == "latency"
        assert MetricType.ACCURACY.value == "accuracy"
        assert MetricType.COST.value == "cost"


# =============================================================================
# MetricValue Tests
# =============================================================================


class TestMetricValue:
    """Tests for MetricValue dataclass."""

    def test_create_metric_value(self) -> None:
        """Test creating a metric value."""
        metric = MetricValue(
            name="test_metric",
            value=42.0,
            metric_type=MetricType.LATENCY,
        )
        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.metric_type == MetricType.LATENCY

    def test_metric_with_tags(self) -> None:
        """Test metric with tags."""
        metric = MetricValue(
            name="test",
            value=1.0,
            metric_type=MetricType.ACCURACY,
            tags={"environment": "test", "agent": "research"},
        )
        assert metric.tags["environment"] == "test"

    def test_metric_timestamp(self) -> None:
        """Test metric has timestamp."""
        metric = MetricValue(
            name="test",
            value=1.0,
            metric_type=MetricType.COST,
        )
        assert metric.timestamp is not None
        assert isinstance(metric.timestamp, datetime)


# =============================================================================
# LatencyMetric Tests
# =============================================================================


class TestLatencyMetric:
    """Tests for LatencyMetric dataclass."""

    def test_create_latency_metric(self) -> None:
        """Test creating a latency metric."""
        now = datetime.now()
        metric = LatencyMetric(
            operation="api_call",
            duration_ms=150.5,
            start_time=now,
            end_time=now,
        )
        assert metric.operation == "api_call"
        assert metric.duration_ms == 150.5
        assert metric.success is True

    def test_latency_with_error(self) -> None:
        """Test latency metric with error."""
        now = datetime.now()
        metric = LatencyMetric(
            operation="failed_call",
            duration_ms=50.0,
            start_time=now,
            end_time=now,
            success=False,
            error="Connection timeout",
        )
        assert metric.success is False
        assert metric.error == "Connection timeout"


# =============================================================================
# AccuracyMetric Tests
# =============================================================================


class TestAccuracyMetric:
    """Tests for AccuracyMetric dataclass."""

    def test_create_accuracy_metric(self) -> None:
        """Test creating an accuracy metric."""
        metric = AccuracyMetric(
            expected="Hello",
            actual="Hello",
            is_correct=True,
            similarity_score=1.0,
        )
        assert metric.is_correct is True
        assert metric.similarity_score == 1.0

    def test_accuracy_with_method(self) -> None:
        """Test accuracy metric with evaluation method."""
        metric = AccuracyMetric(
            expected="test",
            actual="TEST",
            is_correct=False,
            similarity_score=0.0,
            evaluation_method="exact_match",
        )
        assert metric.evaluation_method == "exact_match"


# =============================================================================
# CostMetric Tests
# =============================================================================


class TestCostMetric:
    """Tests for CostMetric dataclass."""

    def test_create_cost_metric(self) -> None:
        """Test creating a cost metric."""
        metric = CostMetric(
            operation="generate",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.0005,
            model="gpt-4o-mini",
        )
        assert metric.total_tokens == 150
        assert metric.cost_usd == 0.0005
        assert metric.model == "gpt-4o-mini"


# =============================================================================
# QualityScore Tests
# =============================================================================


class TestQualityScore:
    """Tests for QualityScore dataclass."""

    def test_create_quality_score(self) -> None:
        """Test creating a quality score."""
        score = QualityScore(
            dimension="relevance",
            score=0.85,
            explanation="Response is highly relevant",
        )
        assert score.dimension == "relevance"
        assert score.score == 0.85

    def test_quality_with_evaluator(self) -> None:
        """Test quality score with evaluator info."""
        score = QualityScore(
            dimension="clarity",
            score=0.9,
            evaluator="llm_judge",
        )
        assert score.evaluator == "llm_judge"


# =============================================================================
# EvaluationResult Tests
# =============================================================================


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_create_evaluation_result(self) -> None:
        """Test creating an evaluation result."""
        result = EvaluationResult(
            run_id="run_001",
            agent_name="test_agent",
        )
        assert result.run_id == "run_001"
        assert result.agent_name == "test_agent"

    def test_overall_quality_calculation(self) -> None:
        """Test overall quality calculation."""
        result = EvaluationResult(
            run_id="run_001",
            agent_name="test_agent",
            quality_scores=[
                QualityScore("relevance", 0.8, ""),
                QualityScore("clarity", 0.9, ""),
                QualityScore("accuracy", 0.7, ""),
            ],
        )
        assert result.overall_quality == pytest.approx(0.8, rel=0.01)

    def test_overall_quality_empty(self) -> None:
        """Test overall quality with no scores."""
        result = EvaluationResult(
            run_id="run_001",
            agent_name="test_agent",
        )
        assert result.overall_quality == 0.0


# =============================================================================
# Evaluator Tests
# =============================================================================


class TestExactMatchEvaluator:
    """Tests for ExactMatchEvaluator."""

    def test_exact_match_success(self) -> None:
        """Test exact match success."""
        evaluator = ExactMatchEvaluator()
        result = evaluator.evaluate("Hello", "Hello")
        assert result.is_correct is True
        assert result.similarity_score == 1.0

    def test_exact_match_failure(self) -> None:
        """Test exact match failure."""
        evaluator = ExactMatchEvaluator()
        result = evaluator.evaluate("Hello", "World")
        assert result.is_correct is False
        assert result.similarity_score == 0.0

    def test_exact_match_with_whitespace(self) -> None:
        """Test exact match handles whitespace."""
        evaluator = ExactMatchEvaluator()
        result = evaluator.evaluate("  Hello  ", "Hello")
        assert result.is_correct is True


class TestContainsEvaluator:
    """Tests for ContainsEvaluator."""

    def test_contains_success(self) -> None:
        """Test contains check success."""
        evaluator = ContainsEvaluator()
        result = evaluator.evaluate("Paris", "The capital is Paris, France.")
        assert result.is_correct is True

    def test_contains_failure(self) -> None:
        """Test contains check failure."""
        evaluator = ContainsEvaluator()
        result = evaluator.evaluate("London", "The capital is Paris.")
        assert result.is_correct is False

    def test_contains_case_insensitive(self) -> None:
        """Test contains is case insensitive."""
        evaluator = ContainsEvaluator()
        result = evaluator.evaluate("PARIS", "paris is beautiful")
        assert result.is_correct is True


class TestSemanticSimilarityEvaluator:
    """Tests for SemanticSimilarityEvaluator."""

    def test_similar_text(self) -> None:
        """Test similar text detection."""
        evaluator = SemanticSimilarityEvaluator(threshold=0.3)
        result = evaluator.evaluate(
            "AI is transforming technology",
            "Artificial intelligence transforms tech industry"
        )
        # Word overlap should give some similarity
        assert result.similarity_score > 0

    def test_different_text(self) -> None:
        """Test different text detection."""
        evaluator = SemanticSimilarityEvaluator(threshold=0.8)
        result = evaluator.evaluate(
            "The quick brown fox",
            "Lorem ipsum dolor sit"
        )
        assert result.is_correct is False

    def test_custom_threshold(self) -> None:
        """Test custom similarity threshold."""
        evaluator = SemanticSimilarityEvaluator(threshold=0.9)
        result = evaluator.evaluate("test", "test word")
        # With high threshold, partial matches fail
        assert result.is_correct is False


# =============================================================================
# CostCalculator Tests
# =============================================================================


class TestOpenAICostCalculator:
    """Tests for OpenAICostCalculator."""

    def test_gpt4o_mini_cost(self) -> None:
        """Test GPT-4o-mini cost calculation."""
        calculator = OpenAICostCalculator()
        cost = calculator.calculate(1000, 500, "gpt-4o-mini")
        # Input: 1000 tokens * $0.00015/1K = $0.00015
        # Output: 500 tokens * $0.0006/1K = $0.0003
        expected = 0.00015 + 0.0003
        assert cost == pytest.approx(expected, rel=0.01)

    def test_gpt4o_cost(self) -> None:
        """Test GPT-4o cost calculation."""
        calculator = OpenAICostCalculator()
        cost = calculator.calculate(1000, 500, "gpt-4o")
        # Input: 1000 * $0.005/1K = $0.005
        # Output: 500 * $0.015/1K = $0.0075
        expected = 0.005 + 0.0075
        assert cost == pytest.approx(expected, rel=0.01)

    def test_unknown_model(self) -> None:
        """Test unknown model defaults to gpt-4o-mini pricing."""
        calculator = OpenAICostCalculator()
        cost = calculator.calculate(1000, 500, "unknown-model")
        # Should use default pricing
        assert cost > 0


# =============================================================================
# MetricsCollector Tests
# =============================================================================


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    @pytest.fixture
    def collector(self) -> MetricsCollector:
        """Create a metrics collector."""
        return MetricsCollector()

    def test_create_collector(self, collector: MetricsCollector) -> None:
        """Test creating a collector."""
        assert collector is not None
        assert isinstance(collector.evaluator, ExactMatchEvaluator)

    def test_measure_latency(self, collector: MetricsCollector) -> None:
        """Test latency measurement."""
        with collector.measure_latency("test_operation"):
            time.sleep(0.01)  # 10ms
        
        metrics = collector.get_metrics(metric_type=MetricType.LATENCY)
        assert len(metrics) == 1
        assert metrics[0].value >= 10  # At least 10ms

    def test_measure_latency_with_error(
        self, collector: MetricsCollector
    ) -> None:
        """Test latency measurement with error."""
        with pytest.raises(ValueError):
            with collector.measure_latency("failing_operation"):
                raise ValueError("Test error")
        
        metrics = collector.get_metrics(metric_type=MetricType.LATENCY)
        assert len(metrics) == 1
        # Error should still be recorded

    def test_record_accuracy(self, collector: MetricsCollector) -> None:
        """Test accuracy recording."""
        result = collector.record_accuracy("test", "test")
        assert result.is_correct is True
        
        metrics = collector.get_metrics(metric_type=MetricType.ACCURACY)
        assert len(metrics) == 1

    def test_record_cost(self, collector: MetricsCollector) -> None:
        """Test cost recording."""
        result = collector.record_cost(
            operation="generate",
            input_tokens=100,
            output_tokens=50,
            model="gpt-4o-mini",
        )
        assert result.total_tokens == 150
        
        metrics = collector.get_metrics(metric_type=MetricType.COST)
        assert len(metrics) == 1

    def test_record_quality(self, collector: MetricsCollector) -> None:
        """Test quality recording."""
        result = collector.record_quality(
            dimension="relevance",
            score=0.9,
            explanation="Good",
        )
        assert result.score == 0.9
        
        metrics = collector.get_metrics(metric_type=MetricType.QUALITY)
        assert len(metrics) == 1

    def test_record_quality_clamping(
        self, collector: MetricsCollector
    ) -> None:
        """Test quality score clamping."""
        result = collector.record_quality(
            dimension="test",
            score=1.5,  # Over 1.0
        )
        assert result.score == 1.0

    def test_record_custom_metric(
        self, collector: MetricsCollector
    ) -> None:
        """Test custom metric recording."""
        result = collector.record_custom(
            name="custom_metric",
            value=42.0,
            metric_type=MetricType.THROUGHPUT,
            tags={"env": "test"},
        )
        assert result.value == 42.0
        assert result.tags["env"] == "test"

    def test_create_evaluation(
        self, collector: MetricsCollector
    ) -> None:
        """Test evaluation creation."""
        evaluation = collector.create_evaluation(
            run_id="test_run",
            agent_name="test_agent",
        )
        assert evaluation.run_id == "test_run"
        
        evaluations = collector.get_evaluations()
        assert len(evaluations) == 1

    def test_get_evaluations_filtered(
        self, collector: MetricsCollector
    ) -> None:
        """Test filtered evaluation retrieval."""
        collector.create_evaluation("run1", "agent_a")
        collector.create_evaluation("run2", "agent_b")
        collector.create_evaluation("run3", "agent_a")
        
        filtered = collector.get_evaluations(agent_name="agent_a")
        assert len(filtered) == 2

    def test_aggregate_metrics(
        self, collector: MetricsCollector
    ) -> None:
        """Test metric aggregation."""
        for i in range(10):
            collector.record_custom(
                name="test_latency",
                value=100 + i * 10,  # 100, 110, 120, ..., 190
                metric_type=MetricType.LATENCY,
            )
        
        agg = collector.aggregate("test_latency")
        assert agg.count == 10
        assert agg.min_value == 100
        assert agg.max_value == 190
        assert agg.avg_value == 145

    def test_aggregate_empty(
        self, collector: MetricsCollector
    ) -> None:
        """Test aggregation with no data."""
        agg = collector.aggregate("nonexistent")
        assert agg.count == 0
        assert agg.avg_value == 0.0

    def test_summary(self, collector: MetricsCollector) -> None:
        """Test metrics summary."""
        collector.record_accuracy("a", "a")
        collector.record_cost("op", 100, 50, "gpt-4o-mini")
        
        summary = collector.summary()
        assert summary["total_metrics"] == 2
        assert "aggregations" in summary

    def test_export_json(self, collector: MetricsCollector) -> None:
        """Test JSON export."""
        collector.record_accuracy("test", "test")
        
        json_str = collector.export_json()
        assert "metrics" in json_str
        assert "evaluations" in json_str
        assert "summary" in json_str

    def test_reset(self, collector: MetricsCollector) -> None:
        """Test collector reset."""
        collector.record_accuracy("a", "a")
        collector.create_evaluation("run", "agent")
        
        collector.reset()
        
        assert len(collector.get_metrics()) == 0
        assert len(collector.get_evaluations()) == 0


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_collector(self) -> None:
        """Test create_collector function."""
        collector = create_collector()
        assert isinstance(collector, MetricsCollector)

    def test_create_collector_custom_evaluator(self) -> None:
        """Test create_collector with custom evaluator."""
        evaluator = ContainsEvaluator()
        collector = create_collector(evaluator=evaluator)
        assert isinstance(collector.evaluator, ContainsEvaluator)

    def test_evaluate_response_exact(self) -> None:
        """Test evaluate_response with exact match."""
        result = evaluate_response("test", "test", "exact_match")
        assert result.is_correct is True

    def test_evaluate_response_contains(self) -> None:
        """Test evaluate_response with contains."""
        result = evaluate_response("test", "this is a test", "contains")
        assert result.is_correct is True

    def test_evaluate_response_semantic(self) -> None:
        """Test evaluate_response with semantic."""
        result = evaluate_response(
            "hello world",
            "hello beautiful world",
            "semantic"
        )
        assert result.similarity_score > 0

    def test_estimate_cost(self) -> None:
        """Test estimate_cost function."""
        cost = estimate_cost(1000, 500, "gpt-4o-mini")
        assert cost > 0
        assert cost < 0.01  # Should be very low for mini


# =============================================================================
# Integration Tests
# =============================================================================


class TestMetricsIntegration:
    """Integration tests for metrics collection."""

    def test_full_evaluation_workflow(self) -> None:
        """Test complete evaluation workflow."""
        collector = create_collector()
        
        # Simulate an agent run
        with collector.measure_latency("agent_run"):
            time.sleep(0.01)
        
        # Record accuracy
        accuracy = collector.record_accuracy(
            expected="The answer is 42",
            actual="The answer is 42",
        )
        
        # Record cost
        cost = collector.record_cost(
            operation="generate",
            input_tokens=500,
            output_tokens=250,
            model="gpt-4o-mini",
        )
        
        # Record quality
        quality = collector.record_quality(
            dimension="helpfulness",
            score=0.9,
        )
        
        # Create evaluation
        now = datetime.now()
        evaluation = collector.create_evaluation(
            run_id="workflow_test",
            agent_name="test_agent",
            latency=LatencyMetric(
                "agent_run", 10.0, now, now
            ),
            accuracy=accuracy,
            cost=cost,
            quality_scores=[quality],
        )
        
        assert evaluation.overall_quality == 0.9
        
        # Get summary
        summary = collector.summary()
        assert summary["total_metrics"] >= 3
        assert summary["total_evaluations"] == 1
