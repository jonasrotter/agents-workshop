"""Unit tests for evaluation metrics module."""

from __future__ import annotations

import os
import pytest
import time
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

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
    # SDK wrapper functions
    create_relevance_evaluator,
    create_coherence_evaluator,
    create_fluency_evaluator,
    create_groundedness_evaluator,
    create_intent_resolution_evaluator,
    create_task_adherence_evaluator,
    create_tool_call_accuracy_evaluator,
    batch_evaluate,
)
from src.common.config import (
    get_model_config,
    get_azure_ai_project,
    validate_model_config,
    get_config_summary,
    ModelConfig,
)
from src.common.exceptions import ConfigurationError


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
            "AI technology is transforming the world",
            "AI technology transforms the modern world"
        )
        # Word overlap should give some similarity (AI, technology, the, world)
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


# =============================================================================
# Evaluation Config Tests
# =============================================================================


class TestGetModelConfig:
    """Tests for get_model_config function.
    
    The get_model_config function now uses Settings from src.common.config
    which reads from AZURE_OPENAI_* environment variables. Tests pass
    a mock Settings object to ensure isolation from .env file.
    """

    def test_get_config_from_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test getting config from environment variables via Settings."""
        # Use the correct AZURE_OPENAI_* env vars (read by Settings)
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        # Clear cached settings so new env vars take effect
        from src.common.config import get_settings
        get_settings.cache_clear()

        config = get_model_config()

        assert config["azure_deployment"] == "gpt-4o"
        assert config["api_key"] == "test-key"
        assert config["azure_endpoint"] == "https://test.openai.azure.com"
        assert config["api_version"] == "2024-02-15-preview"

    def test_get_config_with_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test getting config with parameter overrides."""
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "default-model")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "default-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://default.openai.azure.com")
        
        from src.common.config import get_settings
        get_settings.cache_clear()

        config = get_model_config(
            azure_deployment="override-model",
            api_key="override-key",
        )

        assert config["azure_deployment"] == "override-model"
        assert config["api_key"] == "override-key"
        # Endpoint should still come from env
        assert config["azure_endpoint"] == "https://default.openai.azure.com"

    def test_get_config_raises_on_missing(self) -> None:
        """Test that missing config raises ConfigurationError."""
        # Create a mock Settings with missing values
        from src.common.config import Settings
        
        mock_settings = MagicMock(spec=Settings)
        mock_settings.azure_openai_deployment = ""
        mock_settings.azure_openai_api_key = ""
        mock_settings.azure_openai_endpoint = ""
        mock_settings.azure_openai_api_version = ""

        with pytest.raises(ConfigurationError) as exc_info:
            get_model_config(settings=mock_settings)

        assert "Missing required environment variables" in str(exc_info.value)

    def test_get_config_no_raise_on_missing(self) -> None:
        """Test getting config without raising on missing values."""
        from src.common.config import Settings
        
        mock_settings = MagicMock(spec=Settings)
        mock_settings.azure_openai_deployment = ""
        mock_settings.azure_openai_api_key = ""
        mock_settings.azure_openai_endpoint = ""
        mock_settings.azure_openai_api_version = ""

        config = get_model_config(raise_on_missing=False, settings=mock_settings)

        assert config["azure_deployment"] == ""
        assert config["api_key"] == ""
        assert config["azure_endpoint"] == ""

    def test_default_api_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test default API version is used when not set."""
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "model")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
        
        from src.common.config import get_settings
        get_settings.cache_clear()

        config = get_model_config()

        # Default comes from Settings.azure_openai_api_version default
        assert config["api_version"] == "2024-10-01-preview"


class TestGetAzureAIProject:
    """Tests for get_azure_ai_project function."""

    def test_get_project_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test getting Azure AI project config."""
        monkeypatch.setenv("AZURE_AI_PROJECT_SUBSCRIPTION_ID", "sub-123")
        monkeypatch.setenv("AZURE_AI_PROJECT_RESOURCE_GROUP", "rg-test")
        monkeypatch.setenv("AZURE_AI_PROJECT_NAME", "project-test")
        
        from src.common.config import get_settings
        get_settings.cache_clear()

        project = get_azure_ai_project()

        assert project is not None
        assert project["subscription_id"] == "sub-123"
        assert project["resource_group_name"] == "rg-test"
        assert project["project_name"] == "project-test"

    def test_get_project_returns_none_on_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that missing project config returns None."""
        monkeypatch.delenv("AZURE_AI_PROJECT_SUBSCRIPTION_ID", raising=False)
        monkeypatch.delenv("AZURE_AI_PROJECT_RESOURCE_GROUP", raising=False)
        monkeypatch.delenv("AZURE_AI_PROJECT_NAME", raising=False)
        
        from src.common.config import get_settings
        get_settings.cache_clear()

        project = get_azure_ai_project()

        assert project is None

    def test_get_project_with_partial_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that partial project config returns None."""
        monkeypatch.setenv("AZURE_AI_PROJECT_SUBSCRIPTION_ID", "sub-123")
        # Missing resource group and project name
        monkeypatch.delenv("AZURE_AI_PROJECT_RESOURCE_GROUP", raising=False)
        monkeypatch.delenv("AZURE_AI_PROJECT_NAME", raising=False)
        
        from src.common.config import get_settings
        get_settings.cache_clear()

        project = get_azure_ai_project()

        assert project is None


class TestValidateConfig:
    """Tests for validate_model_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config: ModelConfig = {
            "azure_deployment": "gpt-4o",
            "api_key": "test-key",
            "azure_endpoint": "https://test.openai.azure.com",
            "api_version": "2024-02-15-preview",
        }

        issues = validate_model_config(config)

        assert issues == []

    def test_missing_fields(self) -> None:
        """Test validation catches missing fields."""
        config: ModelConfig = {
            "azure_deployment": "",
            "api_key": "",
            "azure_endpoint": "",
            "api_version": "",
        }

        issues = validate_model_config(config)

        # api_key is optional (credential auth supported), so only 3 required fields
        assert len(issues) == 3
        assert any("azure_deployment" in issue for issue in issues)
        assert not any("api_key" in issue for issue in issues)  # api_key is now optional

    def test_invalid_endpoint(self) -> None:
        """Test validation catches invalid endpoint."""
        config: ModelConfig = {
            "azure_deployment": "gpt-4o",
            "api_key": "test-key",
            "azure_endpoint": "http://insecure.endpoint.com",  # Not https
            "api_version": "2024-02-15-preview",
        }

        issues = validate_model_config(config)

        assert any("https://" in issue for issue in issues)


class TestGetConfigSummary:
    """Tests for get_config_summary function."""

    def test_masks_api_key(self) -> None:
        """Test that API key is masked in summary."""
        config: ModelConfig = {
            "azure_deployment": "gpt-4o",
            "api_key": "super-secret-key",
            "azure_endpoint": "https://test.openai.azure.com",
            "api_version": "2024-02-15-preview",
        }

        summary = get_config_summary(config)

        assert summary["api_key"] == "***"
        assert summary["azure_deployment"] == "gpt-4o"
        assert summary["azure_endpoint"] == "https://test.openai.azure.com"

    def test_shows_not_set_for_empty_key(self) -> None:
        """Test that empty API key shows (not set)."""
        config: ModelConfig = {
            "azure_deployment": "gpt-4o",
            "api_key": "",
            "azure_endpoint": "https://test.openai.azure.com",
            "api_version": "2024-02-15-preview",
        }

        summary = get_config_summary(config)

        assert summary["api_key"] == "(not set)"


# =============================================================================
# SDK Evaluator Wrapper Tests
# =============================================================================


class TestSDKEvaluatorWrappers:
    """Tests for SDK evaluator wrapper functions."""

    @pytest.fixture
    def mock_model_config(self) -> ModelConfig:
        """Create a mock model config."""
        return {
            "azure_deployment": "gpt-4o",
            "api_key": "test-key",
            "azure_endpoint": "https://test.openai.azure.com",
            "api_version": "2024-02-15-preview",
        }

    @patch("src.common.evaluation.RelevanceEvaluator")
    def test_create_relevance_evaluator(
        self, mock_evaluator: MagicMock, mock_model_config: ModelConfig
    ) -> None:
        """Test creating relevance evaluator."""
        evaluator = create_relevance_evaluator(mock_model_config)

        # With api_key set, credential should be None
        mock_evaluator.assert_called_once_with(mock_model_config, credential=None)
        assert evaluator == mock_evaluator.return_value

    @patch("src.common.evaluation.CoherenceEvaluator")
    def test_create_coherence_evaluator(
        self, mock_evaluator: MagicMock, mock_model_config: ModelConfig
    ) -> None:
        """Test creating coherence evaluator."""
        evaluator = create_coherence_evaluator(mock_model_config)

        # With api_key set, credential should be None
        mock_evaluator.assert_called_once_with(mock_model_config, credential=None)
        assert evaluator == mock_evaluator.return_value

    @patch("src.common.evaluation.FluencyEvaluator")
    def test_create_fluency_evaluator(
        self, mock_evaluator: MagicMock, mock_model_config: ModelConfig
    ) -> None:
        """Test creating fluency evaluator."""
        evaluator = create_fluency_evaluator(mock_model_config)

        # With api_key set, credential should be None
        mock_evaluator.assert_called_once_with(mock_model_config, credential=None)
        assert evaluator == mock_evaluator.return_value

    @patch("src.common.evaluation.GroundednessEvaluator")
    def test_create_groundedness_evaluator(
        self, mock_evaluator: MagicMock, mock_model_config: ModelConfig
    ) -> None:
        """Test creating groundedness evaluator."""
        evaluator = create_groundedness_evaluator(mock_model_config)

        # With api_key set, credential should be None
        mock_evaluator.assert_called_once_with(mock_model_config, credential=None)
        assert evaluator == mock_evaluator.return_value

    @patch("src.common.evaluation.IntentResolutionEvaluator")
    def test_create_intent_resolution_evaluator(
        self, mock_evaluator: MagicMock, mock_model_config: ModelConfig
    ) -> None:
        """Test creating intent resolution evaluator."""
        evaluator = create_intent_resolution_evaluator(mock_model_config)

        # With api_key set, credential should be None
        mock_evaluator.assert_called_once_with(
            model_config=mock_model_config,
            credential=None,
            threshold=3,
            is_reasoning_model=False,
        )
        assert evaluator == mock_evaluator.return_value

    @patch("src.common.evaluation.TaskAdherenceEvaluator")
    def test_create_task_adherence_evaluator(
        self, mock_evaluator: MagicMock, mock_model_config: ModelConfig
    ) -> None:
        """Test creating task adherence evaluator."""
        evaluator = create_task_adherence_evaluator(mock_model_config)

        # With api_key set, credential should be None
        mock_evaluator.assert_called_once_with(
            model_config=mock_model_config,
            credential=None,
            threshold=3,
            is_reasoning_model=False,
        )
        assert evaluator == mock_evaluator.return_value

    @patch("src.common.evaluation.ToolCallAccuracyEvaluator")
    def test_create_tool_call_accuracy_evaluator(
        self, mock_evaluator: MagicMock, mock_model_config: ModelConfig
    ) -> None:
        """Test creating tool call accuracy evaluator."""
        evaluator = create_tool_call_accuracy_evaluator(mock_model_config)

        # With api_key set, credential should be None
        mock_evaluator.assert_called_once_with(
            model_config=mock_model_config,
            credential=None,
            is_reasoning_model=False,
        )
        assert evaluator == mock_evaluator.return_value


class TestBatchEvaluate:
    """Tests for batch_evaluate function."""

    @pytest.fixture
    def mock_model_config(self) -> ModelConfig:
        """Create a mock model config."""
        return {
            "azure_deployment": "gpt-4o",
            "api_key": "test-key",
            "azure_endpoint": "https://test.openai.azure.com",
            "api_version": "2024-02-15-preview",
        }

    @pytest.fixture
    def sample_data(self) -> list[dict[str, str]]:
        """Create sample evaluation data."""
        return [
            {
                "query": "What is AI?",
                "context": "AI is artificial intelligence.",
                "response": "AI stands for artificial intelligence.",
            },
            {
                "query": "Define ML",
                "context": "ML is machine learning.",
                "response": "ML means machine learning.",
            },
        ]

    @patch("src.common.evaluation.evaluate")
    def test_batch_evaluate_with_evaluators_dict(
        self,
        mock_evaluate: MagicMock,
        mock_model_config: ModelConfig,
        sample_data: list[dict[str, str]],
    ) -> None:
        """Test batch evaluation with evaluators dictionary."""
        mock_evaluate.return_value = {"metrics": {"relevance": 4.5, "coherence": 4.0}}
        
        # Create mock evaluators
        mock_relevance = MagicMock()
        mock_coherence = MagicMock()
        evaluators = {"relevance": mock_relevance, "coherence": mock_coherence}

        result = batch_evaluate(
            data=sample_data,
            evaluators=evaluators,
        )

        mock_evaluate.assert_called_once()
        assert result == {"metrics": {"relevance": 4.5, "coherence": 4.0}}

    @patch("src.common.evaluation.evaluate")
    def test_batch_evaluate_with_evaluator_config(
        self,
        mock_evaluate: MagicMock,
        mock_model_config: ModelConfig,
    ) -> None:
        """Test batch evaluation with evaluator config."""
        data = [{"q": "test", "r": "response"}]
        evaluator_config = {"relevance": {"column_mapping": {"query": "q", "response": "r"}}}
        mock_evaluate.return_value = {"metrics": {}}
        mock_evaluator = MagicMock()

        batch_evaluate(
            data=data,
            evaluators={"relevance": mock_evaluator},
            evaluator_config=evaluator_config,
        )

        call_kwargs = mock_evaluate.call_args.kwargs
        assert call_kwargs["evaluator_config"] == evaluator_config

    @patch("src.common.evaluation.evaluate")
    def test_batch_evaluate_with_azure_ai_project(
        self,
        mock_evaluate: MagicMock,
        mock_model_config: ModelConfig,
        sample_data: list[dict[str, str]],
    ) -> None:
        """Test batch evaluation with Azure AI project."""
        azure_project = {
            "subscription_id": "sub-123",
            "resource_group_name": "rg-test",
            "project_name": "project-test",
        }
        mock_evaluate.return_value = {"metrics": {}}
        mock_evaluator = MagicMock()

        batch_evaluate(
            data=sample_data,
            evaluators={"relevance": mock_evaluator},
            azure_ai_project=azure_project,
        )

        call_kwargs = mock_evaluate.call_args.kwargs
        assert call_kwargs["azure_ai_project"] == azure_project

    @patch("src.common.evaluation.evaluate")
    def test_batch_evaluate_with_output_path(
        self,
        mock_evaluate: MagicMock,
        sample_data: list[dict[str, str]],
    ) -> None:
        """Test batch evaluation with output path."""
        mock_evaluate.return_value = {"metrics": {}}
        mock_evaluator = MagicMock()
        output_path = "/tmp/results.json"

        batch_evaluate(
            data=sample_data,
            evaluators={"relevance": mock_evaluator},
            output_path=output_path,
        )

        call_kwargs = mock_evaluate.call_args.kwargs
        assert call_kwargs["output_path"] == output_path
