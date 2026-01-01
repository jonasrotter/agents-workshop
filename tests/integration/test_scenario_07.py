"""Integration tests for Scenario 07: Evaluation and Prompt Evolution."""

from __future__ import annotations

import pytest
import time
from datetime import datetime
from typing import Any


# =============================================================================
# Test Module Imports
# =============================================================================


class TestScenario07Imports:
    """Test that all Scenario 07 modules can be imported."""

    def test_import_metrics_collector(self) -> None:
        """Test importing MetricsCollector."""
        from src.common.evaluation import MetricsCollector
        assert MetricsCollector is not None

    def test_import_metric_type(self) -> None:
        """Test importing MetricType."""
        from src.common.evaluation import MetricType
        assert MetricType is not None

    def test_import_evaluators(self) -> None:
        """Test importing evaluators."""
        from src.common.evaluation import (
            ExactMatchEvaluator,
            ContainsEvaluator,
            SemanticSimilarityEvaluator,
        )
        assert ExactMatchEvaluator is not None
        assert ContainsEvaluator is not None
        assert SemanticSimilarityEvaluator is not None

    def test_import_cost_calculator(self) -> None:
        """Test importing OpenAICostCalculator."""
        from src.common.evaluation import OpenAICostCalculator
        assert OpenAICostCalculator is not None

    def test_import_data_classes(self) -> None:
        """Test importing data classes."""
        from src.common.evaluation import (
            MetricValue,
            LatencyMetric,
            AccuracyMetric,
            CostMetric,
            QualityScore,
            EvaluationResult,
        )
        assert MetricValue is not None
        assert LatencyMetric is not None
        assert AccuracyMetric is not None
        assert CostMetric is not None
        assert QualityScore is not None
        assert EvaluationResult is not None

    def test_import_convenience_functions(self) -> None:
        """Test importing convenience functions."""
        from src.common.evaluation import (
            create_collector,
            evaluate_response,
            estimate_cost,
        )
        assert create_collector is not None
        assert evaluate_response is not None
        assert estimate_cost is not None

    def test_import_prompt_tuner(self) -> None:
        """Test importing PromptTuner."""
        from src.common.prompt_tuning import PromptTuner
        assert PromptTuner is not None

    def test_import_prompt_analyzer(self) -> None:
        """Test importing PromptAnalyzer."""
        from src.common.prompt_tuning import PromptAnalyzer
        assert PromptAnalyzer is not None

    def test_import_prompt_template(self) -> None:
        """Test importing PromptTemplate."""
        from src.common.prompt_tuning import PromptTemplate
        assert PromptTemplate is not None

    def test_import_prompt_enums(self) -> None:
        """Test importing prompt-related enums."""
        from src.common.prompt_tuning import (
            PromptStatus,
            ComparisonResult,
            ImprovementType,
        )
        assert PromptStatus is not None
        assert ComparisonResult is not None
        assert ImprovementType is not None

    def test_import_prompt_convenience(self) -> None:
        """Test importing prompt convenience functions."""
        from src.common.prompt_tuning import (
            create_tuner,
            analyze_prompt,
            create_template,
        )
        assert create_tuner is not None
        assert analyze_prompt is not None
        assert create_template is not None


# =============================================================================
# Evaluation Integration Tests
# =============================================================================


class TestEvaluationIntegration:
    """Integration tests for evaluation functionality."""

    def test_full_metrics_workflow(self) -> None:
        """Test complete metrics collection workflow."""
        from src.common.evaluation import (
            MetricsCollector,
            MetricType,
            LatencyMetric,
            QualityScore,
        )

        collector = MetricsCollector()

        # 1. Measure latency
        with collector.measure_latency("test_operation"):
            time.sleep(0.01)

        # 2. Record accuracy
        accuracy = collector.record_accuracy(
            expected="expected output",
            actual="expected output",
        )
        assert accuracy.is_correct

        # 3. Record cost
        cost = collector.record_cost(
            operation="generate",
            input_tokens=1000,
            output_tokens=500,
            model="gpt-4o-mini",
        )
        assert cost.cost_usd > 0

        # 4. Record quality
        quality = collector.record_quality(
            dimension="relevance",
            score=0.9,
            explanation="Very relevant response",
        )

        # 5. Create evaluation
        evaluation = collector.create_evaluation(
            run_id="integration_test",
            agent_name="test_agent",
            accuracy=accuracy,
            cost=cost,
            quality_scores=[quality],
        )

        # 6. Verify summary
        summary = collector.summary()
        assert summary["total_metrics"] >= 3
        assert summary["total_evaluations"] == 1

    def test_multiple_evaluations(self) -> None:
        """Test collecting multiple evaluations."""
        from src.common.evaluation import create_collector

        collector = create_collector()

        # Create multiple evaluations
        for i in range(5):
            collector.record_accuracy(f"test{i}", f"test{i}")
            collector.record_cost(
                f"op{i}", 100 * (i + 1), 50 * (i + 1), "gpt-4o-mini"
            )
            collector.create_evaluation(f"run_{i}", "test_agent")

        # Verify counts
        assert len(collector.get_evaluations()) == 5

    def test_metric_aggregation(self) -> None:
        """Test metric aggregation functionality."""
        from src.common.evaluation import create_collector, MetricType

        collector = create_collector()

        # Record multiple latency metrics
        latencies = [100, 150, 120, 200, 180, 110, 130, 170, 140, 160]
        for lat in latencies:
            collector.record_custom(
                name="api_latency",
                value=float(lat),
                metric_type=MetricType.LATENCY,
            )

        # Aggregate
        agg = collector.aggregate("api_latency")

        assert agg.count == 10
        assert agg.min_value == 100
        assert agg.max_value == 200
        assert agg.avg_value == 146.0  # Average of latencies

    def test_evaluator_comparison(self) -> None:
        """Test comparing different evaluators."""
        from src.common.evaluation import evaluate_response

        test_cases = [
            ("Paris", "Paris", True, True, True),  # All should match
            ("Paris", "The answer is Paris.", False, True, True),  # Contains/semantic
            ("AI", "Artificial Intelligence", False, False, True),  # Semantic only
        ]

        for expected, actual, exact_expected, contains_expected, _ in test_cases:
            exact = evaluate_response(expected, actual, "exact_match")
            contains = evaluate_response(expected, actual, "contains")
            
            assert exact.is_correct == exact_expected
            assert contains.is_correct == contains_expected


# =============================================================================
# Prompt Tuning Integration Tests
# =============================================================================


class TestPromptTuningIntegration:
    """Integration tests for prompt tuning functionality."""

    def test_prompt_versioning_workflow(self) -> None:
        """Test complete prompt versioning workflow."""
        from src.common.prompt_tuning import create_tuner

        tuner = create_tuner()

        # Create initial prompt
        v1 = tuner.create_prompt(
            name="summarize",
            content="Summarize this text.",
        )
        assert v1.version == "v1.0"

        # Iterate
        v2 = tuner.iterate(
            name="summarize",
            new_content="You are a helpful assistant. Summarize this text concisely.",
            changes="Added role definition",
        )
        assert v2.version == "v1.1"

        # Get history
        history = tuner.get_history("summarize")
        assert len(history) == 2

    def test_prompt_analysis(self) -> None:
        """Test prompt analysis functionality."""
        from src.common.prompt_tuning import analyze_prompt

        # Analyze a simple prompt
        simple = analyze_prompt("Do the task.")
        assert simple.word_count < 10
        assert len(simple.suggestions) > 0  # Should have suggestions

        # Analyze a complete prompt
        complete = analyze_prompt("""
            You are an expert analyst.
            
            Analyze the following data and provide insights.
            
            Requirements:
            - Be concise
            - Use bullet points
            
            Example:
            Input: Sales data
            Output: Key trends...
            
            Format: Markdown
        """)
        assert complete.has_role
        assert complete.has_examples
        assert complete.has_constraints

    def test_prompt_template_rendering(self) -> None:
        """Test prompt template rendering."""
        from src.common.prompt_tuning import create_template

        template = create_template(
            template="You are a {role}. {task} Output in {format}.",
            variables=[
                {"name": "role", "description": "Agent role", "required": True},
                {"name": "task", "description": "Task to perform", "required": True},
                {"name": "format", "description": "Output format", "default": "text"},
            ],
        )

        # Render with all variables
        result = template.render(
            role="data analyst",
            task="Analyze this data.",
            format="JSON",
        )
        assert "data analyst" in result
        assert "JSON" in result

        # Render with defaults
        result_default = template.render(
            role="assistant",
            task="Help me.",
        )
        assert "text" in result_default

    def test_ab_testing_setup(self) -> None:
        """Test A/B testing setup."""
        from src.common.prompt_tuning import create_tuner

        tuner = create_tuner()

        # Create two versions
        tuner.create_prompt("test", "Version A content")
        tuner.iterate("test", "Version B content", "Testing")

        # Set up comparison
        config = tuner.compare(
            name="test",
            version_a="v1.0",
            version_b="v1.1",
            sample_size=50,
        )

        assert config.sample_size == 50
        assert config.variant_a.version == "v1.0"
        assert config.variant_b.version == "v1.1"


# =============================================================================
# Cost Estimation Integration Tests
# =============================================================================


class TestCostEstimationIntegration:
    """Integration tests for cost estimation."""

    def test_cost_comparison_models(self) -> None:
        """Test cost comparison across models."""
        from src.common.evaluation import estimate_cost

        models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        costs = {}

        for model in models:
            cost = estimate_cost(1000, 500, model)
            costs[model] = cost

        # GPT-4o-mini should be cheapest
        assert costs["gpt-4o-mini"] < costs["gpt-4o"]
        assert costs["gpt-3.5-turbo"] < costs["gpt-4-turbo"]

    def test_cost_tracking_accumulation(self) -> None:
        """Test accumulating costs over multiple calls."""
        from src.common.evaluation import create_collector, MetricType

        collector = create_collector()

        # Simulate multiple API calls
        operations = [
            ("query1", 200, 100),
            ("query2", 500, 250),
            ("query3", 300, 150),
        ]

        total_cost = 0
        for op, input_t, output_t in operations:
            cost = collector.record_cost(op, input_t, output_t, "gpt-4o-mini")
            total_cost += cost.cost_usd

        # Verify total
        cost_metrics = collector.get_metrics(metric_type=MetricType.COST)
        recorded_total = sum(m.value for m in cost_metrics)
        
        assert recorded_total == pytest.approx(total_cost, rel=0.01)


# =============================================================================
# Export and Serialization Tests
# =============================================================================


class TestExportIntegration:
    """Integration tests for export functionality."""

    def test_metrics_json_export(self) -> None:
        """Test JSON export of metrics."""
        import json
        from src.common.evaluation import create_collector

        collector = create_collector()

        # Add various metrics
        collector.record_accuracy("test", "test")
        collector.record_cost("op", 100, 50, "gpt-4o-mini")
        collector.record_quality("clarity", 0.9)
        collector.create_evaluation("run1", "agent1")

        # Export to JSON
        json_str = collector.export_json()
        
        # Verify valid JSON
        data = json.loads(json_str)
        assert "metrics" in data
        assert "evaluations" in data
        assert "summary" in data

    def test_prompt_export(self) -> None:
        """Test prompt export functionality."""
        from src.common.prompt_tuning import create_tuner

        tuner = create_tuner()

        # Create prompts
        tuner.create_prompt("prompt_a", "Content A")
        tuner.create_prompt("prompt_b", "Content B")
        tuner.iterate("prompt_a", "Content A v2", "Updated")

        # Export
        export = tuner.export()

        assert "prompt_a" in export
        assert "prompt_b" in export
        assert len(export["prompt_a"]["versions"]) == 2


# =============================================================================
# End-to-End Scenario Tests
# =============================================================================


class TestEndToEndScenarios:
    """End-to-end tests for complete scenarios."""

    def test_evaluation_driven_improvement(self) -> None:
        """Test evaluation-driven prompt improvement workflow."""
        from src.common.evaluation import create_collector, evaluate_response
        from src.common.prompt_tuning import create_tuner, analyze_prompt

        # 1. Start with initial prompt
        tuner = create_tuner()
        v1 = tuner.create_prompt(
            "summarize",
            "Summarize the text.",
        )

        # 2. Analyze initial prompt
        analysis = analyze_prompt(v1.content)
        assert len(analysis.suggestions) > 0  # Has room for improvement

        # 3. Improve based on suggestions
        improved_content = """
        You are an expert summarizer.
        
        Summarize the following text in 3 bullet points.
        
        Requirements:
        - Be concise
        - Focus on key points
        
        Example:
        Input: Long article about AI
        Output: 
        - AI is advancing rapidly
        - Key applications in healthcare
        - Ethical concerns remain
        """

        v2 = tuner.iterate("summarize", improved_content, "Applied suggestions")

        # 4. Verify improvement
        analysis_v2 = analyze_prompt(v2.content)
        assert analysis_v2.has_role
        assert analysis_v2.has_examples
        assert analysis_v2.has_constraints

        # 5. Track metrics for both versions
        collector = create_collector()

        # Simulated results for v1
        collector.record_quality(
            dimension="v1_quality",
            score=0.6,
        )

        # Simulated results for v2
        collector.record_quality(
            dimension="v2_quality",
            score=0.85,
        )

        # 6. Promote better version
        tuner.promote("summarize", "v1.1")
        active = tuner.registry.get_active("summarize")
        assert active.version == "v1.1"

    def test_multi_agent_evaluation(self) -> None:
        """Test evaluating multiple agents."""
        from src.common.evaluation import create_collector

        collector = create_collector()

        # Evaluate multiple agents
        agents = ["research_agent", "summarize_agent", "qa_agent"]
        
        for agent in agents:
            # Simulate evaluation
            collector.record_accuracy(f"expected_{agent}", f"expected_{agent}")
            collector.record_cost(f"{agent}_call", 200, 100, "gpt-4o-mini")
            collector.record_quality(f"{agent}_quality", 0.8 + len(agent) * 0.01)
            collector.create_evaluation(f"run_{agent}", agent)

        # Get evaluations by agent
        for agent in agents:
            evals = collector.get_evaluations(agent_name=agent)
            assert len(evals) == 1

    def test_metrics_dashboard_data(self) -> None:
        """Test generating data for a metrics dashboard."""
        from src.common.evaluation import create_collector, MetricType

        collector = create_collector()

        # Simulate a day's worth of metrics
        for hour in range(24):
            base_latency = 100 + (hour % 12) * 10
            collector.record_custom(
                name="hourly_latency",
                value=float(base_latency),
                metric_type=MetricType.LATENCY,
                tags={"hour": str(hour)},
            )
            
            collector.record_cost(
                f"hour_{hour}",
                input_tokens=1000,
                output_tokens=500,
                model="gpt-4o-mini",
            )

        # Generate dashboard data
        latency_agg = collector.aggregate("hourly_latency")
        cost_metrics = collector.get_metrics(metric_type=MetricType.COST)

        assert latency_agg.count == 24
        assert len(cost_metrics) == 24
        
        # Summary for dashboard
        summary = collector.summary()
        assert summary["total_metrics"] >= 48  # At least 2 per hour
