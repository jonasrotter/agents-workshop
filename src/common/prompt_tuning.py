"""Prompt iteration and tuning framework for agentic AI systems.

This module provides tools for:
- Prompt version tracking
- A/B comparison support
- Improvement suggestions
- Prompt templates with variables
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
import re

from opentelemetry import trace

tracer = trace.get_tracer(__name__)


# =============================================================================
# Enums
# =============================================================================


class PromptStatus(str, Enum):
    """Status of a prompt version."""

    DRAFT = "draft"
    TESTING = "testing"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ComparisonResult(str, Enum):
    """Result of A/B comparison."""

    VARIANT_A_BETTER = "variant_a_better"
    VARIANT_B_BETTER = "variant_b_better"
    NO_DIFFERENCE = "no_difference"
    INCONCLUSIVE = "inconclusive"


class ImprovementType(str, Enum):
    """Types of prompt improvements."""

    CLARITY = "clarity"
    SPECIFICITY = "specificity"
    STRUCTURE = "structure"
    EXAMPLES = "examples"
    CONSTRAINTS = "constraints"
    TONE = "tone"
    LENGTH = "length"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PromptVariable:
    """Variable placeholder in a prompt template."""

    name: str
    description: str = ""
    required: bool = True
    default: Optional[str] = None
    validation_pattern: Optional[str] = None


@dataclass
class PromptTemplate:
    """A prompt template with variables."""

    template: str
    variables: list[PromptVariable] = field(default_factory=list)
    name: str = ""
    description: str = ""

    def render(self, **kwargs: Any) -> str:
        """Render template with provided variables."""
        result = self.template
        
        for var in self.variables:
            value = kwargs.get(var.name, var.default)
            
            if value is None and var.required:
                raise ValueError(f"Missing required variable: {var.name}")
            
            if value is not None:
                if var.validation_pattern:
                    if not re.match(var.validation_pattern, str(value)):
                        raise ValueError(
                            f"Variable {var.name} doesn't match pattern: {var.validation_pattern}"
                        )
                result = result.replace(f"{{{var.name}}}", str(value))
        
        return result

    def get_variable_names(self) -> list[str]:
        """Get list of variable names."""
        return [v.name for v in self.variables]


@dataclass
class PromptVersion:
    """A versioned prompt."""

    content: str
    version: str
    created_at: datetime = field(default_factory=datetime.now)
    status: PromptStatus = PromptStatus.DRAFT
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_version: Optional[str] = None
    changes: str = ""
    
    @property
    def hash(self) -> str:
        """Generate content hash."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:12]


@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""

    name: str
    variant_a: PromptVersion
    variant_b: PromptVersion
    sample_size: int = 100
    confidence_level: float = 0.95
    metrics: list[str] = field(default_factory=lambda: ["accuracy", "latency"])


@dataclass
class ABTestResult:
    """Results from an A/B test."""

    config: ABTestConfig
    variant_a_metrics: dict[str, float]
    variant_b_metrics: dict[str, float]
    winner: ComparisonResult
    confidence: float
    sample_count: int
    completed_at: datetime = field(default_factory=datetime.now)


@dataclass
class ImprovementSuggestion:
    """A suggested improvement to a prompt."""

    improvement_type: ImprovementType
    description: str
    original_segment: str = ""
    suggested_segment: str = ""
    rationale: str = ""
    confidence: float = 0.5
    

@dataclass
class PromptAnalysis:
    """Analysis of a prompt."""

    prompt: str
    word_count: int
    sentence_count: int
    has_examples: bool
    has_constraints: bool
    has_role: bool
    has_output_format: bool
    complexity_score: float
    suggestions: list[ImprovementSuggestion] = field(default_factory=list)


# =============================================================================
# Prompt Registry
# =============================================================================


class PromptRegistry:
    """Registry for managing prompt versions."""

    def __init__(self):
        self._prompts: dict[str, list[PromptVersion]] = {}
        self._active: dict[str, str] = {}  # prompt_name -> active_version

    def register(
        self,
        name: str,
        content: str,
        version: Optional[str] = None,
        parent_version: Optional[str] = None,
        changes: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> PromptVersion:
        """Register a new prompt version."""
        with tracer.start_as_current_span("register_prompt"):
            if name not in self._prompts:
                self._prompts[name] = []
            
            # Generate version if not provided
            if version is None:
                existing = len(self._prompts[name])
                version = f"v{existing + 1}.0"
            
            prompt_version = PromptVersion(
                content=content,
                version=version,
                parent_version=parent_version,
                changes=changes,
                metadata=metadata or {},
            )
            
            self._prompts[name].append(prompt_version)
            
            # Set as active if first version
            if name not in self._active:
                self._active[name] = version
                prompt_version.status = PromptStatus.ACTIVE
            
            return prompt_version

    def get(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> Optional[PromptVersion]:
        """Get a prompt by name and optional version."""
        if name not in self._prompts:
            return None
        
        versions = self._prompts[name]
        
        if version is None:
            # Return active version
            active_version = self._active.get(name)
            for v in versions:
                if v.version == active_version:
                    return v
            return versions[-1] if versions else None
        
        for v in versions:
            if v.version == version:
                return v
        
        return None

    def get_active(self, name: str) -> Optional[PromptVersion]:
        """Get the active version of a prompt."""
        return self.get(name, self._active.get(name))

    def set_active(self, name: str, version: str) -> None:
        """Set the active version of a prompt."""
        if name not in self._prompts:
            raise ValueError(f"Prompt not found: {name}")
        
        found = False
        for v in self._prompts[name]:
            if v.version == version:
                v.status = PromptStatus.ACTIVE
                found = True
            elif v.status == PromptStatus.ACTIVE:
                v.status = PromptStatus.DEPRECATED
        
        if not found:
            raise ValueError(f"Version not found: {version}")
        
        self._active[name] = version

    def list_versions(self, name: str) -> list[PromptVersion]:
        """List all versions of a prompt."""
        return self._prompts.get(name, [])

    def list_prompts(self) -> list[str]:
        """List all registered prompt names."""
        return list(self._prompts.keys())

    def get_history(self, name: str) -> list[dict[str, Any]]:
        """Get version history for a prompt."""
        versions = self._prompts.get(name, [])
        return [
            {
                "version": v.version,
                "status": v.status.value,
                "created_at": v.created_at.isoformat(),
                "changes": v.changes,
                "hash": v.hash,
            }
            for v in versions
        ]


# =============================================================================
# Prompt Analyzer
# =============================================================================


class PromptAnalyzer:
    """Analyzer for prompt quality and improvement suggestions."""

    def __init__(self):
        self._patterns = {
            "role": [
                r"you are",
                r"act as",
                r"pretend to be",
                r"your role is",
                r"as a",
            ],
            "examples": [
                r"example:",
                r"for example",
                r"e\.g\.",
                r"such as",
                r"here is an example",
            ],
            "constraints": [
                r"must",
                r"should not",
                r"never",
                r"always",
                r"required",
                r"important:",
            ],
            "output_format": [
                r"format:",
                r"output:",
                r"respond with",
                r"return",
                r"json",
                r"markdown",
            ],
        }

    def analyze(self, prompt: str) -> PromptAnalysis:
        """Analyze a prompt and generate suggestions."""
        with tracer.start_as_current_span("analyze_prompt"):
            words = prompt.split()
            sentences = re.split(r'[.!?]+', prompt)
            
            analysis = PromptAnalysis(
                prompt=prompt,
                word_count=len(words),
                sentence_count=len([s for s in sentences if s.strip()]),
                has_examples=self._has_pattern(prompt, "examples"),
                has_constraints=self._has_pattern(prompt, "constraints"),
                has_role=self._has_pattern(prompt, "role"),
                has_output_format=self._has_pattern(prompt, "output_format"),
                complexity_score=self._calculate_complexity(prompt),
            )
            
            # Generate suggestions
            analysis.suggestions = self._generate_suggestions(analysis)
            
            return analysis

    def _has_pattern(self, text: str, pattern_type: str) -> bool:
        """Check if text contains any patterns of given type."""
        patterns = self._patterns.get(pattern_type, [])
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in patterns)

    def _calculate_complexity(self, prompt: str) -> float:
        """Calculate prompt complexity score (0-1)."""
        factors = []
        
        # Length factor
        word_count = len(prompt.split())
        length_score = min(word_count / 500, 1.0)
        factors.append(length_score)
        
        # Structure factor (presence of lists, headers)
        has_lists = bool(re.search(r'^\s*[-*]\s', prompt, re.MULTILINE))
        has_numbers = bool(re.search(r'^\s*\d+[.)]\s', prompt, re.MULTILINE))
        structure_score = 0.5 if (has_lists or has_numbers) else 0.0
        factors.append(structure_score)
        
        # Technical terms factor
        technical_patterns = [r'\{.*?\}', r'<.*?>', r'\[.*?\]']
        technical_count = sum(
            len(re.findall(p, prompt)) for p in technical_patterns
        )
        technical_score = min(technical_count / 10, 1.0)
        factors.append(technical_score)
        
        return sum(factors) / len(factors) if factors else 0.0

    def _generate_suggestions(
        self, analysis: PromptAnalysis
    ) -> list[ImprovementSuggestion]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []
        
        if not analysis.has_role:
            suggestions.append(
                ImprovementSuggestion(
                    improvement_type=ImprovementType.CLARITY,
                    description="Add a role definition to clarify the AI's perspective",
                    suggested_segment="You are a [specific role]...",
                    rationale="Role definitions help the model adopt appropriate tone and expertise",
                    confidence=0.8,
                )
            )
        
        if not analysis.has_examples:
            suggestions.append(
                ImprovementSuggestion(
                    improvement_type=ImprovementType.EXAMPLES,
                    description="Include examples to demonstrate expected output",
                    suggested_segment="Example:\nInput: ...\nOutput: ...",
                    rationale="Examples significantly improve output consistency",
                    confidence=0.9,
                )
            )
        
        if not analysis.has_constraints:
            suggestions.append(
                ImprovementSuggestion(
                    improvement_type=ImprovementType.CONSTRAINTS,
                    description="Add explicit constraints to guide behavior",
                    suggested_segment="Important: [constraint]",
                    rationale="Constraints help prevent undesired outputs",
                    confidence=0.7,
                )
            )
        
        if not analysis.has_output_format:
            suggestions.append(
                ImprovementSuggestion(
                    improvement_type=ImprovementType.STRUCTURE,
                    description="Specify the desired output format",
                    suggested_segment="Format your response as: [format]",
                    rationale="Clear format specifications reduce parsing errors",
                    confidence=0.85,
                )
            )
        
        if analysis.word_count < 50:
            suggestions.append(
                ImprovementSuggestion(
                    improvement_type=ImprovementType.SPECIFICITY,
                    description="Consider adding more detail to your prompt",
                    rationale="Short prompts may lead to inconsistent results",
                    confidence=0.6,
                )
            )
        
        if analysis.word_count > 500:
            suggestions.append(
                ImprovementSuggestion(
                    improvement_type=ImprovementType.LENGTH,
                    description="Consider condensing the prompt for efficiency",
                    rationale="Very long prompts increase token costs and latency",
                    confidence=0.5,
                )
            )
        
        return suggestions


# =============================================================================
# A/B Testing
# =============================================================================


class ABTestRunner:
    """Runner for A/B testing prompts."""

    def __init__(self):
        self._tests: dict[str, ABTestConfig] = {}
        self._results: dict[str, ABTestResult] = {}

    def create_test(
        self,
        name: str,
        variant_a: PromptVersion,
        variant_b: PromptVersion,
        sample_size: int = 100,
        metrics: Optional[list[str]] = None,
    ) -> ABTestConfig:
        """Create a new A/B test."""
        config = ABTestConfig(
            name=name,
            variant_a=variant_a,
            variant_b=variant_b,
            sample_size=sample_size,
            metrics=metrics or ["accuracy", "latency"],
        )
        self._tests[name] = config
        return config

    def get_variant(
        self,
        test_name: str,
        sample_id: int,
    ) -> PromptVersion:
        """Get the variant to use for a given sample."""
        config = self._tests.get(test_name)
        if not config:
            raise ValueError(f"Test not found: {test_name}")
        
        # Simple alternating assignment
        if sample_id % 2 == 0:
            return config.variant_a
        return config.variant_b

    def record_result(
        self,
        test_name: str,
        variant: str,  # "a" or "b"
        metrics: dict[str, float],
    ) -> None:
        """Record a test result for a variant."""
        # In production, store results for statistical analysis
        pass

    def analyze(self, test_name: str) -> ABTestResult:
        """Analyze test results and determine winner."""
        config = self._tests.get(test_name)
        if not config:
            raise ValueError(f"Test not found: {test_name}")
        
        # Placeholder for actual statistical analysis
        # In production, use proper statistical tests
        result = ABTestResult(
            config=config,
            variant_a_metrics={"accuracy": 0.85, "latency": 150.0},
            variant_b_metrics={"accuracy": 0.82, "latency": 145.0},
            winner=ComparisonResult.VARIANT_A_BETTER,
            confidence=0.95,
            sample_count=config.sample_size,
        )
        
        self._results[test_name] = result
        return result

    def get_results(self, test_name: str) -> Optional[ABTestResult]:
        """Get test results."""
        return self._results.get(test_name)


# =============================================================================
# Prompt Tuner
# =============================================================================


class PromptTuner:
    """High-level prompt tuning interface."""

    def __init__(self):
        self.registry = PromptRegistry()
        self.analyzer = PromptAnalyzer()
        self.ab_runner = ABTestRunner()

    def create_prompt(
        self,
        name: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> PromptVersion:
        """Create and register a new prompt."""
        return self.registry.register(
            name=name,
            content=content,
            metadata=metadata,
        )

    def iterate(
        self,
        name: str,
        new_content: str,
        changes: str = "",
    ) -> PromptVersion:
        """Create a new iteration of an existing prompt."""
        current = self.registry.get_active(name)
        parent_version = current.version if current else None
        
        # Auto-generate version number
        versions = self.registry.list_versions(name)
        if versions:
            last = versions[-1].version
            # Parse version like "v1.0" -> "v2.0"
            match = re.match(r'v(\d+)\.(\d+)', last)
            if match:
                major = int(match.group(1))
                minor = int(match.group(2)) + 1
                new_version = f"v{major}.{minor}"
            else:
                new_version = f"v{len(versions) + 1}.0"
        else:
            new_version = "v1.0"
        
        return self.registry.register(
            name=name,
            content=new_content,
            version=new_version,
            parent_version=parent_version,
            changes=changes,
        )

    def analyze_prompt(self, content: str) -> PromptAnalysis:
        """Analyze a prompt and get suggestions."""
        return self.analyzer.analyze(content)

    def compare(
        self,
        name: str,
        version_a: str,
        version_b: str,
        sample_size: int = 100,
    ) -> ABTestConfig:
        """Set up an A/B comparison between versions."""
        prompt_a = self.registry.get(name, version_a)
        prompt_b = self.registry.get(name, version_b)
        
        if not prompt_a or not prompt_b:
            raise ValueError("Both versions must exist")
        
        return self.ab_runner.create_test(
            name=f"{name}_{version_a}_vs_{version_b}",
            variant_a=prompt_a,
            variant_b=prompt_b,
            sample_size=sample_size,
        )

    def promote(self, name: str, version: str) -> None:
        """Promote a version to active status."""
        self.registry.set_active(name, version)

    def get_history(self, name: str) -> list[dict[str, Any]]:
        """Get prompt version history."""
        return self.registry.get_history(name)

    def export(self) -> dict[str, Any]:
        """Export all prompts and their history."""
        data = {}
        for name in self.registry.list_prompts():
            versions = self.registry.list_versions(name)
            data[name] = {
                "active": self.registry._active.get(name),
                "versions": [
                    {
                        "version": v.version,
                        "content": v.content,
                        "status": v.status.value,
                        "created_at": v.created_at.isoformat(),
                        "changes": v.changes,
                        "hash": v.hash,
                    }
                    for v in versions
                ],
            }
        return data


# =============================================================================
# Convenience Functions
# =============================================================================


def create_tuner() -> PromptTuner:
    """Create a new prompt tuner instance."""
    return PromptTuner()


def analyze_prompt(content: str) -> PromptAnalysis:
    """Analyze a prompt and get improvement suggestions."""
    analyzer = PromptAnalyzer()
    return analyzer.analyze(content)


def create_template(
    template: str,
    variables: Optional[list[dict[str, Any]]] = None,
    name: str = "",
) -> PromptTemplate:
    """Create a prompt template."""
    vars_list = []
    if variables:
        for v in variables:
            vars_list.append(
                PromptVariable(
                    name=v["name"],
                    description=v.get("description", ""),
                    required=v.get("required", True),
                    default=v.get("default"),
                    validation_pattern=v.get("validation_pattern"),
                )
            )
    
    return PromptTemplate(
        template=template,
        variables=vars_list,
        name=name,
    )
