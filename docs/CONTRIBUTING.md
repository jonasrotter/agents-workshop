# Contributing to the Agentic AI Patterns Workshop

Thank you for your interest in contributing to this workshop! This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Adding New Scenarios](#adding-new-scenarios)
- [Adding New Tools](#adding-new-tools)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Please read and follow it in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Set up the development environment (see below)
4. Create a feature branch
5. Make your changes
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.11+
- uv (recommended) or pip
- Azure OpenAI access for testing

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/agents-workshop.git
cd agents-workshop

# Create virtual environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install with development dependencies
uv pip install -e ".[dev]"

# Copy environment file
cp .env.example .env
# Edit .env with your credentials
```

### Verify Setup

```bash
# Run tests
pytest

# Run linting
ruff check src/ tests/

# Run type checking
mypy src/
```

## How to Contribute

### Types of Contributions

1. **Bug Fixes**: Fix issues in existing scenarios
2. **New Scenarios**: Add new workshop modules
3. **New Tools**: Implement additional MCP tools
4. **Documentation**: Improve guides and examples
5. **Tests**: Increase test coverage
6. **Accessibility**: Improve workshop accessibility

### Finding Work

- Check [Issues](../../issues) for open tasks
- Look for `good first issue` labels
- Review `help wanted` labels
- Propose new features via issues first

## Coding Standards

### Python Style

We follow strict coding standards for educational clarity:

```python
# ✅ Good: Clear, typed, documented
def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o",
) -> float:
    """Calculate API cost for given token usage.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model identifier (default: gpt-4o)
        
    Returns:
        Estimated cost in USD
    """
    rates = {"gpt-4o": (0.005, 0.015), "gpt-4o-mini": (0.00015, 0.0006)}
    input_rate, output_rate = rates.get(model, rates["gpt-4o"])
    return (input_tokens * input_rate + output_tokens * output_rate) / 1000


# ❌ Bad: No types, no docs, unclear
def calc(i, o, m="gpt-4o"):
    r = {"gpt-4o": (0.005, 0.015)}
    return (i * r[m][0] + o * r[m][1]) / 1000
```

### Key Requirements

1. **Type Hints**: All functions must have complete type annotations
2. **Docstrings**: All public functions/classes need docstrings
3. **Function Length**: Maximum 30 lines per function
4. **Imports**: Use `from __future__ import annotations`
5. **Async**: Use `async/await` for I/O operations

### Formatting

```bash
# Format code
ruff format src/ tests/

# Check formatting
ruff format --check src/ tests/
```

## Adding New Scenarios

### Scenario Structure

Each scenario consists of:

1. **Notebook**: `notebooks/XX_scenario_name.ipynb`
2. **Support Module**: `src/<area>/module.py` (if needed)
3. **Tests**: `tests/unit/` and `tests/integration/`
4. **Documentation**: Update README and ARCHITECTURE

### Notebook Template

```python
# Cell 1: Markdown Header
"""
# Scenario XX: Your Scenario Name

**Estimated Time**: XX minutes

## Learning Objectives
- Objective 1
- Objective 2
- Objective 3

## Prerequisites
- Completed Scenario 01
- Additional prereqs
"""

# Cell 2: Imports
from __future__ import annotations

from typing import Any
# ... other imports

# Cell 3-N: Scenario Content
# Each section should have:
# - Markdown explanation
# - Code cells with comments
# - Output demonstration

# Final Cell: Exercise
"""
## 🎯 Exercise: Hands-On Task

Your turn! Complete the following:

1. Task step 1
2. Task step 2
3. Task step 3

**Hint**: Consider using...
"""
```

### Scenario Checklist

- [ ] Clear learning objectives
- [ ] Estimated completion time
- [ ] Progressive difficulty
- [ ] Hands-on exercise
- [ ] Connection to other scenarios
- [ ] Test coverage

## Adding New Tools

### Tool Structure

MCP tools follow this pattern:

```python
# src/tools/your_tool.py
"""Your tool description."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class YourToolConfig:
    """Configuration for your tool."""
    
    setting: str = "default"


class YourTool:
    """Tool that does something useful.
    
    Attributes:
        config: Tool configuration
        
    Example:
        >>> tool = YourTool()
        >>> result = tool.execute({"param": "value"})
    """
    
    def __init__(self, config: YourToolConfig | None = None) -> None:
        """Initialize the tool."""
        self.config = config or YourToolConfig()
    
    def get_schema(self) -> dict[str, Any]:
        """Return MCP tool schema."""
        return {
            "name": "your_tool",
            "description": "Does something useful",
            "parameters": {
                "type": "object",
                "properties": {
                    "param": {
                        "type": "string",
                        "description": "Parameter description"
                    }
                },
                "required": ["param"]
            }
        }
    
    def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool with given parameters."""
        # Implementation
        return {"result": "success"}
```

### Tool Checklist

- [ ] MCP-compliant schema
- [ ] Type hints throughout
- [ ] Comprehensive docstrings
- [ ] Error handling
- [ ] Unit tests
- [ ] Integration with at least one scenario

## Testing Guidelines

### Test Structure

```python
# tests/unit/test_your_module.py
"""Unit tests for your module."""

from __future__ import annotations

import pytest

from src.your_area.your_module import YourClass


class TestYourClass:
    """Tests for YourClass."""
    
    @pytest.fixture
    def instance(self) -> YourClass:
        """Create test instance."""
        return YourClass()
    
    def test_basic_functionality(self, instance: YourClass) -> None:
        """Test basic operation."""
        result = instance.method()
        assert result == expected
    
    def test_edge_case(self, instance: YourClass) -> None:
        """Test edge case handling."""
        with pytest.raises(ValueError):
            instance.method(invalid_input)
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific module
pytest tests/unit/test_your_module.py -v

# Only integration tests
pytest tests/integration/ -v
```

### Coverage Requirements

- New modules: Minimum 80% coverage
- Critical paths: 100% coverage
- Integration tests for each scenario

## Documentation

### Code Documentation

- All public APIs need docstrings
- Use Google-style docstrings
- Include examples where helpful

### README Updates

When adding features, update:
- Feature list in main README
- Duration estimates if adding scenarios
- Prerequisites if adding dependencies

### Architecture Updates

Update `docs/ARCHITECTURE.md` when:
- Adding new modules
- Changing data flow
- Adding new protocols

## Pull Request Process

### Before Submitting

1. Run all tests: `pytest`
2. Run linting: `ruff check src/ tests/`
3. Run type checking: `mypy src/`
4. Update documentation
5. Add/update tests

### PR Description Template

```markdown
## Summary
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Tests

## Checklist
- [ ] Tests pass locally
- [ ] Linting passes
- [ ] Documentation updated
- [ ] Follows coding standards

## Related Issues
Fixes #123
```

### Review Process

1. Automated checks must pass
2. At least one maintainer review
3. Address all feedback
4. Squash commits if needed

## Questions?

- Open an issue for questions
- Join discussions for broader topics
- Tag maintainers for urgent issues

Thank you for contributing!
