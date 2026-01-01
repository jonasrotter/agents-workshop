<!--
  Sync Impact Report
  ==================
  Version change: 0.0.0 → 1.0.0 (Initial constitution)
  Modified principles: N/A (new document)
  Added sections: Core Principles (5), Technology Stack, Development Workflow, Governance
  Removed sections: N/A
  Templates requiring updates:
    - .specify/templates/plan-template.md ✅ (aligned with Constitution Check section)
    - .specify/templates/spec-template.md ✅ (requirements structure compatible)
    - .specify/templates/tasks-template.md ✅ (phase structure compatible)
  Follow-up TODOs: None
-->

# Agents Workshop Constitution

## Core Principles

### I. Type Safety & Static Analysis (NON-NEGOTIABLE)

All Python code MUST leverage type hints and static analysis to catch errors before runtime:

- All function signatures MUST include complete type annotations (parameters and return types)
- All class attributes MUST be typed using dataclasses, Pydantic models, or explicit type annotations
- `mypy` (strict mode) or `pyright` MUST pass with zero errors before merge
- Use `typing` module constructs: `Optional`, `Union`, `TypeVar`, `Generic`, `Protocol` as appropriate
- Avoid `Any` type except when interfacing with untyped external libraries (must be documented)

**Rationale**: Type safety prevents entire categories of runtime errors, improves IDE support, and serves as executable documentation.

### II. Test-First Development (NON-NEGOTIABLE)

Test-Driven Development (TDD) is mandatory for all feature work:

- Tests MUST be written before implementation code
- Tests MUST fail initially (Red phase), then pass after implementation (Green phase)
- Refactoring occurs only with passing tests (Refactor phase)
- Minimum test coverage: 80% for new code, measured by `pytest-cov`
- Test categories:
  - **Unit tests**: Isolated function/class behavior (`tests/unit/`)
  - **Integration tests**: Component interactions (`tests/integration/`)
  - **Contract tests**: API/interface compliance (`tests/contract/`)

**Rationale**: TDD ensures code correctness, enables safe refactoring, and produces testable designs by default.

### III. Clean Code & SOLID Principles

Code MUST adhere to Python best practices and SOLID principles:

- **Single Responsibility**: Each module, class, and function has one clear purpose
- **Open/Closed**: Classes open for extension, closed for modification (use composition, protocols)
- **Liskov Substitution**: Subtypes MUST be substitutable for their base types
- **Interface Segregation**: Prefer small, focused protocols over large interfaces
- **Dependency Inversion**: Depend on abstractions (protocols), not concretions

Additional standards:
- Functions MUST be ≤ 30 lines (excluding docstrings); longer requires justification
- Cyclomatic complexity MUST be ≤ 10 per function
- All public APIs MUST have docstrings (Google or NumPy style, consistently applied)
- Use `ruff` for linting and `ruff format` (or `black`) for formatting
- Imports MUST be sorted (`isort` compatible) and grouped: stdlib → third-party → local

**Rationale**: Clean code reduces cognitive load, simplifies maintenance, and enables team scalability.

### IV. Dependency Management & Reproducibility

All dependencies MUST be explicitly declared and version-pinned:

- Use `pyproject.toml` as the single source of truth for project metadata and dependencies
- Production dependencies in `[project.dependencies]`
- Development dependencies in `[project.optional-dependencies.dev]`
- Use `uv`, `pip-tools`, or `poetry` for deterministic lock files
- Pin exact versions in lock files; use compatible ranges in `pyproject.toml`
- Virtual environments MUST be used; never install to system Python
- Document Python version requirement (minimum 3.11 for this project)

**Rationale**: Reproducible environments prevent "works on my machine" issues and ensure consistent CI/CD behavior.

### V. Observability & Error Handling

Production-ready code MUST be observable and handle errors gracefully:

- Use structured logging via `structlog` or `logging` with JSON formatter
- Log levels: DEBUG for development, INFO for operations, WARNING/ERROR for issues
- All exceptions MUST be caught at appropriate boundaries; no silent failures
- Custom exceptions MUST inherit from a project-specific base exception
- Include correlation IDs for request tracing in distributed systems
- Metrics and health checks MUST be exposed for any long-running services

**Rationale**: Observability enables rapid debugging, performance analysis, and production incident response.

## Technology Stack

This constitution applies to Python projects with the following baseline:

| Concern | Tool/Library | Notes |
|---------|--------------|-------|
| Python Version | 3.11+ | Required for modern typing features |
| Package Management | `uv` or `pip-tools` | Lock file generation |
| Project Config | `pyproject.toml` | PEP 621 compliant |
| Testing | `pytest` + `pytest-cov` + `pytest-asyncio` | Coverage required |
| Type Checking | `mypy` (strict) or `pyright` | Zero errors policy |
| Linting/Formatting | `ruff` | Replaces flake8, isort, black |
| Pre-commit | `pre-commit` | Enforce quality gates locally |
| Documentation | `mkdocs` or `sphinx` | For public APIs |
| Logging | `structlog` | Structured JSON logs |
| CLI | `typer` or `click` | When CLI interfaces needed |
| HTTP/API | `httpx` (client), `fastapi` (server) | Async-first |

## Development Workflow

### Quality Gates (Pre-Merge Checklist)

All code changes MUST pass these gates before merge:

1. **Formatting**: `ruff format --check` passes
2. **Linting**: `ruff check` passes with zero errors/warnings
3. **Type Check**: `mypy --strict` or `pyright` passes
4. **Tests**: `pytest` passes with ≥80% coverage on changed files
5. **Documentation**: All public APIs have docstrings; README updated if needed
6. **Commit Messages**: Follow Conventional Commits format (`feat:`, `fix:`, `docs:`, etc.)

### Code Review Standards

- All PRs require at least one approval
- Reviewers MUST verify constitution compliance
- Complexity additions require explicit justification in PR description
- Breaking changes require migration documentation

### Branch Strategy

- `main`: Protected, always deployable
- `feature/*`: Feature development branches
- `fix/*`: Bug fix branches
- `docs/*`: Documentation-only changes

## Governance

This constitution supersedes all other development practices for this project:

- **Authority**: All PRs and code reviews MUST verify constitution compliance
- **Amendments**: Changes to this constitution require:
  1. Written proposal with rationale
  2. Review period (minimum 1 week for significant changes)
  3. Version bump following semantic versioning:
     - MAJOR: Principle removal or backward-incompatible changes
     - MINOR: New principles or significant expansions
     - PATCH: Clarifications, typo fixes, non-semantic updates
  4. Migration plan for existing code (if applicable)
- **Exceptions**: Temporary exceptions MUST be documented with TODO comments and tracked as technical debt
- **Enforcement**: Pre-commit hooks and CI pipelines MUST enforce measurable rules

**Version**: 1.0.0 | **Ratified**: 2026-01-01 | **Last Amended**: 2026-01-01
