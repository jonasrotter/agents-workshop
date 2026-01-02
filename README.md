# Agentic AI Patterns Workshop

A hands-on workshop teaching 7 agentic AI patterns using Python, Azure OpenAI, and the Microsoft Agent Framework. Progress from simple agents through multi-agent orchestration, AG-UI interfaces, A2A interoperability, and evaluation-driven evolution.

## 🎯 Workshop Overview

This workshop consists of 7 interactive Jupyter Notebook scenarios, each teaching a specific agentic AI pattern:

| Scenario | Topic | Duration | What You'll Learn |
|----------|-------|----------|-------------------|
| [01](notebooks/01_simple_agent_mcp.ipynb) | Simple Agent + MCP | 30 min | Base agent pattern, MCP tools, OpenTelemetry observability |
| [02](notebooks/02_agui_interface.ipynb) | AG-UI Protocol | 45 min | Frontend integration with streaming events |
| [03](notebooks/03_a2a_protocol.ipynb) | A2A Interoperability | 45 min | Exposing agents for external consumption |
| [04](notebooks/04_deterministic_workflows.ipynb) | Workflows | 45 min | Multi-agent orchestration, sequential/parallel execution |
| [05](notebooks/05_declarative_agents.ipynb) | Declarative Agents | 30 min | YAML-based agent and workflow configuration |
| [06](notebooks/06_agent_discussions.ipynb) | Agent Discussions | 45 min | Moderated debates, turn-taking, conflict resolution |
| [07](notebooks/07_evaluation_evolution.ipynb) | Evaluation & Evolution | 45 min | Metrics, prompt tuning, A/B testing |

**Total Duration**: ~4-6 hours

## 📋 Prerequisites

### Required
- **Python 3.11+** installed
- **Azure OpenAI API** access (or OpenAI API key)
- **Azure Subscription** for Azure Monitor (observability features)
- Basic Python and async/await knowledge

### Recommended
- VS Code with Python and Jupyter extensions
- Familiarity with LLM concepts (prompts, tokens, completions)

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/agents-workshop.git
cd agents-workshop
```

### 2. Create Virtual Environment

```bash
# Using standard Python
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt"
```

### 3. Configure Environment

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# Azure Monitor (for observability)
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=xxx;IngestionEndpoint=https://...

# Optional: Model configuration
DEFAULT_MODEL=gpt-4o
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=4096
```

### 4. Verify Setup

Run the setup verification notebook:

```bash
jupyter notebook notebooks/00_setup.ipynb
```

Or run tests:

```bash
pytest tests/ -v
```

## 📁 Project Structure

```text
agents-workshop/
├── notebooks/                    # Interactive workshop scenarios
│   ├── 00_setup.ipynb           # Environment verification
│   ├── 01_simple_agent_mcp.ipynb
│   ├── 02_agui_interface.ipynb
│   ├── 03_a2a_protocol.ipynb
│   ├── 04_deterministic_workflows.ipynb
│   ├── 05_declarative_agents.ipynb
│   ├── 06_agent_discussions.ipynb
│   └── 07_evaluation_evolution.ipynb
├── src/
│   ├── common/                  # Shared utilities
│   │   ├── config.py            # Configuration management
│   │   ├── telemetry.py         # OpenTelemetry setup
│   │   ├── exceptions.py        # Custom exceptions
│   │   ├── evaluation.py        # Metrics collection
│   │   └── prompt_tuning.py     # Prompt iteration
│   ├── tools/                   # MCP tool implementations
│   │   ├── search_tool.py
│   │   ├── calculator_tool.py
│   │   └── file_tool.py
│   ├── agents/                  # Agent definitions
│   │   ├── base_agent.py
│   │   ├── research_agent.py
│   │   └── moderator_agent.py
│   ├── protocols/               # Protocol implementations
│   │   ├── agui.py             # AG-UI server
│   │   ├── a2a.py              # A2A protocol
│   │   └── discussion.py       # Discussion protocols
│   └── workflows/               # Workflow engine
│       └── engine.py
├── configs/
│   ├── agents/                  # Declarative agent configs
│   └── workflows/               # Declarative workflow configs
├── tests/
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── contract/                # Protocol contract tests
└── docs/                        # Documentation
```

## 🔧 Technology Stack

- **Python 3.11+** - Modern Python with strict type hints
- **Azure OpenAI SDK** - LLM integration
- **Microsoft Agent Framework** - Agent orchestration
- **FastAPI** - AG-UI and A2A servers
- **Pydantic v2** - Data validation
- **OpenTelemetry** - Distributed tracing
- **Azure Monitor** - Observability backend
- **pytest** - Testing framework


## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/unit/test_base_agent.py -v

# Run integration tests
pytest tests/integration/ -v
```

## 📖 Additional Resources

- [Architecture Documentation](docs/ARCHITECTURE.md)
- [Contributing Guide](docs/CONTRIBUTING.md)
- [Cost Guidance](docs/COST_GUIDANCE.md)
- [Azure OpenAI Documentation](https://learn.microsoft.com/azure/ai-services/openai/)
- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This workshop is for educational purposes. API costs may apply when using Azure OpenAI. See [Cost Guidance](docs/COST_GUIDANCE.md) for estimates.