# Agentic AI Patterns Workshop

A hands-on workshop teaching 7 agentic AI patterns using Python, Azure OpenAI, and the Microsoft Agent Framework. Progress from simple agents through multi-agent orchestration, AG-UI interfaces, A2A interoperability, and evaluation-driven evolution.

## 🎯 Workshop Overview

This workshop consists of 7 interactive Jupyter Notebook scenarios, each teaching a specific agentic AI pattern:

| Scenario | Topic | Duration | What You'll Learn |
|----------|-------|----------|-------------------|
| [01](notebooks/01_simple_agent_mcp.ipynb) | Simple Agent + MCP | 30 min | Base agent pattern, MCP tools, OpenTelemetry observability |
| [02](notebooks/02_agui_interface.ipynb) | AG-UI Protocol | 45 min | Frontend integration with streaming events |
| [03](notebooks/03_a2a_protocol.ipynb) | A2A Interoperability | 45 min | Exposing agents for external consumption |
| [04](notebooks/04_deterministic_workflows.ipynb) | Workflows | 45 min | Multi-agent orchestration with Microsoft Agent Framework |
| [05](notebooks/05_declarative_agents.ipynb) | Declarative Agents | 30 min | YAML-based agent and workflow configuration |
| [06](notebooks/06_agent_discussions.ipynb) | Agent Discussions | 45 min | Moderated debates, turn-taking, conflict resolution |
| [07](notebooks/07_evaluation_evolution.ipynb) | Evaluation & Evolution | 45 min | Metrics, prompt tuning, A/B testing |


## 📋 Prerequisites

### Required
- **Python 3.11+** (tested with Python 3.12)
- **Azure OpenAI** resource with deployed model (e.g., `gpt-4.1-mini`)
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
pip install -r requirements.txt
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
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini

# Authentication (choose one):
# Option 1: API Key (if enabled on your resource)
AZURE_OPENAI_API_KEY=your-api-key

# Option 2: Entra ID (Azure AD) - Recommended
# Leave AZURE_OPENAI_API_KEY empty and ensure you have:
# - Azure CLI logged in: az login
# - RBAC role: "Cognitive Services OpenAI User" on your resource

# Azure Monitor (for observability)
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=xxx;IngestionEndpoint=https://...
```

> **Note**: If your Azure OpenAI resource has `disableLocalAuth=true`, you must use Entra ID authentication. The notebooks automatically detect and use `DefaultAzureCredential` when no API key is provided.

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
│   │   ├── prompt_tuning.py     # Prompt iteration
│   │   └── yaml_loader.py       # YAML configuration loader
│   ├── tools/                   # MCP tool implementations
│   │   ├── mcp_server.py        # FastMCP server wrapper
│   │   ├── search_tool.py
│   │   ├── calculator_tool.py
│   │   └── file_tool.py
│   ├── agents/                  # Agent implementations
│   │   ├── base_agent.py        # Base agent with telemetry
│   │   ├── research_agent.py    # Research agent
│   │   ├── moderator_agent.py   # Discussion moderator
│   │   ├── declarative.py       # YAML-based agent loader
│   │   ├── discussion.py        # Multi-agent discussions
│   │   ├── agui_server.py       # AG-UI protocol server
│   │   ├── a2a_server.py        # A2A protocol server
│   │   └── factories.py         # Agent factory functions
│   └── workflows/               # Workflow engine (deprecated)
│       ├── engine.py            # ⚠️ Use WorkflowBuilder instead
│       ├── steps.py
│       └── builders.py
├── configs/
│   ├── agents/                  # Declarative agent configs
│   │   ├── research_agent.yaml
│   │   └── summarizer_agent.yaml
│   └── workflows/               # Declarative workflow configs
│       └── research_pipeline.yaml
├── tests/
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests (notebook scenarios)
│   └── contract/                # Protocol contract tests
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md
│   ├── CONTRIBUTING.md
│   └── COST_GUIDANCE.md
└── specs/                       # Feature specifications
```

> **Note**: The `src/workflows/` module is deprecated. Notebook 04 now uses Microsoft Agent Framework's `WorkflowBuilder` for workflow orchestration.

## 🔧 Technology Stack

| Package | Version | Purpose |
|---------|---------|---------|
| `agent-framework` | `1.0.0b251120` | Microsoft Agent Framework |
| `azure-ai-projects` | `1.0.0b11` | Azure AI Projects client |
| `openai` | `1.84.0` | OpenAI/Azure OpenAI SDK |
| `pydantic` | `2.11.5` | Data validation |
| `fastapi` | `0.115.12` | AG-UI and A2A servers |
| `opentelemetry-*` | `1.30+` | Distributed tracing |
| `azure-identity` | `1.23.0` | Entra ID authentication |

**Runtime Requirements:**
- Python 3.11+ (3.12 recommended)
- Azure OpenAI deployment with GPT-4o model
- Entra ID authentication (recommended) or API key


## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run unit tests
pytest tests/unit/ -v

# Run integration tests (notebook scenarios)
pytest tests/integration/ -v

# Run contract tests (protocol schemas)
pytest tests/contract/ -v
```


## 🔐 Authentication Options

The workshop supports two authentication methods for Azure OpenAI:

### Option 1: Entra ID Authentication (Recommended)

Uses Azure Identity for secure, token-based authentication. No API keys required.

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_ad_token_provider=token_provider,
    api_version="2025-01-01-preview",
)
```

**Required Azure Role**: `Cognitive Services OpenAI User` on your Azure OpenAI resource.

### Option 2: API Key Authentication

If your Azure OpenAI resource has `disableLocalAuth=false`:

```python
client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2025-01-01-preview",
)
```

> **Note**: Many enterprise Azure OpenAI resources have local authentication disabled (`disableLocalAuth=true`). In this case, you must use Entra ID authentication.

## 📖 Additional Resources

- [Architecture Documentation](docs/ARCHITECTURE.md)
- [Contributing Guide](docs/CONTRIBUTING.md)
- [Cost Guidance](docs/COST_GUIDANCE.md)
- [Azure OpenAI Documentation](https://learn.microsoft.com/azure/ai-services/openai/)
- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This workshop is for educational purposes. API costs may apply when using Azure OpenAI. See [Cost Guidance](docs/COST_GUIDANCE.md) for estimates.