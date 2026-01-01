# Quickstart: Agentic AI Patterns Workshop

**Estimated Setup Time**: 15-20 minutes

This guide walks you through setting up your environment for the workshop.

---

## Prerequisites

### Required

- **Python 3.11+**: Download from [python.org](https://www.python.org/downloads/)
- **Git**: For cloning the repository
- **Azure Subscription**: With access to Azure OpenAI Service
- **Azure CLI**: For authentication ([Install Guide](https://learn.microsoft.com/cli/azure/install-azure-cli))

### Recommended

- **VS Code**: With Python and Jupyter extensions
- **uv**: Fast Python package manager ([Install Guide](https://docs.astral.sh/uv/getting-started/installation/))

---

## Step 1: Clone and Setup Repository

```bash
# Clone the repository
git clone https://github.com/your-org/agents-workshop.git
cd agents-workshop

# Create virtual environment (using uv)
uv venv --python 3.11
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install dependencies
uv sync
```

### Alternative: Using pip

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

pip install -e ".[dev]"
```

---

## Step 2: Azure OpenAI Setup

### 2.1 Create Azure OpenAI Resource

1. Go to [Azure Portal](https://portal.azure.com)
2. Create a new **Azure OpenAI** resource
3. Wait for deployment to complete

### 2.2 Deploy Models

Deploy these models in Azure AI Studio:

| Model | Deployment Name | Purpose |
|-------|-----------------|---------|
| gpt-4o | gpt-4o | Main agent model |
| gpt-4o-mini | gpt-4o-mini | Fast responses |
| text-embedding-ada-002 | text-embedding | Embeddings (optional) |

### 2.3 Get Endpoint and Keys

From your Azure OpenAI resource:
- **Endpoint**: `https://<your-resource>.openai.azure.com/`
- **API Key**: Found under "Keys and Endpoint"

---

## Step 3: Azure Monitor Setup (Observability)

### 3.1 Create Application Insights

1. In Azure Portal, create **Application Insights** resource
2. Select your subscription and resource group
3. Choose **Workspace-based** mode
4. Note the **Connection String**

### 3.2 Get Connection String

From your Application Insights resource:
- Navigate to **Overview** > **Connection String**
- Copy the full connection string

---

## Step 4: Configure Environment

Create a `.env` file in the repository root:

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Model Deployments
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME=gpt-4o

# Azure Monitor (Observability)
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=xxx;IngestionEndpoint=xxx

# Optional: OpenAI (for comparison)
# OPENAI_API_KEY=sk-...
# OPENAI_CHAT_MODEL_ID=gpt-4o
```

### Alternative: Azure CLI Authentication

If you prefer Azure CLI authentication (no API key needed):

```bash
# Login to Azure
az login

# Set subscription (if you have multiple)
az account set --subscription "Your Subscription Name"
```

Then in code, use `AzureCliCredential`:

```python
from azure.identity import AzureCliCredential
credential = AzureCliCredential()
```

---

## Step 5: Verify Installation

Run the verification notebook:

```bash
# Start Jupyter
jupyter lab

# Open notebooks/00_setup.ipynb
```

Or run the verification script:

```bash
python -c "
import os
from dotenv import load_dotenv
load_dotenv()

# Check environment variables
required = [
    'AZURE_OPENAI_ENDPOINT',
    'AZURE_OPENAI_API_KEY',
    'APPLICATIONINSIGHTS_CONNECTION_STRING'
]

missing = [v for v in required if not os.getenv(v)]
if missing:
    print(f'❌ Missing: {missing}')
else:
    print('✅ All environment variables set')

# Check imports
try:
    from agent_framework import ChatAgent
    from agent_framework.azure import AzureOpenAIChatClient
    print('✅ Agent Framework imported')
except ImportError as e:
    print(f'❌ Import error: {e}')

try:
    from mcp.server.fastmcp import FastMCP
    print('✅ MCP SDK imported')
except ImportError as e:
    print(f'❌ Import error: {e}')

try:
    from opentelemetry import trace
    print('✅ OpenTelemetry imported')
except ImportError as e:
    print(f'❌ Import error: {e}')
"
```

---

## Step 6: Test Azure OpenAI Connection

```python
import asyncio
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential

async def test_connection():
    agent = AzureOpenAIResponsesClient(
        credential=AzureCliCredential()
    ).create_agent(
        name="TestAgent",
        instructions="You are a helpful assistant."
    )
    
    result = await agent.run("Say hello!")
    print(f"Agent response: {result}")

asyncio.run(test_connection())
```

Expected output:
```
Agent response: Hello! How can I help you today?
```

---

## Project Structure After Setup

```
agents-workshop/
├── .env                       # Your environment configuration
├── .venv/                     # Python virtual environment
├── pyproject.toml             # Project dependencies
├── src/
│   ├── common/                # Shared utilities
│   ├── tools/                 # MCP tools
│   ├── agents/                # Agent definitions
│   └── workflows/             # Workflow engine
├── notebooks/
│   ├── 00_setup.ipynb         # ← Start here!
│   ├── 01_simple_agent_mcp.ipynb
│   └── ...
├── configs/
│   ├── agents/                # Declarative agent configs
│   └── workflows/             # Workflow definitions
└── tests/
```

---

## Troubleshooting

### "Azure OpenAI resource not found"

- Verify your endpoint URL is correct
- Check the resource is deployed and accessible
- Ensure you have the right subscription selected

### "Model deployment not found"

- Verify the deployment name matches exactly
- Check the model is deployed in Azure AI Studio
- Wait a few minutes after deployment

### "Authentication failed"

For API key auth:
- Verify the API key is correct
- Check it hasn't been rotated

For Azure CLI auth:
- Run `az login` again
- Check `az account show` shows correct subscription

### "Application Insights not logging"

- Verify the connection string is correct
- Check network connectivity to Azure
- Wait 2-5 minutes for telemetry to appear

### Import errors

```bash
# Reinstall dependencies
uv sync --force-reinstall

# Or with pip
pip install -e ".[dev]" --force-reinstall
```

---

## Next Steps

1. Open `notebooks/00_setup.ipynb` to complete setup verification
2. Proceed to `notebooks/01_simple_agent_mcp.ipynb` for your first agent
3. Follow the workshop modules in order (prerequisites are listed in each notebook)

---

## Resource Links

- [Microsoft Agent Framework Docs](https://learn.microsoft.com/agent-framework)
- [Azure OpenAI Service](https://learn.microsoft.com/azure/ai-services/openai/)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [AG-UI Protocol Docs](https://docs.ag-ui.com)
- [A2A Protocol Specification](https://google.github.io/A2A)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
- [Azure Monitor OpenTelemetry](https://learn.microsoft.com/azure/azure-monitor/app/opentelemetry-python)
