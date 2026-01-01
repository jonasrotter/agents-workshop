# Cost Guidance for Workshop

This document provides estimates for API costs when running the workshop scenarios.

## Overview

The workshop uses Azure OpenAI APIs which are billed based on token usage. This guide helps you estimate and manage costs.

## Model Pricing (As of 2024)

| Model | Input (per 1K tokens) | Output (per 1K tokens) |
|-------|----------------------|------------------------|
| gpt-4o | $0.005 | $0.015 |
| gpt-4o-mini | $0.00015 | $0.0006 |
| gpt-4-turbo | $0.01 | $0.03 |
| gpt-3.5-turbo | $0.0005 | $0.0015 |

**Recommendation**: Use `gpt-4o-mini` during learning to minimize costs.

## Cost Estimates by Scenario

### Scenario 01: Simple Agent + MCP
**Estimated Cost**: $0.05 - $0.15

| Activity | Tokens (avg) | Cost (gpt-4o-mini) |
|----------|-------------|-------------------|
| Basic agent calls | ~2,000 | $0.001 |
| Tool usage | ~3,000 | $0.002 |
| Exercise completion | ~5,000 | $0.003 |
| **Total** | ~10,000 | **~$0.01** |

### Scenario 02: AG-UI Interface
**Estimated Cost**: $0.10 - $0.25

| Activity | Tokens (avg) | Cost (gpt-4o-mini) |
|----------|-------------|-------------------|
| Streaming demo | ~5,000 | $0.003 |
| Event handling | ~8,000 | $0.005 |
| Exercise | ~10,000 | $0.006 |
| **Total** | ~23,000 | **~$0.02** |

### Scenario 03: A2A Protocol
**Estimated Cost**: $0.10 - $0.30

| Activity | Tokens (avg) | Cost (gpt-4o-mini) |
|----------|-------------|-------------------|
| Agent card setup | ~3,000 | $0.002 |
| Cross-agent calls | ~10,000 | $0.006 |
| Exercise | ~12,000 | $0.007 |
| **Total** | ~25,000 | **~$0.02** |

### Scenario 04: Workflows
**Estimated Cost**: $0.20 - $0.50

| Activity | Tokens (avg) | Cost (gpt-4o-mini) |
|----------|-------------|-------------------|
| Sequential workflows | ~15,000 | $0.009 |
| Parallel workflows | ~20,000 | $0.012 |
| Exercise | ~15,000 | $0.009 |
| **Total** | ~50,000 | **~$0.04** |

### Scenario 05: Declarative Agents
**Estimated Cost**: $0.10 - $0.25

| Activity | Tokens (avg) | Cost (gpt-4o-mini) |
|----------|-------------|-------------------|
| YAML config loading | ~5,000 | $0.003 |
| Declarative workflows | ~15,000 | $0.009 |
| Exercise | ~8,000 | $0.005 |
| **Total** | ~28,000 | **~$0.02** |

### Scenario 06: Agent Discussions
**Estimated Cost**: $0.30 - $0.75

| Activity | Tokens (avg) | Cost (gpt-4o-mini) |
|----------|-------------|-------------------|
| Debate setup | ~10,000 | $0.006 |
| Multi-turn discussions | ~40,000 | $0.024 |
| Conflict resolution | ~15,000 | $0.009 |
| Exercise | ~20,000 | $0.012 |
| **Total** | ~85,000 | **~$0.06** |

### Scenario 07: Evaluation
**Estimated Cost**: $0.15 - $0.40

| Activity | Tokens (avg) | Cost (gpt-4o-mini) |
|----------|-------------|-------------------|
| Metrics collection | ~10,000 | $0.006 |
| A/B testing | ~20,000 | $0.012 |
| Exercise | ~10,000 | $0.006 |
| **Total** | ~40,000 | **~$0.03** |

## Total Workshop Cost

| Model | Estimated Total |
|-------|-----------------|
| gpt-4o-mini | **$0.20 - $0.50** |
| gpt-4o | **$2.00 - $5.00** |
| gpt-4-turbo | **$4.00 - $10.00** |

## Cost Optimization Tips

### 1. Use gpt-4o-mini for Learning

Set in your `.env`:
```env
DEFAULT_MODEL=gpt-4o-mini
```

### 2. Limit Max Tokens

Reduce max tokens for exercises:
```python
response = await agent.execute(
    prompt="Your prompt",
    max_tokens=500,  # Limit output
)
```

### 3. Cache Responses

During development, cache API responses:
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_completion(prompt_hash: str) -> str:
    # Only call API if not cached
    pass
```

### 4. Use Mock Responses

For testing without API calls:
```python
from unittest.mock import AsyncMock

agent.llm_client = AsyncMock(return_value="Mocked response")
```

### 5. Monitor Usage

Track costs in real-time:
```python
from src.common.evaluation import MetricsCollector

collector = MetricsCollector()
cost = collector.record_cost(
    operation="my_operation",
    input_tokens=100,
    output_tokens=50,
    model="gpt-4o-mini",
)
print(f"This call cost: ${cost.cost_usd:.4f}")
```

## Azure Monitor Costs

If using Azure Monitor for observability:

| Feature | Monthly Cost |
|---------|-------------|
| Basic tier (5 GB included) | Free |
| Data ingestion (per GB) | ~$2.30 |
| Data retention (31 days) | Included |

**Workshop Estimate**: < 1 GB data → **Free tier sufficient**

## Azure Pricing Tiers

### For Individual Learning

- **Pay-As-You-Go**: Best for workshop
- No minimum commitment
- Only pay for usage

### For Team Workshops

- **Reserved Capacity**: Discount for consistent usage
- **PTU (Provisioned Throughput)**: For high-volume training

## Cost Monitoring

### Azure Portal

1. Go to Azure Portal → Cost Management
2. Set up budget alerts at $5, $10, $25
3. Review daily cost breakdowns

### Programmatic Tracking

```python
# Add to notebook start
import os
from src.common.evaluation import MetricsCollector

# Initialize global collector
cost_tracker = MetricsCollector()

# After each major operation
cost = cost_tracker.record_cost(
    operation="scenario_01",
    input_tokens=input_usage,
    output_tokens=output_usage,
    model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
)

# At notebook end
print(f"Total scenario cost: ${sum_costs:.2f}")
```

## Troubleshooting High Costs

### Symptoms
- Unexpectedly high token usage
- Runaway API calls
- Infinite loops

### Solutions

1. **Add timeout limits**:
```python
import asyncio

async with asyncio.timeout(30):
    result = await agent.execute(prompt)
```

2. **Set max iterations**:
```python
MAX_TOOL_CALLS = 10
for i in range(MAX_TOOL_CALLS):
    if done:
        break
```

3. **Review prompts**: Long system prompts add up

4. **Check for loops**: Ensure exit conditions

## Summary

| Scenario | Duration | Est. Cost (mini) |
|----------|----------|------------------|
| 01 - MCP | 30 min | $0.01 |
| 02 - AG-UI | 45 min | $0.02 |
| 03 - A2A | 45 min | $0.02 |
| 04 - Workflows | 45 min | $0.04 |
| 05 - Declarative | 30 min | $0.02 |
| 06 - Discussions | 45 min | $0.06 |
| 07 - Evaluation | 45 min | $0.03 |
| **Total** | ~5 hours | **~$0.20** |

**Bottom Line**: The complete workshop costs approximately **$0.20-$0.50** using gpt-4o-mini, making it very accessible for learning.
