"""
Agent Factory Functions for Thread-Safe Agent Creation.

This module provides factory functions for creating agents using the
Microsoft Agent Framework patterns. Each factory returns a new agent
instance configured for specific use cases.

Key patterns:
- Factory functions return fresh instances (thread-safe)
- Configuration via environment variables or Settings
- AzureOpenAIChatClient for Azure OpenAI integration
- ChatAgent for agent orchestration

Usage:
    from src.agents.factories import create_research_agent, create_summarizer_agent
    
    research_agent = create_research_agent()
    summarizer = create_summarizer_agent()
"""

import logging
from typing import Optional

from agent_framework import ChatAgent, ai_function
from agent_framework.azure import AzureOpenAIChatClient

from src.common.config import get_settings, Settings
from src.common.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


def _create_azure_client(
    settings: Optional[Settings] = None,
    deployment: Optional[str] = None,
) -> AzureOpenAIChatClient:
    """
    Create an AzureOpenAIChatClient from settings.
    
    Args:
        settings: Optional Settings object (uses get_settings() if not provided)
        deployment: Optional deployment name override
        
    Returns:
        Configured AzureOpenAIChatClient
    """
    settings = settings or get_settings()
    
    return AzureOpenAIChatClient(
        endpoint=settings.azure_openai_endpoint,
        deployment=deployment or settings.azure_openai_deployment,
        api_version=settings.azure_openai_api_version,
    )


def create_research_agent(
    name: str = "research_agent",
    instructions: Optional[str] = None,
    temperature: float = 0.7,
    settings: Optional[Settings] = None,
) -> ChatAgent:
    """
    Create a research agent for gathering and analyzing information.
    
    The research agent is optimized for:
    - Web search and information retrieval
    - Document analysis
    - Data synthesis
    
    Args:
        name: Agent identifier
        instructions: Custom system prompt (uses default if not provided)
        temperature: Sampling temperature (0-2)
        settings: Optional Settings object
        
    Returns:
        Configured ChatAgent for research tasks
        
    Example:
        agent = create_research_agent()
        result = await agent.run("Research the latest trends in AI agents")
    """
    default_instructions = """You are a Research Agent specialized in gathering, analyzing, 
and synthesizing information. Your responsibilities include:

1. Searching for relevant information on given topics
2. Analyzing and validating sources
3. Synthesizing findings into clear, structured reports
4. Citing sources and providing evidence

Always be thorough, accurate, and objective in your research."""

    client = _create_azure_client(settings)
    
    return ChatAgent(
        name=name,
        instructions=instructions or default_instructions,
        client=client,
        model_settings={"temperature": temperature},
    )


def create_summarizer_agent(
    name: str = "summarizer_agent",
    instructions: Optional[str] = None,
    temperature: float = 0.5,
    settings: Optional[Settings] = None,
) -> ChatAgent:
    """
    Create a summarizer agent for condensing information.
    
    The summarizer agent is optimized for:
    - Text summarization
    - Key point extraction
    - Executive summaries
    
    Args:
        name: Agent identifier
        instructions: Custom system prompt (uses default if not provided)
        temperature: Sampling temperature (0-2)
        settings: Optional Settings object
        
    Returns:
        Configured ChatAgent for summarization tasks
        
    Example:
        agent = create_summarizer_agent()
        summary = await agent.run(f"Summarize the following: {long_text}")
    """
    default_instructions = """You are a Summarization Agent specialized in condensing 
information into clear, concise summaries. Your responsibilities include:

1. Identifying key points and main ideas
2. Removing redundant information
3. Preserving essential details and context
4. Creating structured, readable summaries

Always maintain accuracy while maximizing clarity and brevity."""

    client = _create_azure_client(settings)
    
    return ChatAgent(
        name=name,
        instructions=instructions or default_instructions,
        client=client,
        model_settings={"temperature": temperature},
    )


def create_analysis_agent(
    name: str = "analysis_agent",
    instructions: Optional[str] = None,
    temperature: float = 0.3,
    settings: Optional[Settings] = None,
) -> ChatAgent:
    """
    Create an analysis agent for critical evaluation.
    
    The analysis agent is optimized for:
    - Critical analysis
    - Logical reasoning
    - Pattern recognition
    - Comparative analysis
    
    Args:
        name: Agent identifier
        instructions: Custom system prompt (uses default if not provided)
        temperature: Lower temperature for more deterministic analysis
        settings: Optional Settings object
        
    Returns:
        Configured ChatAgent for analysis tasks
    """
    default_instructions = """You are an Analysis Agent specialized in critical 
evaluation and logical reasoning. Your responsibilities include:

1. Breaking down complex problems into components
2. Identifying patterns and relationships
3. Evaluating evidence and arguments
4. Providing balanced, well-reasoned conclusions

Always be systematic, objective, and thorough in your analysis."""

    client = _create_azure_client(settings)
    
    return ChatAgent(
        name=name,
        instructions=instructions or default_instructions,
        client=client,
        model_settings={"temperature": temperature},
    )


def create_coordinator_agent(
    name: str = "coordinator_agent",
    instructions: Optional[str] = None,
    temperature: float = 0.5,
    settings: Optional[Settings] = None,
) -> ChatAgent:
    """
    Create a coordinator agent for orchestrating multi-agent workflows.
    
    The coordinator agent is optimized for:
    - Task delegation
    - Workflow management
    - Result aggregation
    - Decision making
    
    Args:
        name: Agent identifier
        instructions: Custom system prompt (uses default if not provided)
        temperature: Sampling temperature (0-2)
        settings: Optional Settings object
        
    Returns:
        Configured ChatAgent for coordination tasks
    """
    default_instructions = """You are a Coordinator Agent specialized in orchestrating 
multi-agent workflows. Your responsibilities include:

1. Delegating tasks to specialized agents
2. Managing workflow execution
3. Aggregating and synthesizing results
4. Making decisions about next steps

Always coordinate efficiently and ensure quality outcomes."""

    client = _create_azure_client(settings)
    
    return ChatAgent(
        name=name,
        instructions=instructions or default_instructions,
        client=client,
        model_settings={"temperature": temperature},
    )


def create_custom_agent(
    name: str,
    instructions: str,
    temperature: float = 0.7,
    deployment: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> ChatAgent:
    """
    Create a custom agent with specified configuration.
    
    This is the most flexible factory function, allowing full
    customization of the agent behavior.
    
    Args:
        name: Agent identifier
        instructions: System prompt defining agent behavior
        temperature: Sampling temperature (0-2)
        deployment: Optional specific Azure OpenAI deployment
        settings: Optional Settings object
        
    Returns:
        Configured ChatAgent
        
    Example:
        agent = create_custom_agent(
            name="creative_writer",
            instructions="You are a creative writing assistant...",
            temperature=0.9
        )
    """
    client = _create_azure_client(settings, deployment)
    
    return ChatAgent(
        name=name,
        instructions=instructions,
        client=client,
        model_settings={"temperature": temperature},
    )


# Registry of available factory functions
AGENT_FACTORIES = {
    "research": create_research_agent,
    "summarizer": create_summarizer_agent,
    "analysis": create_analysis_agent,
    "coordinator": create_coordinator_agent,
}


def get_agent_factory(agent_type: str):
    """
    Get a factory function by agent type.
    
    Args:
        agent_type: Type of agent ("research", "summarizer", "analysis", "coordinator")
        
    Returns:
        Factory function for the specified agent type
        
    Raises:
        ValueError: If agent type is not recognized
    """
    if agent_type not in AGENT_FACTORIES:
        available = ", ".join(AGENT_FACTORIES.keys())
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {available}")
    
    return AGENT_FACTORIES[agent_type]
