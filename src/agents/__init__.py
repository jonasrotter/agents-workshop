"""Agents module for the Agentic AI Patterns Workshop.

This module exports agent implementations:
- BaseAgent: Foundation for all agents with telemetry
- ResearchAgent: Agent for web research tasks
- AGUIServer: AG-UI protocol server for streaming chat
- AGUIEventEmitter: Event emitter for AG-UI streaming
- A2AServer: A2A protocol server for agent interoperability
- create_agui_server: Factory function for AG-UI server
- create_a2a_server: Factory function for A2A server
- DeclarativeAgent: Agent loaded from YAML (legacy format)
- DeclarativeAgentLoader: Load agents from YAML (legacy format)
- AgentFactoryLoader: Load agents using AgentFactory (new format)
- load_agent_from_yaml: Load single agent using AgentFactory

Agent Factories (thread-safe):
- create_research_agent: Factory for research agents
- create_summarizer_agent: Factory for summarization agents
- create_analysis_agent: Factory for analysis agents
- create_coordinator_agent: Factory for coordinator agents
- create_custom_agent: Factory for custom agents

Example:
    from src.agents import ResearchAgent, AGUIServer, A2AServer

    agent = ResearchAgent(name="researcher")
    
    # For streaming UI
    agui = AGUIServer(agent=agent)
    agui_app = agui.create_app()
    
    # For agent-to-agent
    a2a = A2AServer(agent=agent, name="Research Agent")
    a2a_app = a2a.create_app()
    
    # For declarative agents (new format)
    from src.agents import load_agent_from_yaml
    agent = load_agent_from_yaml("configs/agents/research_agent.yaml")
    
    # For thread-safe factory-created agents
    from src.agents import create_research_agent, create_summarizer_agent
    research = create_research_agent()
    summarizer = create_summarizer_agent()
"""

from src.agents.base_agent import BaseAgent
from src.agents.research_agent import ResearchAgent
from src.agents.agui_server import (
    AGUIServer,
    AGUIEventEmitter,
    create_agui_server,
    EventType,
)
from src.agents.a2a_server import (
    A2AServer,
    WorkshopRequestHandler,
    create_a2a_server,
    Skill,  # Alias for AgentSkill
)

# Re-export SDK types for backward compatibility
from a2a.types import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
    Task,
    TaskState,
    TaskStatus,
    Message,
    TextPart,
    Artifact,
)
from src.agents.declarative import (
    DeclarativeAgent,
    DeclarativeAgentLoader,
    DeclarativeWorkflowLoader,
    AgentFactoryLoader,
    load_agent_from_yaml,
    load_agents_from_config,
    load_agents_with_factory,
    load_workflows_from_config,
)
from src.agents.factories import (
    create_research_agent,
    create_summarizer_agent,
    create_analysis_agent,
    create_coordinator_agent,
    create_custom_agent,
    get_agent_factory,
    AGENT_FACTORIES,
)
from src.agents.discussion import (
    DiscussionAgent,
    create_discussion_agent,
    DiscussionRole,
    DiscussionConfig,
    DiscussionProtocol,
    Participant,
    RoundResult,
)
from src.agents.moderator_agent import (
    ModeratorAgent,
    DiscussionPhase,
    DiscussionTurn,
    DiscussionSummary,
    ConflictStrategy,
    AgentProtocol,
    create_moderator_agent,
    DEFAULT_MODERATOR_INSTRUCTIONS,
)

__all__ = [
    "BaseAgent",
    "ResearchAgent",
    "AGUIServer",
    "AGUIEventEmitter",
    "create_agui_server",
    "EventType",
    # A2A Server
    "A2AServer",
    "WorkshopRequestHandler",
    "create_a2a_server",
    # A2A SDK types (re-exports)
    "AgentCard",
    "AgentSkill",
    "AgentCapabilities",
    "Skill",  # Alias for AgentSkill
    "Task",
    "TaskState",
    "TaskStatus",
    "Message",
    "TextPart",
    "Artifact",
    # Declarative agents
    "DeclarativeAgent",
    "DeclarativeAgentLoader",
    "DeclarativeWorkflowLoader",
    "AgentFactoryLoader",
    "load_agent_from_yaml",
    "load_agents_from_config",
    "load_agents_with_factory",
    "load_workflows_from_config",
    # Agent factories
    "create_research_agent",
    "create_summarizer_agent",
    "create_analysis_agent",
    "create_coordinator_agent",
    "create_custom_agent",
    "get_agent_factory",
    "AGENT_FACTORIES",
    # Discussion framework
    "DiscussionAgent",
    "create_discussion_agent",
    "DiscussionRole",
    "DiscussionConfig",
    "DiscussionProtocol",
    "Participant",
    "RoundResult",
    # Moderator
    "ModeratorAgent",
    "DiscussionPhase",
    "DiscussionTurn",
    "DiscussionSummary",
    "ConflictStrategy",
    "AgentProtocol",
    "create_moderator_agent",
    "DEFAULT_MODERATOR_INSTRUCTIONS",
]
