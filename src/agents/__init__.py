"""Agents module for the Agentic AI Patterns Workshop.

This module exports agent implementations:
- BaseAgent: Foundation for all agents with telemetry
- ResearchAgent: Agent for web research tasks
- AGUIServer: AG-UI protocol server for streaming chat
- AGUIEventEmitter: Event emitter for AG-UI streaming
- A2AServer: A2A protocol server for agent interoperability
- create_agui_server: Factory function for AG-UI server
- create_a2a_server: Factory function for A2A server

Example:
    from src.agents import ResearchAgent, AGUIServer, A2AServer

    agent = ResearchAgent(name="researcher")
    
    # For streaming UI
    agui = AGUIServer(agent=agent)
    agui_app = agui.create_app()
    
    # For agent-to-agent
    a2a = A2AServer(agent=agent, name="Research Agent")
    a2a_app = a2a.create_app()
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
    create_a2a_server,
    AgentCard,
    Skill,
    TaskState,
    Task,
)

__all__ = [
    "BaseAgent",
    "ResearchAgent",
    "AGUIServer",
    "AGUIEventEmitter",
    "create_agui_server",
    "EventType",
    "A2AServer",
    "create_a2a_server",
    "AgentCard",
    "Skill",
    "TaskState",
    "Task",
]
