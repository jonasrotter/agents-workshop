"""Research agent implementation with MCP tool integration.

This module provides a specialized agent for research tasks
that can use MCP tools to search the web and gather information.

Uses Microsoft Agent Framework's ChatAgent with @ai_function tools.
"""

import logging
from collections.abc import Callable
from typing import Any

from src.agents.base_agent import BaseAgent
from src.common.config import Settings
from src.common.exceptions import AgentError
from src.common.telemetry import create_span_attributes, get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

# Default system prompt for research agent
RESEARCH_SYSTEM_PROMPT = """You are a research assistant specialized in finding and synthesizing information.

Your capabilities:
1. Search the web for relevant information
2. Get current weather data
3. Perform calculations
4. Read and write files

When given a research task:
1. Break down the task into specific questions
2. Use available tools to gather information
3. Synthesize findings into a clear response
4. Cite your sources when applicable

Always be thorough, accurate, and helpful."""


class ResearchAgent(BaseAgent):
    """Agent specialized for research tasks with tool integration.

    This agent extends BaseAgent with:
    - Tool handling via Microsoft Agent Framework's @ai_function
    - Research-specific system prompt
    - Tool result synthesis

    Example:
        agent = ResearchAgent(name="researcher")
        agent.set_tool_handlers({
            "search_web": search_web,
            "calculate": calculate,
        })
        result = await agent.run("Find information about Python async/await")
    """

    def __init__(
        self,
        name: str = "research_agent",
        system_prompt: str = RESEARCH_SYSTEM_PROMPT,
        model: str | None = None,
        temperature: float = 0.7,
        max_completion_tokens: int = 4096,
        max_tool_iterations: int = 10,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the research agent.

        Args:
            name: Unique identifier for the agent.
            system_prompt: Instructions defining agent behavior.
            model: Azure OpenAI deployment name.
            temperature: Sampling temperature.
            max_completion_tokens: Maximum tokens in response.
            max_tool_iterations: Maximum tool call loops.
            settings: Configuration settings.
        """
        super().__init__(
            name=name,
            instructions=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_completion_tokens,
            settings=settings,
        )
        self.max_tool_iterations = max_tool_iterations
        self._tool_handlers: dict[str, Callable[..., Any]] = {}

    def set_tool_handlers(
        self,
        handlers: dict[str, Callable[..., Any]],
    ) -> None:
        """Set handlers for tool execution.

        This method wraps the handlers with @ai_function decorator
        and registers them with the agent framework.

        Args:
            handlers: Dictionary mapping tool names to async functions.

        Example:
            agent.set_tool_handlers({
                "search_web": search_web_func,
                "calculate": calculate_func,
            })
        """
        self._tool_handlers = handlers

        # Create @ai_function wrapped tools from handlers
        wrapped_tools = self._create_ai_function_tools()
        if wrapped_tools:
            self.register_tools(wrapped_tools)

        logger.info(f"Set {len(handlers)} tool handlers for '{self.name}'")

    def _create_ai_function_tools(self) -> list[Callable[..., Any]]:
        """Create list of tools from registered handlers.

        The handlers should be properly typed async functions that
        work with Microsoft Agent Framework's tool auto-invocation.

        Returns:
            List of callable tools.
        """
        return list(self._tool_handlers.values())

    async def run(
        self,
        task: str,
        **kwargs: Any,
    ) -> str:
        """Execute a research task using available tools.

        With Microsoft Agent Framework, tools are automatically invoked
        by the ChatAgent when the model requests them. This simplifies
        the implementation compared to manual tool loop handling.

        Args:
            task: The research task or question.
            **kwargs: Additional parameters.

        Returns:
            Research results and synthesis.

        Raises:
            AgentError: If the research fails.
        """
        with tracer.start_as_current_span("research_run") as span:
            span.set_attributes(
                create_span_attributes(
                    agent_name=self.name,
                    model=self.model,
                )
            )

            logger.info(f"Starting research task: {task[:100]}...")

            try:
                # ChatAgent handles tool invocation automatically
                # Use thread for conversation history
                response = await self._agent.run(
                    task,
                    thread=self.thread,
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                )

                result_text = response.text or ""
                span.set_attribute("final_response_length", len(result_text))
                logger.info("Research completed")
                return result_text

            except Exception as e:
                logger.error(f"Research task failed: {e}")
                raise AgentError(
                    f"Research failed: {e}",
                    agent_name=self.name,
                    details={"model": self.model, "error": str(e)},
                ) from e
