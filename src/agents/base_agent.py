"""Base agent implementation with Microsoft Agent Framework and telemetry.

This module provides the foundation for all workshop agents,
using Microsoft Agent Framework with Azure OpenAI and OpenTelemetry tracing.

The implementation uses:
- `ChatAgent` from agent-framework for agent orchestration
- `AzureOpenAIChatClient` for Azure OpenAI integration
- `@ai_function` decorator for tool definitions
- OpenTelemetry middleware for observability
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Callable
from typing import Annotated, Any

from agent_framework import ChatAgent, ai_function
from agent_framework._middleware import (
    AgentMiddleware,
    AgentRunContext,
)
from agent_framework._threads import AgentThread
from agent_framework._types import AgentRunResponse, AgentRunResponseUpdate
from agent_framework.azure import AzureOpenAIChatClient

from src.common.config import Settings, get_settings
from src.common.exceptions import AgentError, ConfigurationError
from src.common.telemetry import (
    create_span_attributes,
    get_tracer,
    record_exception,
)

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class TelemetryMiddleware(AgentMiddleware):
    """Middleware to add OpenTelemetry tracing to agent runs."""

    def __init__(self, agent_name: str, model: str = "default") -> None:
        """Initialize telemetry middleware.

        Args:
            agent_name: Name of the agent for span attributes.
            model: Model name for span attributes.
        """
        self.agent_name = agent_name
        self.model = model

    async def process(
        self,
        context: AgentRunContext,
        next: Callable[[AgentRunContext], Any],
    ) -> None:
        """Wrap agent execution with OpenTelemetry span.

        Args:
            context: Agent run context.
            next: Next handler in middleware chain.
        """
        with tracer.start_as_current_span("agent_run") as span:
            span.set_attributes(
                create_span_attributes(
                    agent_name=self.agent_name,
                    model=self.model,
                )
            )
            try:
                await next(context)
            except Exception as e:
                record_exception(span, e)
                raise


class BaseAgent(ABC):
    """Base class for all workshop agents using Microsoft Agent Framework.

    Provides common functionality including:
    - ChatAgent from Microsoft Agent Framework
    - Azure OpenAI integration via AzureOpenAIChatClient
    - OpenTelemetry tracing via middleware
    - Tool integration via @ai_function decorator
    - Conversation history via AgentThread

    Subclasses must implement the `run` method.

    Attributes:
        name: Unique identifier for the agent.
        instructions: System prompt defining agent behavior.
        model: Azure OpenAI deployment name.
        temperature: Sampling temperature (0-2).
        max_tokens: Maximum tokens in response.
    """

    def __init__(
        self,
        name: str,
        instructions: str = "You are a helpful AI assistant.",
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[Callable[..., Any]] | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the base agent.

        Args:
            name: Unique identifier for the agent.
            instructions: System prompt defining agent behavior.
            model: Azure OpenAI deployment name (defaults to config).
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens in response.
            tools: List of @ai_function decorated tools.
            settings: Configuration settings (defaults to get_settings()).

        Raises:
            ConfigurationError: If Azure OpenAI is not configured.
        """
        self.name = name
        self.instructions = instructions
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._settings = settings or get_settings()
        self._tools = tools or []
        self._thread: AgentThread | None = None

        # Set model from config if not specified
        self.model = model or self._settings.azure_openai_deployment

        if not self.model:
            raise ConfigurationError(
                "No model specified and AZURE_OPENAI_DEPLOYMENT not set",
                config_key="AZURE_OPENAI_DEPLOYMENT",
            )

        # Create chat client and agent
        self._chat_client = self._create_chat_client()
        self._agent = self._create_agent()

        logger.info(f"Initialized agent '{name}' with model '{self.model}'")

    # Backward compatibility aliases
    @property
    def system_prompt(self) -> str:
        """Alias for instructions (backward compatibility)."""
        return self.instructions

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        """Set instructions (backward compatibility)."""
        self.instructions = value

    @property
    def max_completion_tokens(self) -> int:
        """Alias for max_tokens (backward compatibility)."""
        return self.max_tokens

    @max_completion_tokens.setter
    def max_completion_tokens(self, value: int) -> None:
        """Set max_tokens (backward compatibility)."""
        self.max_tokens = value

    def _create_chat_client(self) -> AzureOpenAIChatClient:
        """Create Azure OpenAI chat client.

        Returns:
            Configured AzureOpenAIChatClient.

        Raises:
            ConfigurationError: If Azure OpenAI is not configured.
        """
        if not self._settings.is_azure_configured:
            raise ConfigurationError(
                "Azure OpenAI is not configured",
                config_key="AZURE_OPENAI_ENDPOINT",
            )

        # Create client with API key or credential
        if self._settings.azure_openai_api_key:
            return AzureOpenAIChatClient(
                endpoint=self._settings.azure_openai_endpoint,
                api_key=self._settings.azure_openai_api_key,
                deployment_name=self.model,
                api_version=self._settings.azure_openai_api_version,
            )
        else:
            # Use Azure AD authentication
            from azure.identity import DefaultAzureCredential

            return AzureOpenAIChatClient(
                endpoint=self._settings.azure_openai_endpoint,
                credential=DefaultAzureCredential(),
                deployment_name=self.model,
                api_version=self._settings.azure_openai_api_version,
            )

    def _create_agent(self) -> ChatAgent:
        """Create ChatAgent with telemetry middleware.

        Returns:
            Configured ChatAgent instance.
        """
        return ChatAgent(
            chat_client=self._chat_client,
            name=self.name,
            instructions=self.instructions,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=self._tools if self._tools else None,
            middleware=[TelemetryMiddleware(self.name, self.model)],
        )

    @property
    def agent(self) -> ChatAgent:
        """Get the underlying ChatAgent instance.

        Returns:
            The ChatAgent used by this agent.
        """
        return self._agent

    @property
    def thread(self) -> AgentThread:
        """Get or create conversation thread.

        Returns:
            AgentThread for conversation history.
        """
        if self._thread is None:
            self._thread = self._agent.get_new_thread()
        return self._thread

    def register_tools(self, tools: list[Callable[..., Any]]) -> None:
        """Register tools for the agent to use.

        Tools should be decorated with @ai_function.

        Args:
            tools: List of @ai_function decorated callables.

        Example:
            @ai_function
            def search_web(query: str) -> str:
                '''Search the web for information.'''
                return f"Results for: {query}"

            agent.register_tools([search_web])
        """
        self._tools = tools
        # Recreate agent with new tools
        self._agent = self._create_agent()
        logger.info(f"Agent '{self.name}' registered {len(tools)} tools")

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._thread = None
        logger.debug(f"Cleared history for agent '{self.name}'")

    def get_history(self) -> list[dict[str, str]]:
        """Get the conversation history.

        Returns:
            List of message dictionaries.
        """
        if self._thread is None:
            return []
        # Convert thread messages to legacy format
        messages = []
        for msg in self._thread.messages:
            messages.append({
                "role": msg.role,
                "content": msg.content if isinstance(msg.content, str) else str(msg.content),
            })
        return messages

    async def chat(
        self,
        message: str,
        *,
        include_history: bool = True,
        **kwargs: Any,
    ) -> str:
        """Send a message and get a response.

        This method maintains conversation history for multi-turn
        conversations using AgentThread.

        Args:
            message: User message to send.
            include_history: Whether to include conversation history.
            **kwargs: Additional parameters for the completion.

        Returns:
            Assistant's response text.

        Raises:
            AgentError: If the completion fails.
        """
        with tracer.start_as_current_span("agent_chat") as span:
            span.set_attributes(
                create_span_attributes(
                    agent_name=self.name,
                    model=self.model,
                )
            )

            try:
                # Use thread for history, or run without thread
                thread = self.thread if include_history else None

                response = await self._agent.run(
                    message,
                    thread=thread,
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                )

                result_text = response.text or ""
                span.set_attribute("response_length", len(result_text))

                return result_text

            except Exception as e:
                record_exception(span, e)
                logger.error(f"Agent '{self.name}' chat failed: {e}")
                raise AgentError(
                    f"Failed to complete chat: {e}",
                    agent_name=self.name,
                    details={"model": self.model, "error": str(e)},
                ) from e

    async def chat_stream(
        self,
        message: str,
        *,
        include_history: bool = True,
        **kwargs: Any,
    ) -> AsyncIterable[str]:
        """Send a message and stream the response.

        Args:
            message: User message to send.
            include_history: Whether to include conversation history.
            **kwargs: Additional parameters for the completion.

        Yields:
            Response text chunks.

        Raises:
            AgentError: If the completion fails.
        """
        with tracer.start_as_current_span("agent_chat_stream") as span:
            span.set_attributes(
                create_span_attributes(
                    agent_name=self.name,
                    model=self.model,
                )
            )

            try:
                thread = self.thread if include_history else None

                async for chunk in self._agent.run_stream(
                    message,
                    thread=thread,
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                ):
                    if chunk.text:
                        yield chunk.text

            except Exception as e:
                record_exception(span, e)
                logger.error(f"Agent '{self.name}' stream failed: {e}")
                raise AgentError(
                    f"Failed to stream chat: {e}",
                    agent_name=self.name,
                    details={"model": self.model, "error": str(e)},
                ) from e

    @abstractmethod
    async def run(self, task: str, **kwargs: Any) -> str:
        """Execute a task and return the result.

        Subclasses must implement this method to define agent behavior.

        Args:
            task: The task or query to execute.
            **kwargs: Additional task-specific parameters.

        Returns:
            Result of the task execution.
        """
        pass

    async def close(self) -> None:
        """Clean up resources."""
        self._thread = None
        logger.debug(f"Closed agent '{self.name}'")

    async def __aenter__(self) -> "BaseAgent":
        """Enter async context manager."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context manager."""
        await self.close()
