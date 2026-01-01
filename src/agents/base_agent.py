"""Base agent implementation with telemetry integration.

This module provides the foundation for all workshop agents,
including Azure OpenAI integration and OpenTelemetry tracing.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletion

from src.common.config import Settings, get_settings
from src.common.exceptions import AgentError, ConfigurationError
from src.common.telemetry import (
    create_span_attributes,
    get_tracer,
    record_exception,
)

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class BaseAgent(ABC):
    """Base class for all workshop agents.

    Provides common functionality including:
    - Azure OpenAI client management
    - OpenTelemetry tracing
    - Tool integration support
    - Conversation history management

    Subclasses must implement the `run` method.

    Attributes:
        name: Unique identifier for the agent.
        system_prompt: Instructions defining agent behavior.
        model: Azure OpenAI deployment name.
        temperature: Sampling temperature (0-2).
        max_completion_tokens: Maximum tokens in response.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str = "You are a helpful AI assistant.",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_completion_tokens: int = 4096,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the base agent.

        Args:
            name: Unique identifier for the agent.
            system_prompt: Instructions defining agent behavior.
            model: Azure OpenAI deployment name (defaults to config).
            temperature: Sampling temperature (0-2).
            max_completion_tokens: Maximum tokens in response.
            settings: Configuration settings (defaults to get_settings()).

        Raises:
            ConfigurationError: If Azure OpenAI is not configured.
        """
        self.name = name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self._settings = settings or get_settings()
        self._client: Optional[AsyncAzureOpenAI] = None
        self._conversation_history: list[dict[str, str]] = []
        self._tools: list[dict[str, Any]] = []

        # Set model from config if not specified
        self.model = model or self._settings.azure_openai_deployment

        if not self.model:
            raise ConfigurationError(
                "No model specified and AZURE_OPENAI_DEPLOYMENT not set",
                config_key="AZURE_OPENAI_DEPLOYMENT",
            )

        logger.info(f"Initialized agent '{name}' with model '{self.model}'")

    @property
    def client(self) -> AsyncAzureOpenAI:
        """Get or create the Azure OpenAI client.

        Returns:
            Configured AsyncAzureOpenAI client.

        Raises:
            ConfigurationError: If Azure OpenAI is not configured.
        """
        if self._client is None:
            if not self._settings.is_azure_configured:
                raise ConfigurationError(
                    "Azure OpenAI is not configured",
                    config_key="AZURE_OPENAI_ENDPOINT",
                )

            # Create client based on authentication method
            if self._settings.azure_openai_api_key:
                self._client = AsyncAzureOpenAI(
                    azure_endpoint=self._settings.azure_openai_endpoint,
                    api_key=self._settings.azure_openai_api_key,
                    api_version=self._settings.azure_openai_api_version,
                )
            else:
                # Use Azure AD authentication
                from azure.identity.aio import DefaultAzureCredential

                credential = DefaultAzureCredential()
                self._client = AsyncAzureOpenAI(
                    azure_endpoint=self._settings.azure_openai_endpoint,
                    azure_ad_token_provider=self._get_token_provider(credential),
                    api_version=self._settings.azure_openai_api_version,
                )

        return self._client

    def _get_token_provider(self, credential: Any) -> Any:
        """Create a token provider for Azure AD authentication.

        Args:
            credential: Azure credential instance.

        Returns:
            Token provider callable.
        """
        from azure.identity import get_bearer_token_provider

        return get_bearer_token_provider(
            credential,
            "https://cognitiveservices.azure.com/.default",
        )

    def register_tools(self, tools: list[dict[str, Any]]) -> None:
        """Register tools for the agent to use.

        Args:
            tools: List of tool definitions in OpenAI format.

        Example:
            agent.register_tools([
                {
                    "type": "function",
                    "function": {
                        "name": "search_web",
                        "description": "Search the web",
                        "parameters": {...}
                    }
                }
            ])
        """
        self._tools = tools
        logger.info(f"Agent '{self.name}' registered {len(tools)} tools")

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._conversation_history.clear()
        logger.debug(f"Cleared history for agent '{self.name}'")

    def get_history(self) -> list[dict[str, str]]:
        """Get the conversation history.

        Returns:
            List of message dictionaries.
        """
        return self._conversation_history.copy()

    async def _create_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion with telemetry.

        Args:
            messages: List of messages to send.
            **kwargs: Additional parameters for the API call.

        Returns:
            ChatCompletion response.

        Raises:
            AgentError: If the API call fails.
        """
        with tracer.start_as_current_span("agent_completion") as span:
            span.set_attributes(
                create_span_attributes(
                    agent_name=self.name,
                    model=self.model,
                )
            )

            try:
                # Build request parameters
                params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", self.temperature),
                    "max_completion_tokens": kwargs.get("max_completion_tokens", self.max_completion_tokens),
                }

                # Add tools if registered
                if self._tools:
                    params["tools"] = self._tools
                    params["tool_choice"] = kwargs.get("tool_choice", "auto")

                response = await self.client.chat.completions.create(**params)

                # Record token usage
                if response.usage:
                    span.set_attributes(
                        create_span_attributes(
                            prompt_tokens=response.usage.prompt_tokens,
                            completion_tokens=response.usage.completion_tokens,
                        )
                    )
                    logger.debug(
                        f"Tokens used: {response.usage.prompt_tokens} prompt, "
                        f"{response.usage.completion_tokens} completion"
                    )

                return response

            except Exception as e:
                record_exception(span, e)
                logger.error(f"Agent '{self.name}' completion failed: {e}")
                raise AgentError(
                    f"Failed to create completion: {e}",
                    agent_name=self.name,
                    details={"model": self.model, "error": str(e)},
                ) from e

    async def chat(
        self,
        message: str,
        *,
        include_history: bool = True,
        **kwargs: Any,
    ) -> str:
        """Send a message and get a response.

        This method maintains conversation history for multi-turn
        conversations.

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

            # Build messages list
            messages = [{"role": "system", "content": self.system_prompt}]

            if include_history:
                messages.extend(self._conversation_history)

            messages.append({"role": "user", "content": message})

            # Get completion
            response = await self._create_completion(messages, **kwargs)

            # Extract response text
            assistant_message = response.choices[0].message.content or ""

            # Update history
            self._conversation_history.append({"role": "user", "content": message})
            self._conversation_history.append(
                {"role": "assistant", "content": assistant_message}
            )

            span.set_attribute("response_length", len(assistant_message))

            return assistant_message

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
        if self._client:
            await self._client.close()
            self._client = None
        logger.debug(f"Closed agent '{self.name}'")

    async def __aenter__(self) -> "BaseAgent":
        """Enter async context manager."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context manager."""
        await self.close()
