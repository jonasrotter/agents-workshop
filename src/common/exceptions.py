"""Custom exception hierarchy for the Agentic AI Patterns Workshop.

This module defines a structured exception hierarchy following best practices:
- All exceptions inherit from WorkshopError base class
- Each exception type maps to a specific error domain
- Exceptions carry context for debugging

Example:
    from src.common.exceptions import AgentError, ToolError

    try:
        result = await agent.run(task)
    except ToolError as e:
        logger.error(f"Tool failed: {e.tool_name} - {e}")
    except AgentError as e:
        logger.error(f"Agent error: {e}")
"""

from typing import Any, Optional


class WorkshopError(Exception):
    """Base exception for all workshop errors.

    All custom exceptions in the workshop inherit from this class,
    allowing for catch-all error handling while maintaining specificity.

    Attributes:
        message: Human-readable error description.
        details: Additional context about the error.
    """

    def __init__(
        self,
        message: str,
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize WorkshopError.

        Args:
            message: Human-readable error description.
            details: Additional context as key-value pairs.
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """Format error message with details if present."""
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class ConfigurationError(WorkshopError):
    """Raised when configuration is invalid or missing.

    Use this for environment variable issues, invalid settings,
    or missing required configuration.

    Example:
        if not settings.azure_openai_endpoint:
            raise ConfigurationError(
                "Azure OpenAI endpoint not configured",
                details={"env_var": "AZURE_OPENAI_ENDPOINT"}
            )
    """

    def __init__(
        self,
        message: str,
        *,
        config_key: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize ConfigurationError.

        Args:
            message: Description of the configuration issue.
            config_key: The specific configuration key that's problematic.
            details: Additional context.
        """
        details = details or {}
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, details=details)
        self.config_key = config_key


class AgentError(WorkshopError):
    """Raised when an agent operation fails.

    Use this for agent creation, execution, or communication failures.

    Example:
        try:
            response = await agent.run(prompt)
        except Exception as e:
            raise AgentError(
                "Agent execution failed",
                agent_name="research_agent",
                details={"original_error": str(e)}
            ) from e
    """

    def __init__(
        self,
        message: str,
        *,
        agent_name: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize AgentError.

        Args:
            message: Description of the agent failure.
            agent_name: Name/identifier of the failing agent.
            details: Additional context.
        """
        details = details or {}
        if agent_name:
            details["agent_name"] = agent_name
        super().__init__(message, details=details)
        self.agent_name = agent_name


class ToolError(WorkshopError):
    """Raised when a tool operation fails.

    Use this for MCP tool invocation failures, validation errors,
    or tool-specific exceptions.

    Example:
        if not is_valid_query(query):
            raise ToolError(
                "Invalid search query",
                tool_name="search_web",
                details={"query": query, "reason": "empty query"}
            )
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize ToolError.

        Args:
            message: Description of the tool failure.
            tool_name: Name of the failing tool.
            details: Additional context.
        """
        details = details or {}
        if tool_name:
            details["tool_name"] = tool_name
        super().__init__(message, details=details)
        self.tool_name = tool_name


class MCPError(WorkshopError):
    """Raised when MCP protocol operations fail.

    Use this for MCP server connection issues, protocol violations,
    or resource access failures.

    Example:
        except ConnectionRefusedError:
            raise MCPError(
                "Cannot connect to MCP server",
                server_url="http://localhost:8001",
                details={"transport": "http"}
            )
    """

    def __init__(
        self,
        message: str,
        *,
        server_url: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize MCPError.

        Args:
            message: Description of the MCP failure.
            server_url: URL of the MCP server involved.
            details: Additional context.
        """
        details = details or {}
        if server_url:
            details["server_url"] = server_url
        super().__init__(message, details=details)
        self.server_url = server_url


class WorkflowError(WorkshopError):
    """Raised when workflow execution fails.

    Use this for workflow orchestration issues, step failures,
    or data flow problems.

    Example:
        if step.failed:
            raise WorkflowError(
                f"Workflow step '{step.name}' failed",
                workflow_name="research_pipeline",
                step_name=step.name,
                details={"step_index": idx, "error": step.error}
            )
    """

    def __init__(
        self,
        message: str,
        *,
        workflow_name: Optional[str] = None,
        step_name: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize WorkflowError.

        Args:
            message: Description of the workflow failure.
            workflow_name: Name of the failing workflow.
            step_name: Name of the specific failing step.
            details: Additional context.
        """
        details = details or {}
        if workflow_name:
            details["workflow_name"] = workflow_name
        if step_name:
            details["step_name"] = step_name
        super().__init__(message, details=details)
        self.workflow_name = workflow_name
        self.step_name = step_name


class AGUIError(WorkshopError):
    """Raised when AG-UI protocol operations fail.

    Use this for AG-UI streaming errors, event encoding issues,
    or connection problems.

    Example:
        if not encoder.supports(event_type):
            raise AGUIError(
                f"Unsupported event type: {event_type}",
                event_type=event_type
            )
    """

    def __init__(
        self,
        message: str,
        *,
        event_type: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize AGUIError.

        Args:
            message: Description of the AG-UI failure.
            event_type: The AG-UI event type involved.
            details: Additional context.
        """
        details = details or {}
        if event_type:
            details["event_type"] = event_type
        super().__init__(message, details=details)
        self.event_type = event_type


class A2AError(WorkshopError):
    """Raised when A2A protocol operations fail.

    Use this for A2A agent discovery issues, task invocation failures,
    or protocol violations.

    Example:
        if response.status != 200:
            raise A2AError(
                "A2A task invocation failed",
                task_id=task_id,
                details={"status": response.status, "body": response.body}
            )
    """

    def __init__(
        self,
        message: str,
        *,
        task_id: Optional[str] = None,
        agent_url: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize A2AError.

        Args:
            message: Description of the A2A failure.
            task_id: The A2A task ID involved.
            agent_url: URL of the remote agent.
            details: Additional context.
        """
        details = details or {}
        if task_id:
            details["task_id"] = task_id
        if agent_url:
            details["agent_url"] = agent_url
        super().__init__(message, details=details)
        self.task_id = task_id
        self.agent_url = agent_url
