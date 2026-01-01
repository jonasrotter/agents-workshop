"""Research agent implementation with MCP tool integration.

This module provides a specialized agent for research tasks
that can use MCP tools to search the web and gather information.
"""

import json
import logging
from typing import Any, Callable, Optional

from src.agents.base_agent import BaseAgent
from src.common.config import Settings
from src.common.exceptions import AgentError, ToolError
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
    - Automatic tool handling loop
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
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_tool_iterations: int = 10,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the research agent.

        Args:
            name: Unique identifier for the agent.
            system_prompt: Instructions defining agent behavior.
            model: Azure OpenAI deployment name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            max_tool_iterations: Maximum tool call loops.
            settings: Configuration settings.
        """
        super().__init__(
            name=name,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            settings=settings,
        )
        self.max_tool_iterations = max_tool_iterations
        self._tool_handlers: dict[str, Callable[..., Any]] = {}

    def set_tool_handlers(
        self,
        handlers: dict[str, Callable[..., Any]],
    ) -> None:
        """Set handlers for tool execution.

        Args:
            handlers: Dictionary mapping tool names to async functions.

        Example:
            agent.set_tool_handlers({
                "search_web": search_web_func,
                "calculate": calculate_func,
            })
        """
        self._tool_handlers = handlers
        logger.info(f"Set {len(handlers)} tool handlers for '{self.name}'")

    def _create_openai_tools(self) -> list[dict[str, Any]]:
        """Create OpenAI tool definitions from registered handlers.

        Returns:
            List of tool definitions in OpenAI format.
        """
        # Default tool schemas - these match MCP tool schemas
        tool_schemas = {
            "search_web": {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for information on a topic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum results (1-20)",
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            "get_weather": {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City or location name",
                            },
                            "units": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "default": "celsius",
                            },
                        },
                        "required": ["location"],
                    },
                },
            },
            "calculate": {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": [
                                    "add",
                                    "subtract",
                                    "multiply",
                                    "divide",
                                    "power",
                                    "sqrt",
                                ],
                                "description": "Operation to perform",
                            },
                            "a": {
                                "type": "number",
                                "description": "First operand",
                            },
                            "b": {
                                "type": "number",
                                "description": "Second operand (not for sqrt)",
                            },
                        },
                        "required": ["operation", "a"],
                    },
                },
            },
            "read_file": {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to file",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            "write_file": {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to file",
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write",
                            },
                        },
                        "required": ["path", "content"],
                    },
                },
            },
        }

        # Return only schemas for registered handlers
        tools = []
        for name in self._tool_handlers:
            if name in tool_schemas:
                tools.append(tool_schemas[name])
        return tools

    async def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Execute a tool and return the result as string.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments to pass to the tool.

        Returns:
            Tool result as JSON string.

        Raises:
            ToolError: If tool execution fails.
        """
        with tracer.start_as_current_span(f"execute_tool:{tool_name}") as span:
            span.set_attributes(
                create_span_attributes(
                    agent_name=self.name,
                    tool_name=tool_name,
                )
            )

            if tool_name not in self._tool_handlers:
                raise ToolError(
                    f"Unknown tool: {tool_name}",
                    tool_name=tool_name,
                    details={"available": list(self._tool_handlers.keys())},
                )

            handler = self._tool_handlers[tool_name]
            logger.info(f"Executing tool '{tool_name}' with args: {arguments}")

            try:
                result = await handler(**arguments)
                span.set_attribute("success", True)
                return json.dumps(result, indent=2)
            except Exception as e:
                logger.error(f"Tool '{tool_name}' failed: {e}")
                span.set_attribute("success", False)
                span.record_exception(e)
                # Return error as result so agent can handle it
                return json.dumps({
                    "error": str(e),
                    "tool": tool_name,
                })

    async def run(
        self,
        task: str,
        **kwargs: Any,
    ) -> str:
        """Execute a research task using available tools.

        This method:
        1. Sends the task to the model
        2. Handles any tool calls in a loop
        3. Returns the final synthesized response

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

            # Register tools if handlers are set
            if self._tool_handlers and not self._tools:
                tools = self._create_openai_tools()
                self.register_tools(tools)

            # Build initial messages
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": task},
            ]

            # Tool execution loop
            iteration = 0
            while iteration < self.max_tool_iterations:
                iteration += 1
                logger.debug(f"Research iteration {iteration}")

                # Get completion
                response = await self._create_completion(messages)
                message = response.choices[0].message

                # Check if we need to call tools
                if not message.tool_calls:
                    # No more tools to call, return final response
                    final_response = message.content or ""
                    span.set_attribute("iterations", iteration)
                    span.set_attribute("final_response_length", len(final_response))
                    logger.info(f"Research completed in {iteration} iterations")
                    return final_response

                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                })

                # Execute each tool call
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                    # Execute tool
                    result = await self._execute_tool(tool_name, arguments)

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })

            # Max iterations reached
            logger.warning(f"Max tool iterations ({self.max_tool_iterations}) reached")
            raise AgentError(
                "Max tool iterations reached without completion",
                agent_name=self.name,
                details={"max_iterations": self.max_tool_iterations},
            )
