"""MCP Server implementation for the Agentic AI Patterns Workshop.

This module provides an MCP server that exposes workshop tools
to AI agents using the Model Context Protocol.

Example:
    from src.tools.mcp_server import create_mcp_server, run_server

    # Create and run the server
    server = create_mcp_server()
    await run_server(server, port=8001)

    # Or use with FastMCP's built-in runner
    server.run()
"""

import asyncio
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
)

from src.common.config import get_settings
from src.common.exceptions import MCPError, ToolError
from src.common.telemetry import get_tracer, setup_telemetry
from src.tools.calculator_tool import calculate
from src.tools.file_tool import list_files, read_file, write_file
from src.tools.search_tool import get_weather, search_web

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

# Tool definitions following MCP schema
TOOLS: list[Tool] = [
    Tool(
        name="search_web",
        description="Search the web for information on a topic",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                    "minLength": 1,
                    "maxLength": 500,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="get_weather",
        description="Get current weather for a location",
        inputSchema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location",
                    "minLength": 1,
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                },
            },
            "required": ["location"],
        },
    ),
    Tool(
        name="calculate",
        description="Perform mathematical calculations",
        inputSchema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide", "power", "sqrt"],
                    "description": "Mathematical operation to perform",
                },
                "a": {
                    "type": "number",
                    "description": "First operand",
                },
                "b": {
                    "type": "number",
                    "description": "Second operand (not required for sqrt)",
                },
            },
            "required": ["operation", "a"],
        },
    ),
    Tool(
        name="read_file",
        description="Read the contents of a file from the workspace",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file",
                },
                "encoding": {
                    "type": "string",
                    "enum": ["utf-8", "ascii", "latin-1"],
                    "default": "utf-8",
                },
            },
            "required": ["path"],
        },
    ),
    Tool(
        name="write_file",
        description="Write content to a file in the workspace",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write",
                },
                "mode": {
                    "type": "string",
                    "enum": ["overwrite", "append"],
                    "default": "overwrite",
                },
            },
            "required": ["path", "content"],
        },
    ),
    Tool(
        name="list_files",
        description="List files in a directory within the workspace",
        inputSchema={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Relative path to directory",
                    "default": ".",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files",
                    "default": "*",
                },
            },
            "required": [],
        },
    ),
]

# Map tool names to their implementations
TOOL_HANDLERS: dict[str, Any] = {
    "search_web": search_web,
    "get_weather": get_weather,
    "calculate": calculate,
    "read_file": read_file,
    "write_file": write_file,
    "list_files": list_files,
}


def create_mcp_server(name: str = "workshop-tools") -> Server:
    """Create and configure an MCP server with workshop tools.

    Args:
        name: Server name for identification.

    Returns:
        Configured MCP Server instance.

    Example:
        server = create_mcp_server()
        # Server is ready to run
    """
    server = Server(name)

    @server.list_tools()
    async def handle_list_tools() -> list[Tool]:
        """Return list of available tools."""
        logger.info(f"Listing {len(TOOLS)} tools")
        return TOOLS

    @server.call_tool()
    async def handle_call_tool(
        name: str,
        arguments: dict[str, Any] | None,
    ) -> list[TextContent]:
        """Execute a tool and return results.

        Args:
            name: Name of the tool to execute.
            arguments: Tool arguments.

        Returns:
            List of TextContent with results.

        Raises:
            MCPError: If tool not found or execution fails.
        """
        with tracer.start_as_current_span(f"mcp_call_tool:{name}") as span:
            span.set_attribute("tool.name", name)

            logger.info(f"Calling tool: {name} with args: {arguments}")

            if name not in TOOL_HANDLERS:
                raise MCPError(
                    f"Unknown tool: {name}",
                    details={"available_tools": list(TOOL_HANDLERS.keys())},
                )

            handler = TOOL_HANDLERS[name]
            args = arguments or {}

            try:
                result = await handler(**args)
                logger.info(f"Tool {name} completed successfully")

                # Format result as text
                import json

                result_text = json.dumps(result, indent=2)
                return [TextContent(type="text", text=result_text)]

            except ToolError as e:
                logger.error(f"Tool {name} failed: {e}")
                span.record_exception(e)
                raise MCPError(
                    f"Tool execution failed: {e}",
                    details={"tool": name, "error": str(e)},
                ) from e
            except Exception as e:
                logger.error(f"Unexpected error in tool {name}: {e}")
                span.record_exception(e)
                raise MCPError(
                    f"Unexpected error: {e}",
                    details={"tool": name},
                ) from e

    return server


async def run_server_stdio() -> None:
    """Run the MCP server using stdio transport.

    This is the standard way to run MCP servers for use with
    MCP clients like Claude Desktop.
    """
    setup_telemetry()
    server = create_mcp_server()

    logger.info("Starting MCP server with stdio transport")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """Entry point for running the MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    asyncio.run(run_server_stdio())


if __name__ == "__main__":
    main()
