"""Contract tests for MCP tool schemas.

These tests verify that MCP tool implementations conform to
the schemas defined in contracts/mcp-tools.md.
"""

import json
from typing import Any

import pytest

from src.tools.mcp_server import TOOLS


class TestMCPToolSchemas:
    """Tests that verify MCP tool schema compliance."""

    def test_all_tools_have_required_fields(self) -> None:
        """Every tool must have name, description, and inputSchema."""
        for tool in TOOLS:
            assert tool.name, "Tool must have a name"
            assert tool.description, "Tool must have a description"
            assert tool.inputSchema, "Tool must have an inputSchema"

    def test_search_web_schema(self) -> None:
        """search_web tool should match contract schema."""
        tool = next((t for t in TOOLS if t.name == "search_web"), None)
        assert tool is not None, "search_web tool not found"

        schema = tool.inputSchema
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "max_results" in schema["properties"]
        assert "query" in schema["required"]

        # Verify query constraints
        query_schema = schema["properties"]["query"]
        assert query_schema["type"] == "string"
        assert query_schema.get("minLength", 0) >= 0
        assert query_schema.get("maxLength", 1000) <= 1000

        # Verify max_results constraints
        max_results_schema = schema["properties"]["max_results"]
        assert max_results_schema["type"] == "integer"
        assert max_results_schema.get("minimum", 1) >= 1
        assert max_results_schema.get("maximum", 20) <= 100

    def test_calculate_schema(self) -> None:
        """calculate tool should match contract schema."""
        tool = next((t for t in TOOLS if t.name == "calculate"), None)
        assert tool is not None, "calculate tool not found"

        schema = tool.inputSchema
        assert schema["type"] == "object"
        assert "operation" in schema["properties"]
        assert "a" in schema["properties"]
        assert "b" in schema["properties"]
        assert "operation" in schema["required"]
        assert "a" in schema["required"]

        # Verify operation enum
        op_schema = schema["properties"]["operation"]
        assert op_schema["type"] == "string"
        expected_ops = {"add", "subtract", "multiply", "divide", "power", "sqrt"}
        assert set(op_schema["enum"]) == expected_ops

    def test_read_file_schema(self) -> None:
        """read_file tool should match contract schema."""
        tool = next((t for t in TOOLS if t.name == "read_file"), None)
        assert tool is not None, "read_file tool not found"

        schema = tool.inputSchema
        assert schema["type"] == "object"
        assert "path" in schema["properties"]
        assert "path" in schema["required"]

    def test_write_file_schema(self) -> None:
        """write_file tool should match contract schema."""
        tool = next((t for t in TOOLS if t.name == "write_file"), None)
        assert tool is not None, "write_file tool not found"

        schema = tool.inputSchema
        assert schema["type"] == "object"
        assert "path" in schema["properties"]
        assert "content" in schema["properties"]
        assert "path" in schema["required"]
        assert "content" in schema["required"]

    def test_get_weather_schema(self) -> None:
        """get_weather tool should match contract schema."""
        tool = next((t for t in TOOLS if t.name == "get_weather"), None)
        assert tool is not None, "get_weather tool not found"

        schema = tool.inputSchema
        assert schema["type"] == "object"
        assert "location" in schema["properties"]
        assert "units" in schema["properties"]
        assert "location" in schema["required"]

        # Verify units enum
        units_schema = schema["properties"]["units"]
        assert set(units_schema["enum"]) == {"celsius", "fahrenheit"}

    def test_list_files_schema(self) -> None:
        """list_files tool should match contract schema."""
        tool = next((t for t in TOOLS if t.name == "list_files"), None)
        assert tool is not None, "list_files tool not found"

        schema = tool.inputSchema
        assert schema["type"] == "object"
        assert "directory" in schema["properties"]
        assert "pattern" in schema["properties"]


class TestMCPToolOutputs:
    """Tests that verify MCP tool output conformance."""

    @pytest.mark.asyncio
    async def test_search_web_output(self) -> None:
        """search_web output should match expected schema."""
        from src.tools import search_web

        result = await search_web("test query", max_results=3)

        assert "results" in result
        assert "total_found" in result
        assert isinstance(result["results"], list)
        assert isinstance(result["total_found"], int)

        # Verify result item structure
        for item in result["results"]:
            assert "title" in item
            assert "url" in item
            assert "snippet" in item

    @pytest.mark.asyncio
    async def test_calculate_output(self) -> None:
        """calculate output should match expected schema."""
        from src.tools import calculate

        result = await calculate("add", 5, 3)

        assert "result" in result
        assert "operation" in result
        assert "expression" in result
        assert result["result"] == 8
        assert result["operation"] == "add"

    @pytest.mark.asyncio
    async def test_get_weather_output(self) -> None:
        """get_weather output should match expected schema."""
        from src.tools.search_tool import get_weather

        result = await get_weather("Seattle", "celsius")

        assert "location" in result
        assert "temperature" in result
        assert "units" in result
        assert "condition" in result
        assert "humidity" in result
        assert isinstance(result["temperature"], (int, float))


class TestMCPToolValidation:
    """Tests that verify input validation for MCP tools."""

    @pytest.mark.asyncio
    async def test_search_web_empty_query_fails(self) -> None:
        """search_web should reject empty queries."""
        from src.tools import search_web

        with pytest.raises(ValueError, match="empty"):
            await search_web("", max_results=5)

    @pytest.mark.asyncio
    async def test_search_web_max_results_bounds(self) -> None:
        """search_web should enforce max_results bounds."""
        from src.tools import search_web

        with pytest.raises(ValueError):
            await search_web("test", max_results=0)

        with pytest.raises(ValueError):
            await search_web("test", max_results=100)

    @pytest.mark.asyncio
    async def test_calculate_division_by_zero(self) -> None:
        """calculate should handle division by zero."""
        from src.common.exceptions import ToolError
        from src.tools import calculate

        with pytest.raises(ToolError, match="zero"):
            await calculate("divide", 10, 0)

    @pytest.mark.asyncio
    async def test_calculate_sqrt_negative(self) -> None:
        """calculate should reject sqrt of negative numbers."""
        from src.common.exceptions import ToolError
        from src.tools import calculate

        with pytest.raises(ToolError, match="negative"):
            await calculate("sqrt", -4)

    @pytest.mark.asyncio
    async def test_get_weather_empty_location_fails(self) -> None:
        """get_weather should reject empty locations."""
        from src.tools.search_tool import get_weather

        with pytest.raises(ValueError, match="empty"):
            await get_weather("")
