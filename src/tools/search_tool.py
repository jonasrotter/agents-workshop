"""Search tools for MCP integration.

This module provides web search and weather tools that can be
exposed via the MCP server and used with Microsoft Agent Framework.

The implementations are mock/simulated for workshop purposes,
but follow the exact schemas defined in contracts/mcp-tools.md.

Tools use the @ai_function decorator for integration with ChatAgent.
"""

import logging
import random
from typing import Annotated, Any, Literal

from agent_framework import ai_function

from src.common.telemetry import create_span_attributes, get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

# Mock search results for workshop demonstration
MOCK_SEARCH_RESULTS = {
    "python": [
        {
            "title": "Welcome to Python.org",
            "url": "https://www.python.org/",
            "snippet": "The official home of the Python Programming Language.",
        },
        {
            "title": "Python Tutorial - W3Schools",
            "url": "https://www.w3schools.com/python/",
            "snippet": "Python is a popular programming language for web development.",
        },
        {
            "title": "Python Documentation",
            "url": "https://docs.python.org/3/",
            "snippet": "Official Python 3 documentation and tutorials.",
        },
    ],
    "azure": [
        {
            "title": "Microsoft Azure Cloud Platform",
            "url": "https://azure.microsoft.com/",
            "snippet": "Build, run, and manage applications across multiple clouds.",
        },
        {
            "title": "Azure OpenAI Service",
            "url": "https://azure.microsoft.com/products/ai-services/openai-service",
            "snippet": "Apply advanced coding and language models to various use cases.",
        },
    ],
    "agent": [
        {
            "title": "AI Agents - Introduction",
            "url": "https://learn.microsoft.com/ai/agents",
            "snippet": "Learn about AI agents and how to build autonomous systems.",
        },
        {
            "title": "Multi-Agent Systems Overview",
            "url": "https://arxiv.org/abs/agent-systems",
            "snippet": "Research on coordinating multiple AI agents for complex tasks.",
        },
    ],
    "default": [
        {
            "title": "Search Result 1",
            "url": "https://example.com/result1",
            "snippet": "This is a sample search result for your query.",
        },
        {
            "title": "Search Result 2",
            "url": "https://example.com/result2",
            "snippet": "Another relevant result matching your search terms.",
        },
    ],
}

# Mock weather data
MOCK_WEATHER = {
    "seattle": {"temperature": 12, "condition": "Cloudy", "humidity": 75},
    "san francisco": {"temperature": 18, "condition": "Foggy", "humidity": 80},
    "new york": {"temperature": 22, "condition": "Partly Cloudy", "humidity": 65},
    "london": {"temperature": 15, "condition": "Rainy", "humidity": 85},
    "tokyo": {"temperature": 25, "condition": "Sunny", "humidity": 60},
    "default": {"temperature": 20, "condition": "Clear", "humidity": 50},
}


@ai_function
async def search_web(
    query: Annotated[str, "Search query string"],
    max_results: Annotated[int, "Maximum number of results to return (1-20)"] = 5,
) -> dict[str, Any]:
    """Search the web for information.

    This is a mock implementation for workshop purposes. In production,
    this would integrate with a real search API.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (1-20).

    Returns:
        Dictionary with search results and total count.

    Raises:
        ValueError: If query is empty or max_results is out of range.

    Example:
        results = await search_web("python programming", max_results=3)
        for result in results["results"]:
            print(f"{result['title']}: {result['url']}")
    """
    with tracer.start_as_current_span("search_web") as span:
        span.set_attributes(
            create_span_attributes(
                tool_name="search_web",
                query=query,
                max_results=str(max_results),
            )
        )

        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if not 1 <= max_results <= 20:
            raise ValueError("max_results must be between 1 and 20")

        query_lower = query.lower().strip()
        logger.info(f"Searching web for: {query_lower}")

        # Find matching mock results
        results = []
        for keyword, mock_results in MOCK_SEARCH_RESULTS.items():
            if keyword != "default" and keyword in query_lower:
                results.extend(mock_results)

        # Use default results if no matches found
        if not results:
            results = MOCK_SEARCH_RESULTS["default"].copy()

        # Shuffle for variety and limit to max_results
        random.shuffle(results)
        results = results[:max_results]

        response = {
            "results": results,
            "total_found": len(results),
        }

        span.set_attribute("result_count", len(results))
        logger.info(f"Search returned {len(results)} results")

        return response


@ai_function
async def get_weather(
    location: Annotated[str, "City name or location"],
    units: Annotated[Literal["celsius", "fahrenheit"], "Temperature units"] = "celsius",
) -> dict[str, Any]:
    """Get current weather for a location.

    This is a mock implementation for workshop purposes. In production,
    this would integrate with a real weather API.

    Args:
        location: City name or location.
        units: Temperature units ('celsius' or 'fahrenheit').

    Returns:
        Dictionary with weather information.

    Raises:
        ValueError: If location is empty.

    Example:
        weather = await get_weather("Seattle", units="fahrenheit")
        print(f"Temperature: {weather['temperature']}°F")
    """
    with tracer.start_as_current_span("get_weather") as span:
        span.set_attributes(
            create_span_attributes(
                tool_name="get_weather",
                location=location,
                units=units,
            )
        )

        # Validate inputs
        if not location or not location.strip():
            raise ValueError("Location cannot be empty")

        location_lower = location.lower().strip()
        logger.info(f"Getting weather for: {location_lower}")

        # Find matching mock weather
        weather_data = MOCK_WEATHER.get(
            location_lower, MOCK_WEATHER["default"]
        ).copy()

        temperature = weather_data["temperature"]

        # Convert to Fahrenheit if requested
        if units == "fahrenheit":
            temperature = round(temperature * 9 / 5 + 32, 1)

        response = {
            "location": location.strip(),
            "temperature": temperature,
            "units": units,
            "condition": weather_data["condition"],
            "humidity": weather_data["humidity"],
        }

        span.set_attribute("temperature", temperature)
        span.set_attribute("condition", weather_data["condition"])
        logger.info(f"Weather for {location}: {temperature}° {units}")

        return response
