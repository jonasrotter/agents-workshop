"""Calculator tool for MCP integration.

This module provides mathematical calculation capabilities
that can be exposed via the MCP server and used with Microsoft Agent Framework.

Tools use the @ai_function decorator for integration with ChatAgent.
"""

import logging
import math
from typing import Annotated, Any, Literal

from agent_framework import ai_function

from src.common.exceptions import ToolError
from src.common.telemetry import create_span_attributes, get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

OperationType = Literal["add", "subtract", "multiply", "divide", "power", "sqrt"]


@ai_function
async def calculate(
    operation: Annotated[OperationType, "The mathematical operation to perform"],
    a: Annotated[float, "First operand"],
    b: Annotated[float | None, "Second operand (not required for sqrt)"] = None,
) -> dict[str, Any]:
    """Perform mathematical calculations.

    Supports basic arithmetic operations and some mathematical functions.

    Args:
        operation: The operation to perform.
            - add: a + b
            - subtract: a - b
            - multiply: a * b
            - divide: a / b
            - power: a ^ b
            - sqrt: √a (b is ignored)
        a: First operand.
        b: Second operand (not required for sqrt).

    Returns:
        Dictionary with result, operation name, and expression string.

    Raises:
        ToolError: If the operation is invalid or mathematically impossible.

    Example:
        result = await calculate("add", 5, 3)
        print(f"{result['expression']} = {result['result']}")  # 5 + 3 = 8

        sqrt_result = await calculate("sqrt", 16)
        print(f"√{16} = {sqrt_result['result']}")  # √16 = 4
    """
    with tracer.start_as_current_span("calculate") as span:
        span.set_attributes(
            create_span_attributes(
                tool_name="calculate",
                operation=operation,
                operand_a=str(a),
                operand_b=str(b) if b is not None else "N/A",
            )
        )

        logger.info(f"Calculating: {operation}({a}, {b})")

        # Validate operation requires b (except sqrt)
        if operation != "sqrt" and b is None:
            raise ToolError(
                f"Operation '{operation}' requires two operands",
                tool_name="calculate",
                details={"operation": operation, "a": a, "b": b},
            )

        result: float
        expression: str

        try:
            if operation == "add":
                result = a + b  # type: ignore[operator]
                expression = f"{a} + {b}"

            elif operation == "subtract":
                result = a - b  # type: ignore[operator]
                expression = f"{a} - {b}"

            elif operation == "multiply":
                result = a * b  # type: ignore[operator]
                expression = f"{a} × {b}"

            elif operation == "divide":
                if b == 0:
                    raise ToolError(
                        "Division by zero is not allowed",
                        tool_name="calculate",
                        details={"operation": "divide", "a": a, "b": b},
                    )
                result = a / b  # type: ignore[operator]
                expression = f"{a} ÷ {b}"

            elif operation == "power":
                result = math.pow(a, b)  # type: ignore[arg-type]
                expression = f"{a} ^ {b}"

            elif operation == "sqrt":
                if a < 0:
                    raise ToolError(
                        "Cannot calculate square root of negative number",
                        tool_name="calculate",
                        details={"operation": "sqrt", "a": a},
                    )
                result = math.sqrt(a)
                expression = f"√{a}"

            else:
                raise ToolError(
                    f"Unknown operation: {operation}",
                    tool_name="calculate",
                    details={"operation": operation},
                )

        except OverflowError as e:
            raise ToolError(
                "Calculation resulted in overflow",
                tool_name="calculate",
                details={"operation": operation, "a": a, "b": b},
            ) from e

        # Round to reasonable precision to avoid floating point artifacts
        if isinstance(result, float):
            result = round(result, 10)
            # Convert to int if it's a whole number
            if result == int(result):
                result = int(result)

        response = {
            "result": result,
            "operation": operation,
            "expression": expression,
        }

        span.set_attribute("result", str(result))
        logger.info(f"Calculation result: {expression} = {result}")

        return response
