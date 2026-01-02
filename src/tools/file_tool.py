"""File operation tools for MCP integration.

This module provides file read/write capabilities
that can be exposed via the MCP server and used with Microsoft Agent Framework.

For workshop safety, all file operations are sandboxed
to a specific workspace directory.

Tools use the @ai_function decorator for integration with ChatAgent.
"""

import logging
import os
from pathlib import Path
from typing import Annotated, Any, Literal

from agent_framework import ai_function

from src.common.exceptions import ToolError
from src.common.telemetry import create_span_attributes, get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

# Default workspace for file operations
# In production, this would be configurable
DEFAULT_WORKSPACE = Path.cwd() / "workspace"


def _get_safe_path(
    path: str,
    workspace: Path | None = None,
) -> Path:
    """Resolve a path safely within the workspace sandbox.

    Args:
        path: Relative path to resolve.
        workspace: Base workspace directory.

    Returns:
        Resolved absolute path within workspace.

    Raises:
        ToolError: If path attempts to escape workspace.
    """
    workspace = workspace or DEFAULT_WORKSPACE

    # Ensure workspace exists
    workspace.mkdir(parents=True, exist_ok=True)

    # Resolve the path
    resolved = (workspace / path).resolve()

    # Security check: ensure path is within workspace
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError:
        raise ToolError(
            "Path attempts to escape workspace sandbox",
            tool_name="file_operation",
            details={"path": path, "workspace": str(workspace)},
        )

    return resolved


@ai_function
async def read_file(
    path: Annotated[str, "Relative path to the file within the workspace"],
    encoding: Annotated[Literal["utf-8", "ascii", "latin-1"], "File encoding to use"] = "utf-8",
    workspace: Path | None = None,
) -> dict[str, Any]:
    """Read the contents of a file from the workspace.

    Args:
        path: Relative path to the file within the workspace.
        encoding: File encoding to use.
        workspace: Base workspace directory (defaults to ./workspace).

    Returns:
        Dictionary with file content, size, and encoding.

    Raises:
        ToolError: If file doesn't exist or can't be read.

    Example:
        result = await read_file("data/config.json")
        print(result["content"])
    """
    with tracer.start_as_current_span("read_file") as span:
        span.set_attributes(
            create_span_attributes(
                tool_name="read_file",
                path=path,
                encoding=encoding,
            )
        )

        logger.info(f"Reading file: {path}")

        try:
            safe_path = _get_safe_path(path, workspace)

            if not safe_path.exists():
                raise ToolError(
                    f"File not found: {path}",
                    tool_name="read_file",
                    details={"path": path},
                )

            if not safe_path.is_file():
                raise ToolError(
                    f"Path is not a file: {path}",
                    tool_name="read_file",
                    details={"path": path},
                )

            content = safe_path.read_text(encoding=encoding)
            size_bytes = safe_path.stat().st_size

            response = {
                "content": content,
                "size_bytes": size_bytes,
                "encoding": encoding,
            }

            span.set_attribute("size_bytes", size_bytes)
            logger.info(f"Read {size_bytes} bytes from {path}")

            return response

        except ToolError:
            raise
        except UnicodeDecodeError as e:
            raise ToolError(
                f"Failed to decode file with {encoding} encoding",
                tool_name="read_file",
                details={"path": path, "encoding": encoding, "error": str(e)},
            ) from e
        except PermissionError as e:
            raise ToolError(
                f"Permission denied reading file: {path}",
                tool_name="read_file",
                details={"path": path},
            ) from e
        except OSError as e:
            raise ToolError(
                f"Error reading file: {e}",
                tool_name="read_file",
                details={"path": path, "error": str(e)},
            ) from e


@ai_function
async def write_file(
    path: Annotated[str, "Relative path to the file within the workspace"],
    content: Annotated[str, "Content to write to the file"],
    mode: Annotated[Literal["overwrite", "append"], "Write mode"] = "overwrite",
    workspace: Path | None = None,
) -> dict[str, Any]:
    """Write content to a file in the workspace.

    Args:
        path: Relative path to the file within the workspace.
        content: Content to write.
        mode: Write mode - 'overwrite' replaces content, 'append' adds to end.
        workspace: Base workspace directory (defaults to ./workspace).

    Returns:
        Dictionary with success status, bytes written, and path.

    Raises:
        ToolError: If file can't be written.

    Example:
        result = await write_file("output/results.txt", "Analysis complete!")
        print(f"Wrote {result['bytes_written']} bytes")
    """
    with tracer.start_as_current_span("write_file") as span:
        span.set_attributes(
            create_span_attributes(
                tool_name="write_file",
                path=path,
                mode=mode,
                content_length=str(len(content)),
            )
        )

        logger.info(f"Writing file: {path} (mode={mode})")

        try:
            safe_path = _get_safe_path(path, workspace)

            # Create parent directories if needed
            safe_path.parent.mkdir(parents=True, exist_ok=True)

            # Determine write mode
            file_mode = "a" if mode == "append" else "w"

            with open(safe_path, file_mode, encoding="utf-8") as f:
                bytes_written = f.write(content)

            response = {
                "success": True,
                "bytes_written": bytes_written,
                "path": str(safe_path.relative_to(workspace or DEFAULT_WORKSPACE)),
            }

            span.set_attribute("bytes_written", bytes_written)
            logger.info(f"Wrote {bytes_written} bytes to {path}")

            return response

        except ToolError:
            raise
        except PermissionError as e:
            raise ToolError(
                f"Permission denied writing file: {path}",
                tool_name="write_file",
                details={"path": path},
            ) from e
        except OSError as e:
            raise ToolError(
                f"Error writing file: {e}",
                tool_name="write_file",
                details={"path": path, "error": str(e)},
            ) from e


@ai_function
async def list_files(
    directory: Annotated[str, "Relative path to directory within workspace"] = ".",
    pattern: Annotated[str, "Glob pattern to filter files"] = "*",
    workspace: Path | None = None,
) -> dict[str, Any]:
    """List files in a directory within the workspace.

    Args:
        directory: Relative path to directory.
        pattern: Glob pattern to filter files.
        workspace: Base workspace directory.

    Returns:
        Dictionary with list of files.

    Example:
        result = await list_files("data", "*.json")
        for file in result["files"]:
            print(file)
    """
    with tracer.start_as_current_span("list_files") as span:
        span.set_attributes(
            create_span_attributes(
                tool_name="list_files",
                directory=directory,
                pattern=pattern,
            )
        )

        logger.info(f"Listing files in: {directory} with pattern: {pattern}")

        try:
            safe_path = _get_safe_path(directory, workspace)

            if not safe_path.exists():
                raise ToolError(
                    f"Directory not found: {directory}",
                    tool_name="list_files",
                    details={"directory": directory},
                )

            if not safe_path.is_dir():
                raise ToolError(
                    f"Path is not a directory: {directory}",
                    tool_name="list_files",
                    details={"directory": directory},
                )

            files = []
            for item in safe_path.glob(pattern):
                rel_path = item.relative_to(workspace or DEFAULT_WORKSPACE)
                files.append({
                    "name": item.name,
                    "path": str(rel_path),
                    "is_directory": item.is_dir(),
                    "size_bytes": item.stat().st_size if item.is_file() else 0,
                })

            response = {
                "directory": directory,
                "pattern": pattern,
                "files": files,
                "count": len(files),
            }

            span.set_attribute("file_count", len(files))
            logger.info(f"Found {len(files)} files in {directory}")

            return response

        except ToolError:
            raise
        except OSError as e:
            raise ToolError(
                f"Error listing directory: {e}",
                tool_name="list_files",
                details={"directory": directory, "error": str(e)},
            ) from e
