"""Unit tests for file tool operations.

Tests for src/tools/file_tool.py - file operations with workspace safety.
"""

from __future__ import annotations

import pytest
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch, Mock

from src.tools.file_tool import (
    read_file,
    write_file,
    list_files,
    _get_safe_path,
    DEFAULT_WORKSPACE,
)
from src.common.exceptions import ToolError


# =============================================================================
# Test File Tool Imports
# =============================================================================


class TestFileToolImports:
    """Test that file tool module can be imported."""

    def test_import_read_file(self) -> None:
        """Test importing read_file."""
        from src.tools.file_tool import read_file
        assert read_file is not None

    def test_import_write_file(self) -> None:
        """Test importing write_file."""
        from src.tools.file_tool import write_file
        assert write_file is not None

    def test_import_list_files(self) -> None:
        """Test importing list_files."""
        from src.tools.file_tool import list_files
        assert list_files is not None

    def test_import_get_safe_path(self) -> None:
        """Test importing _get_safe_path."""
        from src.tools.file_tool import _get_safe_path
        assert _get_safe_path is not None


# =============================================================================
# Test _get_safe_path Helper
# =============================================================================


class TestGetSafePath:
    """Test the _get_safe_path helper function."""

    def test_returns_path_within_workspace(self, tmp_path: Path) -> None:
        """Test safe path stays within workspace."""
        result = _get_safe_path("test.txt", tmp_path)
        assert result.parent == tmp_path
        assert result.name == "test.txt"

    def test_handles_subdirectory(self, tmp_path: Path) -> None:
        """Test safe path with subdirectory."""
        result = _get_safe_path("subdir/test.txt", tmp_path)
        assert tmp_path in result.parents

    def test_raises_on_path_traversal(self, tmp_path: Path) -> None:
        """Test that path traversal attempts are rejected."""
        with pytest.raises(ToolError) as exc_info:
            _get_safe_path("../../../etc/passwd", tmp_path)
        assert "outside workspace" in str(exc_info.value).lower() or "traversal" in str(exc_info.value).lower()

    def test_raises_on_absolute_path(self, tmp_path: Path) -> None:
        """Test that absolute paths are rejected."""
        with pytest.raises(ToolError):
            _get_safe_path("/absolute/path", tmp_path)


# =============================================================================
# Test read_file
# =============================================================================


class TestReadFile:
    """Test read_file function."""

    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_path: Path) -> None:
        """Test reading an existing file."""
        test_file = tmp_path / "test.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content, encoding="utf-8")

        result = await read_file("test.txt", workspace=tmp_path)

        assert result["content"] == test_content
        assert result["size_bytes"] == len(test_content)
        assert result["encoding"] == "utf-8"

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, tmp_path: Path) -> None:
        """Test reading non-existent file raises ToolError."""
        with pytest.raises(ToolError) as exc_info:
            await read_file("nonexistent.txt", workspace=tmp_path)
        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_read_directory_raises_error(self, tmp_path: Path) -> None:
        """Test reading a directory raises ToolError."""
        (tmp_path / "subdir").mkdir()
        with pytest.raises(ToolError) as exc_info:
            await read_file("subdir", workspace=tmp_path)
        assert "not a file" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_read_with_different_encoding(self, tmp_path: Path) -> None:
        """Test reading with different encoding."""
        test_file = tmp_path / "latin.txt"
        test_content = "Héllo Wörld"
        test_file.write_text(test_content, encoding="latin-1")

        result = await read_file("latin.txt", encoding="latin-1", workspace=tmp_path)

        assert result["content"] == test_content
        assert result["encoding"] == "latin-1"

    @pytest.mark.asyncio
    async def test_read_with_wrong_encoding_raises_error(self, tmp_path: Path) -> None:
        """Test reading with wrong encoding raises ToolError."""
        test_file = tmp_path / "binary.bin"
        test_file.write_bytes(b"\xff\xfe\x00\x01")

        with pytest.raises(ToolError) as exc_info:
            await read_file("binary.bin", encoding="utf-8", workspace=tmp_path)
        # Should raise decode error
        assert "decode" in str(exc_info.value).lower() or "encoding" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_read_file_in_subdirectory(self, tmp_path: Path) -> None:
        """Test reading file in subdirectory."""
        subdir = tmp_path / "data"
        subdir.mkdir()
        test_file = subdir / "config.json"
        test_content = '{"key": "value"}'
        test_file.write_text(test_content)

        result = await read_file("data/config.json", workspace=tmp_path)

        assert result["content"] == test_content


# =============================================================================
# Test write_file
# =============================================================================


class TestWriteFile:
    """Test write_file function."""

    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_path: Path) -> None:
        """Test writing a new file."""
        result = await write_file(
            "output.txt",
            "Hello, World!",
            workspace=tmp_path,
        )

        assert result["success"] is True
        assert result["bytes_written"] == 13

        # Verify file was created
        written_file = tmp_path / "output.txt"
        assert written_file.exists()
        assert written_file.read_text() == "Hello, World!"

    @pytest.mark.asyncio
    async def test_write_overwrite_mode(self, tmp_path: Path) -> None:
        """Test write with overwrite mode."""
        test_file = tmp_path / "existing.txt"
        test_file.write_text("Original content")

        result = await write_file(
            "existing.txt",
            "New content",
            mode="overwrite",
            workspace=tmp_path,
        )

        assert result["success"] is True
        assert test_file.read_text() == "New content"

    @pytest.mark.asyncio
    async def test_write_append_mode(self, tmp_path: Path) -> None:
        """Test write with append mode."""
        test_file = tmp_path / "appendable.txt"
        test_file.write_text("Line 1\n")

        result = await write_file(
            "appendable.txt",
            "Line 2\n",
            mode="append",
            workspace=tmp_path,
        )

        assert result["success"] is True
        assert test_file.read_text() == "Line 1\nLine 2\n"

    @pytest.mark.asyncio
    async def test_write_creates_parent_directories(self, tmp_path: Path) -> None:
        """Test write creates parent directories if needed."""
        result = await write_file(
            "deep/nested/dir/file.txt",
            "Nested content",
            workspace=tmp_path,
        )

        assert result["success"] is True
        nested_file = tmp_path / "deep" / "nested" / "dir" / "file.txt"
        assert nested_file.exists()
        assert nested_file.read_text() == "Nested content"

    @pytest.mark.asyncio
    async def test_write_file_returns_relative_path(self, tmp_path: Path) -> None:
        """Test write returns relative path."""
        result = await write_file(
            "subdir/output.txt",
            "Content",
            workspace=tmp_path,
        )

        assert "subdir" in result["path"]
        assert "output.txt" in result["path"]


# =============================================================================
# Test list_files
# =============================================================================


class TestListFiles:
    """Test list_files function."""

    @pytest.mark.asyncio
    async def test_list_empty_directory(self, tmp_path: Path) -> None:
        """Test listing empty directory."""
        result = await list_files(".", workspace=tmp_path)

        assert result["directory"] == "."
        assert result["count"] == 0
        assert result["files"] == []

    @pytest.mark.asyncio
    async def test_list_files_with_content(self, tmp_path: Path) -> None:
        """Test listing directory with files."""
        (tmp_path / "file1.txt").write_text("content 1")
        (tmp_path / "file2.txt").write_text("content 2")
        (tmp_path / "subdir").mkdir()

        result = await list_files(".", workspace=tmp_path)

        assert result["count"] == 3
        names = [f["name"] for f in result["files"]]
        assert "file1.txt" in names
        assert "file2.txt" in names
        assert "subdir" in names

    @pytest.mark.asyncio
    async def test_list_files_with_pattern(self, tmp_path: Path) -> None:
        """Test listing with glob pattern."""
        (tmp_path / "data.json").write_text("{}")
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "readme.txt").write_text("readme")

        result = await list_files(".", pattern="*.json", workspace=tmp_path)

        assert result["count"] == 2
        names = [f["name"] for f in result["files"]]
        assert "data.json" in names
        assert "config.json" in names
        assert "readme.txt" not in names

    @pytest.mark.asyncio
    async def test_list_files_directory_not_found(self, tmp_path: Path) -> None:
        """Test listing non-existent directory."""
        with pytest.raises(ToolError) as exc_info:
            await list_files("nonexistent", workspace=tmp_path)
        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_list_files_on_file_raises_error(self, tmp_path: Path) -> None:
        """Test listing a file (not directory) raises error."""
        (tmp_path / "file.txt").write_text("content")

        with pytest.raises(ToolError) as exc_info:
            await list_files("file.txt", workspace=tmp_path)
        assert "not a directory" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_list_files_returns_size_info(self, tmp_path: Path) -> None:
        """Test listed files include size information."""
        test_content = "Hello, World!"
        (tmp_path / "sized.txt").write_text(test_content)

        result = await list_files(".", pattern="*.txt", workspace=tmp_path)

        assert result["count"] == 1
        file_info = result["files"][0]
        assert file_info["name"] == "sized.txt"
        assert file_info["size_bytes"] == len(test_content)
        assert file_info["is_directory"] is False

    @pytest.mark.asyncio
    async def test_list_files_marks_directories(self, tmp_path: Path) -> None:
        """Test directories are marked correctly."""
        (tmp_path / "subdir").mkdir()

        result = await list_files(".", workspace=tmp_path)

        dir_entry = next(f for f in result["files"] if f["name"] == "subdir")
        assert dir_entry["is_directory"] is True
        assert dir_entry["size_bytes"] == 0
