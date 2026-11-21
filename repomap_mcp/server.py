# Copyright 2026 RepoMap MCP Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions of this file are based on or derived from Aider (aider/repomap.py)
# Copyright (c) Paul Gauthier, licensed under Apache License 2.0.
# Modified for repomap-mcp.

import argparse
import atexit
import contextlib
import inspect
import os
from pathlib import Path
from typing import Any, Literal

try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        from fastmcp import FastMCP

try:
    from mcp.types import ToolAnnotations
except ImportError:
    ToolAnnotations = None  # type: ignore[assignment,misc]

from . import __version__
from .lsp import LspManager
from .repomap import RepoMap
from .utils import resolve_scope

_mapper_cache: dict[tuple[Path, Path, str], RepoMap] = {}
_lsp_manager = LspManager()


def _get_mapper(git_root: Path, target_dir: Path, max_map_tokens: int = 2048) -> RepoMap:
    key = (git_root, target_dir, os.environ.get("REPOMAP_CACHE_DIR", ""))
    mapper = _mapper_cache.get(key)
    if mapper is None:
        mapper = RepoMap(root=git_root, target_dir=target_dir, max_map_tokens=max_map_tokens)
        _mapper_cache[key] = mapper
    return mapper


def _close_mappers() -> None:
    for mapper in list(_mapper_cache.values()):
        mapper.close()
    _mapper_cache.clear()


def _cleanup_sync() -> None:
    _lsp_manager.terminate_all()
    _close_mappers()


atexit.register(_cleanup_sync)


@contextlib.asynccontextmanager
async def _lifespan(_server: Any):
    try:
        yield
    finally:
        await _lsp_manager.close_all()
        _close_mappers()


mcp = FastMCP(
    "repomap",
    instructions=(
        "Provides token-budgeted, AST-based repository maps with PageRank importance "
        "ranking plus read-only LSP semantic navigation. Use get_repo_map (optionally with "
        "line_numbers=True) to orient across files, and lsp for exact definitions, references, "
        "hover types, and diagnostics."
    ),
    lifespan=_lifespan,
)

_TOOL_KW: dict[str, Any] = {}
if ToolAnnotations is not None and "annotations" in inspect.signature(mcp.tool).parameters:
    _TOOL_KW["annotations"] = ToolAnnotations(readOnlyHint=True)


@mcp.tool(**_TOOL_KW)
def get_repo_map(
    root_path: str = ".",
    include_tests: bool = False,
    types_only: bool = False,
    focus_files: list[str] | None = None,
    other_files: list[str] | None = None,
    mentioned_idents: list[str] | None = None,
    max_map_tokens: int = 2048,
    force_refresh: bool = False,
    auto_focus_git: bool = True,
    line_numbers: bool = False,
) -> str:
    """Generate an Aider-style token-budgeted repository map with PageRank ranking.

    Args:
        root_path: Target directory (scopes candidate files; '.' maps entire repo).
        include_tests: Include unit tests and test directories. Defaults to False.
        types_only: Include only type definitions (structs, classes, interfaces, enums).
        focus_files: Files currently active in editor to bias PageRank towards.
        other_files: Explicit list of files to map. Defaults to discovered files.
        mentioned_idents: Identifiers mentioned in current prompt to prioritize.
        max_map_tokens: Token budget for the output map.
        force_refresh: Force re-parsing of AST tags.
        auto_focus_git: Automatically bias ranking towards git modified/untracked files.
        line_numbers: Prefix mapped source lines with 1-indexed line numbers for lsp handoff.
    Returns:
        Concise code skeleton highlighting key definitions and call signatures.
    """
    git_root, target_dir = resolve_scope(root_path)
    if not target_dir.is_dir():
        return f"Error: Directory not found: {root_path}"

    mapper = _get_mapper(git_root, target_dir, max_map_tokens=max_map_tokens)
    return mapper.get_repo_map(
        focus_files=focus_files,
        other_files=other_files,
        mentioned_idents=mentioned_idents,
        max_map_tokens=max_map_tokens,
        include_tests=include_tests,
        types_only=types_only,
        force_refresh=force_refresh,
        auto_focus_git=auto_focus_git,
        line_numbers=line_numbers,
    )


@mcp.tool(**_TOOL_KW)
def get_tags(file_path: str, root_path: str = ".") -> list[dict[str, Any]]:
    """Extract code definitions and references for a specific file via Tree-sitter.

    Args:
        file_path: Path to the target source file.
        root_path: Repository root or target directory.

    Returns:
        List of extracted tags with line numbers, symbol names, kind, and symbol_type.
    """
    git_root, target_dir = resolve_scope(root_path)
    if not target_dir.is_dir():
        return []

    mapper = _get_mapper(git_root, target_dir)
    raw_path = Path(file_path)
    if raw_path.is_absolute():
        full_path = raw_path.resolve()
    else:
        full_path = (target_dir / file_path).resolve()
        if not full_path.exists():
            full_path = (git_root / file_path).resolve()

    rel_path = (
        full_path.relative_to(git_root).as_posix()
        if full_path.is_relative_to(git_root)
        else file_path
    )

    tags = mapper.get_tags(str(full_path), rel_path)
    return [
        {
            "rel_fname": t.rel_fname,
            "line": t.line,
            "name": t.name,
            "kind": t.kind,
            "symbol_type": t.symbol_type,
        }
        for t in tags
    ]


@mcp.tool(**_TOOL_KW)
async def lsp(
    action: Literal["definition", "references", "hover", "diagnostics", "status"],
    file: str | None = None,
    line: int | None = None,
    symbol: str | None = None,
    root_path: str = ".",
    max_results: int = 50,
    timeout: int = 15,
) -> str:
    """Query a local language server (read-only) for exact semantic code intelligence.

    Args:
        action: One of definition, references, hover, diagnostics, or status.
        file: Source file path (required except for action='status').
        line: 1-indexed line number (required for definition, references, hover).
        symbol: Identifier on `line` (supports `name#N` 1-indexed occurrence).
        root_path: Repository or target directory scope.
        max_results: Cap on returned locations/diagnostics (1..200).
        timeout: Per-call timeout in seconds (5..120).
    Returns:
        Compact plain-text LSP results with relative paths and 1-indexed line:col locations.
    """
    return await _lsp_manager.call(
        action=action,
        file=file,
        line=line,
        symbol=symbol,
        root_path=root_path,
        max_results=max_results,
        timeout=timeout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Local stdio MCP server for repository mapping")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args()

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
