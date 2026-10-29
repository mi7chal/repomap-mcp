# Copyright 2026 RepoMap MCP Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Portions of this file are based on or derived from Aider (aider/repomap.py)
# Copyright (c) Paul Gauthier, licensed under Apache License 2.0.
# Modified for repomap-mcp.

import argparse
from pathlib import Path
from typing import Any

try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        from fastmcp import FastMCP

from . import __version__
from .repomap import RepoMap
from .utils import resolve_scope

mcp = FastMCP(
    "repomap",
    instructions=(
        "Provides token-budgeted, AST-based repository maps with PageRank importance "
        "ranking. Use get_repo_map to orient and locate relevant classes, methods, and functions."
    ),
)


@mcp.tool()
def get_repo_map(
    root_path: str = ".",
    include_tests: bool = False,
    types_only: bool = False,
    focus_files: list[str] | None = None,
    other_files: list[str] | None = None,
    mentioned_idents: list[str] | None = None,
    max_map_tokens: int = 2048,
    force_refresh: bool = False,
) -> str:
    """Generate an Aider-style token-budgeted repository map with PageRank ranking.

    Args:
        root_path: Target directory (scopes candidate files; '.' maps entire repo).
        include_tests: Include unit tests and test directories. Defaults to False.
        types_only: Include only type definitions (structs, classes, interfaces, enums). Defaults to False.
        focus_files: Files currently active in editor to bias PageRank towards.
        other_files: Explicit list of files to map. Defaults to discovered files.
        mentioned_idents: Identifiers mentioned in current prompt to prioritize.
        max_map_tokens: Token budget for the output map.
        force_refresh: Force re-parsing of AST tags.

    Returns:
        Concise code skeleton highlighting key definitions and call signatures.
    """
    git_root, target_dir = resolve_scope(root_path)
    if not target_dir.is_dir():
        return f"Error: Directory not found: {root_path}"

    mapper = RepoMap(root=git_root, target_dir=target_dir, max_map_tokens=max_map_tokens)
    return mapper.get_repo_map(
        focus_files=focus_files,
        other_files=other_files,
        mentioned_idents=mentioned_idents,
        max_map_tokens=max_map_tokens,
        include_tests=include_tests,
        types_only=types_only,
        force_refresh=force_refresh,
    )


@mcp.tool()
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

    mapper = RepoMap(root=git_root, target_dir=target_dir)
    full_path = Path(file_path)
    if not full_path.is_absolute():
        full_path = target_dir / file_path
        if not full_path.exists():
            full_path = git_root / file_path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Local stdio MCP server for repository mapping")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--verbose", action="store_true", help="Log debug information to stderr")
    parser.parse_args()

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
