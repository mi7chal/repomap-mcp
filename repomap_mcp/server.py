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
from .utils import find_git_root

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
    focus_files: list[str] | None = None,
    other_files: list[str] | None = None,
    mentioned_idents: list[str] | None = None,
    max_map_tokens: int = 2048,
    force_refresh: bool = False,
) -> str:
    """Generate an Aider-style token-budgeted repository map with PageRank ranking.

    Args:
        root_path: Target directory (auto-climbs to nearest git root).
        focus_files: Files currently active in editor to bias PageRank towards.
        other_files: Explicit list of files to map. Defaults to git-tracked files.
        mentioned_idents: Identifiers mentioned in current prompt to prioritize.
        max_map_tokens: Token budget for the output map.
        force_refresh: Force re-parsing of AST tags.

    Returns:
        Concise code skeleton highlighting key definitions and call signatures.
    """
    resolved_root = find_git_root(root_path)
    if not resolved_root.is_dir():
        return f"Error: Directory not found: {root_path}"

    mapper = RepoMap(root=resolved_root, max_map_tokens=max_map_tokens)
    return mapper.get_repo_map(
        focus_files=focus_files,
        other_files=other_files,
        mentioned_idents=mentioned_idents,
        max_map_tokens=max_map_tokens,
        force_refresh=force_refresh,
    )


@mcp.tool()
def get_tags(file_path: str, root_path: str = ".") -> list[dict[str, Any]]:
    """Extract code definitions and references for a specific file via Tree-sitter.

    Args:
        file_path: Path to the target source file.
        root_path: Repository root directory.

    Returns:
        List of extracted tags with line numbers, symbol names, and tag kinds ('def' or 'ref').
    """
    resolved_root = find_git_root(root_path)
    if not resolved_root.is_dir():
        return []

    mapper = RepoMap(root=resolved_root)
    full_path = Path(file_path)
    if not full_path.is_absolute():
        full_path = resolved_root / file_path

    rel_path = (
        full_path.relative_to(resolved_root).as_posix()
        if full_path.is_relative_to(resolved_root)
        else file_path
    )

    tags = mapper.get_tags(str(full_path), rel_path)
    return [
        {
            "rel_fname": t.rel_fname,
            "line": t.line,
            "name": t.name,
            "kind": t.kind,
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
