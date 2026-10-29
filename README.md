# repomap-mcp

Local `stdio` Model Context Protocol (MCP) server for Aider's AST Tree-sitter and PageRank repository mapping.

Deterministic, offline, zero API keys. Generates concise code skeletons to orient LLMs within token budgets.

## Setup

Add to your MCP client config (Claude Desktop, Cursor, Cline):

```json
{
  "mcpServers": {
    "repomap": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/repomap-mcp", "repomap-mcp"]
    }
  }
}
```

Or run directly from GitHub without cloning:

```json
{
  "mcpServers": {
    "repomap": {
      "command": "uv",
      "args": [
        "run",
        "--from",
        "git+https://github.com/mi7chal/repomap-mcp",
        "repomap-mcp"
      ]
    }
  }
}
```

## Tools

### `get_repo_map`
Returns a PageRank-ranked code skeleton fitting within `max_map_tokens`.

- `root_path` (string, default: `"."`): Repository directory (auto-climbs to nearest `.git`).
- `focus_files` (list[string]): Currently active/edited files to bias PageRank towards.
- `other_files` (list[string]): Files to include. Defaults to git-tracked files.
- `mentioned_idents` (list[string]): Symbol names to prioritize.
- `max_map_tokens` (int, default: `2048`): Max token budget.
- `force_refresh` (bool, default: `false`): Invalidate tag cache.

### `get_tags`
Extracts Tree-sitter definitions and references for a specific file.

- `file_path` (string): Target source file.
- `root_path` (string, default: `"."`): Repository directory.

## Cache

- Git repos: `.git/repomap-mcp/cache` (natively ignored by git, zero root pollution).
- Non-git / read-only: `~/.cache/repomap-mcp/tags-<hash>`.
- Override: `REPOMAP_CACHE_DIR` environment variable.

## Development

```bash
uv run pytest
uv run prek run --all-files
```

## AI Transparency

This project was developed with AI assistance.

## License

Apache License 2.0. Derived from [Aider](https://github.com/Aider-AI/aider) by Paul Gauthier. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
