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
Returns a PageRank-ranked code skeleton fitting within `max_map_tokens`, prefixed with coverage telemetry.

- `root_path` (string, default: `"."`): Target directory (scopes candidate files; `"."` maps entire repo).
- `include_tests` (bool, default: `false`): Include unit tests and `tests/` directories.
- `types_only` (bool, default: `false`): Filter to type definitions (structs, classes, interfaces, enums).
- `focus_files` (list[string]): Currently active/edited files to bias PageRank towards.
- `other_files` (list[string]): Explicit file list. Defaults to scoped discovered files.
- `mentioned_idents` (list[string]): Symbol names to prioritize.
- `max_map_tokens` (int, default: `2048`): Max token budget.
- `force_refresh` (bool, default: `false`): Invalidate tag cache.
- `auto_focus_git` (bool, default: `true`): Automatically detect and bias ranking towards git modified/untracked files.
- `line_numbers` (bool, default: `false`): Prefix rendered lines with 1-indexed line numbers for direct `lsp` handoff.

### `get_tags`
Extracts Tree-sitter definitions and references for a specific file.

- `file_path` (string): Target source file.
- `root_path` (string, default: `"."`): Repository or target directory.

### `lsp`
Read-only Language Server Protocol (LSP) queries with lazy startup, automatic disk-edit sync (`didOpen`/`didChange`), and `symbol#N` column resolution.

- `action`: `"definition"`, `"references"`, `"hover"`, `"diagnostics"`, or `"status"`.
- `file` (string, optional): Target file path (required except for `status`).
- `line` (int, optional): 1-indexed line number (`definition`, `references`, `hover`).
- `symbol` (string, optional): Identifier on `line` (supports `name#N`, 1-indexed).
- `root_path` (string, default: `"."`): Repository or target directory.
- `max_results` (int, default: `50`): Output cap (`1..200`).
- `timeout` (int, default: `15`): Request timeout in seconds (`5..120`).

Zero-config detection (checks project `.venv/bin` and `node_modules/.bin`, then `$PATH`): Python (`basedpyright-langserver`, `pyright-langserver`, `pylsp`), TypeScript/JS (`typescript-language-server`), Rust (`rust-analyzer`), Go (`gopls`), C/C++ (`clangd`). Override via `REPOMAP_LSP_SERVERS` JSON env var (e.g. `'{"python": ["pyright-langserver", "--stdio"]}'`).

## Cache

- Git repos: `.git/repomap-mcp/cache` (natively ignored by git, zero root pollution).
- Non-git / read-only: `~/.cache/repomap-mcp/tags-v4-<hash>`.
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
