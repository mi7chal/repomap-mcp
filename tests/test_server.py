"""Tests for repomap-mcp."""

import json
import os
import subprocess
import sys
from pathlib import Path

import anyio
import pytest

from repomap_mcp.repomap import RepoMap, compute_pagerank
from repomap_mcp.server import get_repo_map, get_tags, mcp
from repomap_mcp.utils import count_tokens, find_git_root, get_repo_files


def test_token_counter():
    """Verify deterministic token counting without network calls."""
    assert count_tokens("") == 0
    cnt = count_tokens("def calculate(x, y): return x + y")
    assert cnt > 0
    assert count_tokens("a" * 100) > 0


def test_git_root_and_files():
    """Verify git root detection and repository file scanning."""
    root = find_git_root(".")
    assert (root / ".git").exists()

    files = get_repo_files(root)
    assert len(files) > 0
    assert any("server.py" in f for f in files)
    assert not any(".git/" in f for f in files)


def test_pure_pagerank():
    """Verify pure-Python PageRank implementation."""
    import networkx as nx

    g = nx.MultiDiGraph()
    g.add_node("a")
    g.add_node("b")
    g.add_node("c")
    g.add_edge("a", "b")
    g.add_edge("c", "b")

    ranks = compute_pagerank(g)
    assert len(ranks) == 3
    # 'b' has 2 in-edges, should have highest rank
    assert ranks["b"] > ranks["a"]
    assert ranks["b"] > ranks["c"]


def test_repomap_generation(tmp_path: Path):
    """Test AST parsing, PageRank ranking, and token budget fitting in a mock codebase."""
    app_py = tmp_path / "app.py"
    app_py.write_text(
        "from models import DatabaseUser\n\n"
        "def run_app():\n"
        "    user = DatabaseUser('alice')\n"
        "    return user.get_display_name()\n",
        encoding="utf-8",
    )

    models_py = tmp_path / "models.py"
    models_py.write_text(
        "class DatabaseUser:\n"
        "    def __init__(self, username):\n"
        "        self.username = username\n\n"
        "    def get_display_name(self):\n"
        "        return self.username.capitalize()\n",
        encoding="utf-8",
    )

    mapper = RepoMap(root=tmp_path, max_map_tokens=1000)
    repomap_out = mapper.get_repo_map(
        focus_files=["app.py"],
        other_files=["app.py", "models.py"],
        mentioned_idents=["DatabaseUser"],
    )

    assert "models.py" in repomap_out
    assert "DatabaseUser" in repomap_out
    assert "get_display_name" in repomap_out

    # Verify token budgeting works when restricted
    small_map = mapper.get_repo_map(
        other_files=["app.py", "models.py"],
        max_map_tokens=20,
    )
    assert count_tokens(small_map) <= 75


def test_get_tags(tmp_path: Path):
    """Verify Tree-sitter tag extraction."""
    test_py = tmp_path / "service.py"
    test_py.write_text(
        "class PaymentService:\n    def process(self, amount):\n        pass\n",
        encoding="utf-8",
    )

    tags = get_tags("service.py", root_path=str(tmp_path))
    def_names = [t["name"] for t in tags if t["kind"] == "def"]
    assert "PaymentService" in def_names
    assert "process" in def_names


def test_cache_location_inside_git(tmp_path: Path):
    """Verify default cache path is inside .git/repomap-mcp/cache."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir(parents=True)

    mapper = RepoMap(root=tmp_path)
    expected_cache = tmp_path / ".git" / "repomap-mcp" / "cache"
    assert mapper.cache_dir == expected_cache
    assert expected_cache.exists()


def test_cache_fallback_no_git(tmp_path: Path):
    """Verify non-git directories fall back to ~/.cache/repomap-mcp/tags-v3-<hash>."""
    import hashlib

    no_git_dir = tmp_path / "plain_dir"
    no_git_dir.mkdir()

    mapper = RepoMap(root=no_git_dir)
    h = hashlib.sha256(str(no_git_dir).encode("utf-8")).hexdigest()[:12]
    expected_cache = Path.home() / ".cache" / "repomap-mcp" / f"tags-v3-{h}"
    assert mapper.cache_dir == expected_cache
    assert expected_cache.exists()


def test_cache_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify cache respects custom REPOMAP_CACHE_DIR and falls back properly."""
    custom_cache = tmp_path / "custom_cache"
    monkeypatch.setenv("REPOMAP_CACHE_DIR", str(custom_cache))

    mapper = RepoMap(root=tmp_path)
    assert mapper.cache_dir == custom_cache
    assert custom_cache.exists()


def test_subdirectory_scoping(tmp_path: Path):
    """Verify root_path scopes candidate files to requested folder."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "api.py").write_text("def start_server(): pass\n")

    web_dir = tmp_path / "web"
    web_dir.mkdir()
    (web_dir / "app.py").write_text("def render_ui(): pass\n")

    # Scoped to web: only web/app.py should be mapped
    scoped_out = get_repo_map(root_path=str(web_dir))
    assert "app.py" in scoped_out
    assert "api.py" not in scoped_out

    # Scoped to repo root: both mapped
    all_out = get_repo_map(root_path=str(tmp_path))
    assert "app.py" in all_out
    assert "api.py" in all_out


def test_telemetry_header(tmp_path: Path):
    """Verify telemetry JSON header metadata is always present."""
    (tmp_path / "main.py").write_text("def run(): pass\n")

    out = get_repo_map(root_path=str(tmp_path))
    assert out.startswith("```json")
    assert '"total_repo_files":' in out
    assert '"mapped_files":' in out
    assert '"dropped_files":' in out
    assert '"budget_exhausted":' in out


def test_prune_test_files(tmp_path: Path):
    """Verify automatic pruning of tests/ dirs and test files."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "core.py").write_text("def core_logic(): pass\n")

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_core.py").write_text("def test_core(): pass\n")

    # Pruned by default
    pruned = get_repo_map(root_path=str(tmp_path), include_tests=False)
    assert "core_logic" in pruned
    assert "test_core" not in pruned

    # Included when requested
    included = get_repo_map(root_path=str(tmp_path), include_tests=True)
    assert "core_logic" in included
    assert "test_core" in included


def test_types_only(tmp_path: Path):
    """Verify types_only filter renders only structs/classes/types, skipping function bodies."""
    (tmp_path / "models.py").write_text(
        "class UserModel:\n    id: int\n\n\n\n\ndef helper_func():\n    pass\n"
    )

    all_out = get_repo_map(root_path=str(tmp_path), types_only=False)
    assert "UserModel" in all_out
    assert "helper_func" in all_out

    types_out = get_repo_map(root_path=str(tmp_path), types_only=True)
    assert "UserModel" in types_out
    assert "helper_func" not in types_out


def test_mcp_tools_registered():
    """Verify MCP tools are registered with valid input schemas."""
    tools = anyio.run(mcp.list_tools)
    tool_names = [t.name for t in tools]
    assert "get_repo_map" in tool_names
    assert "get_tags" in tool_names


def test_stdio_jsonrpc_handshake_and_tool_call(tmp_path: Path):
    """End-to-end test verifying full JSON-RPC over stdio subprocess."""
    test_file = tmp_path / "calc.py"
    test_file.write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )

    cmd = [sys.executable, "-m", "repomap_mcp"]
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)}

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    try:

        def send(req):
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            return json.loads(line)

        # 1. Initialize
        init_res = send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0"},
                },
            }
        )
        assert init_res["id"] == 1
        assert "serverInfo" in init_res["result"]

        # 2. Initialized notification
        proc.stdin.write(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        )
        proc.stdin.flush()

        # 3. Call get_repo_map tool
        call_res = send(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "get_repo_map",
                    "arguments": {
                        "root_path": str(tmp_path),
                        "max_map_tokens": 500,
                    },
                },
            }
        )
        assert call_res["id"] == 2
        content_items = call_res["result"]["content"]
        assert len(content_items) > 0
        text = content_items[0]["text"]
        assert "calc.py" in text
        assert "add" in text
    finally:
        proc.stdin.close()
        proc.wait(timeout=5)
