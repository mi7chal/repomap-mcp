"""Hermetic tests for minimal oh-my-pi-inspired LSP integration."""

import json
import shutil
import sys
from pathlib import Path

import anyio
import pytest

from repomap_mcp.lsp import LspManager, resolve_symbol_column
from repomap_mcp.server import lsp

MOCK_SERVER_CODE = """
import json
import sys

pull_mode = "--pull" in sys.argv
crash_hover = "--crash-hover" in sys.argv
docs = {}
config_acked = False

def read_msg():
    content_length = None
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\\r\\n", b"\\n"):
            break
        s = line.decode("ascii", errors="ignore").strip()
        if s.lower().startswith("content-length:"):
            content_length = int(s.split(":", 1)[1].strip())
    if not content_length:
        return None
    body = sys.stdin.buffer.read(content_length)
    return json.loads(body.decode("utf-8"))

def send_msg(payload):
    body = json.dumps(payload).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\\r\\n\\r\\n".encode("ascii") + body)
    sys.stdout.buffer.flush()

def publish_diags(uri, text):
    if pull_mode:
        return
    diags = []
    for idx, ln in enumerate(text.splitlines()):
        if "BROKEN" in ln:
            diags.append({
                "range": {"start": {"line": idx, "character": 0}, "end": {"line": idx, "character": 6}},
                "severity": 1,
                "source": "mocklsp",
                "code": "E001",
                "message": "broken marker found",
            })
    send_msg({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics", "params": {"uri": uri, "diagnostics": diags}})

while True:
    msg = read_msg()
    if msg is None:
        break
    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        caps = {"hoverProvider": True, "definitionProvider": True, "referencesProvider": True}
        if pull_mode:
            caps["diagnosticProvider"] = {"interFileDependencies": False}
        send_msg({"jsonrpc": "2.0", "id": msg_id, "result": {"capabilities": caps}})
    elif method == "initialized":
        send_msg({"jsonrpc": "2.0", "id": 9001, "method": "workspace/configuration", "params": {"items": [{"section": "python"}]}})
    elif msg_id == 9001 and "result" in msg:
        config_acked = True
    elif method == "textDocument/didOpen":
        td = params["textDocument"]
        docs[td["uri"]] = {"version": td["version"], "text": td["text"]}
        publish_diags(td["uri"], td["text"])
    elif method == "textDocument/didChange":
        td = params["textDocument"]
        text = params["contentChanges"][0]["text"]
        docs[td["uri"]] = {"version": td["version"], "text": text}
        publish_diags(td["uri"], text)
    elif method == "textDocument/definition":
        uri = params["textDocument"]["uri"]
        send_msg({
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": [{
                "targetUri": uri,
                "targetRange": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 10}},
                "targetSelectionRange": {"start": {"line": 0, "character": 4}, "end": {"line": 0, "character": 10}},
            }],
        })
    elif method == "textDocument/references":
        uri = params["textDocument"]["uri"]
        locs = [
            {"uri": uri, "range": {"start": {"line": i, "character": 0}, "end": {"line": i, "character": 3}}}
            for i in range(6)
        ]
        send_msg({"jsonrpc": "2.0", "id": msg_id, "result": locs})
    elif method == "textDocument/hover":
        if crash_hover:
            sys.exit(1)
        uri = params["textDocument"]["uri"]
        pos = params["position"]
        ver = docs.get(uri, {}).get("version", 0)
        send_msg({
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "contents": {
                    "kind": "markdown",
                    "value": f"line={pos['line']} char={pos['character']} ver={ver} cfg={config_acked}",
                }
            },
        })
    elif method == "textDocument/diagnostic":
        uri = params["textDocument"]["uri"]
        text = docs.get(uri, {}).get("text", "")
        items = []
        for idx, ln in enumerate(text.splitlines()):
            if "BROKEN" in ln:
                items.append({
                    "range": {"start": {"line": idx, "character": 0}, "end": {"line": idx, "character": 6}},
                    "severity": 2,
                    "source": "mockpull",
                    "message": "pull warning",
                })
        send_msg({"jsonrpc": "2.0", "id": msg_id, "result": {"kind": "full", "items": items}})
    elif method == "shutdown":
        send_msg({"jsonrpc": "2.0", "id": msg_id, "result": None})
    elif method == "exit":
        break
"""


def test_resolve_symbol_column_and_utf16():
    """Verify #N occurrence matching, word boundaries, and UTF-16 code-unit offsets."""
    line = "foobar = foo(x) + foo(y)"
    # Skips 'foobar' and resolves exact 1st and 2nd 'foo'
    assert resolve_symbol_column(line, "foo") == 9
    assert resolve_symbol_column(line, "foo#2") == 18
    assert resolve_symbol_column(line, "missing") is None

    # Surrogate-pair emoji takes 2 UTF-16 code units in LSP Position.character
    emoji_line = "msg = '🚀'; target()"
    idx = emoji_line.index("target")
    expected_utf16 = len(emoji_line[:idx].encode("utf-16-le")) // 2
    assert expected_utf16 == idx + 1
    assert resolve_symbol_column(emoji_line, "target") == expected_utf16


def test_lsp_missing_binary_graceful(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify missing language server returns helpful guidance instead of raising."""
    src = tmp_path / "hello.rs"
    src.write_text("fn main() {}\n", encoding="utf-8")
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)

    async def run():
        mgr = LspManager()
        try:
            res = await mgr.call(
                "definition", file="hello.rs", line=1, symbol="main", root_path=str(tmp_path)
            )
            assert "No language server found" in res
            assert "rust-analyzer" in res
        finally:
            await mgr.close_all()

    anyio.run(run)


def test_lsp_push_lifecycle_sync_and_truncation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test handshake, server->client config request, didChange, hover, def, refs, push diags."""
    mock_script = tmp_path / "mock_lsp.py"
    mock_script.write_text(MOCK_SERVER_CODE, encoding="utf-8")
    monkeypatch.setenv(
        "REPOMAP_LSP_SERVERS",
        json.dumps({"python": [sys.executable, str(mock_script)]}),
    )

    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    src = tmp_path / "app.py"
    src.write_text(
        "def target():\n    return 1\nx = '🚀'; target()\ny = 2\nz = 3\nw = 4\n",
        encoding="utf-8",
    )

    async def run():
        mgr = LspManager()
        try:
            # 1. Hover verifies UTF-16 emoji offset and didOpen version=1
            hov = await mgr.call(
                "hover", file="app.py", line=3, symbol="target", root_path=str(tmp_path)
            )
            assert "line=2 char=10 ver=1" in hov

            # 2. Definition parses LocationLink targetSelectionRange
            dfn = await mgr.call(
                "definition", file="app.py", line=3, symbol="target", root_path=str(tmp_path)
            )
            assert "app.py:1:5" in dfn
            assert "def target():" in dfn

            # 3. References enforces max_results cap with truncation trailer
            refs = await mgr.call(
                "references",
                file="app.py",
                line=1,
                symbol="target",
                root_path=str(tmp_path),
                max_results=3,
            )
            assert "showing 3 of 6 results" in refs

            # 4. External disk edit triggers didChange (version=2) and push diagnostics
            src.write_text(
                "def target():\n    return 1\nBROKEN = True\n",
                encoding="utf-8",
            )
            diags = await mgr.call("diagnostics", file="app.py", root_path=str(tmp_path))
            assert "app.py:3:1 [error] broken marker found (mocklsp:E001)" in diags

            hov2 = await mgr.call(
                "hover", file="app.py", line=1, symbol="target", root_path=str(tmp_path)
            )
            assert "ver=2 cfg=True" in hov2

            # 5. Status shows active server
            status = await mgr.call("status", root_path=str(tmp_path))
            assert "python (alive)" in status
        finally:
            await mgr.close_all()

    anyio.run(run)


def test_lsp_pull_diagnostics_and_crash_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test LSP 3.17 pull diagnostics and automatic eviction/restart after server crash."""
    mock_script = tmp_path / "mock_lsp.py"
    mock_script.write_text(MOCK_SERVER_CODE, encoding="utf-8")

    (tmp_path / "pyproject.toml").write_text("[project]\nname='pull'\n", encoding="utf-8")
    src = tmp_path / "mod.py"
    src.write_text("BROKEN = 1\ndef ok(): pass\n", encoding="utf-8")

    monkeypatch.setenv(
        "REPOMAP_LSP_SERVERS",
        json.dumps({"python": [sys.executable, str(mock_script), "--pull"]}),
    )

    async def run_pull():
        mgr = LspManager()
        try:
            diags = await mgr.call("diagnostics", file="mod.py", root_path=str(tmp_path))
            assert "mod.py:1:1 [warning] pull warning (mockpull)" in diags
        finally:
            await mgr.close_all()

    anyio.run(run_pull)

    # Crash on hover -> evicted cleanly -> next definition restarts and succeeds
    monkeypatch.setenv(
        "REPOMAP_LSP_SERVERS",
        json.dumps({"python": [sys.executable, str(mock_script), "--crash-hover"]}),
    )

    async def run_crash():
        mgr = LspManager()
        try:
            res1 = await mgr.call(
                "hover", file="mod.py", line=2, symbol="ok", root_path=str(tmp_path)
            )
            assert "LSP hover failed" in res1
            assert len(mgr._clients) == 0

            res2 = await mgr.call(
                "definition", file="mod.py", line=2, symbol="ok", root_path=str(tmp_path)
            )
            assert "mod.py:1:5" in res2
        finally:
            await mgr.close_all()

    anyio.run(run_crash)


def test_mcp_lsp_tool_async_smoke(tmp_path: Path):
    """Verify the FastMCP async `lsp` tool executes end-to-end."""
    out = anyio.run(lsp, "status", None, None, None, str(tmp_path))
    assert "Active LSP servers:" in out
    assert "Detected binaries:" in out
