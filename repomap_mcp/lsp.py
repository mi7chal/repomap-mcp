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

"""Minimal read-only LSP client & manager inspired by oh-my-pi."""

import asyncio
import json
import os
import re
import shutil
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .utils import find_git_root, is_too_large, resolve_scope

_SEVERITY_NAMES = {1: "error", 2: "warning", 3: "info", 4: "hint"}
_HOVER_MAX_CHARS = 4000


@dataclass(frozen=True)
class ServerSpec:
    id: str
    extensions: dict[str, str]
    commands: tuple[tuple[str, tuple[str, ...]], ...]
    root_markers: tuple[str, ...]
    install_hint: str


SERVERS: tuple[ServerSpec, ...] = (
    ServerSpec(
        id="python",
        extensions={".py": "python", ".pyi": "python"},
        commands=(
            ("basedpyright-langserver", ("--stdio",)),
            ("pyright-langserver", ("--stdio",)),
            ("pylsp", ()),
        ),
        root_markers=("pyproject.toml", "setup.py", "pyrightconfig.json", ".git"),
        install_hint="pip install basedpyright (or pyright)",
    ),
    ServerSpec(
        id="typescript",
        extensions={
            ".ts": "typescript",
            ".tsx": "typescriptreact",
            ".js": "javascript",
            ".jsx": "javascriptreact",
            ".mjs": "javascript",
            ".cjs": "javascript",
        },
        commands=(("typescript-language-server", ("--stdio",)),),
        root_markers=("tsconfig.json", "jsconfig.json", "package.json", ".git"),
        install_hint="npm i -g typescript-language-server typescript",
    ),
    ServerSpec(
        id="rust",
        extensions={".rs": "rust"},
        commands=(("rust-analyzer", ()),),
        root_markers=("Cargo.toml",),
        install_hint="rustup component add rust-analyzer",
    ),
    ServerSpec(
        id="go",
        extensions={".go": "go"},
        commands=(("gopls", ()),),
        root_markers=("go.mod",),
        install_hint="go install golang.org/x/tools/gopls@latest",
    ),
    ServerSpec(
        id="c_cpp",
        extensions={
            ".c": "c",
            ".h": "c",
            ".cc": "cpp",
            ".cpp": "cpp",
            ".cxx": "cpp",
            ".hpp": "cpp",
        },
        commands=(("clangd", ()),),
        root_markers=("compile_commands.json", "CMakeLists.txt", ".git"),
        install_hint="install clangd via llvm / system package manager",
    ),
)

_EXT_TO_SPEC: dict[str, ServerSpec] = {ext: spec for spec in SERVERS for ext in spec.extensions}


def _path_to_uri(path: Path) -> str:
    return path.resolve().as_uri()


def _uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    path_str = unquote(parsed.path)
    if len(path_str) >= 3 and path_str[0] == "/" and path_str[2] == ":":
        path_str = path_str[1:]
    return Path(path_str).resolve()


def _normalize_uri(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return uri
    return _path_to_uri(_uri_to_path(uri))


def resolve_symbol_column(line_text: str, symbol: str) -> int | None:
    """Resolve 0-based UTF-16 column for `symbol` (supports `name#N`, 1-indexed)."""
    if not symbol:
        return None
    name = symbol
    occ = 1
    if "#" in symbol:
        base, _, occ_str = symbol.rpartition("#")
        if base and occ_str.isdigit() and int(occ_str) >= 1:
            name = base
            occ = int(occ_str)

    escaped = re.escape(name)
    patterns = (
        (rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", 0),
        (escaped, 0),
        (escaped, re.IGNORECASE),
    )
    for pat, flags in patterns:
        matches = list(re.finditer(pat, line_text, flags))
        if len(matches) >= occ:
            idx = matches[occ - 1].start()
            return len(line_text[:idx].encode("utf-16-le")) // 2
    return None


def _env_overrides() -> dict[str, list[str]]:
    raw = os.environ.get("REPOMAP_LSP_SERVERS", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {
                str(k): [str(x) for x in v] for k, v in data.items() if isinstance(v, list) and v
            }
    except Exception as e:
        print(f"[LSP] Invalid REPOMAP_LSP_SERVERS JSON: {e}", file=sys.stderr)
    return {}


def _find_binary(cmd: str, workspace_root: Path) -> str | None:
    if os.path.isabs(cmd) or "/" in cmd or "\\" in cmd:
        p = Path(cmd).expanduser()
        return str(p) if p.is_file() and os.access(p, os.X_OK) else None

    local_candidates = (
        workspace_root / ".venv" / "bin" / cmd,
        workspace_root / "node_modules" / ".bin" / cmd,
        Path(sys.executable).parent / cmd,
    )
    for cand in local_candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return shutil.which(cmd)


def _resolve_workspace_root(file_path: Path, markers: tuple[str, ...], scope_root: Path) -> Path:
    git_root = find_git_root(scope_root)
    stop_at = git_root if file_path.is_relative_to(git_root) else scope_root.resolve()
    cur = file_path.parent.resolve()
    while True:
        if any((cur / m).exists() for m in markers):
            return cur
        if cur == stop_at or cur.parent == cur:
            break
        cur = cur.parent
    return stop_at


class LspClient:
    """Async JSON-RPC 2.0 client over stdio for a single language server."""

    def __init__(
        self,
        server_id: str,
        binary: str,
        args: tuple[str, ...],
        workspace_root: Path,
    ):
        self.server_id = server_id
        self.binary = binary
        self.args = args
        self.workspace_root = workspace_root.resolve()
        self.proc: asyncio.subprocess.Process | None = None
        self.capabilities: dict[str, Any] = {}
        self._next_id = 1
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._open_docs: dict[str, tuple[int, tuple[int, int]]] = {}
        self._diagnostics: dict[str, list[dict[str, Any]]] = {}
        self._diag_events: dict[str, asyncio.Event] = {}
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._write_lock = asyncio.Lock()
        self._tasks: list[asyncio.Task[Any]] = []
        self._dead = False

    @property
    def alive(self) -> bool:
        return not self._dead and self.proc is not None and self.proc.returncode is None

    async def start(self, timeout: float = 15.0) -> None:
        self.proc = await asyncio.create_subprocess_exec(
            self.binary,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workspace_root),
        )
        self._tasks = [
            asyncio.create_task(self._read_loop()),
            asyncio.create_task(self._drain_stderr()),
        ]
        root_uri = _path_to_uri(self.workspace_root)
        init_res = await self.request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": root_uri,
                "workspaceFolders": [{"uri": root_uri, "name": self.workspace_root.name}],
                "capabilities": {
                    "general": {"positionEncodings": ["utf-16"]},
                    "textDocument": {
                        "synchronization": {"dynamicRegistration": False, "didSave": False},
                        "hover": {"contentFormat": ["markdown", "plaintext"]},
                        "definition": {"linkSupport": True},
                        "references": {},
                        "publishDiagnostics": {"relatedInformation": True},
                        "diagnostic": {},
                    },
                    "workspace": {"configuration": True, "workspaceFolders": True},
                },
            },
            timeout=timeout,
        )
        if isinstance(init_res, dict):
            self.capabilities = init_res.get("capabilities") or {}
        await self.notify("initialized", {})

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if not self.alive or self.proc is None or self.proc.stdin is None:
            raise ConnectionError("LSP process is not running")
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        frame = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
        async with self._write_lock:
            self.proc.stdin.write(frame)
            await self.proc.stdin.drain()

    async def notify(self, method: str, params: Any) -> None:
        await self._send_json({"jsonrpc": "2.0", "method": method, "params": params})

    async def request(self, method: str, params: Any, timeout: float = 15.0) -> Any:
        if not self.alive:
            raise ConnectionError("LSP process is not running")
        req_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        self._pending[req_id] = fut
        try:
            await self._send_json(
                {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
            )
            return await asyncio.wait_for(fut, timeout=timeout)
        except Exception:
            self._pending.pop(req_id, None)
            if not fut.done():
                fut.cancel()
            raise

    async def _handle_server_request(self, req_id: Any, method: str, params: Any) -> None:
        result: Any = None
        if method == "workspace/configuration":
            items = (params or {}).get("items") or [{}]
            result = [{} for _ in items]
        elif method == "workspace/workspaceFolders":
            result = [{"uri": _path_to_uri(self.workspace_root), "name": self.workspace_root.name}]
        try:
            await self._send_json({"jsonrpc": "2.0", "id": req_id, "result": result})
        except Exception:
            pass

    async def _read_loop(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        stdout = self.proc.stdout
        try:
            while True:
                content_length = None
                while True:
                    line = await stdout.readline()
                    if not line:
                        return
                    if line in (b"\r\n", b"\n"):
                        break
                    text = line.decode("ascii", errors="ignore").strip()
                    if text.lower().startswith("content-length:"):
                        content_length = int(text.split(":", 1)[1].strip())

                if content_length is None or content_length <= 0:
                    continue
                body = await stdout.readexactly(content_length)
                msg = json.loads(body.decode("utf-8", errors="replace"))

                if "id" in msg and "method" in msg:
                    await self._handle_server_request(msg["id"], msg["method"], msg.get("params"))
                elif "id" in msg:
                    fut = self._pending.pop(msg["id"], None)
                    if fut is not None and not fut.done():
                        if "error" in msg and msg["error"] is not None:
                            err = msg["error"]
                            err_msg = (
                                err.get("message", str(err)) if isinstance(err, dict) else str(err)
                            )
                            fut.set_exception(RuntimeError(err_msg))
                        else:
                            fut.set_result(msg.get("result"))
                elif msg.get("method") == "textDocument/publishDiagnostics":
                    params = msg.get("params") or {}
                    raw_uri = params.get("uri")
                    if raw_uri:
                        norm_uri = _normalize_uri(raw_uri)
                        self._diagnostics[norm_uri] = params.get("diagnostics") or []
                        ev = self._diag_events.setdefault(norm_uri, asyncio.Event())
                        ev.set()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._fail_pending(e)
        finally:
            self._dead = True
            self._fail_pending(ConnectionError(self._exit_reason()))

    def _exit_reason(self) -> str:
        tail = " | ".join(list(self._stderr_tail)[-3:])
        return f"LSP server '{self.server_id}' disconnected" + (f": {tail}" if tail else "")

    def _fail_pending(self, exc: Exception) -> None:
        for req_id, fut in list(self._pending.items()):
            self._pending.pop(req_id, None)
            if not fut.done():
                fut.set_exception(exc)

    async def _drain_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        try:
            while True:
                line = await self.proc.stderr.readline()
                if not line:
                    return
                self._stderr_tail.append(line.decode("utf-8", errors="replace").rstrip())
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def sync_file(self, path: Path, language_id: str) -> tuple[str, str]:
        norm_path = path.resolve()
        uri = norm_path.as_uri()
        st = norm_path.stat()
        stamp = (st.st_mtime_ns, st.st_size)
        text = norm_path.read_text(encoding="utf-8", errors="replace")

        prev = self._open_docs.get(uri)
        if prev is None:
            self._diag_events[uri] = asyncio.Event()
            await self.notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": language_id,
                        "version": 1,
                        "text": text,
                    }
                },
            )
            self._open_docs[uri] = (1, stamp)
        elif prev[1] != stamp:
            new_ver = prev[0] + 1
            self._diag_events[uri] = asyncio.Event()
            await self.notify(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": new_ver},
                    "contentChanges": [{"text": text}],
                },
            )
            self._open_docs[uri] = (new_ver, stamp)
        return uri, text

    async def get_diagnostics(
        self, path: Path, language_id: str, timeout: float
    ) -> list[dict[str, Any]]:
        uri, _ = await self.sync_file(path, language_id)

        if self.capabilities.get("diagnosticProvider"):
            try:
                res = await self.request(
                    "textDocument/diagnostic",
                    {"textDocument": {"uri": uri}},
                    timeout=timeout,
                )
                if isinstance(res, dict):
                    if res.get("kind") == "full":
                        items = res.get("items") or []
                        self._diagnostics[uri] = items
                        return items
                    if res.get("kind") == "unchanged":
                        return self._diagnostics.get(uri, [])
            except Exception:
                pass

        ev = self._diag_events.setdefault(uri, asyncio.Event())
        if not ev.is_set():
            try:
                await asyncio.wait_for(ev.wait(), timeout=min(timeout, 2.5))
            except asyncio.TimeoutError:
                pass
        return self._diagnostics.get(uri, [])

    async def close(self) -> None:
        self._dead = True
        if self.proc is not None and self.proc.returncode is None:
            try:
                await self.request("shutdown", None, timeout=1.5)
                await self.notify("exit", None)
            except Exception:
                pass
            if self.proc.returncode is None:
                self.proc.terminate()
                try:
                    await asyncio.wait_for(self.proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    self.proc.kill()
        for t in self._tasks:
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def terminate_sync(self) -> None:
        self._dead = True
        if self.proc is not None and self.proc.returncode is None:
            try:
                self.proc.terminate()
            except Exception:
                pass


class LspManager:
    """Lazy multi-language LSP manager for read-only MCP queries."""

    def __init__(self) -> None:
        self._clients: dict[tuple[str, Path], LspClient] = {}
        self._locks: dict[tuple[str, Path], asyncio.Lock] = {}

    async def _get_client(
        self, spec: ServerSpec, workspace_root: Path, ext: str, timeout: float
    ) -> tuple[LspClient | None, str | None]:
        overrides = _env_overrides()
        custom_cmd = overrides.get(ext) or overrides.get(spec.id)
        candidates: list[tuple[str, tuple[str, ...]]] = (
            [(custom_cmd[0], tuple(custom_cmd[1:]))] if custom_cmd else list(spec.commands)
        )

        chosen: tuple[str, tuple[str, ...]] | None = None
        for cmd, args in candidates:
            resolved = _find_binary(cmd, workspace_root)
            if resolved:
                chosen = (resolved, args)
                break

        if chosen is None:
            checked = ", ".join(c for c, _ in candidates)
            return None, (
                f"No language server found for '{ext}' (checked: {checked}). "
                f"Install: {spec.install_hint}. Or use get_repo_map / get_tags for AST structure."
            )

        key = (spec.id, workspace_root)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            client = self._clients.get(key)
            if client is not None and client.alive:
                return client, None
            if client is not None:
                await client.close()
                self._clients.pop(key, None)

            client = LspClient(spec.id, chosen[0], chosen[1], workspace_root)
            try:
                await client.start(timeout=timeout)
            except Exception as e:
                await client.close()
                return None, f"Failed to start {spec.id} LSP ({chosen[0]}): {e}"
            self._clients[key] = client
            return client, None

    async def call(
        self,
        action: str,
        file: str | None = None,
        line: int | None = None,
        symbol: str | None = None,
        root_path: str = ".",
        max_results: int = 50,
        timeout: int = 15,
    ) -> str:
        clamped_timeout = float(max(5, min(120, timeout)))
        cap = max(1, min(200, max_results))
        git_root, target_dir = resolve_scope(root_path)

        if action == "status":
            return self._format_status(target_dir)

        if not file:
            return f"Error: 'file' is required for action='{action}'."

        raw_path = Path(file)
        file_path = raw_path if raw_path.is_absolute() else target_dir / raw_path
        if not file_path.exists() and not raw_path.is_absolute():
            file_path = git_root / raw_path
        file_path = file_path.resolve()

        if not file_path.is_file():
            return f"Error: File not found: {file}"
        if is_too_large(file_path):
            return f"Error: File exceeds size limit: {file}"

        ext = file_path.suffix.lower()
        spec = _EXT_TO_SPEC.get(ext)
        if spec is None:
            return (
                f"No LSP mapping for extension '{ext or '<none>'}'. "
                "Use get_repo_map / get_tags for AST structure."
            )

        language_id = spec.extensions[ext]
        workspace_root = _resolve_workspace_root(file_path, spec.root_markers, target_dir)

        client, err = await self._get_client(spec, workspace_root, ext, clamped_timeout)
        if err or client is None:
            return err or "Failed to initialize LSP client."

        try:
            if action == "diagnostics":
                diags = await client.get_diagnostics(file_path, language_id, clamped_timeout)
                return self._format_diagnostics(file_path, workspace_root, diags, cap)

            if line is None or line < 1:
                return f"Error: 'line' (>= 1) is required for action='{action}'."
            if not symbol:
                return f"Error: 'symbol' is required for action='{action}'."

            uri, text = await client.sync_file(file_path, language_id)
            lines = text.splitlines()
            if line > len(lines):
                return f"Error: line {line} out of range (file has {len(lines)} lines)."

            line_text = lines[line - 1]
            col = resolve_symbol_column(line_text, symbol)
            if col is None:
                return f"Symbol '{symbol}' not found on {file}:{line}: {line_text.strip()}"

            pos_params = {
                "textDocument": {"uri": uri},
                "position": {"line": line - 1, "character": col},
            }

            if action == "definition":
                res = await client.request(
                    "textDocument/definition", pos_params, timeout=clamped_timeout
                )
                return self._format_locations(
                    res, workspace_root, cap, empty_msg="No definition found."
                )
            if action == "references":
                res = await client.request(
                    "textDocument/references",
                    {**pos_params, "context": {"includeDeclaration": True}},
                    timeout=clamped_timeout,
                )
                return self._format_locations(
                    res, workspace_root, cap, empty_msg="No references found."
                )
            if action == "hover":
                res = await client.request(
                    "textDocument/hover", pos_params, timeout=clamped_timeout
                )
                return self._format_hover(res)

            return f"Error: Unknown action '{action}'."
        except Exception as e:
            if not client.alive:
                self._clients.pop((spec.id, workspace_root), None)
                await client.close()
            return f"LSP {action} failed ({spec.id}): {e}"

    def _rel_display(self, path: Path, root: Path) -> str:
        try:
            return path.resolve().relative_to(root).as_posix()
        except ValueError:
            return str(path)

    def _read_line_preview(self, path: Path, line_idx: int) -> str:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for i, txt in enumerate(f):
                    if i == line_idx:
                        return txt.strip()
        except OSError:
            pass
        return ""

    def _format_locations(self, raw: Any, workspace_root: Path, cap: int, empty_msg: str) -> str:
        if not raw:
            return empty_msg
        items = raw if isinstance(raw, list) else [raw]
        entries: list[str] = []
        seen: set[tuple[str, int, int]] = set()

        for item in items:
            if not isinstance(item, dict):
                continue
            uri = item.get("targetUri") or item.get("uri")
            rng = item.get("targetSelectionRange") or item.get("targetRange") or item.get("range")
            if not uri or not isinstance(rng, dict):
                continue
            start = rng.get("start") or {}
            line_idx = int(start.get("line", 0))
            col_idx = int(start.get("character", 0))
            target_path = _uri_to_path(uri)
            rel = self._rel_display(target_path, workspace_root)
            dedup = (rel, line_idx, col_idx)
            if dedup in seen:
                continue
            seen.add(dedup)
            snippet = self._read_line_preview(target_path, line_idx)
            loc = f"{rel}:{line_idx + 1}:{col_idx + 1}"
            entries.append(f"{loc}  {snippet}" if snippet else loc)

        if not entries:
            return empty_msg
        total = len(entries)
        shown = entries[:cap]
        if total > cap:
            shown.append(f"... truncated, showing {cap} of {total} results")
        return "\n".join(shown)

    def _format_hover(self, raw: Any) -> str:
        if not isinstance(raw, dict) or not raw.get("contents"):
            return "No hover information available."
        contents = raw["contents"]
        parts: list[str] = []
        items = contents if isinstance(contents, list) else [contents]
        for item in items:
            if isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, dict) and item.get("value"):
                lang = item.get("language")
                val = str(item["value"]).strip()
                parts.append(f"```{lang}\n{val}\n```" if lang else val)
        text = "\n\n".join(p for p in parts if p)
        if not text:
            return "No hover information available."
        if len(text) > _HOVER_MAX_CHARS:
            return text[:_HOVER_MAX_CHARS] + "\n... [truncated]"
        return text

    def _format_diagnostics(
        self,
        file_path: Path,
        workspace_root: Path,
        diags: list[dict[str, Any]],
        cap: int,
    ) -> str:
        rel = self._rel_display(file_path, workspace_root)
        if not diags:
            return f"No diagnostics in {rel}."
        lines: list[str] = []
        for d in diags[:cap]:
            start = (d.get("range") or {}).get("start") or {}
            line_no = int(start.get("line", 0)) + 1
            col_no = int(start.get("character", 0)) + 1
            sev = _SEVERITY_NAMES.get(int(d.get("severity", 1)), "error")
            msg = str(d.get("message", "")).replace("\n", " ").strip()
            src = d.get("source")
            code = d.get("code")
            meta = ":".join(str(x) for x in (src, code) if x is not None)
            suffix = f" ({meta})" if meta else ""
            lines.append(f"{rel}:{line_no}:{col_no} [{sev}] {msg}{suffix}")
        if len(diags) > cap:
            lines.append(f"... truncated, showing {cap} of {len(diags)} diagnostics")
        return "\n".join(lines)

    def _format_status(self, target_dir: Path) -> str:
        lines = ["Active LSP servers:"]
        if not self._clients:
            lines.append("  (none)")
        else:
            for (sid, ws), client in sorted(self._clients.items(), key=lambda x: str(x[0])):
                state = "alive" if client.alive else "dead"
                lines.append(
                    f"  - {sid} ({state}) root={ws} bin={client.binary} docs={len(client._open_docs)}"
                )

        overrides = _env_overrides()
        lines.append("Detected binaries:")
        for spec in SERVERS:
            override_cmd = overrides.get(spec.id) or next(
                (overrides[ext] for ext in spec.extensions if ext in overrides), None
            )
            cmds = [(override_cmd[0], ())] if override_cmd else list(spec.commands)
            found = next((_find_binary(c, target_dir) for c, _ in cmds), None)
            lines.append(f"  - {spec.id}: {found or 'not found'}")
        return "\n".join(lines)

    async def close_all(self) -> None:
        for client in list(self._clients.values()):
            await client.close()
        self._clients.clear()

    def terminate_all(self) -> None:
        for client in list(self._clients.values()):
            client.terminate_sync()
        self._clients.clear()
