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
# LSP tool UX and symbol-column resolution inspired by oh-my-pi
# (https://github.com/can1357/oh-my-pi), licensed under the MIT License.

"""Minimal read-only LSP client and manager for MCP."""

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .utils import DEFAULT_IGNORED_DIRS, find_git_root, is_too_large, resolve_scope

_SEVERITY_NAMES = {1: "error", 2: "warning", 3: "info", 4: "hint"}
_HOVER_MAX_CHARS = 4000

# server_id -> (extensions_to_lang_id, commands, root_markers, install_hint)
SERVERS: dict[
    str, tuple[dict[str, str], tuple[tuple[str, tuple[str, ...]], ...], tuple[str, ...], str]
] = {
    "python": (
        {".py": "python", ".pyi": "python"},
        (
            ("basedpyright-langserver", ("--stdio",)),
            ("pyright-langserver", ("--stdio",)),
            ("pylsp", ()),
        ),
        ("pyproject.toml", "setup.py", "pyrightconfig.json", ".git"),
        "pip install basedpyright",
    ),
    "typescript": (
        {
            ".ts": "typescript",
            ".tsx": "typescriptreact",
            ".js": "javascript",
            ".jsx": "javascriptreact",
            ".mjs": "javascript",
            ".cjs": "javascript",
        },
        (("typescript-language-server", ("--stdio",)),),
        ("tsconfig.json", "jsconfig.json", "package.json", ".git"),
        "npm i -g typescript-language-server typescript",
    ),
    "svelte": (
        {".svelte": "svelte"},
        (("svelteserver", ("--stdio",)),),
        ("svelte.config.js", "svelte.config.ts", "package.json", ".git"),
        "npm i -g svelte-language-server",
    ),
    "rust": (
        {".rs": "rust"},
        (("rust-analyzer", ()),),
        ("Cargo.toml",),
        "rustup component add rust-analyzer",
    ),
    "go": (
        {".go": "go"},
        (("gopls", ()),),
        ("go.mod",),
        "go install golang.org/x/tools/gopls@latest",
    ),
    "c_cpp": (
        {".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp"},
        (("clangd", ()),),
        ("compile_commands.json", "CMakeLists.txt", ".git"),
        "install clangd",
    ),
}

_EXT_TO_SERVER: dict[str, tuple[str, str]] = {
    ext: (sid, lang_id) for sid, (exts, _, _, _) in SERVERS.items() for ext, lang_id in exts.items()
}


def _uri_to_path(uri: str) -> Path:
    path_str = unquote(urlparse(uri).path)
    if len(path_str) >= 3 and path_str[0] == "/" and path_str[2] == ":":
        path_str = path_str[1:]
    return Path(path_str).resolve()


def resolve_symbol_column(line_text: str, symbol: str) -> int | None:
    """Resolve 0-based UTF-16 column for `symbol` (supports `name#N`, 1-indexed)."""
    if not symbol:
        return None
    name, occ = symbol, 1
    if "#" in symbol:
        base, _, occ_str = symbol.rpartition("#")
        if base and occ_str.isdigit() and int(occ_str) >= 1:
            name, occ = base, int(occ_str)

    esc = re.escape(name)
    for pat, flags in (
        (rf"(?<![A-Za-z0-9_]){esc}(?![A-Za-z0-9_])", 0),
        (esc, 0),
        (esc, re.IGNORECASE),
    ):
        matches = list(re.finditer(pat, line_text, flags))
        if len(matches) >= occ:
            return len(line_text[: matches[occ - 1].start()].encode("utf-16-le")) // 2
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


def _find_binary(cmd: str, workspace_root: Path, file_dir: Path | None = None) -> str | None:
    if os.path.isabs(cmd) or "/" in cmd or "\\" in cmd:
        p = Path(cmd).expanduser()
        return str(p) if p.is_file() and os.access(p, os.X_OK) else None

    search_dirs: list[Path] = []
    if file_dir is not None:
        cur = file_dir.resolve()
        ws = workspace_root.resolve()
        while True:
            search_dirs.extend((cur / ".venv" / "bin", cur / "node_modules" / ".bin"))
            if cur == ws or cur.parent == cur:
                break
            cur = cur.parent
    search_dirs.extend(
        (
            workspace_root / ".venv" / "bin",
            workspace_root / "node_modules" / ".bin",
            Path(sys.executable).parent,
        )
    )

    resolved: str | None = None
    for d in search_dirs:
        cand = d / cmd
        if cand.is_file() and os.access(cand, os.X_OK):
            resolved = str(cand)
            break
    if resolved is None:
        resolved = shutil.which(cmd)

    # Resolve rustup proxy directly; catches missing rust-analyzer in pinned rust-toolchain.toml
    if (
        resolved
        and cmd == "rust-analyzer"
        and "rustup" in resolved
        and "toolchains" not in resolved
    ):
        try:
            r = subprocess.run(
                ["rustup", "which", "rust-analyzer"],
                cwd=str(workspace_root),
                capture_output=True,
                text=True,
                timeout=2,
            )
            return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
        except Exception:
            return None
    return resolved


def _resolve_workspace_root(file_path: Path, markers: tuple[str, ...], scope_root: Path) -> Path:
    git_root = find_git_root(scope_root)
    stop_at = git_root if file_path.is_relative_to(git_root) else scope_root.resolve()
    cur = file_path.parent.resolve()
    while True:
        if any((cur / m).exists() for m in markers):
            return cur
        if cur == stop_at or cur.parent == cur:
            return stop_at
        cur = cur.parent


def _rust_excludes(workspace_root: Path) -> list[str]:
    excludes: list[str] = []
    for entry in workspace_root.iterdir():
        if not entry.is_dir():
            continue
        if entry.name in DEFAULT_IGNORED_DIRS:
            excludes.append(entry.name)
        else:
            try:
                for sub in entry.iterdir():
                    if sub.is_dir() and sub.name in DEFAULT_IGNORED_DIRS:
                        excludes.append(f"{entry.name}/{sub.name}")
            except OSError:
                pass
    return sorted(set(excludes))


class LspClient:
    """Async JSON-RPC 2.0 client over stdio for a single language server."""

    def __init__(self, server_id: str, binary: str, args: tuple[str, ...], workspace_root: Path):
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
        self._quiescent: bool | None = None
        self._active_progress: set[Any] = set()
        self._ready = asyncio.Event()
        self._ready.set()
        self._config = (
            {"files": {"exclude": _rust_excludes(self.workspace_root)}}
            if server_id == "rust"
            else {}
        )

    @property
    def alive(self) -> bool:
        return not self._dead and self.proc is not None and self.proc.returncode is None

    async def start(self, timeout: float = 25.0) -> None:
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
        root_uri = self.workspace_root.as_uri()
        init_res = await self.request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": root_uri,
                "workspaceFolders": [{"uri": root_uri, "name": self.workspace_root.name}],
                "initializationOptions": self._config,
                "capabilities": {
                    "general": {"positionEncodings": ["utf-16"]},
                    "window": {"workDoneProgress": True},
                    "textDocument": {
                        "synchronization": {"dynamicRegistration": False, "didSave": False},
                        "hover": {"contentFormat": ["markdown", "plaintext"]},
                        "definition": {"linkSupport": True},
                        "references": {},
                        "publishDiagnostics": {"relatedInformation": True},
                        "diagnostic": {},
                    },
                    "workspace": {"configuration": True, "workspaceFolders": True},
                    "experimental": {"serverStatusNotification": True},
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

    async def request(self, method: str, params: Any, timeout: float = 25.0) -> Any:
        if not self.alive:
            raise ConnectionError("LSP process is not running")
        req_id = self._next_id
        self._next_id += 1
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        try:
            await self._send_json(
                {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
            )
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            tail = " | ".join(list(self._stderr_tail)[-3:])
            raise TimeoutError(
                f"LSP '{method}' timed out after {timeout:.0f}s" + (f" ({tail})" if tail else "")
            ) from None
        finally:
            self._pending.pop(req_id, None)
            if not fut.done():
                fut.cancel()

    async def wait_ready(self, timeout: float) -> None:
        if not self._ready.is_set() and timeout > 0:
            try:
                await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass

    async def _handle_server_request(self, req_id: Any, method: str, params: Any) -> None:
        result: Any = None
        if method == "workspace/configuration":
            items = (params or {}).get("items") or [{}]
            result = [self._config for _ in items]
        elif method == "workspace/workspaceFolders":
            result = [{"uri": self.workspace_root.as_uri(), "name": self.workspace_root.name}]
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

                if not content_length or content_length <= 0:
                    continue
                msg = json.loads(
                    (await stdout.readexactly(content_length)).decode("utf-8", errors="replace")
                )
                method = msg.get("method")

                if "id" in msg and method:
                    await self._handle_server_request(msg["id"], method, msg.get("params"))
                elif "id" in msg:
                    fut = self._pending.pop(msg["id"], None)
                    if fut is not None and not fut.done():
                        err = msg.get("error")
                        if err is not None:
                            err_msg = (
                                err.get("message", str(err)) if isinstance(err, dict) else str(err)
                            )
                            fut.set_exception(RuntimeError(err_msg))
                        else:
                            fut.set_result(msg.get("result"))
                elif method == "textDocument/publishDiagnostics":
                    params = msg.get("params") or {}
                    raw_uri = params.get("uri")
                    if raw_uri:
                        norm_uri = (
                            _uri_to_path(raw_uri).as_uri()
                            if urlparse(raw_uri).scheme == "file"
                            else raw_uri
                        )
                        self._diagnostics[norm_uri] = params.get("diagnostics") or []
                        self._diag_events.setdefault(norm_uri, asyncio.Event()).set()
                elif method == "experimental/serverStatus":
                    q = (msg.get("params") or {}).get("quiescent")
                    if isinstance(q, bool):
                        self._quiescent = q
                        self._ready.set() if q else self._ready.clear()
                elif method == "$/progress":
                    p = msg.get("params") or {}
                    kind = (p.get("value") or {}).get("kind")
                    tok = p.get("token")
                    if kind == "begin":
                        self._active_progress.add(tok)
                        self._ready.clear()
                    elif kind == "end":
                        self._active_progress.discard(tok)
                        if not self._active_progress and self._quiescent is not False:
                            self._ready.set()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(e)
            self._pending.clear()
        finally:
            self._dead = True
            self._ready.set()
            tail = " | ".join(list(self._stderr_tail)[-3:])
            exc = ConnectionError(
                f"LSP server '{self.server_id}' disconnected" + (f": {tail}" if tail else "")
            )
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(exc)
            self._pending.clear()

    async def _drain_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        try:
            while line := await self.proc.stderr.readline():
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
                    "textDocument/diagnostic", {"textDocument": {"uri": uri}}, timeout=timeout
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


class LspManager:
    """Lazy multi-language LSP manager for read-only MCP queries."""

    def __init__(self) -> None:
        self._clients: dict[tuple[str, Path], LspClient] = {}
        self._locks: dict[tuple[str, Path], asyncio.Lock] = {}

    async def _get_client(
        self, sid: str, workspace_root: Path, file_dir: Path, ext: str, timeout: float
    ) -> tuple[LspClient | None, str | None]:
        _, default_cmds, _, hint = SERVERS[sid]
        overrides = _env_overrides()
        custom = overrides.get(ext) or overrides.get(sid)
        candidates = [(custom[0], tuple(custom[1:]))] if custom else list(default_cmds)

        chosen = next(
            (
                (resolved, args)
                for cmd, args in candidates
                if (resolved := _find_binary(cmd, workspace_root, file_dir))
            ),
            None,
        )
        if chosen is None:
            checked = ", ".join(c for c, _ in candidates)
            return None, (
                f"No language server found for '{ext}' (checked: {checked}). "
                f"Install: {hint}. Or use get_repo_map / get_tags for AST structure."
            )

        key = (sid, workspace_root)
        async with self._locks.setdefault(key, asyncio.Lock()):
            client = self._clients.get(key)
            if client is not None and client.alive:
                return client, None
            if client is not None:
                await client.close()
                self._clients.pop(key, None)

            client = LspClient(sid, chosen[0], chosen[1], workspace_root)
            try:
                await client.start(timeout=timeout)
            except Exception as e:
                await client.close()
                return None, f"Failed to start {sid} LSP ({chosen[0]}): {e}"
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
        timeout: int = 25,
    ) -> str:
        budget = float(max(5, min(120, timeout)))
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
        mapping = _EXT_TO_SERVER.get(ext)
        if mapping is None:
            return f"No LSP mapping for '{ext or '<none>'}'. Use get_repo_map / get_tags for AST structure."

        sid, language_id = mapping
        _, _, markers, _ = SERVERS[sid]
        workspace_root = _resolve_workspace_root(file_path, markers, target_dir)

        deadline = time.monotonic() + budget
        client, err = await self._get_client(sid, workspace_root, file_path.parent, ext, budget)
        if err or client is None:
            return err or "Failed to initialize LSP client."

        try:
            await client.sync_file(file_path, language_id)
            await client.wait_ready(max(0.0, deadline - time.monotonic() - 3.0))

            if action == "diagnostics":
                diags = await client.get_diagnostics(
                    file_path, language_id, max(2.0, deadline - time.monotonic())
                )
                if not diags and client._quiescent is False and deadline - time.monotonic() > 3.0:
                    await client.wait_ready(deadline - time.monotonic() - 2.0)
                    diags = await client.get_diagnostics(
                        file_path, language_id, max(2.0, deadline - time.monotonic())
                    )
                return self._format_diagnostics(file_path, workspace_root, diags, cap)

            if line is None or line < 1:
                return f"Error: 'line' (>= 1) is required for action='{action}'."
            if not symbol:
                return f"Error: 'symbol' is required for action='{action}'."

            uri, text = await client.sync_file(file_path, language_id)
            lines = text.splitlines()
            if line > len(lines):
                return f"Error: line {line} out of range (file has {len(lines)} lines)."

            col = resolve_symbol_column(lines[line - 1], symbol)
            if col is None:
                return f"Symbol '{symbol}' not found on {file}:{line}: {lines[line - 1].strip()}"

            pos = {"textDocument": {"uri": uri}, "position": {"line": line - 1, "character": col}}
            method, params = {
                "definition": ("textDocument/definition", pos),
                "references": (
                    "textDocument/references",
                    {**pos, "context": {"includeDeclaration": True}},
                ),
                "hover": ("textDocument/hover", pos),
            }.get(action, (None, None))
            if method is None:
                return f"Error: Unknown action '{action}'."

            res = await self._request_with_retry(client, method, params, deadline)
            if action == "hover":
                return self._format_hover(res)
            empty_msg = "No definition found." if action == "definition" else "No references found."
            return self._format_locations(res, workspace_root, cap, empty_msg)
        except Exception as e:
            if not client.alive:
                self._clients.pop((sid, workspace_root), None)
                await client.close()
            return f"LSP {action} failed ({sid}): {e or type(e).__name__}"

    async def _request_with_retry(
        self, client: LspClient, method: str, params: Any, deadline: float
    ) -> Any:
        for attempt in range(2):
            rem = max(2.0, deadline - time.monotonic())
            try:
                res = await client.request(method, params, timeout=rem)
            except RuntimeError as e:
                if attempt == 0 and any(
                    k in str(e).lower() for k in ("content modified", "waiting", "-32801")
                ):
                    await asyncio.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
                    continue
                raise
            if (
                not res
                and attempt == 0
                and (client._quiescent is False or not client._ready.is_set())
            ):
                rem_wait = max(0.0, deadline - time.monotonic() - 2.0)
                if rem_wait > 0.2:
                    await client.wait_ready(rem_wait)
                    continue
            return res
        return None

    def _format_locations(self, raw: Any, workspace_root: Path, cap: int, empty_msg: str) -> str:
        if not raw:
            return empty_msg
        entries: list[str] = []
        seen: set[tuple[str, int, int]] = set()
        for item in raw if isinstance(raw, list) else [raw]:
            if not isinstance(item, dict):
                continue
            uri = item.get("targetUri") or item.get("uri")
            rng = item.get("targetSelectionRange") or item.get("targetRange") or item.get("range")
            if not uri or not isinstance(rng, dict):
                continue
            start = rng.get("start") or {}
            ln, col = int(start.get("line", 0)), int(start.get("character", 0))
            target = _uri_to_path(uri)
            try:
                rel = target.relative_to(workspace_root).as_posix()
            except ValueError:
                rel = str(target)
            if (rel, ln, col) in seen:
                continue
            seen.add((rel, ln, col))
            snippet = ""
            try:
                with open(target, encoding="utf-8", errors="replace") as f:
                    for i, txt in enumerate(f):
                        if i == ln:
                            snippet = txt.strip()
                            break
            except OSError:
                pass
            loc = f"{rel}:{ln + 1}:{col + 1}"
            entries.append(f"{loc}  {snippet}" if snippet else loc)
        if not entries:
            return empty_msg
        shown = entries[:cap]
        if len(entries) > cap:
            shown.append(f"... truncated, showing {cap} of {len(entries)} results")
        return "\n".join(shown)

    def _format_hover(self, raw: Any) -> str:
        if not isinstance(raw, dict) or not raw.get("contents"):
            return "No hover information available."
        contents = raw["contents"]
        parts = []
        for item in contents if isinstance(contents, list) else [contents]:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict) and item.get("value"):
                lang, val = item.get("language"), str(item["value"]).strip()
                parts.append(f"```{lang}\n{val}\n```" if lang else val)
        text = "\n\n".join(parts)
        if not text:
            return "No hover information available."
        return (
            text[:_HOVER_MAX_CHARS] + "\n... [truncated]" if len(text) > _HOVER_MAX_CHARS else text
        )

    def _format_diagnostics(
        self, file_path: Path, workspace_root: Path, diags: list[dict[str, Any]], cap: int
    ) -> str:
        try:
            rel = file_path.relative_to(workspace_root).as_posix()
        except ValueError:
            rel = str(file_path)
        if not diags:
            return f"No diagnostics in {rel}."
        lines = []
        for d in diags[:cap]:
            start = (d.get("range") or {}).get("start") or {}
            ln, col = int(start.get("line", 0)) + 1, int(start.get("character", 0)) + 1
            sev = _SEVERITY_NAMES.get(int(d.get("severity", 1)), "error")
            msg = str(d.get("message", "")).replace("\n", " ").strip()
            meta = ":".join(str(x) for x in (d.get("source"), d.get("code")) if x is not None)
            lines.append(f"{rel}:{ln}:{col} [{sev}] {msg}" + (f" ({meta})" if meta else ""))
        if len(diags) > cap:
            lines.append(f"... truncated, showing {cap} of {len(diags)} diagnostics")
        return "\n".join(lines)

    def _format_status(self, target_dir: Path) -> str:
        lines = ["Active LSP servers:"]
        if not self._clients:
            lines.append("  (none)")
        else:
            for (sid, ws), c in sorted(self._clients.items(), key=lambda x: str(x[0])):
                lines.append(
                    f"  - {sid} ({'alive' if c.alive else 'dead'}) root={ws} bin={c.binary} docs={len(c._open_docs)}"
                )
        overrides = _env_overrides()
        lines.append("Detected binaries:")
        for sid, (exts, cmds, _, _) in SERVERS.items():
            ov = overrides.get(sid) or next((overrides[e] for e in exts if e in overrides), None)
            check = [(ov[0], ())] if ov else list(cmds)
            found = next((_find_binary(c, target_dir) for c, _ in check), None)
            lines.append(f"  - {sid}: {found or 'not found'}")
        return "\n".join(lines)

    async def close_all(self) -> None:
        for client in list(self._clients.values()):
            await client.close()
        self._clients.clear()

    def terminate_all(self) -> None:
        for client in list(self._clients.values()):
            if client.proc is not None and client.proc.returncode is None:
                try:
                    client.proc.terminate()
                except Exception:
                    pass
        self._clients.clear()
