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
# Portions derived from Aider (https://github.com/Aider-AI/aider),
# Copyright (c) Paul Gauthier, licensed under Apache License 2.0.

import os
import subprocess
import sys
from pathlib import Path

import tiktoken

MAX_FILE_BYTES = 1_000_000

DEFAULT_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".pnpm-store",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    "dist",
    "build",
    "target",
    ".idea",
    ".vscode",
    ".repomap-mcp",
}
BINARY_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".bin",
    ".wasm",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".svg",
    ".webp",
    ".bmp",
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
    ".mov",
    ".mkv",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".otf",
    ".lock",
}
NON_CODE_EXTENSIONS = {
    ".md",
    ".txt",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".xml",
    ".csv",
    ".sql",
    ".env",
}
TEST_DIR_NAMES = {"test", "tests", "__tests__"}
TEST_FILE_PATTERNS = ("_test.", ".test.", "_spec.", ".spec.")

_tokenizer_cache: dict[str, tiktoken.Encoding | None] = {}


def get_tokenizer(model: str = "gpt-4o") -> tiktoken.Encoding | None:
    if model not in _tokenizer_cache:
        try:
            _tokenizer_cache[model] = tiktoken.encoding_for_model(model)
        except Exception:
            try:
                _tokenizer_cache[model] = tiktoken.get_encoding("cl100k_base")
            except Exception:
                _tokenizer_cache[model] = None
    return _tokenizer_cache[model]


def _encode_tokens(text: str, model: str) -> int:
    enc = get_tokenizer(model)
    if enc is not None:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass
    return max(1, len(text) // 4)


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    if not text:
        return 0
    if len(text) > 10_000:
        lines = text.splitlines(keepends=True)
        if len(lines) >= 200:
            step = max(1, len(lines) // 100)
            sample = "".join(lines[::step])
            if sample:
                return max(1, round(_encode_tokens(sample, model) * (len(text) / len(sample))))
    return _encode_tokens(text, model)


def is_test_path(path_str: str) -> bool:
    parts = path_str.replace("\\", "/").lower().split("/")
    if any(p in TEST_DIR_NAMES for p in parts[:-1]):
        return True
    fname = parts[-1]
    return fname.startswith(("test_", "spec_")) or any(pat in fname for pat in TEST_FILE_PATTERNS)


def find_git_root(start_path: str | Path | None = None) -> Path:
    raw = "" if start_path is None else str(start_path).strip()
    p = Path.cwd().resolve() if raw in ("", ".") else Path(raw).expanduser().resolve()
    if p.is_file():
        p = p.parent
    cur = p
    while True:
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            return p
        cur = cur.parent


def resolve_scope(start_path: str | Path | None = None) -> tuple[Path, Path]:
    raw = "" if start_path is None else str(start_path).strip()
    target = Path.cwd().resolve() if raw in ("", ".") else Path(raw).expanduser().resolve()
    if target.is_file():
        target = target.parent
    return find_git_root(target), target


def is_too_large(path: Path) -> bool:
    try:
        return path.stat().st_size > MAX_FILE_BYTES
    except OSError:
        return True


def is_binary_file(path: Path) -> bool:
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return True
    try:
        with open(path, "rb") as f:
            return b"\x00" in f.read(1024)
    except OSError:
        return True


def _is_code_file(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() not in NON_CODE_EXTENSIONS
        and not is_too_large(path)
        and not is_binary_file(path)
    )


def _is_hidden_path(rel_path: str) -> bool:
    return any(
        part.startswith(".") and not (idx == 0 and part == ".github")
        for idx, part in enumerate(rel_path.split("/"))
    )


def get_git_modified_files(root_path: Path) -> list[str]:
    """Retrieve modified, staged, or untracked files in the git repository."""
    root_path = Path(root_path).resolve()
    git_root = find_git_root(root_path)
    if not (git_root / ".git").exists():
        return []

    modified: set[str] = set()
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain", "-z", "-uall"],
            cwd=git_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if res.returncode == 0:
            entries = res.stdout.split("\0")
            i = 0
            while i < len(entries):
                entry = entries[i]
                i += 1
                if len(entry) < 4:
                    continue
                status, path_part = entry[:2], entry[3:]
                if "R" in status or "C" in status:
                    i += 1
                if "D" in status:
                    continue
                full_path = git_root / path_part
                if _is_code_file(full_path):
                    try:
                        modified.add(full_path.relative_to(root_path).as_posix())
                    except ValueError:
                        pass
    except Exception as e:
        print(f"Warning: git status failed: {e}", file=sys.stderr)

    return sorted(modified)


def get_repo_files(
    root_path: Path,
    target_dir: Path | None = None,
    include_tests: bool = False,
) -> list[str]:
    root_path = Path(root_path).resolve()
    target = Path(target_dir).resolve() if target_dir is not None else root_path

    rel_prefix = ""
    if target != root_path:
        try:
            rel_prefix = target.relative_to(root_path).as_posix().rstrip("/") + "/"
        except ValueError:
            rel_prefix = ""

    def accept(rel_path: str) -> bool:
        if rel_prefix and not rel_path.startswith(rel_prefix):
            return False
        return include_tests or not is_test_path(rel_path)

    if (root_path / ".git").exists():
        try:
            files: set[str] = set()
            git_ok = True
            for args in (
                ["git", "ls-files", "-z"],
                ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            ):
                res = subprocess.run(
                    args, cwd=root_path, capture_output=True, text=True, check=False, timeout=10
                )
                if res.returncode != 0:
                    git_ok = False
                    break
                for clean in res.stdout.split("\0"):
                    if clean and not _is_hidden_path(clean) and accept(clean):
                        files.add(clean)
            if git_ok:
                return sorted(f for f in files if _is_code_file(root_path / f))
        except Exception as e:
            print(f"Warning: git command failed: {e}", file=sys.stderr)

    walk_root = target if target.is_dir() else root_path
    valid_files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(walk_root):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in DEFAULT_IGNORED_DIRS
            and (not d.startswith(".") or (Path(dirpath) == root_path and d == ".github"))
            and (include_tests or d not in TEST_DIR_NAMES)
        ]
        rel_dir = Path(dirpath).relative_to(root_path)
        for fname in filenames:
            if fname.startswith("."):
                continue
            rel_path = (rel_dir / fname).as_posix().removeprefix("./")
            if accept(rel_path) and _is_code_file(root_path / rel_path):
                valid_files.append(rel_path)

    return sorted(valid_files)
