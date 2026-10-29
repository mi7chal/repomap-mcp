import os
import subprocess
import sys
from pathlib import Path

import tiktoken

DEFAULT_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
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
    ".wasm",
}

_tokenizer_cache = {}


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


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    if not text:
        return 0
    enc = get_tokenizer(model)
    if enc is not None:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass
    return max(1, len(text) // 4)


def find_git_root(start_path: str | Path | None = None) -> Path:
    if start_path is None or str(start_path).strip() in ("", "."):
        p = Path.cwd().resolve()
    else:
        p = Path(start_path).expanduser().resolve()

    if p.is_file():
        p = p.parent

    current = p
    while True:
        if (current / ".git").exists():
            return current
        if current.parent == current:
            break
        current = current.parent

    return p


def is_binary_file(path: Path) -> bool:
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return True
    try:
        with open(path, "rb") as f:
            return b"\x00" in f.read(1024)
    except OSError:
        return True


def get_repo_files(root_path: Path) -> list[str]:
    root_path = Path(root_path).resolve()
    files: set[str] = set()

    if (root_path / ".git").exists():
        try:
            for args in (
                ["git", "ls-files"],
                ["git", "ls-files", "--others", "--exclude-standard"],
            ):
                res = subprocess.run(
                    args, cwd=root_path, capture_output=True, text=True, check=False
                )
                if res.returncode == 0:
                    for line in res.stdout.splitlines():
                        clean = line.strip()
                        if clean and not clean.startswith("."):
                            files.add(clean)

            if files:
                valid_files = [
                    f
                    for f in files
                    if (root_path / f).is_file() and not is_binary_file(root_path / f)
                ]
                valid_files.sort()
                return valid_files
        except Exception as e:
            print(f"Warning: git command failed: {e}", file=sys.stderr)

    valid_files = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [
            d for d in dirnames if d not in DEFAULT_IGNORED_DIRS and not d.startswith(".")
        ]
        rel_dir = Path(dirpath).relative_to(root_path)
        for fname in filenames:
            if fname.startswith("."):
                continue
            rel_path = (rel_dir / fname).as_posix()
            if rel_path.startswith("./"):
                rel_path = rel_path[2:]
            full_path = root_path / rel_path
            if full_path.is_file() and not is_binary_file(full_path):
                valid_files.append(rel_path)

    valid_files.sort()
    return valid_files
