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

MAX_FILE_BYTES = 1_000_000

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


def _encode_tokens(text: str, model: str = "gpt-4o") -> int:
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


TEST_DIR_NAMES = {"test", "tests", "__tests__"}
TEST_FILE_PATTERNS = ("_test.", ".test.", "_spec.", ".spec.")


def is_test_path(path_str: str) -> bool:
    norm = path_str.replace("\\", "/").lower()
    parts = norm.split("/")
    if any(p in TEST_DIR_NAMES for p in parts[:-1]):
        return True
    fname = parts[-1]
    if fname.startswith("test_") or fname.startswith("spec_"):
        return True
    return any(pat in fname for pat in TEST_FILE_PATTERNS)


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


def resolve_scope(start_path: str | Path | None = None) -> tuple[Path, Path]:
    if start_path is None or str(start_path).strip() in ("", "."):
        target_dir = Path.cwd().resolve()
    else:
        target_dir = Path(start_path).expanduser().resolve()

    if target_dir.is_file():
        target_dir = target_dir.parent

    git_root = find_git_root(target_dir)
    return git_root, target_dir


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


def _is_hidden_path(rel_path: str) -> bool:
    for idx, part in enumerate(rel_path.split("/")):
        if part.startswith(".") and not (idx == 0 and part == ".github"):
            return True
    return False


def get_git_modified_files(root_path: Path) -> list[str]:
    """Retrieve modified, staged, or untracked files in the git repository."""
    root_path = Path(root_path).resolve()
    git_root = find_git_root(root_path)
    if not (git_root / ".git").exists():
        return []

    modified_files = set()
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
                status = entry[:2]
                path_part = entry[3:]
                if "R" in status or "C" in status:
                    i += 1
                if "D" in status:
                    continue
                full_path = git_root / path_part
                if (
                    full_path.is_file()
                    and not is_too_large(full_path)
                    and not is_binary_file(full_path)
                ):
                    try:
                        rel_path = full_path.relative_to(root_path).as_posix()
                        modified_files.add(rel_path)
                    except ValueError:
                        pass
    except Exception as e:
        print(f"Warning: git status failed: {e}", file=sys.stderr)

    return sorted(modified_files)


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
            rel_prefix = target.relative_to(root_path).as_posix()
            if not rel_prefix.endswith("/"):
                rel_prefix += "/"
        except ValueError:
            rel_prefix = ""

    def accept(rel_path: str) -> bool:
        if rel_prefix and not rel_path.startswith(rel_prefix):
            return False
        return not (not include_tests and is_test_path(rel_path))

    files: set[str] = set()

    if (root_path / ".git").exists():
        try:
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
                valid_files = [
                    f
                    for f in files
                    if (root_path / f).is_file()
                    and not is_too_large(root_path / f)
                    and not is_binary_file(root_path / f)
                ]
                valid_files.sort()
                return valid_files
        except Exception as e:
            print(f"Warning: git command failed: {e}", file=sys.stderr)

    walk_root = target if target.is_dir() else root_path
    valid_files = []
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
            rel_path = (rel_dir / fname).as_posix()
            if rel_path.startswith("./"):
                rel_path = rel_path[2:]
            full_path = root_path / rel_path
            if (
                full_path.is_file()
                and not is_too_large(full_path)
                and not is_binary_file(full_path)
                and accept(rel_path)
            ):
                valid_files.append(rel_path)

    valid_files.sort()
    return valid_files
