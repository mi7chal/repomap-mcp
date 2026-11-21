# Copyright 2026 RepoMap MCP Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Portions of this file are based on or derived from Aider (aider/repomap.py)
# Copyright (c) Paul Gauthier, licensed under Apache License 2.0.
# Modified for repomap-mcp.

import hashlib
import json
import math
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grep_ast
import pygments
import tree_sitter as ts
import tree_sitter_language_pack as tslp
from diskcache import Cache
from pygments.lexers import guess_lexer_for_filename

from .utils import (
    count_tokens,
    find_git_root,
    get_git_modified_files,
    get_repo_files,
    is_too_large,
)

CACHE_VERSION = 4
SQLITE_ERRORS = (sqlite3.Error, OSError)
TYPE_KINDS = ("class", "struct", "interface", "enum", "type")
_HEADER_OVERHEAD_TOKENS = count_tokens(
    '```json\n{\n  "total_repo_files": 9999,\n  "mapped_files": 9999,\n'
    '  "dropped_files": 9999,\n  "budget_exhausted": false\n}\n```\n\n'
)


@dataclass
class Tag:
    rel_fname: str
    fname: str
    line: int
    name: str
    kind: str  # "def" | "ref"
    symbol_type: str = "other"  # "class", "struct", "interface", "enum", "type", "func"


def compute_pagerank(
    nodes: list[str],
    edges: dict[str, dict[str, float]],
    damping: float = 0.85,
    personalization: dict[str, float] | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Pure-Python weighted PageRank with personalization and dangling redistribution."""
    n_nodes = len(nodes)
    if n_nodes == 0:
        return {}

    node_set = set(nodes)
    if personalization is None:
        p_weights = {n: 1.0 / n_nodes for n in nodes}
    else:
        total_p = sum(max(0.0, personalization.get(n, 0.0)) for n in nodes)
        p_weights = {
            n: (max(0.0, personalization.get(n, 0.0)) / total_p if total_p > 0 else 1.0 / n_nodes)
            for n in nodes
        }

    out_weight: dict[str, float] = {}
    incoming: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for src in nodes:
        total_w = 0.0
        for dst, w in edges.get(src, {}).items():
            if w > 0 and dst in node_set:
                total_w += w
                incoming[dst].append((src, w))
        out_weight[src] = total_w

    dangling_nodes = [n for n in nodes if out_weight[n] == 0.0]
    ranks = {n: 1.0 / n_nodes for n in nodes}

    for _ in range(max_iter):
        dangling_sum = sum(ranks[n] for n in dangling_nodes)
        base_factor = 1.0 - damping + damping * dangling_sum
        new_ranks: dict[str, float] = {}

        for n in nodes:
            in_sum = sum(ranks[src] * (w / out_weight[src]) for src, w in incoming[n])
            new_ranks[n] = damping * in_sum + base_factor * p_weights[n]

        err = sum(abs(new_ranks[n] - ranks[n]) for n in nodes)
        ranks = new_ranks
        if err < tol:
            break

    total = sum(ranks.values()) or 1.0
    return {n: r / total for n, r in ranks.items()}


def _symbol_type_from_capture(kind_str: str) -> str:
    for sym_type in TYPE_KINDS:
        if sym_type in kind_str:
            return sym_type
    if "method" in kind_str or "function" in kind_str:
        return "func"
    return "other"


def _extract_ts_tags(
    content: bytes,
    rel_fname: str,
    fname: str,
    matches: list[tuple[int, dict[str, list[Any]]]],
) -> list[Tag]:
    """Extract normalized Tag objects from tree-sitter QueryCursor matches."""
    tags: list[Tag] = []
    seen: set[tuple[int, str, str]] = set()

    for _, captures in matches:
        name_nodes = captures.get("name")
        kind_str: str | None = None

        if name_nodes:
            other_keys = [k for k in captures if k != "name" and k != "doc"]
            def_keys = [k for k in other_keys if "definition" in k.lower()]
            ref_keys = [k for k in other_keys if "reference" in k.lower()]
            if def_keys:
                kind_str = def_keys[0].lower()
            elif ref_keys:
                kind_str = ref_keys[0].lower()
        else:
            for k in captures:
                kl = k.lower()
                if "name.definition" in kl:
                    name_nodes = captures[k]
                    kind_str = kl
                    break
            if not name_nodes:
                for k in captures:
                    kl = k.lower()
                    if "name.reference" in kl:
                        name_nodes = captures[k]
                        kind_str = kl
                        break

        if not name_nodes or not kind_str:
            continue

        kind = "def" if "definition" in kind_str else ("ref" if "reference" in kind_str else None)
        if kind is None:
            continue

        symbol_type = _symbol_type_from_capture(kind_str)
        for name_node in name_nodes:
            name = content[name_node.start_byte : name_node.end_byte].decode(
                "utf-8", errors="replace"
            )
            if not name or not name.isidentifier():
                continue
            line = name_node.start_point[0] + 1
            key = (line, name, kind)
            if key in seen:
                continue
            seen.add(key)
            tags.append(
                Tag(
                    rel_fname=rel_fname,
                    fname=fname,
                    line=line,
                    name=name,
                    kind=kind,
                    symbol_type=symbol_type,
                )
            )

    return tags


def _pygments_tags(
    content: bytes,
    fname: str,
    rel_fname: str,
    refs_only: bool = False,
) -> list[Tag]:
    tags: list[Tag] = []
    try:
        text_content = content.decode("utf-8", errors="replace")
        lexer = guess_lexer_for_filename(rel_fname, text_content)
        cur_line = 1
        for token_type, val in pygments.lex(text_content, lexer):
            lines_in_val = val.count("\n")
            clean_name = val.strip()
            t_str = str(token_type)
            if clean_name.isidentifier():
                if not refs_only and t_str in (
                    "Token.Name.Class",
                    "Token.Name.Function",
                    "Token.Name.Function.Magic",
                ):
                    sym_type = "class" if t_str == "Token.Name.Class" else "func"
                    tags.append(
                        Tag(
                            rel_fname=rel_fname,
                            fname=fname,
                            line=cur_line,
                            name=clean_name,
                            kind="def",
                            symbol_type=sym_type,
                        )
                    )
                elif t_str.startswith("Token.Name"):
                    tags.append(
                        Tag(
                            rel_fname=rel_fname,
                            fname=fname,
                            line=cur_line,
                            name=clean_name,
                            kind="ref",
                        )
                    )
            cur_line += lines_in_val
    except Exception:
        pass
    return tags


def _normalize_rel(path_str: str, root: Path) -> str:
    p = Path(path_str)
    if p.is_absolute():
        try:
            return p.resolve().relative_to(root).as_posix()
        except ValueError:
            return p.as_posix()
    clean = path_str.replace("\\", "/")
    while clean.startswith("./"):
        clean = clean[2:]
    return clean


class RepoMap:
    def __init__(
        self,
        root: str | Path | None = None,
        target_dir: str | Path | None = None,
        max_map_tokens: int = 2048,
        verbose: bool = False,
    ):
        self.root = Path(find_git_root(root)).resolve()
        self.target_dir = Path(target_dir).resolve() if target_dir is not None else self.root
        self.max_map_tokens = max_map_tokens
        self.verbose = verbose
        self.cache, self.cache_dir = self._init_cache()
        self._queries_cache: dict[str, str | None] = {}

    def _init_cache(self) -> tuple[Cache | None, Path]:
        override = os.environ.get("REPOMAP_CACHE_DIR")
        if override:
            p = Path(override).expanduser().resolve()
            p.mkdir(parents=True, exist_ok=True)
            return self._open_cache(p), p

        git_dir = self.root / ".git"
        if git_dir.is_dir():
            repo_cache = git_dir / "repomap-mcp" / "cache"
            try:
                repo_cache.mkdir(parents=True, exist_ok=True)
                test_file = repo_cache / ".perm_check"
                test_file.touch()
                test_file.unlink()
                return self._open_cache(repo_cache), repo_cache
            except OSError:
                pass

        h = hashlib.sha256(str(self.root).encode("utf-8")).hexdigest()[:12]
        user_cache = Path.home() / ".cache" / "repomap-mcp" / f"tags-v{CACHE_VERSION}-{h}"
        user_cache.mkdir(parents=True, exist_ok=True)
        return self._open_cache(user_cache), user_cache

    def _open_cache(self, path: Path) -> Cache | None:
        try:
            return Cache(str(path))
        except SQLITE_ERRORS as e:
            self.log(f"Cache init error at {path}: {e}")
            return None

    def _cache_get(self, key: str) -> Any:
        if self.cache is None:
            return None
        try:
            return self.cache.get(key)
        except SQLITE_ERRORS:
            return None

    def _cache_set(self, key: str, value: Any) -> None:
        if self.cache is None:
            return
        try:
            self.cache[key] = value
        except SQLITE_ERRORS:
            pass

    def _cache_clear(self) -> None:
        if self.cache is None:
            return
        try:
            self.cache.clear()
        except SQLITE_ERRORS:
            pass

    def close(self) -> None:
        if self.cache is not None:
            try:
                self.cache.close()
            except SQLITE_ERRORS:
                pass
            self.cache = None

    def log(self, *args: Any, **kwargs: Any) -> None:
        if self.verbose:
            print("[RepoMap]", *args, file=sys.stderr, **kwargs)

    def _get_query_text(self, lang: str) -> str | None:
        if lang in self._queries_cache:
            return self._queries_cache[lang]

        query_text = None
        queries_dir = Path(__file__).parent / "queries"
        for candidate in (lang, "typescript" if lang == "tsx" else None):
            if not candidate:
                continue
            query_path = queries_dir / f"{candidate}.scm"
            try:
                if query_path.is_file():
                    query_text = query_path.read_text(encoding="utf-8")
                    break
            except OSError:
                pass

        if not query_text:
            try:
                query_text = tslp.get_tags_query(lang)
            except Exception:
                pass

        self._queries_cache[lang] = query_text
        return query_text

    def get_tags(self, fname: str, rel_fname: str) -> list[Tag]:
        full_path = Path(fname)
        if not full_path.is_file() or is_too_large(full_path):
            return []

        try:
            mtime = full_path.stat().st_mtime
        except OSError:
            return []

        cached = self._cache_get(rel_fname)
        if isinstance(cached, dict) and cached.get("mtime") == mtime:
            data = cached.get("data")
            if isinstance(data, list):
                return data

        try:
            content = full_path.read_bytes()
        except OSError:
            return []

        lang = grep_ast.filename_to_lang(rel_fname)
        tags: list[Tag] = []
        ts_failed = False

        if lang:
            try:
                parser = tslp.get_parser(lang)
                ts_lang = tslp.get_language(lang)
                query_str = self._get_query_text(lang)

                if parser and ts_lang and query_str:
                    tree = parser.parse(content)
                    query = ts.Query(ts_lang, query_str)
                    cursor = ts.QueryCursor(query)
                    tags = _extract_ts_tags(
                        content, rel_fname, fname, cursor.matches(tree.root_node)
                    )
            except Exception as e:
                ts_failed = True
                if self.verbose or "DownloadError" in type(e).__name__:
                    print(
                        f"[RepoMap] Tree-sitter unavailable for {rel_fname}: {e}", file=sys.stderr
                    )

        saw_def = any(t.kind == "def" for t in tags)
        saw_ref = any(t.kind == "ref" for t in tags)

        if ts_failed or not lang:
            tags = _pygments_tags(content, fname, rel_fname, refs_only=False)
        elif saw_def and not saw_ref:
            tags.extend(_pygments_tags(content, fname, rel_fname, refs_only=True))

        # Never poison DiskCache with degraded fallback when Tree-sitter failed to load a parser.
        if not ts_failed:
            self._cache_set(rel_fname, {"mtime": mtime, "data": tags})

        return tags

    def build_graph(
        self,
        all_files: list[str],
        chat_files: set[str],
        mentioned_idents: set[str],
    ) -> tuple[dict[str, float], dict[tuple[str, str], float], dict[str, list[Tag]]]:
        file_tags: dict[str, list[Tag]] = {}
        defines: dict[str, set[str]] = defaultdict(set)
        references: dict[str, list[str]] = defaultdict(list)

        for rel_fname in all_files:
            full_path = (self.root / rel_fname).as_posix()
            tags = self.get_tags(full_path, rel_fname)
            file_tags[rel_fname] = tags

            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                elif tag.kind == "ref":
                    references[tag.name].append(rel_fname)

        if not references:
            references = {k: list(v) for k, v in defines.items()}

        edges: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        edge_items: list[tuple[str, str, str, float]] = []

        for ident in set(defines.keys()) & set(references.keys()):
            def_files = defines[ident]
            mul = 1.0
            is_camel = any(c.islower() for c in ident) and any(c.isupper() for c in ident)
            if len(ident) >= 8 and ("_" in ident or "-" in ident or is_camel):
                mul *= 10.0
            if ident.startswith("_"):
                mul *= 0.1
            if len(def_files) > 5:
                mul *= 0.1
            if ident in mentioned_idents:
                mul *= 10.0

            ref_counts = Counter(references[ident])
            has_cross = any(rf != df for rf in ref_counts for df in def_files)

            for rf, count in ref_counts.items():
                base_w = mul * math.sqrt(min(count, 100)) * (50.0 if rf in chat_files else 1.0)
                for df in def_files:
                    if rf == df:
                        if not has_cross and len(def_files) == 1:
                            w = 0.1 * mul
                            edges[rf][df] += w
                            edge_items.append((rf, df, ident, w))
                        continue
                    edges[rf][df] += base_w
                    edge_items.append((rf, df, ident, base_w))

        n_files = len(all_files) or 1
        personalization: dict[str, float] = {}
        for f in all_files:
            w = 1.0 / n_files
            if f in chat_files:
                w += 10.0 / n_files
            elif mentioned_idents and any(
                t.kind == "def" and t.name in mentioned_idents for t in file_tags.get(f, [])
            ):
                w += 10.0 / n_files
            personalization[f] = w

        file_ranks = compute_pagerank(all_files, edges, personalization=personalization)
        out_weights = {src: sum(dsts.values()) for src, dsts in edges.items()}

        ranked_definitions: dict[tuple[str, str], float] = defaultdict(float)
        for src, dst, ident, w in edge_items:
            ow = out_weights.get(src, 0.0)
            if ow > 0:
                ranked_definitions[(dst, ident)] += file_ranks.get(src, 0.0) * (w / ow)

        return file_ranks, dict(ranked_definitions), file_tags

    def render_tree(
        self,
        rel_fname: str,
        tags: list[Tag],
        types_only: bool = False,
        lines_of_interest: list[int] | None = None,
        margin: int | None = None,
        line_numbers: bool = False,
        _code_cache: dict[str, str] | None = None,
        _tc_cache: dict[str, Any] | None = None,
    ) -> str | None:
        if _code_cache is not None and rel_fname in _code_cache:
            code = _code_cache[rel_fname]
        else:
            full_path = self.root / rel_fname
            if is_too_large(full_path):
                return None
            try:
                code = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return None
            if _code_cache is not None:
                _code_cache[rel_fname] = code

        if lines_of_interest is not None:
            lois = sorted(set(lines_of_interest))
            if not lois and types_only:
                return None
        elif types_only:
            lois = [t.line for t in tags if t.kind == "def" and t.symbol_type in TYPE_KINDS]
            if not lois:
                return None
        else:
            lois = [t.line for t in tags if t.kind == "def"] or [1]

        if margin is None:
            margin = 0 if types_only else 3

        tc = None
        if _tc_cache is not None and rel_fname in _tc_cache:
            tc = _tc_cache[rel_fname]
        else:
            try:
                tc = grep_ast.TreeContext(
                    filename=rel_fname,
                    code=code,
                    color=False,
                    line_number=line_numbers,
                    mark_lois=False,
                    last_line=False,
                    margin=margin,
                    child_context=margin > 0,
                    loi_pad=1 if margin > 0 else 0,
                )
            except Exception as e:
                self.log(f"TreeContext error on {rel_fname}: {e}")
                tc = None
            if _tc_cache is not None:
                _tc_cache[rel_fname] = tc

        if tc is not None:
            tc.lines_of_interest.clear()
            tc.show_lines.clear()
            tc.output_lines.clear()
            tc.margin = margin
            tc.child_context = margin > 0
            tc.loi_pad = 1 if margin > 0 else 0
            tc.line_number = line_numbers

            tc_lois = {max(0, line_no - 1) for line_no in lois}
            tc.add_lines_of_interest(tc_lois)
            tc.add_context()
            formatted = tc.format()
            if formatted:
                return f"{rel_fname}:\n{formatted}\n"

        lines = code.splitlines()
        if types_only or lines_of_interest is not None:
            selected = [
                f"{idx: 3}│{lines[idx - 1]}" if line_numbers else lines[idx - 1]
                for idx in sorted(set(lois))
                if 1 <= idx <= len(lines)
            ]
            if not selected:
                return None
            return f"{rel_fname}:\n" + "\n".join(selected) + "\n"

        preview_lines = lines[:10]
        if line_numbers:
            preview = "\n".join(f"{i + 1: 3}│{line}" for i, line in enumerate(preview_lines))
        else:
            preview = "\n".join(preview_lines)
        return f"{rel_fname}:\n{preview}\n...\n"

    def get_repo_map(
        self,
        focus_files: list[str] | None = None,
        other_files: list[str] | None = None,
        mentioned_idents: list[str] | None = None,
        max_map_tokens: int | None = None,
        include_tests: bool = False,
        types_only: bool = False,
        force_refresh: bool = False,
        auto_focus_git: bool = True,
        line_numbers: bool = False,
    ) -> str:
        if force_refresh:
            self._cache_clear()

        max_tokens = max_map_tokens if max_map_tokens is not None else self.max_map_tokens
        if max_tokens <= 0:
            return ""

        body_budget = max(1, max_tokens - _HEADER_OVERHEAD_TOKENS)

        if other_files is not None:
            all_files = [_normalize_rel(f, self.root) for f in other_files]
        else:
            all_files = get_repo_files(
                self.root,
                target_dir=self.target_dir,
                include_tests=include_tests,
            )

        if not all_files:
            return "No files found in repository."

        chat_set = {_normalize_rel(f, self.root) for f in (focus_files or [])}
        if auto_focus_git:
            git_modified = get_git_modified_files(self.root)
            all_files_set = set(all_files)
            for gm in git_modified:
                if gm in all_files_set or other_files is None:
                    chat_set.add(gm)

        ident_set = set(mentioned_idents or [])

        file_ranks, ranked_definitions, file_tags = self.build_graph(all_files, chat_set, ident_set)
        ranked_files = sorted(all_files, key=lambda f: file_ranks.get(f, 0.0), reverse=True)

        candidate_tags_by_file: dict[str, list[Tag]] = {}
        for f in ranked_files:
            tags = file_tags.get(f, [])
            if types_only:
                c_tags = [t for t in tags if t.kind == "def" and t.symbol_type in TYPE_KINDS]
            else:
                c_tags = [t for t in tags if t.kind == "def"]
            candidate_tags_by_file[f] = c_tags

        def get_tag_priority(t: Tag) -> float:
            rank = (
                ranked_definitions.get((t.rel_fname, t.name), 0.0)
                + file_ranks.get(t.rel_fname, 0.0) * 0.01
            )
            if ident_set and t.name in ident_set:
                rank *= 2.0
            if t.symbol_type in TYPE_KINDS:
                multiplier = 3.0
            elif t.symbol_type == "func":
                multiplier = 1.5
            else:
                multiplier = 1.0
            return rank * multiplier

        all_ranked_tags = [t for f in ranked_files for t in candidate_tags_by_file.get(f, [])]
        all_ranked_tags.sort(key=get_tag_priority, reverse=True)

        files_to_map = (
            [f for f in ranked_files if candidate_tags_by_file[f]] if types_only else ranked_files
        )
        if not files_to_map:
            return "No relevant symbols found matching criteria."

        code_cache: dict[str, str] = {}
        tc_cache: dict[str, Any] = {}

        def _render_within_budget(
            files_list: list[str],
            margin: int,
            lois_map: dict[str, list[int]] | None = None,
        ) -> tuple[str, list[str], bool]:
            chunks: list[str] = []
            used_files: list[str] = []
            running_tokens = 0
            for f in files_list:
                lois = lois_map[f] if lois_map is not None else None
                r = self.render_tree(
                    f,
                    candidate_tags_by_file[f],
                    types_only=types_only,
                    lines_of_interest=lois,
                    margin=margin,
                    line_numbers=line_numbers,
                    _code_cache=code_cache,
                    _tc_cache=tc_cache,
                )
                if not r:
                    continue
                running_tokens += count_tokens(r) + 1
                chunks.append(r)
                used_files.append(f)
                if running_tokens > body_budget:
                    return "\n".join(chunks), used_files, True
            output = "\n".join(chunks).strip()
            exceeded = bool(output) and count_tokens(output) > body_budget
            return output, used_files, exceeded

        mapped_files: list[str] = []
        best_output = ""
        pruned_or_truncated = False

        # Tier 1: Full context (margin=3, or 0 for types_only)
        tier1_margin = 0 if types_only else 3
        t1_out, t1_files, t1_exceeded = _render_within_budget(files_to_map, tier1_margin)
        if not t1_exceeded and t1_out:
            best_output = t1_out
            mapped_files = t1_files
        else:
            # Tier 2: Compact context (margin=0) if not already tried in Tier 1
            if not types_only:
                t2_out, t2_files, t2_exceeded = _render_within_budget(files_to_map, 0)
                if not t2_exceeded and t2_out:
                    best_output = t2_out
                    mapped_files = t2_files

            # Tier 3: Hierarchical symbol pruning via binary search over ranked tags
            if not best_output and all_ranked_tags:
                pruned_or_truncated = True
                low = 1
                high = len(all_ranked_tags)

                while low <= high:
                    mid = (low + high) // 2
                    lois_by_file: dict[str, list[int]] = defaultdict(list)
                    for t in all_ranked_tags[:mid]:
                        lois_by_file[t.rel_fname].append(t.line)

                    cand_files = [f for f in files_to_map if f in lois_by_file]
                    cand_out, used, exceeded = _render_within_budget(
                        cand_files, 0, lois_map=lois_by_file
                    )
                    if not exceeded and cand_out:
                        best_output = cand_out
                        mapped_files = used
                        pruned_or_truncated = mid < len(all_ranked_tags)
                        low = mid + 1
                    else:
                        high = mid - 1

            # Tier 4: Fallback for extremely constrained token budgets
            if not best_output:
                pruned_or_truncated = True
                first = files_to_map[0]
                first_render = (
                    self.render_tree(
                        first,
                        candidate_tags_by_file[first],
                        types_only=types_only,
                        margin=0,
                        line_numbers=line_numbers,
                        _code_cache=code_cache,
                        _tc_cache=tc_cache,
                    )
                    or ""
                )
                truncated: list[str] = []
                for line in first_render.splitlines():
                    truncated.append(line)
                    if count_tokens("\n".join(truncated)) > body_budget:
                        truncated.pop()
                        break
                best_output = "\n".join(truncated)
                mapped_files = [first]

        telemetry = {
            "total_repo_files": len(all_files),
            "mapped_files": len(mapped_files),
            "dropped_files": len(all_files) - len(mapped_files),
            "budget_exhausted": (len(mapped_files) < len(all_files)) or pruned_or_truncated,
        }
        header = f"```json\n{json.dumps(telemetry, indent=2)}\n```\n\n"
        return header + best_output.strip()
