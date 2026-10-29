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
import os
import sys
from collections import defaultdict, namedtuple
from pathlib import Path

import grep_ast
import networkx as nx
import pygments
import tree_sitter as ts
import tree_sitter_language_pack as tslp
from diskcache import Cache
from pygments.lexers import guess_lexer_for_filename

from .utils import count_tokens, find_git_root, get_repo_files

Tag = namedtuple("Tag", ["rel_fname", "fname", "line", "name", "kind"])


def compute_pagerank(
    graph: nx.MultiDiGraph,
    damping: float = 0.85,
    personalization: dict[str, float] | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, float]:
    nodes = list(graph.nodes())
    n_nodes = len(nodes)
    if n_nodes == 0:
        return {}
    if personalization is None:
        p_weights = {n: 1.0 / n_nodes for n in nodes}
    else:
        total_p = sum(personalization.get(n, 0.0) for n in nodes)
        p_weights = {
            n: (personalization.get(n, 0.0) / total_p if total_p > 0 else 1.0 / n_nodes)
            for n in nodes
        }

    ranks = {n: 1.0 / n_nodes for n in nodes}
    out_degree = {n: graph.out_degree(n) for n in nodes}

    for _ in range(max_iter):
        new_ranks = {}
        dangling_sum = sum(ranks[n] for n in nodes if out_degree[n] == 0)
        dangling_contrib = damping * dangling_sum

        for n in nodes:
            in_sum = sum(ranks[pred] / out_degree[pred] for pred in graph.predecessors(n))
            new_ranks[n] = (
                (1.0 - damping) * p_weights[n] + dangling_contrib * p_weights[n] + damping * in_sum
            )

        err = sum(abs(new_ranks[n] - ranks[n]) for n in nodes)
        ranks = new_ranks
        if err < tol:
            break
    return ranks


class RepoMap:
    def __init__(
        self,
        root: str | Path | None = None,
        max_map_tokens: int = 2048,
        verbose: bool = False,
    ):
        self.root = Path(find_git_root(root)).resolve()
        self.max_map_tokens = max_map_tokens
        self.verbose = verbose
        self.cache, self.cache_dir = self._init_cache()
        self._queries_cache: dict[str, str | None] = {}

    def _init_cache(self) -> tuple[Cache, Path]:
        custom_cache = os.environ.get("REPOMAP_CACHE_DIR")
        if custom_cache:
            cache_path = Path(custom_cache)
            cache_path.mkdir(parents=True, exist_ok=True)
            return Cache(str(cache_path)), cache_path

        git_dir = self.root / ".git"
        if git_dir.is_dir():
            repo_cache = git_dir / "repomap-mcp" / "cache"
            try:
                repo_cache.mkdir(parents=True, exist_ok=True)
                test_file = repo_cache / ".write_test"
                test_file.touch()
                test_file.unlink()
                return Cache(str(repo_cache)), repo_cache
            except (OSError, PermissionError):
                pass

        h = hashlib.sha256(str(self.root).encode("utf-8")).hexdigest()[:12]
        user_cache = Path.home() / ".cache" / "repomap-mcp" / f"tags-{h}"
        user_cache.mkdir(parents=True, exist_ok=True)
        return Cache(str(user_cache)), user_cache

    def log(self, *args, **kwargs):
        if self.verbose:
            print("[RepoMap]", *args, file=sys.stderr, **kwargs)

    def _get_query_text(self, lang: str) -> str | None:
        if lang in self._queries_cache:
            return self._queries_cache[lang]

        query_text = None
        try:
            query_path = Path(__file__).parent / "queries" / f"{lang}.scm"
            if query_path.is_file():
                query_text = query_path.read_text(encoding="utf-8")
        except Exception:
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
        if not full_path.is_file():
            return []

        try:
            mtime = full_path.stat().st_mtime
        except OSError:
            return []

        cache_key = (rel_fname, mtime)
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            content = full_path.read_bytes()
        except OSError:
            return []

        lang = grep_ast.filename_to_lang(rel_fname)
        tags = []

        if lang:
            try:
                parser = tslp.get_parser(lang)
                ts_lang = tslp.get_language(lang)
                query_str = self._get_query_text(lang)

                if parser and ts_lang and query_str:
                    tree = parser.parse(content)
                    query = ts.Query(ts_lang, query_str)
                    cursor = ts.QueryCursor(query)

                    for _, captures in cursor.matches(tree.root_node):
                        name_nodes = captures.get("name")
                        if not name_nodes:
                            continue
                        name_node = name_nodes[0]

                        kind_keys = [k for k in captures.keys() if k != "name"]
                        if not kind_keys:
                            continue
                        kind_str = kind_keys[0].lower()

                        kind = "def" if "definition" in kind_str else "ref"
                        name = content[name_node.start_byte : name_node.end_byte].decode(
                            "utf-8", errors="replace"
                        )
                        line = name_node.start_point[0] + 1

                        if name and name.isidentifier():
                            tags.append(
                                Tag(
                                    rel_fname=rel_fname,
                                    fname=fname,
                                    line=line,
                                    name=name,
                                    kind=kind,
                                )
                            )
            except Exception as e:
                self.log(f"Tree-sitter parse error on {rel_fname}: {e}")

        if not any(t.kind == "def" for t in tags):
            try:
                text_content = content.decode("utf-8", errors="replace")
                lexer = guess_lexer_for_filename(rel_fname, text_content)
                tokens = list(pygments.lex(text_content, lexer))
                cur_line = 1
                for token_type, val in tokens:
                    lines_in_val = val.count("\n")
                    clean_name = val.strip()
                    if clean_name.isidentifier():
                        if str(token_type) in ("Token.Name.Class", "Token.Name.Function"):
                            tags.append(
                                Tag(
                                    rel_fname=rel_fname,
                                    fname=fname,
                                    line=cur_line,
                                    name=clean_name,
                                    kind="def",
                                )
                            )
                        elif str(token_type).startswith("Token.Name"):
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

        self.cache[cache_key] = tags
        return tags

    def build_graph(
        self,
        all_files: list[str],
        chat_files: set[str],
        mentioned_idents: set[str],
    ) -> tuple[dict[str, float], dict[str, list[Tag]]]:
        file_tags: dict[str, list[Tag]] = {}
        defines: dict[str, set[str]] = defaultdict(set)
        references: dict[str, set[str]] = defaultdict(set)

        for rel_fname in all_files:
            full_path = (self.root / rel_fname).as_posix()
            tags = self.get_tags(full_path, rel_fname)
            file_tags[rel_fname] = tags

            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                elif tag.kind == "ref":
                    references[tag.name].add(rel_fname)

        graph = nx.MultiDiGraph()
        for f in all_files:
            graph.add_node(f)

        for ident, ref_files in references.items():
            def_files = defines.get(ident)
            if not def_files:
                continue
            for rf in ref_files:
                for df in def_files:
                    if rf != df:
                        graph.add_edge(rf, df, ident=ident)

        personalization = {}
        for f in all_files:
            weight = 1.0
            if f in chat_files:
                weight += 50.0
            file_defs = {t.name for t in file_tags.get(f, []) if t.kind == "def"}
            if mentioned_idents and (file_defs & mentioned_idents):
                weight += 20.0
            personalization[f] = weight

        try:
            ranks = compute_pagerank(graph, personalization=personalization)
        except Exception as e:
            self.log(f"PageRank fallback: {e}")
            total = len(all_files) or 1
            ranks = {f: 1.0 / total for f in all_files}

        return ranks, file_tags

    def render_tree(self, rel_fname: str, tags: list[Tag]) -> str | None:
        full_path = self.root / rel_fname
        try:
            code = full_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

        lines_of_interest = [t.line for t in tags if t.kind == "def"] or [1]

        try:
            tc = grep_ast.TreeContext(
                filename=rel_fname,
                code=code,
                color=False,
                line_number=False,
                mark_lois=False,
            )
            tc.add_lines_of_interest(lines_of_interest)
            tc.add_context()
            formatted = tc.format()
            if formatted:
                return f"{rel_fname}:\n{formatted}\n"
        except Exception as e:
            self.log(f"TreeContext error on {rel_fname}: {e}")

        lines = code.splitlines()
        preview = "\n".join(lines[:10])
        return f"{rel_fname}:\n{preview}\n...\n"

    def get_repo_map(
        self,
        focus_files: list[str] | None = None,
        other_files: list[str] | None = None,
        mentioned_idents: list[str] | None = None,
        max_map_tokens: int | None = None,
        force_refresh: bool = False,
    ) -> str:
        if force_refresh:
            self.cache.clear()

        max_tokens = max_map_tokens if max_map_tokens is not None else self.max_map_tokens
        if max_tokens <= 0:
            return ""

        all_files = list(other_files) if other_files is not None else get_repo_files(self.root)
        if not all_files:
            return "No files found in repository."

        chat_set = set(focus_files or [])
        ident_set = set(mentioned_idents or [])

        ranks, file_tags = self.build_graph(all_files, chat_set, ident_set)
        ranked_files = sorted(all_files, key=lambda f: ranks.get(f, 0.0), reverse=True)

        rendered_cache = {}
        for f in ranked_files:
            rendered = self.render_tree(f, file_tags.get(f, []))
            if rendered:
                rendered_cache[f] = rendered

        # Fast path: check if everything fits
        full_output = "\n".join(rendered_cache[f] for f in ranked_files if f in rendered_cache)
        if count_tokens(full_output) <= max_tokens:
            return full_output.strip()

        # Binary search for optimal file subset within token budget
        low = 1
        high = len(ranked_files)
        best_output = ""

        while low <= high:
            mid = (low + high) // 2
            candidate_files = ranked_files[:mid]
            candidate_output = "\n".join(
                rendered_cache[f] for f in candidate_files if f in rendered_cache
            )
            cand_tokens = count_tokens(candidate_output)

            if cand_tokens <= max_tokens:
                best_output = candidate_output
                low = mid + 1
            else:
                high = mid - 1

        if not best_output and ranked_files:
            first = ranked_files[0]
            if first in rendered_cache:
                lines = rendered_cache[first].splitlines()
                truncated = []
                for line in lines:
                    truncated.append(line)
                    if count_tokens("\n".join(truncated)) > max_tokens:
                        truncated.pop()
                        break
                best_output = "\n".join(truncated)

        return best_output.strip()
