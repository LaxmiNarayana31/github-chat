"""Hierarchical Repository Map with PageRank ranking inspired by Aider and GitHub Copilot Workspace."""

import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from adalflow.core.types import Document as AdalDocument

log = logging.getLogger(__name__)


class RepoMapGenerator:
    """Constructs a directed symbol dependency graph and renders a PageRank-ranked repository map."""

    def __init__(self, max_tokens: int = 1024):
        self.max_tokens = max_tokens
        self.char_budget = max_tokens * 4
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.file_symbols: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.pagerank_scores: Dict[str, float] = {}

    def build_from_chunks(self, chunks: List[AdalDocument]) -> 'RepoMapGenerator':
        """Construct dependency graph from chunk metadata (symbols, paths, and imports)."""
        self.graph.clear()
        self.file_symbols.clear()

        import_pattern = re.compile(
            r"^(?:from\s+([a-zA-Z0-9_\.]+)\s+import|import\s+([a-zA-Z0-9_\.]+)|#include\s+[\"<]([a-zA-Z0-9_\./\\]+)[\">]|require\(['\"]([a-zA-Z0-9_\./\\]+)['\"]\))",
            re.MULTILINE,
        )

        for doc in chunks:
            meta = doc.meta_data or {}
            file_path = meta.get("file_path", "")
            if not file_path:
                continue

            symbol_name = meta.get("symbol_name")
            symbol_type = meta.get("symbol_type")
            line_range = meta.get("line_range")

            if symbol_name and symbol_type in ["function", "method", "class", "declaration"]:
                self.file_symbols[file_path].append(
                    {
                        "name": symbol_name,
                        "type": symbol_type,
                        "lines": line_range,
                    }
                )

            # Extract imports to create graph edges
            text = doc.text or ""
            matches = import_pattern.findall(text)
            for m in matches:
                imported = next((item for item in m if item), "")
                if imported:
                    # Map imported module name to potential file node
                    imported_clean = imported.replace(".", "/").split("/")[-1]
                    self.graph[file_path].add(imported_clean)

        self._compute_pagerank()
        return self

    def _compute_pagerank(self, iterations: int = 25, damping: float = 0.85) -> None:
        """Compute PageRank scores over the file and symbol dependency graph."""
        nodes = list(set(self.file_symbols.keys()) | set(self.graph.keys()))
        if not nodes:
            return

        n = len(nodes)
        scores = {node: 1.0 / n for node in nodes}

        # Reverse adjacency for incoming links
        incoming: Dict[str, Set[str]] = defaultdict(set)
        outgoing_counts: Dict[str, int] = defaultdict(int)

        for src, targets in self.graph.items():
            outgoing_counts[src] = len(targets)
            for tgt in targets:
                # Match target loosely to known files
                for node in nodes:
                    if tgt in node or node.endswith(f"{tgt}.py"):
                        incoming[node].add(src)

        base_score = (1.0 - damping) / n
        for _ in range(iterations):
            new_scores = {}
            for node in nodes:
                rank_sum = 0.0
                for inc in incoming[node]:
                    out_cnt = outgoing_counts[inc] or 1
                    rank_sum += scores[inc] / out_cnt
                new_scores[node] = base_score + (damping * rank_sum)
            scores = new_scores

        self.pagerank_scores = scores

    def generate_repo_map(self, focus_query: Optional[str] = None) -> str:
        """Render a token-budgeted ASCII repository tree ranked by architectural PageRank centrality."""
        if not self.file_symbols:
            return ""

        # Personalized PageRank boost for symbols appearing in the user query
        boosted_scores = dict(self.pagerank_scores)
        if focus_query:
            query_terms = set(re.findall(r"\b[A-Za-z0-9_]{3,}\b", focus_query.lower()))
            for file_path, symbols in self.file_symbols.items():
                match_count = sum(1 for s in symbols if any(term in s["name"].lower() for term in query_terms))
                if match_count > 0:
                    boosted_scores[file_path] = boosted_scores.get(file_path, 0.01) * (1.5 + match_count)

        # Sort files by PageRank score descending
        sorted_files = sorted(
            self.file_symbols.keys(),
            key=lambda f: boosted_scores.get(f, 0.0),
            reverse=True,
        )

        lines: List[str] = ["# Repository Architecture Map (Ranked by Centrality)"]
        current_chars = len(lines[0])

        for file_path in sorted_files:
            symbols = self.file_symbols[file_path]
            symbols_summary = []
            for s in symbols[:6]:  # Limit top symbols per file to conserve budget
                s_name = s["name"]
                s_type = s["type"]
                symbols_summary.append(f"{s_type} {s_name}")

            entry = f"├── {file_path}"
            if symbols_summary:
                entry += f" ({', '.join(symbols_summary)})"

            if current_chars + len(entry) + 1 > self.char_budget:
                remaining = len(sorted_files) - sorted_files.index(file_path)
                lines.append(f"└── ... [{remaining} more files omitted for context budget]")
                break

            lines.append(entry)
            current_chars += len(entry) + 1

        return "\n".join(lines)

