"""AST-aware structural code chunker preserving function and class boundaries with symbol metadata."""

import ast
import logging
import re
from typing import Any, List, Optional

from adalflow.core.types import Document as AdalDocument

try:
    import tree_sitter_languages
except ImportError:
    tree_sitter_languages = None

log = logging.getLogger(__name__)


class ASTChunker:
    """Parses source code into semantic AST chunks (classes, functions, methods) with line-level citations."""

    def __init__(self, max_chunk_chars: int = 1500, min_chunk_chars: int = 80):
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars

    def chunk_file(
        self,
        content: str,
        file_path: str,
        file_type: str,
    ) -> List[AdalDocument]:
        """Chunk a file into semantic units based on language structure."""
        if not content or not content.strip():
            return []

        ext = f".{file_type.lstrip('.')}".lower()
        if ext == ".py":
            chunks = self._chunk_python(content, file_path)
            if chunks:
                return chunks

        # Universal Tree-sitter AST parsing for enterprise polyglot codebases
        ts_chunks = self._chunk_tree_sitter(content, file_path, ext)
        if ts_chunks:
            return ts_chunks

        # Fallback to multi-language structural regex chunker
        return self._chunk_structural_code(content, file_path, ext)

    def _chunk_python(self, content: str, file_path: str) -> List[AdalDocument]:
        """Parse Python source into AST nodes (classes, functions, methods) with precise line numbers."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            log.debug(f"AST parse error in {file_path}, falling back to structural chunking.")
            return []

        lines = content.splitlines()
        chunks: List[AdalDocument] = []

        # Extract top-level module docstrings and imports as header chunk
        header_lines: List[str] = []
        module_imports: List[str] = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)):
                start = getattr(node, "lineno", 1) - 1
                end = getattr(node, "end_lineno", start + 1)
                header_lines.extend(lines[start:end])
                if isinstance(node, ast.Import):
                    module_imports.append(", ".join([a.name for a in node.names]))
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    module_imports.append(f"{mod}: " + ", ".join([a.name for a in node.names]))
            elif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = getattr(node, "lineno", 1) - 1
                end = getattr(node, "end_lineno", start + 1)
                header_lines.extend(lines[start:end])

        imports_summary = " | ".join(module_imports[:5]) if module_imports else ""

        header_text = "\n".join(header_lines).strip()
        if header_text and len(header_text) >= self.min_chunk_chars:
            chunks.append(
                AdalDocument(
                    text=f"# [Context: {file_path} | Scope: module_header]\n" + header_text,
                    meta_data={
                        "file_path": file_path,
                        "symbol_name": "module_header",
                        "symbol_type": "module",
                        "line_range": [1, min(len(header_lines), len(lines))],
                        "is_code": True,
                        "title": f"{file_path} (Module Header)",
                    },
                )
            )

        # Extract functions and classes
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.extend(self._extract_function_chunk(node, lines, file_path, imports_summary=imports_summary))
            elif isinstance(node, ast.ClassDef):
                chunks.extend(self._extract_class_chunks(node, lines, file_path, imports_summary=imports_summary))

        return chunks

    def _extract_function_chunk(
        self,
        node: Any,
        lines: List[str],
        file_path: str,
        parent_class: Optional[str] = None,
        imports_summary: str = "",
    ) -> List[AdalDocument]:
        """Extract a function or method AST node into an AdalDocument chunk with contextual headers."""
        start_line = getattr(node, "lineno", 1)
        end_line = getattr(node, "end_lineno", start_line)
        chunk_lines = lines[start_line - 1 : end_line]
        raw_code = "\n".join(chunk_lines).strip()

        if not raw_code:
            return []

        symbol_type = "method" if parent_class else "function"
        full_name = f"{parent_class}.{node.name}" if parent_class else node.name
        docstring = ast.get_docstring(node) or ""

        # Sub-divide if function exceeds maximum character length
        if len(raw_code) > self.max_chunk_chars:
            return self._subdivide_large_code(
                text=raw_code,
                file_path=file_path,
                symbol_name=full_name,
                symbol_type=symbol_type,
                start_line=start_line,
                comment_char="#",
            )

        # Build contextual breadcrumb header
        header = f"# [Context: {file_path} | Scope: {full_name} | Lines: {start_line}-{end_line}]"
        if imports_summary:
            header += f"\n# Context Imports: {imports_summary}"
        contextual_text = f"{header}\n{raw_code}"

        return [
            AdalDocument(
                text=contextual_text,
                meta_data={
                    "file_path": file_path,
                    "symbol_name": full_name,
                    "symbol_type": symbol_type,
                    "line_range": [start_line, end_line],
                    "docstring": docstring[:200],
                    "is_code": True,
                    "title": f"{file_path}:{start_line}-{end_line} ({full_name})",
                },
            )
        ]

    def _extract_class_chunks(
        self,
        node: ast.ClassDef,
        lines: List[str],
        file_path: str,
        imports_summary: str = "",
    ) -> List[AdalDocument]:
        """Extract class definition and its internal methods as structured chunks."""
        start_line = getattr(node, "lineno", 1)
        end_line = getattr(node, "end_lineno", start_line)
        class_lines = lines[start_line - 1 : end_line]
        class_text = "\n".join(class_lines).strip()

        # If class is compact, store it as a single chunk
        if len(class_text) <= self.max_chunk_chars:
            header = f"# [Context: {file_path} | Scope: class {node.name} | Lines: {start_line}-{end_line}]"
            if imports_summary:
                header += f"\n# Context Imports: {imports_summary}"
            return [
                AdalDocument(
                    text=f"{header}\n{class_text}",
                    meta_data={
                        "file_path": file_path,
                        "symbol_name": node.name,
                        "symbol_type": "class",
                        "line_range": [start_line, end_line],
                        "docstring": (ast.get_docstring(node) or "")[:200],
                        "is_code": True,
                        "title": f"{file_path}:{start_line}-{end_line} (class {node.name})",
                    },
                )
            ]

        # For larger classes, extract header followed by each method
        chunks: List[AdalDocument] = []
        header_lines: List[str] = []
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                s = getattr(item, "lineno", 1) - 1
                e = getattr(item, "end_lineno", s + 1)
                header_lines.extend(lines[s:e])

        header_text = "\n".join(header_lines).strip()
        if header_text:
            cls_header = f"# [Context: {file_path} | Scope: class {node.name} (definition) | Lines: {start_line}-{start_line + len(header_lines)}]"
            if imports_summary:
                cls_header += f"\n# Context Imports: {imports_summary}"
            chunks.append(
                AdalDocument(
                    text=f"{cls_header}\nclass {node.name}:\n    " + header_text,
                    meta_data={
                        "file_path": file_path,
                        "symbol_name": node.name,
                        "symbol_type": "class_definition",
                        "line_range": [start_line, start_line + len(header_lines)],
                        "is_code": True,
                        "title": f"{file_path} (class {node.name})",
                    },
                )
            )

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.extend(
                    self._extract_function_chunk(
                        node=item,
                        lines=lines,
                        file_path=file_path,
                        parent_class=node.name,
                        imports_summary=imports_summary,
                    )
                )

        return chunks

    def _chunk_tree_sitter(
        self,
        content: str,
        file_path: str,
        ext: str,
    ) -> List[AdalDocument]:
        """Universal Tree-sitter AST parser for high-precision syntax chunking across enterprise languages."""
        lang_map = {
            ".js": "javascript",
            ".jsx": "javascript",
            ".mjs": "javascript",
            ".cjs": "javascript",
            ".ts": "typescript",
            ".tsx": "tsx",
            ".mts": "typescript",
            ".cts": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".cpp": "cpp",
            ".hpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".c": "c",
            ".h": "c",
            ".cs": "c_sharp",
        }
        lang_name = lang_map.get(ext)
        if not lang_name or tree_sitter_languages is None:
            return []

        try:
            parser = tree_sitter_languages.get_parser(lang_name)
        except Exception as e:
            log.debug(f"Tree-sitter parser unavailable for {lang_name}: {e}")
            return []

        try:
            tree = parser.parse(content.encode("utf-8", errors="replace"))
        except Exception as e:
            log.warning(f"Tree-sitter parse error in {file_path}: {e}")
            return []

        lines = content.splitlines()
        if not lines:
            return []

        comment_char = "#" if ext in (".py", ".sh", ".rb", ".yaml", ".yml") else "//"

        # Extract import headers
        import_lines = [
            l.strip()
            for l in lines[:30]
            if re.match(r"^(?:import|#include|package|using|from|use)\b", l.strip())
        ]
        imports_summary = " | ".join(import_lines[:5]) if import_lines else ""

        declaration_types = {
            "function_declaration",
            "method_definition",
            "method_declaration",
            "class_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "type_declaration",
            "function_item",
            "impl_item",
            "struct_item",
            "trait_item",
            "enum_item",
            "enum_declaration",
            "class_specifier",
            "struct_specifier",
            "function_definition",
        }

        chunks: List[AdalDocument] = []

        def extract_symbol_info(node: Any) -> tuple[str, str]:
            sym_name = ""
            name_node = node.child_by_field_name("name")
            if name_node:
                sym_name = name_node.text.decode("utf-8", errors="replace")
            else:
                for c in node.children:
                    if c.type in ("identifier", "type_identifier"):
                        sym_name = c.text.decode("utf-8", errors="replace")
                        break

            sym_type = "function" if "function" in node.type or "method" in node.type else "class"
            if "interface" in node.type:
                sym_type = "interface"
            elif "struct" in node.type:
                sym_type = "struct"
            return sym_name or "symbol", sym_type

        # Scan top-level nodes and exported statements
        target_nodes = []
        for child in tree.root_node.children:
            if child.type == "export_statement":
                for sub in child.children:
                    if sub.type in declaration_types:
                        target_nodes.append(sub)
            elif child.type in declaration_types:
                target_nodes.append(child)

        for node in target_nodes:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            chunk_slice = lines[start_line - 1 : end_line]
            raw_text = "\n".join(chunk_slice).strip()

            if len(raw_text) < self.min_chunk_chars:
                continue

            sym_name, sym_type = extract_symbol_info(node)

            if len(raw_text) > self.max_chunk_chars:
                chunks.extend(
                    self._subdivide_large_code(
                        text=raw_text,
                        file_path=file_path,
                        symbol_name=sym_name,
                        symbol_type=sym_type,
                        start_line=start_line,
                        comment_char=comment_char,
                    )
                )
            else:
                header = f"{comment_char} [Context: {file_path} | Scope: {sym_name} | Lines: {start_line}-{end_line}]"
                if imports_summary:
                    header += f"\n{comment_char} Context Imports: {imports_summary}"
                contextual_text = f"{header}\n{raw_text}"
                chunks.append(
                    AdalDocument(
                        text=contextual_text,
                        meta_data={
                            "file_path": file_path,
                            "symbol_name": sym_name,
                            "symbol_type": sym_type,
                            "line_range": [start_line, end_line],
                            "is_code": True,
                            "title": f"{file_path}:{start_line}-{end_line} ({sym_name})",
                        },
                    )
                )

        return chunks

    def _chunk_structural_code(
        self,
        content: str,
        file_path: str,
        ext: str,
    ) -> List[AdalDocument]:
        """Split multi-language code files (C/C++, JS/TS, Go, Rust, Java) respecting declaration boundaries."""
        lines = content.splitlines()
        if not lines:
            return []

        comment_char = "#" if ext in (".py", ".sh", ".bash", ".rb", ".yaml", ".yml") else "//"

        # Extract import/include header lines
        import_lines = [
            l.strip()
            for l in lines[:30]
            if re.match(r"^(?:import|#include|package|using|from)\b", l.strip())
        ]
        imports_summary = " | ".join(import_lines[:5]) if import_lines else ""

        # Regex for common function/class/struct boundaries across languages
        boundary_pattern = re.compile(
            r"^(?:"
            r"(?:export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_$]+)|"
            r"(?:public|protected|private|static|\s)*class\s+([A-Za-z0-9_$]+)|"
            r"(?:struct|interface|enum)\s+([A-Za-z0-9_$]+)|"
            r"func\s+(?:\(.*?\)\s*)?([A-Za-z0-9_]+)|"
            r"(?:pub\s+)?fn\s+([A-Za-z0-9_]+)|"
            r"(?:[A-Za-z0-9_:<>\*]+\s+)+([A-Za-z0-9_]+)\s*\([^;]*?\)\s*\{"
            r")",
            re.MULTILINE,
        )

        boundaries = []
        for i, line in enumerate(lines):
            match = boundary_pattern.search(line)
            if match:
                symbol = next((g for g in match.groups() if g), "symbol")
                boundaries.append((i, symbol))

        if not boundaries:
            return self._sliding_window_chunks(lines, file_path, comment_char=comment_char)

        chunks: List[AdalDocument] = []
        for idx, (line_idx, symbol) in enumerate(boundaries):
            next_line_idx = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(lines)
            chunk_slice = lines[line_idx:next_line_idx]
            raw_text = "\n".join(chunk_slice).strip()

            if len(raw_text) < self.min_chunk_chars and idx + 1 < len(boundaries):
                continue

            start = line_idx + 1
            end = next_line_idx

            if len(raw_text) > self.max_chunk_chars:
                chunks.extend(
                    self._subdivide_large_code(
                        text=raw_text,
                        file_path=file_path,
                        symbol_name=symbol,
                        symbol_type="structural_block",
                        start_line=start,
                        comment_char=comment_char,
                    )
                )
            else:
                header = f"{comment_char} [Context: {file_path} | Scope: {symbol} | Lines: {start}-{end}]"
                if imports_summary:
                    header += f"\n{comment_char} Context Imports: {imports_summary}"
                contextual_text = f"{header}\n{raw_text}"
                chunks.append(
                    AdalDocument(
                        text=contextual_text,
                        meta_data={
                            "file_path": file_path,
                            "symbol_name": symbol,
                            "symbol_type": "declaration",
                            "line_range": [start, end],
                            "is_code": True,
                            "title": f"{file_path}:{start}-{end} ({symbol})",
                        },
                    )
                )

        return chunks or self._sliding_window_chunks(lines, file_path, comment_char=comment_char)

    def _subdivide_large_code(
        self,
        text: str,
        file_path: str,
        symbol_name: str,
        symbol_type: str,
        start_line: int,
        comment_char: str = "//",
    ) -> List[AdalDocument]:
        """Subdivide oversized code blocks into bounded chunks with continuous line accounting."""
        sub_chunks: List[AdalDocument] = []
        lines = text.splitlines()
        step = 40
        for i in range(0, len(lines), step):
            sub_lines = lines[i : i + step]
            sub_text = "\n".join(sub_lines).strip()
            cur_start = start_line + i
            cur_end = cur_start + len(sub_lines) - 1
            if sub_text:
                sub_chunks.append(
                    AdalDocument(
                        text=f"{comment_char} [Context: {file_path} | Scope: {symbol_name} | Lines: {cur_start}-{cur_end}]\n" + sub_text,
                        meta_data={
                            "file_path": file_path,
                            "symbol_name": symbol_name,
                            "symbol_type": symbol_type,
                            "line_range": [cur_start, cur_end],
                            "is_code": True,
                            "title": f"{file_path}:{cur_start}-{cur_end} ({symbol_name})",
                        },
                    )
                )
        return sub_chunks

    def _sliding_window_chunks(
        self,
        lines: List[str],
        file_path: str,
        comment_char: str = "//",
    ) -> List[AdalDocument]:
        """Produce overlapping line windows for files lacking clear declaration keywords."""
        chunks: List[AdalDocument] = []
        window_size = 50
        overlap = 10
        total_lines = len(lines)

        for i in range(0, total_lines, window_size - overlap):
            chunk_slice = lines[i : i + window_size]
            raw_text = "\n".join(chunk_slice).strip()
            start = i + 1
            end = min(i + window_size, total_lines)
            if raw_text and len(raw_text) >= self.min_chunk_chars:
                header = f"{comment_char} [Context: {file_path} | Lines: {start}-{end}]"
                chunks.append(
                    AdalDocument(
                        text=f"{header}\n{raw_text}",
                        meta_data={
                            "file_path": file_path,
                            "symbol_name": f"lines_{start}_{end}",
                            "symbol_type": "text_block",
                            "line_range": [start, end],
                            "is_code": True,
                            "title": f"{file_path}:{start}-{end}",
                        },
                    )
                )
            if end >= total_lines:
                break
        return chunks

    @staticmethod
    def fold_code_skeleton(code: str, ext: str = ".py") -> str:
        """Compress large code files into folded AST interface skeletons for secondary prompt contexts."""
        if not code or not code.strip():
            return ""

        clean_ext = f".{ext.lstrip('.')}".lower()

        # Python AST folding
        if clean_ext == ".py":
            try:
                tree = ast.parse(code)
                lines = []
                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        lines.append(f"class {node.name}:")
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                doc = ast.get_docstring(item)
                                doc_line = f'        """{doc}"""\n' if doc else ""
                                lines.append(f"    def {item.name}(...): ...\n{doc_line}".rstrip())
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        doc = ast.get_docstring(node)
                        doc_line = f'    """{doc}"""\n' if doc else ""
                        lines.append(f"def {node.name}(...): ...\n{doc_line}".rstrip())
                if lines:
                    return "\n".join(lines)
            except Exception:
                pass

        # Structural regex signature folding for other languages
        sig_pattern = re.compile(
            r"^\s*(?:export\s+)?(?:public|private|protected|static|\s)*(?:async\s+)?(?:function|class|interface|struct|def|fn|func)\s+([A-Za-z0-9_$]+)[^;{]*",
            re.MULTILINE,
        )
        matches = sig_pattern.findall(code)
        if matches:
            skeleton_lines = [f"// {m} {{ ... }}" for m in matches[:15]]
            return "\n".join(skeleton_lines)

        code_lines = code.splitlines()
        return "\n".join(code_lines[:10]) + "\n// ... [remaining body folded for context budget]"
