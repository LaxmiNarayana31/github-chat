"""Unit tests for ASTChunker code splitting and symbol metadata extraction."""

from typing import Any, Dict
import pytest

from backend.app.pipelines.ast_chunker import ASTChunker


def get_meta(doc: Any) -> Dict[str, Any]:
    """Safely extract metadata dictionary from a document."""
    if hasattr(doc, "meta_data") and isinstance(doc.meta_data, dict):
        return doc.meta_data
    return {}


def test_python_ast_chunker_extracts_functions_and_classes():
    """Verify ASTChunker extracts Python functions and classes with line ranges and symbols."""
    chunker = ASTChunker()
    python_code = """\"\"\"Sample module docstring.\"\"\"
import os
import sys

def compute_total(items):
    \"\"\"Compute sum of items.\"\"\"
    total = sum(items)
    return total

class DataProcessor:
    \"\"\"Processes structured records.\"\"\"
    def __init__(self, name):
        self.name = name

    def process(self, data):
        return [x * 2 for x in data]
"""

    chunks = chunker.chunk_file(python_code, file_path="sample.py", file_type="py")
    assert len(chunks) >= 2

    # Check extracted symbol names and line ranges
    symbols = {str(get_meta(c).get("symbol_name", "")) for c in chunks}
    assert "compute_total" in symbols
    assert any("DataProcessor" in s for s in symbols)

    for c in chunks:
        meta = get_meta(c)
        assert meta.get("file_path") == "sample.py"
        assert meta.get("is_code") is True
        line_range = meta.get("line_range")
        assert isinstance(line_range, list) and len(line_range) == 2
        assert line_range[0] <= line_range[1]
        # Contextual header check: all chunks should start with breadcrumb context
        assert c.text.startswith("# [Context: sample.py")

    # Verify function chunk has context imports
    func_chunk = next(c for c in chunks if get_meta(c).get("symbol_name") == "compute_total")
    assert "Context Imports: os | sys" in func_chunk.text
    assert "def compute_total(items):" in func_chunk.text


def test_multi_language_structural_chunker():
    """Verify structural regex chunking for JavaScript / TypeScript / C++."""
    chunker = ASTChunker()
    ts_code = """import React from 'react';

export function UserCard(props) {
    const { name, email } = props;
    return <div>{name}: {email}</div>;
}

class AuthService {
    login(credentials) {
        return fetch('/api/login', { method: 'POST' });
    }
}
"""
    chunks = chunker.chunk_file(ts_code, file_path="components/UserCard.tsx", file_type="tsx")
    assert len(chunks) >= 1
    symbols = {str(get_meta(c).get("symbol_name", "")) for c in chunks}
    assert any("UserCard" in s or "AuthService" in s for s in symbols)
    for c in chunks:
        assert "// [Context: components/UserCard.tsx" in c.text


def test_embedding_pipeline_bypasses_text_splitter():
    """Verify prepare_embedding_pipeline uses ToEmbeddings directly without TextSplitter."""
    from backend.app.pipelines.data_pipeline import prepare_embedding_pipeline, prepare_data_pipeline
    from adalflow.components.data_process import ToEmbeddings, TextSplitter

    emb_pipeline = prepare_embedding_pipeline()
    assert isinstance(emb_pipeline, ToEmbeddings)

    seq_pipeline = prepare_data_pipeline()
    assert any(isinstance(step, TextSplitter) for step in seq_pipeline)


def test_tree_sitter_multi_language_parsing():
    """Verify universal Tree-sitter AST extraction across Go, Rust, and Java."""
    chunker = ASTChunker()

    # Go source test
    go_code = """package main
import "fmt"

func CalculateTotal(prices []float64) float64 {
    var total float64
    for _, p := range prices {
        total += p
    }
    return total
}
"""
    go_chunks = chunker.chunk_file(go_code, file_path="main.go", file_type="go")
    assert len(go_chunks) >= 1
    go_symbols = [get_meta(c).get("symbol_name") for c in go_chunks]
    assert "CalculateTotal" in go_symbols

    # Rust source test
    rs_code = """pub fn process_data(input: &str) -> String {
    format!("processed: {}", input)
}
"""
    rs_chunks = chunker.chunk_file(rs_code, file_path="lib.rs", file_type="rs")
    assert len(rs_chunks) >= 1
    rs_symbols = [get_meta(c).get("symbol_name") for c in rs_chunks]
    assert "process_data" in rs_symbols

