"""Unit tests for RepoMapGenerator and PageRank architectural ranking."""

import pytest
from adalflow.core.types import Document as AdalDocument

from backend.app.rag.repo_map import RepoMapGenerator


def test_repo_map_generator_ranks_central_files():
    """Verify RepoMapGenerator computes PageRank and renders tree with high-centrality files."""
    chunks = [
        AdalDocument(
            text="from services.rag_service import RAGService\nfrom api.health_routes import router",
            meta_data={
                "file_path": "backend/app/main.py",
                "symbol_name": "create_app",
                "symbol_type": "function",
                "line_range": [1, 25],
            },
        ),
        AdalDocument(
            text="from embeddings.qdrant_manager import QdrantManager\nclass RAGService:\n    def query(self): pass",
            meta_data={
                "file_path": "backend/app/services/rag_service.py",
                "symbol_name": "RAGService",
                "symbol_type": "class",
                "line_range": [1, 100],
            },
        ),
        AdalDocument(
            text="class QdrantManager:\n    def hybrid_search(self): pass",
            meta_data={
                "file_path": "backend/app/embeddings/qdrant_manager.py",
                "symbol_name": "QdrantManager",
                "symbol_type": "class",
                "line_range": [1, 50],
            },
        ),
    ]

    gen = RepoMapGenerator(max_tokens=500)
    gen.build_from_chunks(chunks)
    rendered = gen.generate_repo_map()

    assert "# Repository Architecture Map" in rendered
    assert "backend/app/main.py" in rendered
    assert "backend/app/services/rag_service.py" in rendered


def test_repo_map_focus_query_boosting():
    """Verify user query terms boost matching file PageRank in the repo map."""
    chunks = [
        AdalDocument(
            text="def setup_auth(): pass",
            meta_data={
                "file_path": "auth/security.py",
                "symbol_name": "setup_auth",
                "symbol_type": "function",
                "line_range": [1, 20],
            },
        ),
        AdalDocument(
            text="def format_date(): pass",
            meta_data={
                "file_path": "utils/dates.py",
                "symbol_name": "format_date",
                "symbol_type": "function",
                "line_range": [1, 10],
            },
        ),
    ]

    gen = RepoMapGenerator(max_tokens=300)
    gen.build_from_chunks(chunks)
    rendered = gen.generate_repo_map(focus_query="how does setup_auth work?")

    assert "auth/security.py" in rendered

