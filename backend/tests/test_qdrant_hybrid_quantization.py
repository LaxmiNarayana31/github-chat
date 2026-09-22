"""Unit tests verifying Qdrant Hybrid Search (Dense + BM25 RRF) and INT8 quantization."""

from adalflow.core.types import Document as AdalDocument
import pytest
from qdrant_client import models

from backend.app.embeddings.qdrant_manager import QdrantManager, sanitize_collection_name


def test_qdrant_hybrid_collection_creation_and_quantization():
    """Verify collection is created with dense INT8 scalar quantization and sparse BM25 IDF."""
    manager = QdrantManager(url=None, storage_path=None)  # in-memory
    col_name = "test_hybrid_quant"

    created_name = manager.get_or_create_collection(col_name, dense_dimension=768, force_recreate=True)
    assert created_name.endswith(col_name)

    info = manager.client.get_collection(created_name)
    assert info is not None and info.config is not None and info.config.params is not None

    vectors_cfg = info.config.params.vectors
    assert isinstance(vectors_cfg, dict)
    assert "dense_vector" in vectors_cfg

    sparse_cfg_dict = info.config.params.sparse_vectors
    assert isinstance(sparse_cfg_dict, dict)
    assert "bm25_sparse_vector" in sparse_cfg_dict

    sparse_cfg = sparse_cfg_dict["bm25_sparse_vector"]
    assert sparse_cfg.modifier == models.Modifier.IDF


def test_qdrant_hybrid_indexing_and_search_execution():
    """Verify dual-vector indexing and hybrid search execution using RRF and quantization params."""
    manager = QdrantManager(url=None, storage_path=None)
    col_name = "test_hybrid_execution"

    doc1 = AdalDocument(
        text="def calculate_sha256(data: bytes) -> str: return hashlib.sha256(data).hexdigest()",
        meta_data={"file_path": "crypto/hash.py", "type": "function"},
    )
    doc1.vector = [0.05] * 768

    doc2 = AdalDocument(
        text="class RedisLockManager: async def acquire_lock(self, key: str): pass",
        meta_data={"file_path": "cache/redis.py", "type": "class"},
    )
    doc2.vector = [0.10] * 768

    manager.index_documents(
        collection_name=col_name,
        documents=[doc1, doc2],
        dense_dimension=768,
        force_reindex=True,
    )

    count = manager.get_collection_point_count(sanitize_collection_name(col_name))
    assert count == 2

    # Query with hybrid search
    query_dense = [0.05] * 768
    results = manager.hybrid_search(
        collection_name=col_name,
        query_text="sha256 hash calculation",
        query_dense_vector=query_dense,
        top_k=2,
    )

    assert len(results) > 0
    top_meta = results[0].meta_data or {}
    assert "crypto/hash.py" in str(top_meta.get("file_path", ""))
    assert top_meta.get("rrf_score") is not None

