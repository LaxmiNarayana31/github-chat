"""Unit tests for Git-blob content-addressable deduplication."""

from unittest.mock import MagicMock
import pytest

from backend.app.pipelines.blob_deduplicator import BlobDeduplicator


def test_compute_blob_hash_deterministic():
    """Verify Git-blob SHA-1 hash is deterministic and content-dependent."""
    dedup = BlobDeduplicator.get_instance()
    code1 = "def add(a, b):\n    return a + b\n"
    code2 = "def add(a, b):\n    return a + b\n"
    code3 = "def multiply(a, b):\n    return a * b\n"

    hash1 = dedup.compute_blob_hash(code1)
    hash2 = dedup.compute_blob_hash(code2)
    hash3 = dedup.compute_blob_hash(code3)

    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 40  # SHA-1 40-character hex string


def test_blob_deduplicator_cache_cycle():
    """Verify storing and retrieving chunks by blob hash via Redis cache."""
    mock_redis = MagicMock()
    mock_redis.is_connected = True
    dedup = BlobDeduplicator(redis_mgr=mock_redis)

    test_content = "class Greeter:\n    def greet(self): pass"
    blob_hash = dedup.compute_blob_hash(test_content)
    test_chunks = [
        {
            "text": test_content,
            "vector": [0.1] * 768,
            "meta_data": {"symbol_name": "Greeter"},
        }
    ]

    mock_redis.get.return_value = None
    assert dedup.get_cached_chunks(blob_hash) is None

    dedup.cache_chunks(blob_hash, test_chunks)
    mock_redis.setex.assert_called_once()

