import time
import pytest

from backend.app.services.redis_manager import RedisCacheManager


def test_redis_cache_manager_fallback():
    """Verify that RedisCacheManager operates seamlessly using local fallback when no Redis server is present."""
    mgr = RedisCacheManager(redis_url="")
    mgr.clear()

    # Embedding caching test
    test_text = "def hello_world(): return 'hello'"
    test_vec = [0.1] * 768

    assert mgr.get_embedding(test_text) is None
    mgr.set_embedding(test_text, test_vec, ttl=60)
    cached_vec = mgr.get_embedding(test_text)
    assert cached_vec == test_vec


def test_redis_query_cache():
    """Verify RAG query response caching and retrieval."""
    mgr = RedisCacheManager(redis_url="")
    mgr.clear()

    repo_slug = "facebook_react_c4ca4238"
    query = "Where is the reconciliation algorithm defined?"
    payload = {
        "answer": "React Fiber handles reconciliation in ReactFiberWorkLoop.js",
        "rationale": "Located in packages/react-reconciler",
        "contexts": [{"text": "function workLoopConcurrent()", "meta_data": {"file_path": "ReactFiberWorkLoop.js"}}],
    }

    assert mgr.get_query_cache(repo_slug, query) is None
    mgr.set_query_cache(repo_slug, query, payload, ttl=60)

    cached = mgr.get_query_cache(repo_slug, query)
    assert cached is not None
    assert cached["answer"] == payload["answer"]
    assert len(cached["contexts"]) == 1


def test_redis_distributed_lock():
    """Verify distributed lock acquisition and release semantics."""
    mgr = RedisCacheManager(redis_url="")
    mgr.clear()

    lock_key = "test_repo_indexing"
    # First acquisition succeeds
    assert mgr.acquire_lock(lock_key, timeout_seconds=10) is True

    # Second acquisition fails while held
    assert mgr.acquire_lock(lock_key, timeout_seconds=10) is False

    # After release, acquisition succeeds again
    mgr.release_lock(lock_key)
    assert mgr.acquire_lock(lock_key, timeout_seconds=10) is True
    mgr.release_lock(lock_key)


def test_redis_job_status():
    """Verify asynchronous ingestion job tracking."""
    mgr = RedisCacheManager(redis_url="")
    mgr.clear()

    job_id = "job_12345"
    status_data = {
        "status": "chunking",
        "progress": 45,
        "files_processed": 90,
        "total_files": 200,
    }

    assert mgr.get_job_status(job_id) is None
    mgr.set_job_status(job_id, status_data, ttl=60)

    loaded = mgr.get_job_status(job_id)
    assert loaded is not None
    assert loaded["status"] == "chunking"
    assert loaded["progress"] == 45
    assert "updated_at" in loaded

