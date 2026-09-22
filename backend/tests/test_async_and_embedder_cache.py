import time
from unittest.mock import MagicMock, patch
import pytest
from adalflow.core.types import ModelType
from fastapi.testclient import TestClient

from backend.app.models.gemini_embedder import GeminiEmbedderClient
from backend.app.services.redis_manager import RedisCacheManager
from backend.main import app
from backend.app.services.rag_service import RAGService


def test_gemini_embedder_redis_cache_hit():
    """Verify GeminiEmbedderClient skips API calls entirely when vectors exist in Redis cache."""
    redis_mgr = RedisCacheManager.get_instance()
    redis_mgr.clear()

    known_text = "def calculate_hash(x): return hash(x)"
    known_vector = [0.123] * 768
    redis_mgr.set_embedding(known_text, known_vector)

    embedder = GeminiEmbedderClient(api_key="mock_key")
    # Mock _sync_client so we can be 100% sure it is NEVER called on cache hit
    embedder._sync_client = MagicMock()

    results = embedder.call(
        {"input": [known_text]},
        model_type=ModelType.EMBEDDER,
    )

    assert len(results) == 1
    assert results[0]["values"] == known_vector
    embedder._sync_client.models.embed_content.assert_not_called()

    # Verify parse_embedding_response parses the cached vector properly
    output = embedder.parse_embedding_response(results)
    assert len(output.data) == 1
    assert output.data[0].embedding == known_vector
    assert output.data[0].index == 0


def test_gemini_embedder_caches_new_embeddings():
    """Verify GeminiEmbedderClient writes computed vectors into Redis after an API call."""
    redis_mgr = RedisCacheManager.get_instance()
    redis_mgr.clear()

    new_text = "def unique_computation(): return 42"
    computed_vector = [0.999] * 768

    embedder = GeminiEmbedderClient(api_key="mock_key")

    mock_response = MagicMock()
    mock_item = MagicMock()
    mock_item.values = computed_vector
    mock_response.embeddings = [mock_item]

    with patch.object(embedder, "_embed_batch_with_retry_and_fallback", return_value=mock_response):
        results = embedder.call({"input": [new_text]}, model_type=ModelType.EMBEDDER)

    assert len(results) == 1
    assert results[0]["values"] == computed_vector

    # Ensure Redis now contains the vector
    cached = redis_mgr.get_embedding(new_text)
    assert cached == computed_vector


def test_rag_service_async_init_and_status():
    """Verify background repository indexing creates job tracking and completes successfully."""
    service = RAGService.get_instance()

    with patch.object(service, "initialize_repository", return_value="https://github.com/psf/requests"):
        job_id = service.initialize_repository_async(
            repo_url="https://github.com/psf/requests",
            session_id="test_session_async",
        )

        assert job_id.startswith("job_")

        # Check job status immediately
        status = service.get_job_status(job_id)
        assert status is not None
        assert status["job_id"] == job_id
        assert status["repo_url"] == "https://github.com/psf/requests"

        # Wait briefly for the worker thread to finish
        for _ in range(20):
            time.sleep(0.05)
            status = service.get_job_status(job_id)
            if status and status["status"] in ("completed", "failed"):
                break

        assert status is not None
        assert status["status"] == "completed"
        assert status["progress"] == 1.0


def test_fastapi_async_init_and_status_routes():
    """Verify /init/async and /status/{job_id} REST endpoints work as expected."""
    client = TestClient(app)

    with patch.object(RAGService.get_instance(), "initialize_repository_async", return_value="job_test999"):
        response = client.post(
            "/init/async",
            json={"repo_url": "https://github.com/psf/requests", "session_id": "test_api"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert data["job_id"] == "job_test999"

    # Pre-populate status in Redis
    RedisCacheManager.get_instance().set_job_status(
        "job_test999",
        {
            "job_id": "job_test999",
            "status": "processing",
            "progress": 0.5,
            "message": "Indexing chunks in Qdrant...",
            "repo_url": "https://github.com/psf/requests",
            "slug": "psf_requests_12345678",
        },
    )

    status_resp = client.get("/status/job_test999")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["job_id"] == "job_test999"
    assert status_data["status"] == "processing"
    assert status_data["progress"] == 0.5
