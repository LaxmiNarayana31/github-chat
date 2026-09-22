"""Comprehensive test suite verifying route and service level exception handling."""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.app.services.rag_service import RAGService
from backend.app.services.system_service import SystemService


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


def test_root_endpoint_exception_handling(client):
    """Verify root endpoint catches service errors and raises HTTP 500."""
    with patch.object(SystemService.get_instance(), "get_api_overview", side_effect=RuntimeError("System error")):
        response = client.get("/")
        assert response.status_code == 500
        assert "System error" in response.json()["detail"]


def test_health_endpoint_exception_handling(client):
    """Verify health endpoint catches service errors and raises HTTP 500."""
    with patch.object(SystemService.get_instance(), "get_health_status", side_effect=RuntimeError("Health check crash")):
        response = client.get("/health")
        assert response.status_code == 500
        assert "Health check crash" in response.json()["detail"]


def test_init_validation_error_returns_400(client):
    """Verify service ValueError raises 400 Bad Request."""
    with patch.object(RAGService.get_instance(), "handle_init_sync", side_effect=ValueError("Invalid repo URL format")):
        response = client.post("/init", json={"repo_url": "https://github.com/invalid"})
        assert response.status_code == 400
        assert "Invalid repo URL format" in response.json()["detail"]


def test_init_service_error_returns_500(client):
    """Verify internal failure in /init raises 500 Internal Server Error."""
    with patch.object(RAGService.get_instance(), "handle_init_sync", side_effect=RuntimeError("Indexing crash")):
        response = client.post("/init", json={"repo_url": "https://github.com/fastapi/fastapi"})
        assert response.status_code == 500
        assert "Indexing crash" in response.json()["detail"]


def test_init_async_validation_error_returns_400(client):
    """Verify service ValueError in async init raises 400 Bad Request."""
    with patch.object(RAGService.get_instance(), "handle_init_async", side_effect=ValueError("Invalid async repo")):
        response = client.post("/init/async", json={"repo_url": "https://github.com/invalid"})
        assert response.status_code == 400
        assert "Invalid async repo" in response.json()["detail"]


def test_init_async_service_error_returns_500(client):
    """Verify internal failure in /init/async raises 500 Internal Server Error."""
    with patch.object(RAGService.get_instance(), "handle_init_async", side_effect=RuntimeError("Queue error")):
        response = client.post("/init/async", json={"repo_url": "https://github.com/fastapi/fastapi"})
        assert response.status_code == 500
        assert "Queue error" in response.json()["detail"]


def test_status_not_found_returns_404(client):
    """Verify non-existent job ID returns 404 Not Found."""
    with patch.object(RAGService.get_instance(), "get_job_status_response", return_value=None):
        response = client.get("/status/job_nonexistent_999")
        assert response.status_code == 404
        assert "was not found" in response.json()["detail"]


def test_status_service_error_returns_500(client):
    """Verify internal error in status endpoint returns 500."""
    with patch.object(RAGService.get_instance(), "get_job_status_response", side_effect=RuntimeError("Redis down")):
        response = client.get("/status/job_test_123")
        assert response.status_code == 500
        assert "Redis down" in response.json()["detail"]


def test_query_validation_error_returns_400(client):
    """Verify service ValueError in query returns 400 Bad Request."""
    with patch.object(RAGService.get_instance(), "execute_query", side_effect=ValueError("Repository not indexed")):
        response = client.post("/query", json={"query": "test query"})
        assert response.status_code == 400
        assert "Repository not indexed" in response.json()["detail"]


def test_query_service_error_returns_500(client):
    """Verify unexpected failure in /query returns 500."""
    with patch.object(RAGService.get_instance(), "execute_query", side_effect=RuntimeError("Query pipeline failure")):
        response = client.post("/query", json={"query": "test query"})
        assert response.status_code == 500
        assert "Query pipeline failure" in response.json()["detail"]


def test_query_stream_service_error_returns_500(client):
    """Verify unexpected failure creating stream returns 500."""
    with patch.object(RAGService.get_instance(), "create_streaming_response", side_effect=RuntimeError("Stream init failed")):
        response = client.post("/query/stream", json={"query": "test query"})
        assert response.status_code == 500
        assert "Stream init failed" in response.json()["detail"]


def test_clear_memory_error_returns_500(client):
    """Verify unexpected error clearing memory returns 500."""
    with patch.object(RAGService.get_instance(), "clear_conversation_memory", side_effect=RuntimeError("Memory store down")):
        response = client.post("/clear-memory", json={})
        assert response.status_code == 500
        assert "Memory store down" in response.json()["detail"]


def test_set_context_error_returns_500(client):
    """Verify unexpected error setting context returns 500."""
    with patch.object(RAGService.get_instance(), "restore_context", side_effect=RuntimeError("Context parser error")):
        response = client.post("/set-context", json=[{"role": "user", "content": "hello"}])
        assert response.status_code == 500
        assert "Context parser error" in response.json()["detail"]

