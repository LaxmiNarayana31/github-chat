"""Verification tests for the GenAi Clean System package architecture using clean, direct imports."""

import pytest

# 1. LLM Models package
from backend.app.llm_models.gemini_embedder import GeminiEmbedderClient
from backend.app.llm_models.groq_client import MultiProviderLLMClient

# 2. Embeddings package
from backend.app.embeddings.qdrant_manager import QdrantManager, sanitize_collection_name

# 3. Prompts package
from backend.app.prompts.system_prompt import RAG_TEMPLATE, SYSTEM_PROMPT

# 4. Pipelines package
from backend.app.pipelines.data_pipeline import (
    DatabaseManager,
    download_github_repo,
    read_all_documents,
)

# 5. Services package
from backend.app.services.memory_manager import MemoriManager
from backend.app.services.rag_service import RAGService
from backend.app.services.redis_manager import RedisCacheManager
from backend.app.services.system_service import SystemService

# 6. RAG package
from backend.app.rag.agentic_rag import LangGraphAgenticRAG
from backend.app.rag.rag import Memory, RAG

# 7. Config package
from backend.app.config.config import DEFAULT_EMBEDDING_MODEL, config
from backend.app.config.constants import CODE_EXTENSIONS, IGNORED_DIRS

# 8. API package
from backend.app.api.health_routes import router as health_router
from backend.app.api.rag_routes import router as rag_router

# 9. DTO package
from backend.app.dto.document_dto import Document, DocumentMetadata
from backend.app.dto.rag_dto import QueryRequest, QueryResponse
from backend.app.dto.system_dto import HealthResponse

# 10. Utils package
from backend.app.utils.env_utils import check_api_keys
from backend.app.utils.repo_utils import get_repo_slug, normalize_repo_url


def test_clean_direct_modular_imports():
    """Verify that every domain module in backend.app can be cleanly imported directly without __init__.py bloat."""
    # 1. Models
    assert GeminiEmbedderClient is not None
    assert MultiProviderLLMClient is not None

    # 2. Embeddings
    assert QdrantManager is not None
    assert callable(sanitize_collection_name)

    # 3. Prompts
    assert "Repository Intelligence Bot" in SYSTEM_PROMPT
    assert "<SYS>" in RAG_TEMPLATE

    # 4. Pipelines
    assert DatabaseManager is not None
    assert callable(download_github_repo)
    assert callable(read_all_documents)

    # 5. Services
    assert RAGService is not None
    assert RedisCacheManager is not None
    assert MemoriManager is not None
    assert SystemService is not None

    # 6. RAG
    assert RAG is not None
    assert Memory is not None
    assert LangGraphAgenticRAG is not None

    # 7. Config
    assert isinstance(config, dict)
    assert DEFAULT_EMBEDDING_MODEL == "gemini-embedding-2"
    assert ".py" in CODE_EXTENSIONS
    assert ".git" in IGNORED_DIRS

    # 8. API
    assert health_router is not None
    assert rag_router is not None

    # 9. DTO
    assert Document is not None
    assert QueryResponse is not None
    assert HealthResponse is not None

    # 10. Utils
    assert callable(check_api_keys)
    assert callable(get_repo_slug)
    assert callable(normalize_repo_url)
