from unittest.mock import MagicMock, patch
import pytest

from backend.app.rag.rag import RAG
from backend.app.utils.repo_utils import get_repo_slug, sanitize_collection_slug


def test_skip_reembedding_when_collection_exists():
    """Verify that if Qdrant already contains persistent points for the collection, prepare_retriever skips indexing."""
    with patch("backend.app.rag.rag.QdrantManager") as mock_qdrant_cls, \
         patch("backend.app.rag.rag.MemoriManager") as mock_memori_cls, \
         patch("backend.app.rag.rag.DatabaseManager") as mock_db_cls, \
         patch("backend.app.rag.rag.adal.Embedder"), \
         patch("backend.app.rag.rag.adal.Generator"), \
         patch("backend.app.rag.rag.LangGraphAgenticRAG"):

        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        rag = RAG(entity_id="test_user", process_id="test_proc")

        repo_url = "https://github.com/facebook/react"
        slug = get_repo_slug(repo_url)
        col_name = sanitize_collection_slug(slug)

        # Simulate collection already existing with 150 points
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.get_collection_point_count.return_value = 150

        # Call prepare_retriever
        rag.prepare_retriever(repo_url, force_reindex=False)

        # Verify Qdrant existence was queried
        mock_qdrant.collection_exists.assert_called_with(col_name)
        mock_qdrant.get_collection_point_count.assert_called_with(col_name)

        # Crucial check: db_manager.prepare_database must NEVER be called (no cloning, no re-embedding)
        mock_db.prepare_database.assert_not_called()
        mock_qdrant.index_documents.assert_not_called()

        # Retriever is set to qdrant_manager
        assert rag.retriever == mock_qdrant
        assert rag.current_collection == col_name


def test_force_reindex_rebuilds_collection():
    """Verify that when force_reindex=True, indexing executes even if collection exists."""
    with patch("backend.app.rag.rag.QdrantManager") as mock_qdrant_cls, \
         patch("backend.app.rag.rag.MemoriManager"), \
         patch("backend.app.rag.rag.DatabaseManager") as mock_db_cls, \
         patch("backend.app.rag.rag.adal.Embedder"), \
         patch("backend.app.rag.rag.adal.Generator"), \
         patch("backend.app.rag.rag.LangGraphAgenticRAG"):


        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        # Mock transformed docs with valid vector of 768 dimensions
        mock_doc = MagicMock()
        mock_doc.vector = [0.1] * 768
        mock_db.prepare_database.return_value = [mock_doc]

        rag = RAG(entity_id="test_user", process_id="test_proc")
        repo_url = "https://github.com/facebook/react"

        rag.prepare_retriever(repo_url, force_reindex=True)

        # Database preparation and Qdrant index_documents must be invoked
        mock_db.prepare_database.assert_called_once()
        mock_qdrant.index_documents.assert_called_once()

