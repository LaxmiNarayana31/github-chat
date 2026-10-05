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


def test_same_repo_url_multi_user_deduplication():
    """Verify that when User A indexes a repo, User B with the same URL completely bypasses chunking and embedding."""
    from backend.app.services.rag_service import RAGService
    service = RAGService.get_instance()

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

        # User A indexes repository
        mock_qdrant.collection_exists.return_value = False
        mock_doc = MagicMock()
        mock_doc.vector = [0.1] * 768
        mock_db.prepare_database.return_value = [mock_doc]

        repo_url = "https://github.com/pallets/flask"
        service.initialize_repository(repo_url=repo_url, session_id="user_a")
        assert mock_db.prepare_database.call_count == 1

        # Now collection exists in Qdrant with points
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.get_collection_point_count.return_value = 120

        # User B provides same URL (even with trailing .git or whitespace)
        service.initialize_repository(repo_url="https://github.com/pallets/flask.git", session_id="user_b")

        # Crucial: prepare_database was NOT called again! call_count remains 1!
        assert mock_db.prepare_database.call_count == 1
        # User B's session now has flask ready
        assert service._session_repos.get("user_b") == "https://github.com/pallets/flask"


