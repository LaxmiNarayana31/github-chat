"""Qdrant vector database manager providing hybrid search and quantization."""

import atexit
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import uuid

from adalflow.core.types import Document as AdalDocument
from adalflow.utils import printc
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct

from backend.app.config.config import config

log = logging.getLogger(__name__)


def sanitize_collection_name(name: str) -> str:
    """Ensure collection name meets Qdrant naming standards."""
    prefix = config.get("qdrant", {}).get("collection_prefix", "github_chat_")
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_").lower()
    full_name = clean if clean.startswith(prefix) else f"{prefix}{clean}"
    return full_name[:64]


class QdrantManager:
    """Manages Qdrant vector database operations, hybrid collections, and RRF search."""

    _instance: Optional['QdrantManager'] = None

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        storage_path: Optional[str] = None,
        cloud_inference: Optional[bool] = None,
    ):
        """Initialize Qdrant client connection parameters and fallback options."""
        qdrant_cfg: Dict[str, Any] = config.get("qdrant", {}) if isinstance(config.get("qdrant"), dict) else {}
        raw_url = url or qdrant_cfg.get("url") or os.getenv("QDRANT_URL", "").strip() or None
        self.url: Optional[str] = str(raw_url).strip() if raw_url else None

        raw_api_key = api_key or qdrant_cfg.get("api_key") or os.getenv("QDRANT_API_KEY", "").strip() or None
        self.api_key: Optional[str] = str(raw_api_key).strip() if raw_api_key else None

        raw_storage = storage_path or qdrant_cfg.get("storage_path") or None
        self.storage_path: Optional[str] = str(raw_storage) if raw_storage else None

        self.cloud_inference: bool = bool(
            cloud_inference if cloud_inference is not None else qdrant_cfg.get("cloud_inference", True)
        )
        self.bm25_model_name: str = str(qdrant_cfg.get("bm25_model", "Qdrant/bm25"))
        self._bm25_model: Optional[SparseTextEmbedding] = None
        self._client: Optional[QdrantClient] = None
        self._init_client()

    def _init_client(self):
        """Initialize connection to Qdrant cluster, local disk, or in-memory fallback."""
        if self.url:
            self._client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                cloud_inference=self.cloud_inference,
                timeout=60,
            )
        elif self.storage_path:
            abs_storage = os.path.abspath(self.storage_path)
            os.makedirs(abs_storage, exist_ok=True)
            self._client = QdrantClient(path=abs_storage)
        else:
            self._client = QdrantClient(location=":memory:")

        atexit.register(self.close)

    def close(self):
        """Cleanly close client connection to prevent file lock issues on Windows."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception as e:
                log.warning(f"Error closing Qdrant client: {e}")
            finally:
                self._client = None

    @property
    def client(self) -> QdrantClient:
        """Return initialized QdrantClient instance."""
        if self._client is None:
            self._init_client()
        assert self._client is not None, "Qdrant client could not be initialized."
        return self._client

    @property
    def bm25_embedder(self) -> SparseTextEmbedding:
        """Lazy load local FastEmbed BM25 model for client-side sparse vector generation."""
        if self._bm25_model is None:
            self._bm25_model = SparseTextEmbedding(model_name=self.bm25_model_name)
        return self._bm25_model

    def embed_sparse_texts(self, texts: List[str]) -> List[models.SparseVector]:
        """Generate BM25 sparse vectors for a batch of text chunks."""
        embeddings = list(self.bm25_embedder.embed(texts))
        return [
            models.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())
            for emb in embeddings
        ]

    def embed_sparse_query(self, query: str) -> models.SparseVector:
        """Generate BM25 sparse vector for a search query string."""
        results = self.embed_sparse_texts([query])
        return results[0] if results else models.SparseVector(indices=[], values=[])

    def collection_exists(self, collection_name: str) -> bool:
        """Check whether a collection exists in Qdrant."""
        try:
            return self.client.collection_exists(collection_name)
        except Exception as e:
            log.warning(f"Error checking collection existence for {collection_name}: {e}")
            return False

    def get_collection_point_count(self, collection_name: str) -> int:
        """Retrieve total point count within a specified collection."""
        try:
            if not self.collection_exists(collection_name):
                return 0
            info = self.client.get_collection(collection_name)
            return info.points_count or 0
        except Exception as e:
            log.warning(f"Error getting point count for {collection_name}: {e}")
            return 0

    def get_or_create_collection(
        self,
        collection_name: str,
        dense_dimension: int = 768,
        force_recreate: bool = False,
    ) -> str:
        """Ensure collection exists with dense scalar quantization and sparse BM25 IDF."""
        clean_name = sanitize_collection_name(collection_name)
        exists = self.collection_exists(clean_name)

        if exists and force_recreate:
            self.client.delete_collection(clean_name)
            exists = False

        if not exists:
            # Configure INT8 scalar quantization to reduce memory usage 4x with minimal accuracy loss
            quantization_config = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    always_ram=True,
                    quantile=0.99,
                )
            )

            # Create hybrid collection with dense cosine vector and sparse BM25 IDF vector
            self.client.create_collection(
                collection_name=clean_name,
                vectors_config={
                    "dense_vector": models.VectorParams(
                        size=dense_dimension,
                        distance=models.Distance.COSINE,
                        quantization_config=quantization_config,
                    )
                },
                sparse_vectors_config={
                    "bm25_sparse_vector": models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                },
            )
        return clean_name

    def index_documents(
        self,
        collection_name: str,
        documents: List[AdalDocument],
        dense_dimension: int = 768,
        batch_size: int = 40,
        force_reindex: bool = False,
    ) -> int:
        """Index code chunks into collection with dual dense and BM25 sparse vectors."""
        clean_name = self.get_or_create_collection(
            collection_name, dense_dimension=dense_dimension, force_recreate=force_reindex
        )

        existing_count = self.get_collection_point_count(clean_name)
        if existing_count > 0 and not force_reindex:
            return existing_count

        total_docs = len(documents)
        total_upserted = 0

        for i in range(0, total_docs, batch_size):
            batch = documents[i : i + batch_size]
            texts = [doc.text for doc in batch]
            sparse_vectors = self.embed_sparse_texts(texts)

            points: List[PointStruct] = []
            for doc_idx, (doc, sparse_vec) in enumerate(zip(batch, sparse_vectors)):
                dense_vec = getattr(doc, "vector", None)
                if not dense_vec or len(dense_vec) != dense_dimension:
                    continue

                meta = doc.meta_data or {}
                unique_str = f"{clean_name}_{meta.get('file_path', '')}_{i + doc_idx}"
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))

                payload = {
                    "text": doc.text,
                    "file_path": meta.get("file_path", ""),
                    "type": meta.get("type", ""),
                    "is_code": meta.get("is_code", True),
                    "is_implementation": meta.get("is_implementation", True),
                    "title": meta.get("title", meta.get("file_path", "")),
                    "symbol_name": meta.get("symbol_name", ""),
                    "symbol_type": meta.get("symbol_type", ""),
                    "line_range": meta.get("line_range", []),
                    "docstring": meta.get("docstring", ""),
                    "chunk_index": i + doc_idx,
                }

                # Index dual named vectors for native hybrid search in Qdrant
                point = PointStruct(
                    id=point_id,
                    vector={
                        "dense_vector": dense_vec,
                        "bm25_sparse_vector": sparse_vec,
                    },
                    payload=payload,
                )
                points.append(point)

            if points:
                self.client.upsert(collection_name=clean_name, points=points)
                total_upserted += len(points)

        return total_upserted

    def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        query_dense_vector: List[float],
        top_k: int = 5,
    ) -> List[AdalDocument]:
        """Execute hybrid search fusing dense Cosine vector and sparse BM25 with RRF."""
        clean_name = sanitize_collection_name(collection_name)
        if not self.collection_exists(clean_name):
            return []

        sparse_query_vec = self.embed_sparse_texts([query_text])[0]
        search_params = models.SearchParams(
            quantization=models.QuantizationSearchParams(
                rescore=True,
                oversampling=2.0,
            )
        )

        prefetch_list = []
        if query_dense_vector:
            prefetch_list.append(
                models.Prefetch(
                    query=query_dense_vector,
                    using="dense_vector",
                    limit=top_k * 2,
                    params=search_params,
                )
            )

        if sparse_query_vec.indices:
            prefetch_list.append(
                models.Prefetch(
                    query=sparse_query_vec,
                    using="bm25_sparse_vector",
                    limit=top_k * 2,
                )
            )

        if not prefetch_list:
            return []

        # Fuse candidate sets with Reciprocal Rank Fusion (RRF)
        if len(prefetch_list) > 1:
            results = self.client.query_points(
                collection_name=clean_name,
                prefetch=prefetch_list,
                query=models.RrfQuery(rrf=models.Rrf()),
                limit=top_k,
                with_payload=True,
            )
        else:
            results = self.client.query_points(
                collection_name=clean_name,
                query=query_dense_vector if query_dense_vector else sparse_query_vec,
                using="dense_vector" if query_dense_vector else "bm25_sparse_vector",
                limit=top_k,
                params=search_params if query_dense_vector else None,
                with_payload=True,
            )

        matched_points = results.points or []
        documents: List[AdalDocument] = []
        for p in matched_points:
            payload = p.payload or {}
            meta = {
                "file_path": payload.get("file_path", ""),
                "type": payload.get("type", ""),
                "is_code": payload.get("is_code", True),
                "is_implementation": payload.get("is_implementation", True),
                "title": payload.get("title", ""),
                "symbol_name": payload.get("symbol_name", ""),
                "symbol_type": payload.get("symbol_type", ""),
                "line_range": payload.get("line_range", []),
                "docstring": payload.get("docstring", ""),
                "rrf_score": getattr(p, "score", None),
                "point_id": p.id,
            }
            documents.append(AdalDocument(text=payload.get("text", ""), meta_data=meta))

        return documents

    def get_indexed_files(self, collection_name: str, limit: int = 100) -> List[str]:
        """Retrieve distinct file paths stored within the collection."""
        clean_name = sanitize_collection_name(collection_name)
        if not self.collection_exists(clean_name):
            return []
        try:
            records, _ = self.client.scroll(
                collection_name=clean_name,
                limit=limit,
                with_payload=["file_path"],
                with_vectors=False,
            )
            files = {r.payload["file_path"] for r in records if r.payload and r.payload.get("file_path")}
            return sorted(list(files))
        except Exception as e:
            log.warning(f"Error scrolling files for {clean_name}: {e}")
            return []
