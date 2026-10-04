"""Qdrant vector database manager providing hybrid search and quantization."""

import atexit
import logging
import os
import re
from typing import Any, Dict, List, Optional
import uuid

from adalflow.core.types import Document as AdalDocument
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct

from backend.app.config.config import config

log = logging.getLogger(__name__)


def sanitize_collection_name(name: str) -> str:
    """Sanitizes collection name to meet Qdrant standards."""
    try:
        prefix = (
            config.get("qdrant", {}).get("collection_prefix", "github_chat_")
            if isinstance(config.get("qdrant"), dict)
            else "github_chat_"
        )
        clean = re.sub(r"[^a-zA-Z0-9_-]", "_", str(name or "")).strip("_").lower()
        full_name = clean if clean.startswith(prefix) else f"{prefix}{clean}"
        return full_name[:64] or f"{prefix}default"
    except Exception as e:
        log.error(f"Error sanitizing collection name: {e}")
        return "github_chat_default"


class QdrantManager:
    """Manages Qdrant vector database operations, hybrid collections, and RRF search."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        storage_path: Optional[str] = None,
        cloud_inference: Optional[bool] = None,
    ):
        """Initializes Qdrant client connection and FastEmbed sparse model."""
        try:
            qdrant_cfg: Dict[str, Any] = (
                config.get("qdrant", {}) if isinstance(config.get("qdrant"), dict) else {}
            )
            raw_url = url or qdrant_cfg.get("url") or os.getenv("QDRANT_URL", "").strip() or None
            raw_api_key = (
                api_key or qdrant_cfg.get("api_key") or os.getenv("QDRANT_API_KEY", "").strip() or None
            )
            raw_storage = storage_path or qdrant_cfg.get("storage_path") or None
            use_cloud = bool(
                cloud_inference if cloud_inference is not None else qdrant_cfg.get("cloud_inference", True)
            )

            # Initialize client based on provided configuration with fallback to memory
            if raw_url:
                self.client = QdrantClient(
                    url=str(raw_url).strip(),
                    api_key=str(raw_api_key).strip() if raw_api_key else None,
                    cloud_inference=use_cloud,
                    timeout=60,
                )
            elif raw_storage:
                abs_storage = os.path.abspath(str(raw_storage))
                os.makedirs(abs_storage, exist_ok=True)
                self.client = QdrantClient(path=abs_storage)
            else:
                self.client = QdrantClient(location=":memory:")

            self.bm25_model_name: str = str(qdrant_cfg.get("bm25_model", "Qdrant/bm25"))
            self._bm25_model: Optional[SparseTextEmbedding] = None
            atexit.register(self.close)
        except Exception as e:
            log.warning(f"Error initializing Qdrant client: {e}. Falling back to in-memory instance.")
            self.client = QdrantClient(location=":memory:")
            self.bm25_model_name = "Qdrant/bm25"
            self._bm25_model = None
            atexit.register(self.close)

    def close(self):
        """Cleanly closes Qdrant client connection."""
        try:
            if hasattr(self, "client") and self.client is not None:
                self.client.close()
        except Exception as e:
            log.warning(f"Error closing Qdrant client: {e}")

    @property
    def bm25_embedder(self) -> SparseTextEmbedding:
        """Lazy loads local FastEmbed BM25 sparse model."""
        try:
            if self._bm25_model is None:
                self._bm25_model = SparseTextEmbedding(model_name=self.bm25_model_name)
            return self._bm25_model
        except Exception as e:
            log.error(f"Failed to load BM25 model: {e}")
            raise

    def embed_sparse_texts(self, texts: List[str]) -> List[models.SparseVector]:
        """Generates BM25 sparse vectors for text chunks."""
        try:
            if not texts:
                return []
            sanitized = [str(t) if t is not None else "" for t in texts]
            embeddings = list(self.bm25_embedder.embed(sanitized))
            return [
                models.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())
                for emb in embeddings
            ]
        except Exception as e:
            log.warning(f"Sparse embedding error: {e}. Returning empty vectors.")
            return [models.SparseVector(indices=[], values=[]) for _ in range(len(texts))]

    def collection_exists(self, collection_name: str) -> bool:
        """Checks whether a collection exists in Qdrant."""
        try:
            clean_name = sanitize_collection_name(collection_name)
            return self.client.collection_exists(clean_name)
        except Exception as e:
            log.warning(f"Error checking collection {collection_name}: {e}")
            return False

    def get_collection_point_count(self, collection_name: str) -> int:
        """Retrieves total point count within collection."""
        try:
            clean_name = sanitize_collection_name(collection_name)
            if not self.collection_exists(clean_name):
                return 0
            info = self.client.get_collection(clean_name)
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
        """Ensures collection exists with quantization and BM25 sparse vector configuration."""
        try:
            clean_name = sanitize_collection_name(collection_name)
            exists = self.collection_exists(clean_name)

            if exists and force_recreate:
                self.client.delete_collection(clean_name)
                exists = False

            if not exists:
                quantization = models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        always_ram=True,
                        quantile=0.99,
                    )
                )

                try:
                    self.client.create_collection(
                        collection_name=clean_name,
                        vectors_config={
                            "dense_vector": models.VectorParams(
                                size=dense_dimension,
                                distance=models.Distance.COSINE,
                                on_disk=True,
                                quantization_config=quantization,
                            )
                        },
                        sparse_vectors_config={
                            "bm25_sparse_vector": models.SparseVectorParams(
                                modifier=models.Modifier.IDF,
                            )
                        },
                        on_disk_payload=True,
                    )
                except Exception as err:
                    log.debug(f"Retrying collection creation without on_disk: {err}")
                    self.client.create_collection(
                        collection_name=clean_name,
                        vectors_config={
                            "dense_vector": models.VectorParams(
                                size=dense_dimension,
                                distance=models.Distance.COSINE,
                                quantization_config=quantization,
                            )
                        },
                        sparse_vectors_config={
                            "bm25_sparse_vector": models.SparseVectorParams(
                                modifier=models.Modifier.IDF,
                            )
                        },
                    )

                for field in ["file_path", "symbol_name", "is_code"]:
                    try:
                        self.client.create_payload_index(
                            collection_name=clean_name,
                            field_name=field,
                            field_schema=models.PayloadSchemaType.KEYWORD,
                        )
                    except Exception:
                        pass

            return clean_name
        except Exception as e:
            log.error(f"Error ensuring collection {collection_name}: {e}")
            raise

    def index_documents(
        self,
        collection_name: str,
        documents: List[AdalDocument],
        dense_dimension: int = 768,
        batch_size: int = 40,
        force_reindex: bool = False,
    ) -> int:
        """Indexes code chunks into collection with dual dense and BM25 sparse vectors."""
        try:
            clean_name = self.get_or_create_collection(
                collection_name, dense_dimension=dense_dimension, force_recreate=force_reindex
            )

            existing_count = self.get_collection_point_count(clean_name)
            if existing_count > 0 and not force_reindex:
                return existing_count

            total_upserted = 0
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                sparse_vectors = self.embed_sparse_texts([doc.text for doc in batch])

                points: List[PointStruct] = []
                for doc_idx, (doc, sparse_vec) in enumerate(zip(batch, sparse_vectors)):
                    dense_vec = getattr(doc, "vector", None)
                    if not dense_vec or len(dense_vec) != dense_dimension:
                        continue

                    meta = doc.meta_data or {}
                    unique_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_DNS,
                            f"{clean_name}_{meta.get('file_path', '')}_{i + doc_idx}",
                        )
                    )
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
                    points.append(
                        PointStruct(
                            id=unique_id,
                            vector={"dense_vector": dense_vec, "bm25_sparse_vector": sparse_vec},
                            payload=payload,
                        )
                    )

                if points:
                    self.client.upsert(collection_name=clean_name, points=points)
                    total_upserted += len(points)

            return total_upserted
        except Exception as e:
            log.error(f"Error indexing documents in {collection_name}: {e}")
            raise

    def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        query_dense_vector: List[float],
        top_k: int = 5,
        path_prefix: Optional[str] = None,
    ) -> List[AdalDocument]:
        """Executes hybrid search fusing dense vector and BM25 sparse with RRF."""
        try:
            clean_name = sanitize_collection_name(collection_name)
            if not self.collection_exists(clean_name):
                return []

            sparse_results = self.embed_sparse_texts([query_text])
            sparse_vec = sparse_results[0] if sparse_results else models.SparseVector(indices=[], values=[])

            search_params = models.SearchParams(
                quantization=models.QuantizationSearchParams(rescore=True, oversampling=2.0)
            )

            fetch_limit = top_k * 4 if path_prefix else top_k * 2
            prefetch_list = []
            if query_dense_vector:
                prefetch_list.append(
                    models.Prefetch(
                        query=query_dense_vector,
                        using="dense_vector",
                        limit=fetch_limit,
                        params=search_params,
                    )
                )
            if sparse_vec.indices:
                prefetch_list.append(
                    models.Prefetch(query=sparse_vec, using="bm25_sparse_vector", limit=fetch_limit)
                )

            if not prefetch_list:
                return []

            query_limit = top_k * 4 if path_prefix else top_k
            if len(prefetch_list) > 1:
                results = self.client.query_points(
                    collection_name=clean_name,
                    prefetch=prefetch_list,
                    query=models.RrfQuery(rrf=models.Rrf()),
                    limit=query_limit,
                    with_payload=True,
                )
            else:
                results = self.client.query_points(
                    collection_name=clean_name,
                    query=query_dense_vector if query_dense_vector else sparse_vec,
                    using="dense_vector" if query_dense_vector else "bm25_sparse_vector",
                    limit=query_limit,
                    params=search_params if query_dense_vector else None,
                    with_payload=True,
                )

            matched_points = results.points or []
            clean_prefix = path_prefix.strip().replace("\\", "/").rstrip("/") if path_prefix else ""
            if clean_prefix:
                matched_points = [
                    p
                    for p in matched_points
                    if clean_prefix in (p.payload or {}).get("file_path", "")
                ]

            documents: List[AdalDocument] = []
            for p in matched_points[:top_k]:
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
        except Exception as e:
            log.error(f"Error during hybrid search in {collection_name}: {e}")
            return []

    def get_indexed_files(self, collection_name: str, limit: int = 100) -> List[str]:
        """Retrieves distinct file paths stored within collection."""
        try:
            clean_name = sanitize_collection_name(collection_name)
            if not self.collection_exists(clean_name):
                return []
            records, _ = self.client.scroll(
                collection_name=clean_name,
                limit=limit,
                with_payload=["file_path"],
                with_vectors=False,
            )
            files = {r.payload["file_path"] for r in records if r.payload and r.payload.get("file_path")}
            return sorted(list(files))
        except Exception as e:
            log.warning(f"Error scrolling indexed files: {e}")
            return []

    def lookup_symbol(
        self,
        collection_name: str,
        symbol_name: str,
        limit: int = 5,
        path_prefix: Optional[str] = None,
    ) -> List[AdalDocument]:
        """Looks up exact code symbol via indexed Qdrant payload filters."""
        try:
            clean_name = sanitize_collection_name(collection_name)
            clean_sym = symbol_name.strip()
            if not self.collection_exists(clean_name) or not clean_sym:
                return []

            scroll_filter = models.Filter(
                must=[models.FieldCondition(key="symbol_name", match=models.MatchValue(value=clean_sym))]
            )
            records, _ = self.client.scroll(
                collection_name=clean_name,
                scroll_filter=scroll_filter,
                limit=limit * 5 if path_prefix else limit,
                with_payload=True,
                with_vectors=False,
            )

            clean_prefix = path_prefix.strip().replace("\\", "/").rstrip("/") if path_prefix else ""
            documents: List[AdalDocument] = []
            for record in records:
                payload = record.payload or {}
                f_path = payload.get("file_path", "")
                if clean_prefix and clean_prefix not in f_path:
                    continue
                meta = {
                    "file_path": f_path,
                    "type": payload.get("type", ""),
                    "is_code": payload.get("is_code", True),
                    "is_implementation": payload.get("is_implementation", True),
                    "title": payload.get("title", ""),
                    "symbol_name": payload.get("symbol_name", ""),
                    "symbol_type": payload.get("symbol_type", ""),
                    "line_range": payload.get("line_range", []),
                    "docstring": payload.get("docstring", ""),
                    "point_id": record.id,
                    "is_symbol_jump": True,
                }
                documents.append(AdalDocument(text=payload.get("text", ""), meta_data=meta))
                if len(documents) >= limit:
                    break

            return documents
        except Exception as e:
            log.warning(f"Error looking up symbol '{symbol_name}': {e}")
            return []

    def delete_points_by_file_path(
        self,
        collection_name: str,
        file_path: str,
    ) -> bool:
        """Deletes all points matching a specific file path."""
        try:
            clean_name = sanitize_collection_name(collection_name)
            clean_file = file_path.strip()
            if not self.collection_exists(clean_name) or not clean_file:
                return False

            delete_filter = models.Filter(
                must=[models.FieldCondition(key="file_path", match=models.MatchValue(value=clean_file))]
            )
            self.client.delete(
                collection_name=clean_name,
                points_selector=models.FilterSelector(filter=delete_filter),
            )
            log.info(f"Qdrant: Deleted points for file '{clean_file}' in {clean_name}")
            return True
        except Exception as e:
            log.warning(f"Error deleting points for file {file_path}: {e}")
            return False
