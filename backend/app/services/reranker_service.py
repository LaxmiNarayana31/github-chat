"""Fast local Cross-Encoder Reranker service using FlashRank for low-latency code retrieval scoring."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from adalflow.core.types import Document as AdalDocument
from flashrank import Ranker, RerankRequest

log = logging.getLogger(__name__)


class RerankerService:
    """Provides high-throughput, low-latency cross-encoder reranking for code search candidates."""

    _instance: Optional['RerankerService'] = None

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        self.model_name = model_name
        self._ranker: Optional[Ranker] = None
        self._available: bool = False
        self._init_ranker()

    @classmethod
    def get_instance(cls) -> 'RerankerService':
        """Singleton accessor for RerankerService."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_ranker(self) -> None:
        """Initialize FlashRank cross-encoder in ONNX runtime."""
        try:
            self._ranker = Ranker(model_name=self.model_name)
            self._available = True
            log.info(f"RerankerService initialized with model '{self.model_name}'")
        except Exception as e:
            log.warning(f"RerankerService: FlashRank initialization failed, falling back to heuristic scoring: {e}")
            self._ranker = None
            self._available = False

    @property
    def is_available(self) -> bool:
        """Check if local cross-encoder model is ready."""
        return self._available and self._ranker is not None

    def rerank(
        self,
        query: str,
        documents: List[AdalDocument],
        top_k: Optional[int] = None,
    ) -> List[Tuple[AdalDocument, float]]:
        """Rerank candidate documents against the query using cross-encoder scores.

        Returns:
            List of tuples (doc, score) sorted in descending order of relevance.
        """
        if not documents:
            return []

        ranker = self._ranker
        if not self.is_available or ranker is None:
            # Fallback heuristic: uniform default score with existing order preserved
            return [(doc, 0.5) for doc in (documents[:top_k] if top_k else documents)]

        try:
            passages: List[Dict[str, Any]] = []
            doc_map: Dict[int, AdalDocument] = {}

            for idx, doc in enumerate(documents):
                doc_map[idx] = doc
                passages.append({
                    "id": idx,
                    "text": (doc.text or "")[:1200],
                })

            req = RerankRequest(query=query, passages=passages)
            ranked_results = ranker.rerank(req)

            scored_docs: List[Tuple[AdalDocument, float]] = []
            for item in ranked_results:
                doc_id = item.get("id")
                score = float(item.get("score", 0.0))
                if doc_id in doc_map:
                    scored_docs.append((doc_map[doc_id], score))

            if top_k is not None:
                scored_docs = scored_docs[:top_k]

            return scored_docs
        except Exception as e:
            log.warning(f"RerankerService error during rerank: {e}")
            return [(doc, 0.5) for doc in (documents[:top_k] if top_k else documents)]
