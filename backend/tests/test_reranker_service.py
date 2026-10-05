"""Unit tests for RerankerService cross-encoder ranking."""

import pytest
from adalflow.core.types import Document as AdalDocument

from backend.app.services.reranker_service import RerankerService


def test_reranker_service_initialization():
    """Verify RerankerService initializes and reports availability."""
    reranker = RerankerService.get_instance()
    assert reranker is not None
    assert reranker.is_available is True


def test_reranker_service_ranks_relevant_code_highest():
    """Verify Cross-Encoder assigns higher relevance scores to matching code passages."""
    reranker = RerankerService.get_instance()

    docs = [
        AdalDocument(text="def compute_square_root(x):\n    import math\n    return math.sqrt(x)"),
        AdalDocument(text="The weather in Paris is pleasant during autumn with occasional rain."),
        AdalDocument(text="class UserDatabase:\n    def connect(self): pass"),
    ]

    query = "function to calculate square root in math"
    ranked_results = reranker.rerank(query=query, documents=docs)

    assert len(ranked_results) == 3
    # Top document should be compute_square_root
    top_doc, top_score = ranked_results[0]
    assert "compute_square_root" in top_doc.text
    # Unrelated document should have lower score than matching doc
    unrelated_score = next(score for d, score in ranked_results if "weather" in d.text)
    assert top_score > unrelated_score
