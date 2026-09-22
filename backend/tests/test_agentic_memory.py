from unittest.mock import MagicMock
import pytest

from backend.app.rag.agentic_rag import LangGraphAgenticRAG
from backend.app.rag.rag import Memory


def test_memory_component_turns():
    """Verify dialog turns are stored and converted into dictionary lists for agentic context."""
    memory = Memory(memori_manager=None)
    memory.add_dialog_turn(uq="What does LangGraphAgenticRAG do?", ar="It orchestrates multi-step agentic RAG.")
    memory.add_dialog_turn(uq="How does it handle grading?", ar="It uses CRAG document grading.")

    history = memory.get_history_turns()
    assert len(history) == 2
    assert history[0]["user"] == "What does LangGraphAgenticRAG do?"
    assert "orchestrates" in history[0]["assistant"]
    assert history[1]["user"] == "How does it handle grading?"


def test_contextualize_query_with_pronoun():
    """Verify that ambiguous follow-up questions referencing previous turns are rewritten with explicit topics."""
    mock_llm = MagicMock()
    mock_llm.call.return_value = "How does LangGraphAgenticRAG handle grading?"

    agent = LangGraphAgenticRAG(
        embedder=MagicMock(),
        qdrant_manager=MagicMock(),
        memori_manager=MagicMock(),
        llm_client=mock_llm,
    )

    history = [
        {"user": "What is LangGraphAgenticRAG?", "assistant": "LangGraphAgenticRAG is the core autonomous router."}
    ]

    rewritten = agent._contextualize_query("How does it handle grading?", history)
    assert "LangGraphAgenticRAG" in rewritten


def test_contextualize_query_leaves_independent_query_unchanged():
    """Verify that independent self-contained queries are not unnecessarily altered."""
    agent = LangGraphAgenticRAG(
        embedder=MagicMock(),
        qdrant_manager=MagicMock(),
        memori_manager=MagicMock(),
        llm_client=MagicMock(),
    )

    history = [
        {"user": "What is react?", "assistant": "React is a JavaScript library."}
    ]

    # No pronouns or follow-up markers
    independent_query = "List all files in the backend directory"
    result = agent._contextualize_query(independent_query, history)
    assert result == independent_query

