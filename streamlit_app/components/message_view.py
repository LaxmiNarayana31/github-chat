import logging
from typing import Any, Dict, List, Optional

import streamlit as st

log = logging.getLogger(__name__)

LANGUAGE_MAP = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "jsx": "javascript",
    "md": "markdown",
    "json": "json",
    "yml": "yaml",
    "yaml": "yaml",
    "go": "go",
    "rs": "rust",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "h": "c",
    "hpp": "cpp",
    "sql": "sql",
    "sh": "bash",
}


def render_thought_steps(steps: List[Dict[str, Any]]) -> None:
    """Render the step-by-step thinking and execution trace of the Agentic RAG workflow."""
    try:
        if not steps:
            return
        with st.expander(f"🧠 Agent Reasoning & Execution Steps ({len(steps)} steps)", expanded=False):
            for idx, step in enumerate(steps):
                title = step.get("title", f"Step {idx + 1}")
                detail = step.get("detail", "")
                status = step.get("status", "completed")
                icon = "✅" if status == "completed" else "⚡" if status == "running" else "❌"

                st.markdown(
                    f"""
                    <div style="margin-bottom: 8px; padding: 8px 12px; background: rgba(30, 41, 59, 0.45); border-left: 3px solid #3b82f6; border-radius: 4px;">
                        <div style="font-weight: 600; font-size: 0.86rem; color: #93c5fd;">
                            {icon} {title}
                        </div>
                        {f'<div style="font-size: 0.78rem; color: #cbd5e1; margin-top: 3px; font-family: monospace;">{detail}</div>' if detail else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    except Exception as e:
        log.warning(f"Error rendering thought steps: {e}")


def render_messages(messages: List[Dict[str, Any]]) -> None:
    """Render the full conversation history with avatars and rich formatting."""
    try:
        for message in messages:
            role = message.get("role", "user")

            with st.chat_message(role):
                if role == "user":
                    st.markdown(message.get("content", ""))
                else:
                    # Step-by-step Execution Steps
                    if message.get("steps"):
                        render_thought_steps(message["steps"])

                    # Reasoning / Rationale block
                    if message.get("rationale"):
                        with st.expander("Analysis & Summary", expanded=False):
                            st.markdown(
                                f"""
                                <div style="background: rgba(99, 102, 241, 0.08); border-left: 3px solid #6366f1; border-radius: 4px; padding: 10px 14px; margin-bottom: 8px;">
                                    {message['rationale']}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    # Main answer content
                    st.markdown(message.get("content", ""))

                    # Source citations
                    context_docs = message.get("context")
                    if context_docs:
                        render_sources(context_docs)
    except Exception as e:
        log.error(f"Error rendering messages: {e}")


def render_sources(docs: List[Any]) -> None:
    """Extract and render deduplicated file paths with code snippet previews."""
    try:
        if not docs:
            return

        unique_docs = []
        seen_paths = set()
        for doc in docs:
            meta = getattr(doc, "meta_data", None) or {}
            if isinstance(meta, dict):
                file_path = meta.get("file_path", "unknown")
                file_type = meta.get("type", "txt")
            else:
                file_path = getattr(meta, "file_path", "unknown")
                file_type = getattr(meta, "type", "txt")

            if file_path not in seen_paths:
                seen_paths.add(file_path)
                unique_docs.append({"path": file_path, "type": file_type, "text": getattr(doc, "text", "")})

        if unique_docs:
            with st.expander(f"Referenced Sources ({len(unique_docs)} files)", expanded=False):
                for item in unique_docs:
                    fp = item["path"]
                    ext = item["type"]
                    lang = LANGUAGE_MAP.get(str(ext).lower(), "")

                    st.markdown(
                        f"""
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 8px; margin-bottom: 4px;">
                            <span style="font-weight: 600; font-size: 0.85rem; color: #93c5fd; font-family: monospace;">{fp}</span>
                            <span style="font-size: 0.72rem; padding: 2px 8px; border-radius: 4px; background: rgba(99, 102, 241, 0.2); color: #c7d2fe; text-transform: uppercase;">{ext}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if item.get("text"):
                        st.code(item["text"][:600] + ("..." if len(item["text"]) > 600 else ""), language=lang or "text")
    except Exception as e:
        log.warning(f"Error rendering sources: {e}")


def render_assistant_response(
    response: Any,
    docs: Optional[List[Any]] = None,
    steps: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Format and render a newly generated assistant response."""
    try:
        answer_content = (
            response.answer
            if hasattr(response, "answer") and response.answer
            else getattr(response, "raw_response", str(response))
        )
        rationale_content = getattr(response, "rationale", None)

        if steps:
            render_thought_steps(steps)

        if rationale_content:
            with st.expander("Analysis & Summary", expanded=False):
                st.markdown(
                    f"""
                    <div style="background: rgba(99, 102, 241, 0.08); border-left: 3px solid #6366f1; border-radius: 4px; padding: 10px 14px; margin-bottom: 8px;">
                        {rationale_content}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(answer_content)

        context_docs = []
        if docs and hasattr(docs[0], "documents"):
            context_docs = docs[0].documents
            render_sources(context_docs)

        return {
            "role": "assistant",
            "rationale": rationale_content,
            "content": answer_content,
            "context": context_docs,
            "steps": steps or [],
        }
    except Exception as e:
        log.error(f"Error rendering assistant response: {e}")
        return {
            "role": "assistant",
            "rationale": "",
            "content": str(response),
            "context": [],
            "steps": steps or [],
        }
