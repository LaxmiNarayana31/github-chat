import logging
import os
from typing import Any, Dict, List, Tuple

import streamlit as st

log = logging.getLogger(__name__)


def render_sidebar(
    conversations: List[Dict[str, Any]],
    active_conv_id: str,
    messages: List[Dict[str, Any]],
) -> Tuple[bool, str, bool, str]:
    """Render a clean, modern Chat History and configuration sidebar.
    
    Args:
        conversations: List of all active conversation metadata dicts.
        active_conv_id: ID of the currently selected conversation.
        messages: Active conversation message history for chat export.
        
    Returns:
        Tuple: (force_reindex, selected_conv_id, new_chat_clicked, delete_conv_id)
    """
    new_chat_clicked = False
    selected_conv_id = active_conv_id
    delete_conv_id = ""
    force_reindex = False

    with st.sidebar:
        # Header Branding
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
                <div style="width: 4px; height: 26px; background: #6366f1; border-radius: 2px;"></div>
                <div>
                    <h2 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #ffffff; letter-spacing: -0.02em;">GithubChat</h2>
                    <span style="font-size: 0.72rem; color: #94a3b8;">Repository Intelligence & Chat</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # New Chat Action Button
        if st.button("➕  New Chat", key="sidebar_new_chat_btn", use_container_width=True, type="primary"):
            new_chat_clicked = True

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; margin-bottom: 8px;'>Chat History</div>",
            unsafe_allow_html=True,
        )

        # Conversation History Thread List
        if not conversations:
            st.markdown(
                "<div style='font-size: 0.82rem; color: #64748b; padding: 10px 4px;'>No previous chats yet. Start a conversation to view history.</div>",
                unsafe_allow_html=True,
            )
        else:
            for conv in conversations:
                c_id = conv.get("id", "")
                c_title = conv.get("title") or "New Conversation"

                col_item, col_del = st.columns([5, 1])
                with col_item:
                    btn_label = f"💬  {c_title[:24]}..." if len(c_title) > 24 else f"💬  {c_title}"
                    if st.button(
                        btn_label,
                        key=f"conv_select_{c_id}",
                        use_container_width=True,
                        help=f"Switch to: {c_title}",
                    ):
                        selected_conv_id = c_id

                with col_del:
                    if st.button("🗑️", key=f"conv_del_{c_id}", help="Delete chat"):
                        delete_conv_id = c_id

        st.markdown("---")

        # Model Connection Diagnostics
        st.markdown(
            "<div style='font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; margin-bottom: 8px;'>Model Providers</div>",
            unsafe_allow_html=True,
        )
        gemini_present = bool(os.getenv("GEMINI_API_KEY"))
        groq_present = bool(os.getenv("GROQ_API_KEY"))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""
                <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; padding: 6px; text-align: center;">
                    <div style="font-size: 0.68rem; color: #7dd3fc; font-weight: 600;">GEMINI</div>
                    <div style="font-size: 0.8rem; font-weight: 700; color: {'#34d399' if gemini_present else '#f87171'};">
                        {'Connected' if gemini_present else 'Missing'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div style="background: rgba(249, 115, 22, 0.08); border: 1px solid rgba(249, 115, 22, 0.2); border-radius: 8px; padding: 6px; text-align: center;">
                    <div style="font-size: 0.68rem; color: #fdba74; font-weight: 600;">GROQ</div>
                    <div style="font-size: 0.8rem; font-weight: 700; color: {'#34d399' if groq_present else '#f87171'};">
                        {'Connected' if groq_present else 'Missing'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("API Keys Config", expanded=(not gemini_present or not groq_present)):
            user_gemini = st.text_input("Gemini API Key", type="password", key="sidebar_gemini_input")
            user_groq = st.text_input("Groq API Key", type="password", key="sidebar_groq_input")
            if user_gemini:
                os.environ["GEMINI_API_KEY"] = user_gemini
            if user_groq:
                os.environ["GROQ_API_KEY"] = user_groq
            st.caption("Keys entered here are kept in memory for your current session.")

        st.markdown("---")

        # Indexing and Cache Settings
        with st.expander("Index & Cache Settings", expanded=False):
            force_reindex = st.checkbox(
                "Force Re-index (Purge Cache)",
                value=False,
                help="Purges Qdrant vector embeddings and forces a fresh clone and re-index.",
            )

        # Export Conversation Transcript
        if messages:
            st.markdown("---")
            st.markdown("#### Export Conversation")
            md_lines = ["# GithubChat Conversation Transcript\n\n"]
            for m in messages:
                role = "User" if m.get("role") == "user" else "Assistant"
                md_lines.append(f"### {role}\n\n")
                if m.get("rationale"):
                    md_lines.append(f"> **Reasoning:** {m['rationale']}\n\n")
                md_lines.append(f"{m.get('content', '')}\n\n")
                if m.get("context"):
                    md_lines.append("**Sources Referenced:**\n")
                    for doc in m["context"]:
                        meta = getattr(doc, "meta_data", {})
                        fp = meta.get("file_path", "unknown") if isinstance(meta, dict) else getattr(meta, "file_path", "unknown")
                        md_lines.append(f"- `{fp}`\n")
                    md_lines.append("\n")
                md_lines.append("---\n\n")

            st.download_button(
                label="📥 Export Chat (Markdown)",
                data="".join(md_lines),
                file_name="github_chat_transcript.md",
                mime="text/markdown",
                use_container_width=True,
            )

        st.markdown(
            """
            <div style="margin-top: 24px; text-align: center; font-size: 0.7rem; color: #475569;">
                <b>Agentic RAG</b> • Memory • CRAG • Self-RAG
            </div>
            """,
            unsafe_allow_html=True,
        )

        return force_reindex, selected_conv_id, new_chat_clicked, delete_conv_id
