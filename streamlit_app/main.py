import logging
import os
import sys
import traceback
from typing import Any, Optional
import uuid
import warnings

from dotenv import load_dotenv
import streamlit as st

# Ensure project root, backend, and app dir are in sys.path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
for p in [PROJECT_ROOT, BACKEND_DIR, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.app.config.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_GROQ_MODEL
from backend.app.rag.rag import RAG
from streamlit_app.components.message_view import render_messages, render_sources
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.styles import apply_custom_styles

log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
load_dotenv(verbose=False)

QUICK_PROMPTS = [
    {"label": "Architecture", "prompt": "Provide a high-level architectural overview of this repository, including its core components and data flow."},
    {"label": "API Routes", "prompt": "What are the key API routes and endpoints exposed by this project? Explain their request and response formats."},
    {"label": "Error Handling", "prompt": "How does this codebase handle exceptions, validation, and failure scenarios?"},
    {"label": "Dependencies", "prompt": "What are the key external dependencies and libraries used in this project, and what are their purposes?"},
]


def sync_secrets() -> None:
    """Sync Streamlit Cloud secrets into os.environ for in-process library execution."""
    try:
        if hasattr(st, "secrets"):
            secret_keys = [
                "GEMINI_API_KEY",
                "GROQ_API_KEY",
                "QDRANT_URL",
                "QDRANT_API_KEY",
                "DB_USER",
                "DB_PASSWORD",
                "DB_HOST",
                "DB_PORT",
                "DB_NAME",
                "SSL_MODE",
                "DATABASE_URL",
                "GEMINI_EMBEDDING_MODEL",
                "GROQ_MODEL",
                "GEMINI_EMBEDDING_DIMENSIONS",
            ]
            for k in secret_keys:
                if k in st.secrets and not os.getenv(k):
                    os.environ[k] = str(st.secrets[k])
    except Exception as e:
        log.debug(f"Streamlit secrets sync notice: {e}")


def init_rag(repo_path_or_url: str, force_reindex: bool = False) -> RAG:
    """Validate credentials and instantiate the RAG engine directly in-process."""
    if not repo_path_or_url or not repo_path_or_url.strip():
        raise ValueError("Please enter a valid GitHub repository URL or local folder path.")

    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if not gemini_key:
        raise ValueError(
            "GEMINI_API_KEY is not configured. Please add it to your Streamlit secrets or sidebar inputs."
        )
    if not groq_key:
        raise ValueError(
            "GROQ_API_KEY is not configured. Please add it to your Streamlit secrets or sidebar inputs."
        )

    try:
        rag = RAG(entity_id="streamlit_user", process_id="streamlit_app")
        rag.prepare_retriever(repo_url_or_path=repo_path_or_url.strip(), force_reindex=force_reindex)
        return rag
    except Exception as e:
        log.error(f"Error initializing RAG in Streamlit: {e}")
        raise


def run_app():
    """Main application runner for Streamlit UI."""
    try:
        st.set_page_config(
            page_title="GithubChat • AI Repository Exploration",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass

    # Inject glassmorphic custom CSS styling
    apply_custom_styles()
    sync_secrets()

    # Session State Initialization
    if "conversations" not in st.session_state:
        initial_id = str(uuid.uuid4())
        st.session_state.conversations = [
            {"id": initial_id, "title": "New Chat", "messages": []}
        ]
        st.session_state.active_conv_id = initial_id

    if "active_conv_id" not in st.session_state:
        st.session_state.active_conv_id = st.session_state.conversations[0]["id"]

    if "messages" not in st.session_state:
        active_conv = next(
            (c for c in st.session_state.conversations if c["id"] == st.session_state.active_conv_id),
            None,
        )
        st.session_state.messages = list(active_conv["messages"]) if active_conv else []

    if "rag" not in st.session_state:
        st.session_state.rag = None
    if "current_repo" not in st.session_state:
        st.session_state.current_repo = ""
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    # Render interactive Sidebar
    force_reindex, selected_conv_id, new_chat_clicked, delete_conv_id = render_sidebar(
        st.session_state.conversations,
        st.session_state.active_conv_id,
        st.session_state.messages,
    )

    # Handle Conversation Actions
    if new_chat_clicked:
        new_id = str(uuid.uuid4())
        st.session_state.conversations.insert(0, {"id": new_id, "title": "New Chat", "messages": []})
        st.session_state.active_conv_id = new_id
        st.session_state.messages = []
        if st.session_state.rag and hasattr(st.session_state.rag, "clear_memory"):
            st.session_state.rag.clear_memory()
        st.rerun()

    if delete_conv_id:
        st.session_state.conversations = [
            c for c in st.session_state.conversations if c["id"] != delete_conv_id
        ]
        if not st.session_state.conversations:
            new_id = str(uuid.uuid4())
            st.session_state.conversations = [{"id": new_id, "title": "New Chat", "messages": []}]
            st.session_state.active_conv_id = new_id
            st.session_state.messages = []
        elif st.session_state.active_conv_id == delete_conv_id:
            st.session_state.active_conv_id = st.session_state.conversations[0]["id"]
            st.session_state.messages = list(st.session_state.conversations[0]["messages"])
        st.rerun()

    if selected_conv_id and selected_conv_id != st.session_state.active_conv_id:
        st.session_state.active_conv_id = selected_conv_id
        active_c = next((c for c in st.session_state.conversations if c["id"] == selected_conv_id), None)
        st.session_state.messages = list(active_c["messages"]) if active_c else []
        st.rerun()

    # Hero Header Container
    st.markdown(
        f"""
        <div class="hero-container">
            <h1 class="hero-title">GitHubChat</h1>
            <p class="hero-subtitle">
                Explore, query, and understand any GitHub repository with state-of-the-art semantic search and retrieval-augmented generation.
            </p>
            <div class="badge-group">
                <span class="model-badge badge-gemini">{DEFAULT_EMBEDDING_MODEL}</span>
                <span class="model-badge badge-groq">{DEFAULT_GROQ_MODEL}</span>
                <span class="model-badge badge-faiss">Qdrant Cloud Hybrid Search</span>
                <span class="model-badge badge-gemini">LangGraph Agentic RAG</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Repository Input Form & Action Controls
    with st.container():
        c_input, c_btn = st.columns([4, 1])
        with c_input:
            initial_repo_val: str = str(st.session_state.get("repo_input_value") or "")
            raw_repo = st.text_input(
                label="Repository URL or Local Folder",
                value=initial_repo_val,
                placeholder="e.g. https://github.com/fastapi/fastapi or local/folder/path",
                label_visibility="collapsed",
            )
            repo_path = str(raw_repo or "").strip()

        with c_btn:
            load_clicked = st.button("Index Repo", type="primary", use_container_width=True)

    # Repository Initialization Action
    if load_clicked:
        if not repo_path:
            st.warning("Please provide a valid GitHub repository URL or directory path.")
        else:
            with st.status("Analyzing and indexing repository...", expanded=True) as status_box:
                st.write("Cloning repository into isolated workspace...")
                try:
                    rag_instance = init_rag(repo_path, force_reindex=force_reindex)
                    st.session_state.rag = rag_instance
                    st.session_state.current_repo = repo_path
                    total_chunks = len(getattr(rag_instance, "transformed_docs", []))
                    status_box.update(
                        label=f"Repository indexed successfully ({total_chunks} code chunks ready)",
                        state="complete",
                        expanded=False,
                    )
                    st.toast("Repository indexed and ready for questions.")
                except Exception as e:
                    status_box.update(label="Indexing failed", state="error")
                    st.error(f"Error initializing repository: {e}")
                    with st.expander("Detailed Traceback"):
                        st.code(traceback.format_exc(), language="python")

    # Repository Active Stats Dashboard
    if st.session_state.rag and st.session_state.current_repo:
        chunks_count = len(getattr(st.session_state.rag, "transformed_docs", []))
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-val">{st.session_state.current_repo.split('/')[-1] or 'Active'}</div>
                    <div class="stat-label">Repository</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-val">{chunks_count}</div>
                    <div class="stat-label">Indexed Chunks</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s3:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-val">{len(st.session_state.messages)}</div>
                    <div class="stat-label">Messages</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Interactive Quick Suggestion Chips
        st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='prompt-chips-label'>Quick Questions:</div>", unsafe_allow_html=True)
        q_cols = st.columns(len(QUICK_PROMPTS))
        for idx, q_item in enumerate(QUICK_PROMPTS):
            with q_cols[idx]:
                if st.button(q_item["label"], key=f"chip_{idx}", use_container_width=True):
                    st.session_state.pending_query = q_item["prompt"]
                    st.rerun()

    st.markdown("---")

    # Render Conversation History
    render_messages(st.session_state.messages)

    # Determine Active Query
    user_query = None
    if st.session_state.pending_query:
        user_query = st.session_state.pending_query
        st.session_state.pending_query = None
    else:
        user_query = st.chat_input("Ask a question about this repository's code, structure, or implementation...")

    # Process Query Execution
    if user_query:
        if not st.session_state.rag:
            st.info("Please enter and index a GitHub repository first using the input above.")
        else:
            active_conv = next(
                (c for c in st.session_state.conversations if c["id"] == st.session_state.active_conv_id),
                None,
            )
            if active_conv and (active_conv.get("title") == "New Chat" or not active_conv.get("title")):
                active_conv["title"] = user_query[:28] + ("..." if len(user_query) > 28 else "")

            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query})
            if active_conv:
                active_conv["messages"] = list(st.session_state.messages)

            # Assistant response generation
            with st.chat_message("assistant"):
                status_box = st.status("🧠 Agent Analyzing & Thinking...", expanded=True)
                answer_placeholder = st.empty()
                accumulated_text = ""
                final_rationale = ""
                final_contexts = []
                final_steps = []

                try:
                    rag_client: Any = st.session_state.rag
                    if hasattr(rag_client, "stream_call"):
                        for event in rag_client.stream_call(user_query):
                            event_type = event.get("type", "")

                            if event_type == "step":
                                step_data = event.get("step", {})
                                title = step_data.get("title", "Processing")
                                detail = step_data.get("detail", "")
                                status = step_data.get("status", "running")
                                if status == "running":
                                    status_box.write(f"⚡ **{title}** — *{detail}*")
                                elif status == "completed":
                                    status_box.write(f"✅ **{title}** — {detail}")

                            elif event_type == "status":
                                msg = event.get("message", "Processing...")
                                status_box.update(label=f"🧠 Agent: {msg}", state="running")

                            elif event_type == "token":
                                token = event.get("token", "")
                                accumulated_text += token
                                answer_placeholder.markdown(accumulated_text + " ▌")

                            elif event_type == "done":
                                accumulated_text = event.get("answer", accumulated_text)
                                final_rationale = event.get("rationale", "")
                                final_contexts = event.get("contexts", [])
                                final_steps = event.get("steps", [])
                                total_steps_count = len(final_steps) if final_steps else 4
                                status_box.update(
                                    label=f"Completed workflow in {total_steps_count} steps",
                                    state="complete",
                                    expanded=False,
                                )

                            elif event_type == "error":
                                err = event.get("error", "An error occurred.")
                                status_box.update(label=f"Execution error: {err}", state="error")
                                st.error(err)

                        answer_placeholder.markdown(accumulated_text)
                    else:
                        response, docs = rag_client(user_query)
                        accumulated_text = getattr(response, "answer", "") or str(response)
                        final_rationale = getattr(response, "rationale", "")
                        final_contexts = docs[0].documents if docs and hasattr(docs[0], "documents") else []
                        answer_placeholder.markdown(accumulated_text)
                        status_box.update(label="Response generated", state="complete", expanded=False)

                    # Render sources if contexts available
                    if final_contexts:
                        render_sources(final_contexts)

                    # Append assistant message to session state
                    msg_data = {
                        "role": "assistant",
                        "content": accumulated_text,
                        "rationale": final_rationale,
                        "context": final_contexts,
                        "steps": final_steps,
                    }
                    st.session_state.messages.append(msg_data)
                    if active_conv:
                        active_conv["messages"] = list(st.session_state.messages)

                except Exception as err:
                    err_text = f"An error occurred while answering: {err}"
                    status_box.update(label="Query execution failed", state="error")
                    st.error(err_text)
                    st.session_state.messages.append({"role": "assistant", "content": err_text})
                    if active_conv:
                        active_conv["messages"] = list(st.session_state.messages)


if __name__ == "__main__":
    run_app()
