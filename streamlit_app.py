import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

from adalflow.core.types import Document
import streamlit as st
import os
# from src.rag import RAG
from app.rag import RAG
from dotenv import load_dotenv
from typing import List

load_dotenv(verbose=True)

def init_rag(repo_path_or_url: str):
    os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

    rag = RAG()
    print(f"Loading repository from: {repo_path_or_url}")
    rag.prepare_retriever(repo_url_or_path=repo_path_or_url)
    return rag


st.title("Github-Chat")
st.caption("Learn a repo with RAG assistant")

repo_path = st.text_input(
    "Repository Path",
    help="Github repo URL",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag" not in st.session_state:
    st.session_state.rag = None

if st.button("Initialize local RAG"):
    try:
        st.session_state.rag = init_rag(repo_path)
        if st.session_state.rag:
            st.toast("Repository loaded successfully!")
    except Exception as e:
        st.error(f"Load failed for repository at: {repo_path}")
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")

# TODO: Better reset the conversation
if st.button("Clear Chat"):
    st.session_state.messages = []
    if st.session_state.rag:
        st.session_state.rag.memory.current_conversation.dialog_turns.clear()


def display_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Show user messages as-is
            if message["role"] == "user":
                st.write(message["content"])
            else:
                # Show assistant messages with rationale if available
                if "rationale" in message and message["rationale"]:
                    with st.expander("Reasoning", expanded=False):
                        st.markdown(message["rationale"])
                st.markdown(message["content"])
                # Show context as unique file paths (not all chunks)
                if "context" in message and message["context"]:
                    # Get unique file paths
                    unique_files = []
                    seen_paths = set()
                    for doc in message["context"]:
                        file_path = doc.meta_data.get('file_path', 'unknown')
                        if file_path not in seen_paths:
                            seen_paths.add(file_path)
                            unique_files.append(file_path)
                    
                    with st.expander(f"Sources ({len(unique_files)} files)", expanded=False):
                        for file_path in unique_files:
                            st.write(f"• `{file_path}`")




def form_context(context: List[Document]):
    formatted_context = ""
    for doc in context:
        formatted_context += ""
        f"file_path: {doc.meta_data.get('file_path', 'unknown')} \n"
        f"language: {doc.meta_data.get('type', 'python')} \n"
        f"content: {doc.text} \n"
    return formatted_context


# Display all previous messages first
display_messages()

# Then handle new input
if st.session_state.rag and (
    query := st.chat_input(
        "Ask questions about this GitHub repository"
    )
):
    # Display the new user message
    with st.chat_message("user"):
        st.write(query)
    
    # Add to history
    st.session_state.messages.append({"role": "user", "content": query})

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing code..."):
            response, docs = st.session_state.rag(query)

            # Handle case when API returns an error (response is None)
            if response is None:
                error_msg = "Sorry, I couldn't process your request. This might be due to API rate limits or the request being too large. Please try again in a moment or ask a shorter question."
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
            # Show relevant context first, then the explanation
            elif docs and docs[0].documents:
                context = docs[0].documents
                
                # Get the answer content
                answer_content = (
                    response.answer
                    if hasattr(response, "answer")
                    else response.raw_response
                )
                rationale_content = (
                    response.rationale
                    if hasattr(response, "rationale")
                    else None
                )
                
                # Display immediately
                if rationale_content:
                    with st.expander("Reasoning", expanded=False):
                        st.markdown(rationale_content)
                st.markdown(answer_content)
                
                # Show sources
                unique_files = []
                seen_paths = set()
                for doc in context:
                    file_path = doc.meta_data.get('file_path', 'unknown')
                    if file_path not in seen_paths:
                        seen_paths.add(file_path)
                        unique_files.append(file_path)
                with st.expander(f"Sources ({len(unique_files)} files)", expanded=False):
                    for file_path in unique_files:
                        st.write(f"• `{file_path}`")

                # Add to chat history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "rationale": rationale_content,
                        "content": answer_content,
                        "context": context,
                    }
                )
            else:
                st.markdown(str(response))
                st.session_state.messages.append(
                    {"role": "assistant", "content": str(response)}
                )
elif not st.session_state.rag:
    st.info("Please load a repository first!")
