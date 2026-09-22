import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, cast
from uuid import uuid4

import adalflow as adal
from adalflow.core.types import (
    AssistantResponse,
    Conversation,
    DialogTurn,
    ModelType,
    UserQuery,
)
from adalflow.core.types import Document as AdalDocument
from adalflow.utils import printc

from backend.app.rag.agentic_rag import LangGraphAgenticRAG
from backend.app.config.config import config
from backend.app.embeddings.qdrant_manager import QdrantManager
from backend.app.pipelines.data_pipeline import DatabaseManager
from backend.app.prompts.system_prompt import RAG_TEMPLATE, SYSTEM_PROMPT
from backend.app.services.memory_manager import MemoriManager
from backend.app.utils.repo_utils import (
    get_repo_slug,
    normalize_repo_url,
    sanitize_collection_slug,
)

log = logging.getLogger(__name__)


# Memory component
class Memory(adal.DataComponent):
    def __init__(self, memori_manager: Optional[MemoriManager] = None):
        super().__init__()
        self.current_conversation = Conversation()
        self.memori_manager = memori_manager

    def call(self):
        return self.current_conversation.dialog_turns

    def add_dialog_turn(self, uq: str, ar: str):
        self.current_conversation.append_dialog_turn(DialogTurn(
            id=str(uuid4()),
            user_query=UserQuery(query_str=uq),
            assistant_response=AssistantResponse(response_str=ar)
        ))
        if self.memori_manager:
            self.memori_manager.capture_dialog_turn(user_query=uq, assistant_response=ar)

    def get_history_turns(self) -> List[Dict[str, str]]:
        """Return conversation dialog turns as a list of dicts [{'user': ..., 'assistant': ...}]."""
        turns = []
        dialog_items = (
            self.current_conversation.dialog_turns.values()
            if isinstance(self.current_conversation.dialog_turns, dict)
            else self.current_conversation.dialog_turns
        )
        for dt in dialog_items:
            uq = getattr(getattr(dt, "user_query", None), "query_str", "") or ""
            ar = getattr(getattr(dt, "assistant_response", None), "response_str", "") or ""
            if uq or ar:
                turns.append({"user": uq, "assistant": ar})
        return turns


    def load_from_memori(self, session_id: Optional[str] = None):
        """Restore conversation dialog turns from Memori SQL storage."""
        if not self.memori_manager:
            return
        turns = self.memori_manager.get_conversation_turns(session_id=session_id)
        self.current_conversation.dialog_turns.clear()
        for uq, ar in turns:
            self.current_conversation.append_dialog_turn(DialogTurn(
                id=str(uuid4()),
                user_query=UserQuery(query_str=uq),
                assistant_response=AssistantResponse(response_str=ar)
            ))
        if turns:
            printc(f"Memory: Restored {len(turns)} dialog turns from Memori Labs database.", color="green")



@dataclass
class RAGAnswer(adal.DataClass):
    rationale: str = field(default="", metadata={"desc": "Rationale."})
    answer: str = field(default="", metadata={"desc": "Answer."})
    __output_fields__ = ["rationale", "answer"]


@dataclass
class HybridRetrieverOutput:
    """Retriever output envelope compatible with AdalFlow and RAGService."""
    documents: List[AdalDocument] = field(default_factory=list)


# RAG component
class RAG(adal.Component):
    def __init__(self, entity_id: Optional[str] = None, process_id: Optional[str] = None):
        super().__init__()
        self.memori_manager = MemoriManager(entity_id=entity_id, process_id=process_id)
        self.memory = Memory(memori_manager=self.memori_manager)
        
        # Google Gemini Embedder (defaults to gemini-embedding-2)
        self.embedder = adal.Embedder(
            model_client=config["embedder"]["model_client"](),
            model_kwargs=config["embedder"]["model_kwargs"],
        )
        
        # Primary persistent vector database: Qdrant
        self.qdrant_manager = QdrantManager()
        self.db_manager = DatabaseManager()
        self.transformed_docs: List[AdalDocument] = []
        self.current_collection: Optional[str] = None
        self.retriever: Optional[QdrantManager] = None
        self.repo_url: str = ""

        data_parser = adal.DataClassParser(data_class=cast(Any, RAGAnswer), return_data_class=True)
        
        # Create generator model client (Groq / Gemini)
        model_client = config["generator"]["model_client"]()
        
        # Register model client with Memori for tracking
        self.memori_manager.register_llm(model_client)
        
        self._configure_message_parser(model_client)
        
        self.generator = adal.Generator(
            template=RAG_TEMPLATE,
            prompt_kwargs={
                "output_format_str": data_parser.get_output_format_str(),
                "conversation_history": self.memory(),
                "system_prompt": SYSTEM_PROMPT,
                "contexts": None,
            },
            model_client=model_client,
            model_kwargs=config["generator"]["model_kwargs"],
            output_processors=data_parser,
        )

        # Restore history from Memori if available
        self.memory.load_from_memori()

        # Autonomous LangGraph Agentic RAG engine
        self.agentic_rag = LangGraphAgenticRAG(
            embedder=self.embedder,
            qdrant_manager=self.qdrant_manager,
            memori_manager=self.memori_manager,
            llm_client=model_client,
            memory_component=self.memory,
        )
    
    # Configure the model client to parse template tags into chat messages
    def _configure_message_parser(self, model_client):
        """Configure the model client to parse template tags into chat messages."""
        original_convert = model_client.convert_inputs_to_api_kwargs
        
        def patched_convert(input=None, model_kwargs={}, model_type=ModelType.UNDEFINED):
            final_model_kwargs = model_kwargs.copy()
            if model_type == ModelType.LLM and input:
                messages = []
                
                # Extract system message from <SYS>...</SYS>
                sys_match = re.search(r'<SYS>(.*?)</SYS>', input, re.DOTALL)
                if sys_match:
                    system_content = sys_match.group(1).strip()
                    messages.append({"role": "system", "content": system_content})
                
                # Extract middle content (context, history) between </SYS> and <USER>
                middle_match = re.search(r'</SYS>(.*?)<USER>', input, re.DOTALL)
                if middle_match:
                    middle_content = middle_match.group(1).strip()
                    if middle_content and messages:
                        messages[0]["content"] += "\n\n" + middle_content
                    elif middle_content:
                        messages.append({"role": "system", "content": middle_content})
                
                # Extract user message from <USER>...</USER> - MUST be last
                user_match = re.search(r'<USER>(.*?)</USER>', input, re.DOTALL)
                if user_match:
                    user_content = user_match.group(1).strip()
                    messages.append({"role": "user", "content": user_content})
                else:
                    # Fallback: add the whole input as user message
                    messages.append({"role": "user", "content": input})
                
                final_model_kwargs["messages"] = messages
                return final_model_kwargs
            else:
                return original_convert(input, model_kwargs, model_type)
        
        model_client.convert_inputs_to_api_kwargs = patched_convert

    def prepare_retriever(self, repo_url_or_path: str, force_reindex: bool = False, token: Optional[str] = None):
        """Index repository into Qdrant hybrid collection with deduplication and caching."""
        normalized_url = normalize_repo_url(repo_url_or_path)
        slug = get_repo_slug(normalized_url)
        collection_name = sanitize_collection_slug(slug)
        self.current_collection = collection_name
        self.repo_url = normalized_url
        expected_dim = config["embedder"].get("dimensions", 768)

        # Check if collection already exists in Qdrant with points
        if not force_reindex and self.qdrant_manager.collection_exists(collection_name):
            point_count = self.qdrant_manager.get_collection_point_count(collection_name)
            if point_count > 0:
                printc(
                    f"RAG: Found existing Qdrant collection '{collection_name}' with {point_count} persistent points. "
                    "Skipping duplicate parsing.",
                    color="green",
                )
                self.retriever = self.qdrant_manager
                return

        # Prepare and embed repository documents using data pipeline
        printc(f"RAG: Preparing repository code documents for '{normalized_url}' (slug: {slug})...", color="blue")
        self.transformed_docs = self.db_manager.prepare_database(
            normalized_url, force_reindex=force_reindex, token=token
        )


        valid_docs = [
            doc for doc in (self.transformed_docs or [])
            if hasattr(doc, "vector") and doc.vector and len(doc.vector) == expected_dim
        ]

        if not valid_docs:
            raise RuntimeError(
                f"Cannot build retriever: No documents have valid embeddings of dimension {expected_dim}. "
                "Please verify your GEMINI_API_KEY and repository files."
            )

        self.transformed_docs = valid_docs
        printc(
            f"RAG: Ingesting {len(self.transformed_docs)} documents into Qdrant hybrid collection '{collection_name}'...",
            color="blue",
        )

        self.qdrant_manager.index_documents(
            collection_name=collection_name,
            documents=self.transformed_docs,
            dense_dimension=expected_dim,
            force_reindex=force_reindex,
        )

        self.retriever = self.qdrant_manager
        printc(
            f"RAG: Qdrant Hybrid Retriever initialized successfully with collection '{collection_name}'.",
            color="green",
        )

    def stream_call(self, query: str) -> Iterator[Dict[str, Any]]:
        """Execute stream query via LangGraph Agentic RAG engine."""
        printc(f"Agentic RAG (Streaming): Processing query: '{query}'", color="green")
        if self.retriever is None or not self.current_collection:
            raise RuntimeError("Retriever has not been initialized. Please load a repository first.")

        repo_summary = None
        if self.transformed_docs:
            file_paths = [
                d.meta_data.get("file_path", "")
                for d in self.transformed_docs[:50]
                if hasattr(d, "meta_data") and d.meta_data
            ]
            repo_summary = "\n".join(file_paths[:30])
        elif self.current_collection:
            indexed_files = self.qdrant_manager.get_indexed_files(self.current_collection, limit=60)
            if indexed_files:
                repo_summary = "\n".join(indexed_files[:40])

        history_turns = self.memory.get_history_turns() if hasattr(self, "memory") else []
        yield from self.agentic_rag.stream_invoke(
            query=query,
            repo_url=getattr(self, "repo_url", ""),
            collection_name=self.current_collection,
            transformed_docs_summary=repo_summary,
            conversation_history=history_turns,
        )

    def call(self, query: str) -> Any:
        printc(f"Agentic RAG: Processing query: '{query}'", color="green")
        if self.retriever is None or not self.current_collection:
            raise RuntimeError("Retriever has not been initialized. Please load a repository first.")

        # Autonomous execution via LangGraph Agentic RAG:
        # Route Query -> Qdrant Hybrid Retrieve -> CRAG Grade Docs -> Query Rewrite Loop -> Grounded Generate -> Save Memory
        if hasattr(self, "agentic_rag") and self.agentic_rag:
            try:
                repo_summary = None
                if self.transformed_docs:
                    file_paths = [
                        d.meta_data.get("file_path", "")
                        for d in self.transformed_docs[:50]
                        if hasattr(d, "meta_data") and d.meta_data
                    ]
                    repo_summary = "\n".join(file_paths[:30])
                elif self.current_collection:
                    indexed_files = self.qdrant_manager.get_indexed_files(self.current_collection, limit=60)
                    if indexed_files:
                        repo_summary = "\n".join(indexed_files[:40])

                history_turns = self.memory.get_history_turns() if hasattr(self, "memory") else []
                final_state = self.agentic_rag.invoke(
                    query=query,
                    repo_url=getattr(self, "repo_url", ""),
                    collection_name=self.current_collection,
                    transformed_docs_summary=repo_summary,
                    conversation_history=history_turns,
                )


                retrieved_docs = final_state.get("retrieved_documents") or []
                ans = final_state.get("answer") or ""
                rat = final_state.get("rationale") or ""
                
                final = RAGAnswer(rationale=rat, answer=ans)
                retriever_output = [HybridRetrieverOutput(documents=retrieved_docs)]
                return final, retriever_output
            except Exception as e:
                log.warning(f"LangGraph Agentic RAG execution failed ({e}), falling back to classic pipeline.")

        expected_dim = config["embedder"].get("dimensions", 768)

        # Embed query using GeminiEmbedderClient
        embed_output = self.embedder(query, model_kwargs={"task_type": "RETRIEVAL_QUERY"})
        if not embed_output or not embed_output.data:
            return RAGAnswer(rationale="", answer="Failed to generate query embedding. Please check your GEMINI_API_KEY."), []

        vectors = [emb.embedding for emb in embed_output.data if hasattr(emb, "embedding") and emb.embedding]
        if not vectors or len(vectors[0]) != expected_dim:
            return RAGAnswer(rationale="", answer=f"Embedding error: Expected {expected_dim} dimensions for query."), []

        query_dense = vectors[0]
        top_k = config.get("retriever", {}).get("top_k", 4)

        # Execute Qdrant Hybrid Search (Dense vector + BM25 sparse vector fused with RRF)
        retrieved_docs = self.qdrant_manager.hybrid_search(
            collection_name=self.current_collection,
            query_text=query,
            query_dense_vector=query_dense,
            top_k=top_k,
        )

        if not retrieved_docs:
            return RAGAnswer(rationale="", answer="No relevant code context found in repository."), []

        printc(f"RAG: Qdrant hybrid search retrieved {len(retrieved_docs)} relevant code documents", color="green")

        # Retrieve semantic memory from Memori Labs (Postgres/SQLite)
        semantic_memories = self.memori_manager.recall_memories(query=query, limit=3)
        context_list = list(retrieved_docs)
        if semantic_memories:
            for mem_text in semantic_memories:
                context_list.append(
                    AdalDocument(
                        text=f"[Known Fact / Semantic Memory]: {mem_text}",
                        meta_data={"file_path": "memory://semantic_facts", "is_code": False},
                    )
                )
            printc(f"RAG: Integrated {len(semantic_memories)} semantic memories into context.", color="green")

        # Generate Answer with LLM
        prompt_kwargs = {
            "input_str": query,
            "contexts": context_list,
            "conversation_history": self.memory(),
        }

        response = self.generator(prompt_kwargs=prompt_kwargs)
        printc(f"Raw response: {response.raw_response}", color="yellow")
        printc(f"Parsed data: {response.data}", color="yellow")
        if response.error:
            printc(f"Generator Error: {response.error}", color="red")

        final = response.data

        # Fallback: if parsing failed, extract from raw JSON
        raw_str = str(response.raw_response or "")
        if final and hasattr(final, "rationale") and hasattr(final, "answer"):
            if not getattr(final, "rationale", None) and not getattr(final, "answer", None) and raw_str:
                try:
                    raw = raw_str
                    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
                    if json_match:
                        raw = json_match.group(1)
                    parsed = json.loads(raw)
                    final = RAGAnswer(
                        rationale=str(parsed.get("rationale", "")),
                        answer=str(parsed.get("answer", "")),
                    )
                except (json.JSONDecodeError, AttributeError, ValueError) as e:
                    log.debug(f"RAG: Could not parse response as structured JSON: {e}")
                    final = RAGAnswer(rationale="", answer=raw_str)
        elif not final and raw_str:
            try:
                raw = raw_str
                json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
                if json_match:
                    raw = json_match.group(1)
                parsed = json.loads(raw)
                final = RAGAnswer(
                    rationale=str(parsed.get("rationale", "")),
                    answer=str(parsed.get("answer", "")),
                )
            except (json.JSONDecodeError, AttributeError, ValueError) as e:
                log.debug(f"RAG: Could not parse raw response as fallback JSON: {e}")
                final = RAGAnswer(rationale="", answer=raw_str)
        elif not final:
            final = RAGAnswer(
                rationale="",
                answer="No answer could be generated. Please check your GROQ_API_KEY and model configuration.",
            )

        # Record dialog turn in both conversation memory and Memori Labs SQL storage
        answer_text = str(getattr(final, "answer", None) or final)
        self.memory.add_dialog_turn(uq=query, ar=answer_text)

        # Return final answer and wrapped retriever output
        retriever_output = [HybridRetrieverOutput(documents=retrieved_docs)]
        return final, retriever_output
