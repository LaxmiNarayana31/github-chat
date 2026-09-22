"""LangGraph Agentic RAG System for GithubChat.

Implements an autonomous, multi-step Agentic RAG workflow using LangGraph:
1. Autonomous Query Router: Routes queries to Direct Chat, Repo Overview, or Code Search.
2. Qdrant Hybrid Search: Combines Dense Google Gemini Embeddings + BM25 Sparse Vectors with RRF.
3. Corrective RAG (CRAG) Document Grader: Assesses chunk relevance before generation.
4. Self-Correction Query Rewriter Loop: Reformulates search terms if retrieved docs are insufficient.
5. Grounded Agentic Generator: Multi-provider LLM fallback (Gemma 4, GPT-OSS, Gemini 3.7/3.5) with rationale and citation.
6. Self-RAG Hallucination Check: Assesses factual grounding.
7. Memori Labs SQL Memory: Commits conversational & semantic knowledge to PostgreSQL / SQLite.
"""

import json
import logging
import re
from typing import Any, Dict, Iterator, List, Literal, Optional, TypedDict, cast

from adalflow.core.types import Document as AdalDocument
from langgraph.graph import END, START, StateGraph

from backend.app.config.config import config
from backend.app.embeddings.qdrant_manager import QdrantManager
from backend.app.prompts.system_prompt import SYSTEM_PROMPT
from backend.app.services.memory_manager import MemoriManager
from backend.app.services.redis_manager import RedisCacheManager

log = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Execution state passed between nodes in the LangGraph Agentic RAG."""
    query: str
    effective_query: str
    repo_url: str
    collection_name: str
    route: Literal["code_search", "direct_chat", "repo_overview"]
    retrieved_documents: List[AdalDocument]
    semantic_memories: List[str]
    doc_grade: Literal["relevant", "not_relevant"]
    retry_count: int
    max_retries: int
    rationale: str
    answer: str
    hallucination_grade: Literal["grounded", "hallucinated"]
    conversation_history: List[Dict[str, str]]
    transformed_docs_summary: Optional[str]
    repo_map: Optional[str]


class LangGraphAgenticRAG:
    """Production LangGraph Agentic RAG pipeline."""

    def __init__(
        self,
        embedder: Any,
        qdrant_manager: QdrantManager,
        memori_manager: MemoriManager,
        llm_client: Any,
        memory_component: Any = None,
    ):
        self.embedder = embedder
        self.qdrant_manager = qdrant_manager
        self.memori_manager = memori_manager
        self.llm_client = llm_client
        self.memory_component = memory_component
        self.graph = self._build_graph()

    def _call_llm_text(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Invoke multi-provider LLM client to generate response text with optional history."""
        try:
            messages = [{"role": "system", "content": system_prompt}]
            if history:
                for turn in history[-4:]:
                    if isinstance(turn, dict):
                        u_content = turn.get("user") or turn.get("content")
                        a_content = turn.get("assistant")
                        if u_content:
                            messages.append({"role": "user", "content": u_content})
                        if a_content:
                            messages.append({"role": "assistant", "content": a_content})
            messages.append({"role": "user", "content": user_prompt})

            response = self.llm_client.call(
                api_kwargs={"messages": messages, "temperature": 0.2}
            )
            # UnifiedChatCompletion or raw response string
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    return str(choice.message.content or "").strip()
            if hasattr(response, "text"):
                return str(response.text).strip()
            return str(response).strip()
        except Exception as e:
            log.warning(f"Agentic LLM call encountered error: {e}")
            return ""

    def _contextualize_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """Reformulate follow-up queries referencing prior dialog context into self-contained search queries."""
        if not history:
            return query
        last_turn = history[-1]
        last_uq = last_turn.get("user", "")
        last_ar = (last_turn.get("assistant", "") or "")[:350]

        lowered = query.lower()
        followup_signals = [
            r"\b(it|this|that|these|those|its|their|the function|the class|the method|the file|the component|the endpoint)\b",
            r"^(how to|how do i|what about|can you show|give an example|where is|why is|explain more)\b",
        ]
        if not any(re.search(pat, lowered) for pat in followup_signals):
            return query

        system_prompt = (
            "You are a technical query contextualizer for a GitHub codebase assistant. "
            "Given the previous conversation turn and the user's latest follow-up question, "
            "rewrite the latest question to replace ambiguous pronouns ('it', 'this', 'that', 'its') "
            "with the explicit function name, class identifier, file, or topic discussed in the previous turn. "
            "If the question is already specific and self-contained, return it unchanged. "
            "Output ONLY the rewritten search query text without quotes or explanation."
        )
        user_prompt = f"Previous Question: {last_uq}\nPrevious Answer Context: {last_ar}\nFollow-up Question: {query}"
        rewritten = self._call_llm_text(system_prompt, user_prompt)
        cleaned = rewritten.strip().replace('"', '').replace("'", "")
        return cleaned if cleaned else query


    def _build_graph(self):
        """Construct and compile the LangGraph StateGraph."""
        builder = StateGraph(AgentState)

        # Register Graph Nodes
        builder.add_node("route_query", self.route_query_node)
        builder.add_node("direct_chat", self.direct_chat_node)
        builder.add_node("repo_overview", self.repo_overview_node)
        builder.add_node("retrieve", self.retrieve_node)
        builder.add_node("grade_documents", self.grade_documents_node)
        builder.add_node("rewrite_query", self.rewrite_query_node)
        builder.add_node("generate", self.generate_node)
        builder.add_node("evaluate_hallucination", self.evaluate_hallucination_node)
        builder.add_node("save_memory", self.save_memory_node)

        # Wire Graph Edges
        builder.add_edge(START, "route_query")

        # Dynamic Route Decision
        def route_decision(state: AgentState) -> str:
            return state["route"]

        builder.add_conditional_edges(
            "route_query",
            route_decision,
            {
                "direct_chat": "direct_chat",
                "repo_overview": "repo_overview",
                "code_search": "retrieve",
            },
        )

        builder.add_edge("retrieve", "grade_documents")

        # Corrective RAG (CRAG) Decision: Loop back to rewrite or proceed to generate
        def grade_decision(state: AgentState) -> str:
            if state["doc_grade"] == "relevant" or state["retry_count"] >= state["max_retries"]:
                return "generate"
            return "rewrite_query"

        builder.add_conditional_edges(
            "grade_documents",
            grade_decision,
            {
                "generate": "generate",
                "rewrite_query": "rewrite_query",
            },
        )

        # Query rewrite loops back to retrieve
        builder.add_edge("rewrite_query", "retrieve")
        builder.add_edge("repo_overview", "generate")

        # Self-RAG Hallucination Check
        builder.add_edge("generate", "evaluate_hallucination")

        def hallucination_decision(state: AgentState) -> str:
            if state.get("hallucination_grade") == "hallucinated" and state.get("retry_count", 0) < state.get("max_retries", 2):
                return "rewrite_query"
            return "save_memory"

        builder.add_conditional_edges(
            "evaluate_hallucination",
            hallucination_decision,
            {
                "rewrite_query": "rewrite_query",
                "save_memory": "save_memory",
            },
        )

        builder.add_edge("direct_chat", "save_memory")
        builder.add_edge("save_memory", END)

        return builder.compile()

    # -------------------------------------------------------------------------
    # Node Implementations
    # -------------------------------------------------------------------------

    def route_query_node(self, state: AgentState) -> Dict[str, Any]:
        """Classify user intent: direct conversational chat, repository structure overview, or codebase search."""
        query = state["query"].strip()
        lowered = query.lower()

        # Fast heuristic checks for high-speed routing
        chat_patterns = [
            r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))\b",
            r"^(who\s+are\s+you|what\s+can\s+you\s+do|help|how\s+do\s+you\s+work)\b",
            r"^(thanks|thank\s+you|bye|goodbye)\b",
        ]
        if any(re.search(pat, lowered) for pat in chat_patterns):
            log.info("Agent Router: Fast-path direct_chat")
            return {"route": "direct_chat", "effective_query": query}

        overview_patterns = [
            r"\b(show|list|view|give\s+me)\b.*\b(repo(sitory)?\s+(structure|tree|layout|overview|files))\b",
            r"\bwhat\s+(files|folders|directories)\s+(are|exist)\b",
            r"\bproject\s+(structure|architecture|organization)\b",
        ]
        if any(re.search(pat, lowered) for pat in overview_patterns):
            log.info("Agent Router: Fast-path repo_overview")
            return {"route": "repo_overview", "effective_query": query}

        # LLM Router for nuanced queries
        system_prompt = (
            "You are an expert query router for a GitHub Codebase RAG system. "
            "Classify the user prompt into exactly ONE category:\n"
            "- 'direct_chat': Casual greetings, compliments, meta questions about your identity or capabilities.\n"
            "- 'repo_overview': Questions inquiring about folder layout, directories, high-level project organization, or file lists.\n"
            "- 'code_search': Questions asking about code implementation, functions, classes, bugs, logic, APIs, or configurations.\n"
            "Output JSON only: {\"route\": \"direct_chat\" | \"repo_overview\" | \"code_search\"}"
        )
        resp = self._call_llm_text(system_prompt, f"User Prompt: {query}")
        route = "code_search"
        try:
            match = re.search(r"\{.*?\}", resp, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                parsed_route = data.get("route", "").strip().lower()
                if parsed_route in ["direct_chat", "repo_overview", "code_search"]:
                    route = parsed_route
        except (json.JSONDecodeError, AttributeError, ValueError) as parse_err:
            log.debug(f"Router output JSON parse failed, defaulting to code_search: {parse_err}")
            route = "code_search"
        except Exception as err:
            log.warning(f"Unexpected error in router node, defaulting to code_search: {err}")
            route = "code_search"

        log.info(f"Agent Router: Selected route '{route}' for query '{query}'")
        history = state.get("conversation_history") or []
        effective_query = self._contextualize_query(query, history) if (history and route == "code_search") else query
        if effective_query != query:
            log.info(f"Agent Router: Contextualized query '{query}' -> '{effective_query}'")
        return {"route": route, "effective_query": effective_query}


    def retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute Qdrant Hybrid Search (Dense + BM25 RRF) and recall Memori Labs memories."""
        query_to_search = state["effective_query"]
        collection = state.get("collection_name") or ""
        top_k = config.get("retriever", {}).get("top_k", 4)

        retrieved_docs: List[AdalDocument] = []
        if collection:
            try:
                # Generate dense vector with Google Gemini embedder
                try:
                    embed_output = self.embedder(query_to_search, model_kwargs={"task_type": "RETRIEVAL_QUERY"})
                except Exception as emb_err:
                    log.debug(f"Embedder call with task_type failed, retrying with input kwargs: {emb_err}")
                    embed_output = self.embedder(model_kwargs={"input": [query_to_search]})

                if embed_output and embed_output.data:
                    dense_vector = embed_output.data[0].embedding
                    # Qdrant Hybrid Search (Dense + BM25 RRF)
                    retrieved_docs = self.qdrant_manager.hybrid_search(
                        collection_name=collection,
                        query_text=query_to_search,
                        query_dense_vector=dense_vector,
                        top_k=top_k,
                    )
            except Exception as e:
                log.error(f"Hybrid retrieval failed: {e}")

        # Recall persistent semantic memories from Memori Labs
        semantic_memories = []
        try:
            semantic_memories = self.memori_manager.recall_memories(query=query_to_search, limit=3)
        except Exception as e:
            log.warning(f"Memori recall failed: {e}")

        log.info(
            f"Agent Retriever: Retrieved {len(retrieved_docs)} code chunks & {len(semantic_memories)} semantic memories "
            f"(attempt {state['retry_count'] + 1})"
        )
        return {
            "retrieved_documents": retrieved_docs,
            "semantic_memories": semantic_memories,
        }

    def grade_documents_node(self, state: AgentState) -> Dict[str, Any]:
        """Corrective RAG (CRAG) Grader: Evaluates retrieved chunk relevance to the query."""
        docs = state["retrieved_documents"]
        if not docs:
            log.info("Agent Grader: Zero documents retrieved -> marked not_relevant")
            return {"doc_grade": "not_relevant"}

        query = state["effective_query"]
        doc_samples = "\n---\n".join([d.text[:300] for d in docs[:3]])

        system_prompt = (
            "You are a code relevance evaluator in an Agentic RAG system. "
            "Assess whether the retrieved code snippets contain relevant context to answer the user query. "
            "Respond ONLY in JSON: {\"is_relevant\": true | false, \"reason\": \"brief explanation\"}"
        )
        user_prompt = f"Query: {query}\n\nRetrieved Code Snippets:\n{doc_samples}"
        resp = self._call_llm_text(system_prompt, user_prompt)

        is_relevant = True
        try:
            match = re.search(r"\{.*?\}", resp, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                is_relevant = bool(data.get("is_relevant", True))
        except (json.JSONDecodeError, AttributeError, ValueError) as parse_err:
            log.debug(f"Document grader output JSON parse failed, defaulting to relevant: {parse_err}")
            is_relevant = True
        except Exception as err:
            log.warning(f"Unexpected error in document grader parsing: {err}")
            is_relevant = True

        grade = "relevant" if is_relevant else "not_relevant"
        log.info(f"Agent Grader: Evaluated documents as '{grade}'")
        return {"doc_grade": grade}

    def rewrite_query_node(self, state: AgentState) -> Dict[str, Any]:
        """Self-Correction Query Rewriter: Reformulates technical query terms to optimize search recall."""
        original_query = state["query"]
        prev_effective = state["effective_query"]
        retry_count = state["retry_count"] + 1

        system_prompt = (
            "You are a technical query optimizer for GitHub code search. "
            "The previous search query failed to retrieve relevant code snippets. "
            "Analyze the intent and formulate a concise, technical search query containing exact function names, "
            "class identifiers, library keywords, or file terms. "
            "Output ONLY the rewritten search query text without quotation marks or conversational commentary."
        )
        user_prompt = f"Original Query: {original_query}\nPrevious Attempt: {prev_effective}\nOptimized Query:"
        rewritten = self._call_llm_text(system_prompt, user_prompt)
        rewritten = rewritten.strip().replace('"', '').replace("'", "")
        if not rewritten:
            rewritten = original_query

        log.info(f"Agent Rewriter: Attempt #{retry_count} -> Rewrote '{original_query}' to '{rewritten}'")
        return {
            "effective_query": rewritten,
            "retry_count": retry_count,
        }

    def _get_repo_map(self, collection_name: str, query: Optional[str] = None) -> str:
        """Fetch PageRank repository map from Redis or build fallback from indexed files."""
        try:
            redis_mgr = RedisCacheManager.get_instance()
            if redis_mgr and redis_mgr.is_connected and collection_name:
                cached_map = redis_mgr.get(f"repomap:{collection_name}")
                if cached_map:
                    return cached_map.decode("utf-8") if isinstance(cached_map, bytes) else str(cached_map)
        except Exception as e:
            log.warning(f"Error reading repo map from Redis: {e}")

        # Fallback to indexed files listing
        if collection_name and hasattr(self, "qdrant_manager") and self.qdrant_manager:
            try:
                files = self.qdrant_manager.get_indexed_files(collection_name, limit=60)
                if files:
                    return "# Repository File Structure\n" + "\n".join(f"├── {f}" for f in files)
            except Exception as e:
                log.warning(f"Error getting indexed files for repo map: {e}")
        return ""

    def repo_overview_node(self, state: AgentState) -> Dict[str, Any]:
        """Provide repository structural overview based on PageRank repo map and ingested file summaries."""
        repo_map = state.get("repo_map") or self._get_repo_map(state.get("collection_name", ""), query=state["query"])
        summary = state.get("transformed_docs_summary") or repo_map or "Repository codebase indexed in Qdrant."
        repo = state.get("repo_url") or "Repository"

        synthetic_doc = AdalDocument(
            text=f"Repository: {repo}\n\nArchitecture & Structure Overview:\n{summary}",
            meta_data={"file_path": "repo://overview", "is_code": False},
        )
        return {"retrieved_documents": [synthetic_doc], "repo_map": repo_map}

    def direct_chat_node(self, state: AgentState) -> Dict[str, Any]:
        """Handle greetings and non-code conversations directly without vector search."""
        query = state["query"]
        system_prompt = (
            "You are GithubChat, an intelligent Agentic RAG assistant for analyzing and discussing GitHub repositories. "
            "Respond naturally, courteously, and concisely to greetings, compliments, or general inquiries. "
            "Briefly invite the user to ask any questions about the repository's code, architecture, functions, or dependencies."
        )
        resp = self._call_llm_text(system_prompt, f"User says: {query}")
        return {
            "rationale": "Direct conversational response without vector database retrieval.",
            "answer": resp or "Hello! I am GithubChat. Ask me anything about this repository's code, structure, or implementation.",
            "hallucination_grade": "grounded",
        }

    def generate_node(self, state: AgentState) -> Dict[str, Any]:
        """Synthesize response using the multi-provider LLM fallback chain with rationale and citations."""
        query = state["query"]
        docs = state.get("retrieved_documents") or []
        memories = state.get("semantic_memories") or []
        repo_map = state.get("repo_map") or self._get_repo_map(state.get("collection_name", ""), query=query)

        # Construct structured context block with precise line and symbol citations
        context_parts = []
        if repo_map:
            context_parts.append(f"### [Repository Architecture Overview (PageRank Centrality)]\n{repo_map}")

        for i, doc in enumerate(docs):
            meta = doc.meta_data or {}
            file_path = meta.get("file_path", f"file_{i}")
            lines = meta.get("line_range")
            line_info = f":L{lines[0]}-L{lines[1]}" if lines and len(lines) >= 2 else ""
            symbol_name = meta.get("symbol_name")
            symbol_type = meta.get("symbol_type")
            symbol_info = f" [{symbol_type}: {symbol_name}]" if symbol_name else ""
            context_parts.append(f"### [Code Snippet {i+1}] File: {file_path}{line_info}{symbol_info}\n{doc.text}")

        for m in memories:
            context_parts.append(f"### [Semantic Memory / Known Fact]\n{m}")

        contexts_str = "\n\n".join(context_parts) if context_parts else "No direct code context found."

        system_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "You are operating as an Agentic RAG system for GitHub codebases. "
            "Synthesize an accurate, well-grounded response citing source files when available. "
            "Ground your answer in the provided code snippets and architecture map. "
            "Always cite relevant source files and line numbers (e.g. `[path/to/file.py:L12-L34]`) and explain precisely where functions, classes, and logic reside.\n"
            "You must respond in valid JSON format with exactly these two keys:\n"
            "{\n"
            '  "rationale": "Step-by-step thinking explaining how the answer was derived from the code.",\n'
            '  "answer": "Comprehensive, technical, markdown-formatted answer for the user."\n'
            "}"
        )

        user_prompt = f"User Question: {query}\n\nRetrieved Repository Context:\n{contexts_str}"
        raw_resp = self._call_llm_text(
            system_prompt,
            user_prompt,
            history=state.get("conversation_history"),
        )

        rationale = ""

        answer = raw_resp
        try:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_resp, re.DOTALL)
            clean_json = match.group(1) if match else raw_resp
            json_match = re.search(r"\{.*?\}", clean_json, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                rationale = parsed.get("rationale", "")
                answer = parsed.get("answer", raw_resp)
        except (json.JSONDecodeError, AttributeError, ValueError) as parse_err:
            log.debug(f"Generator JSON parsing failed, using raw response: {parse_err}")
            answer = raw_resp
        except Exception as err:
            log.warning(f"Unexpected error in generator output parsing: {err}")
            answer = raw_resp

        return {
            "rationale": rationale,
            "answer": answer,
        }

    def evaluate_hallucination_node(self, state: AgentState) -> Dict[str, Any]:
        """Self-RAG Evaluator: Checks if answer is factually aligned with retrieved context."""
        docs = state.get("retrieved_documents") or []
        answer = state.get("answer", "")
        route = state.get("route", "code_search")

        # Conversational chat or overview without docs is grounded by default
        if route == "direct_chat" or not docs or not answer:
            return {"hallucination_grade": "grounded"}

        doc_snippets = "\n---\n".join([d.text[:250] for d in docs[:3]])
        system_prompt = (
            "You are a factual grounding evaluator for an Agentic RAG system. "
            "Determine if the assistant's answer is grounded in and supported by the retrieved code context. "
            "Respond ONLY in JSON: {\"is_grounded\": true | false, \"reason\": \"brief explanation\"}"
        )
        user_prompt = f"Retrieved Context:\n{doc_snippets}\n\nGenerated Answer:\n{answer[:400]}"
        resp = self._call_llm_text(system_prompt, user_prompt)

        is_grounded = True
        try:
            match = re.search(r"\{.*?\}", resp, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                is_grounded = bool(data.get("is_grounded", True))
        except (json.JSONDecodeError, AttributeError, ValueError) as parse_err:
            log.debug(f"Hallucination evaluator JSON parse failed, defaulting to grounded: {parse_err}")
            is_grounded = True
        except Exception as err:
            log.warning(f"Unexpected error in hallucination evaluator parsing: {err}")
            is_grounded = True

        grade = "grounded" if is_grounded else "hallucinated"
        log.info(f"Agent Self-RAG: Evaluated factual grounding as '{grade}'")
        return {"hallucination_grade": grade}

    def save_memory_node(self, state: AgentState) -> Dict[str, Any]:
        """Persist dialog turn and extracted knowledge into Memori Labs SQL database."""
        query = state["query"]
        answer = state.get("answer", "")
        try:
            if self.memory_component and hasattr(self.memory_component, "add_dialog_turn"):
                self.memory_component.add_dialog_turn(uq=query, ar=answer)
            elif self.memori_manager:
                self.memori_manager.capture_dialog_turn(user_query=query, assistant_response=answer)
        except Exception as e:
            log.warning(f"Failed to record turn into Memori memory: {e}")
        return {}

    def invoke(
        self,
        query: str,
        repo_url: str = "",
        collection_name: str = "",
        transformed_docs_summary: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> AgentState:
        """Execute the LangGraph Agentic RAG pipeline for a given user query."""
        initial_state: AgentState = {
            "query": query,
            "effective_query": query,
            "repo_url": repo_url,
            "collection_name": collection_name,
            "route": "code_search",
            "retrieved_documents": [],
            "semantic_memories": [],
            "doc_grade": "relevant",
            "retry_count": 0,
            "max_retries": 2,
            "rationale": "",
            "answer": "",
            "hallucination_grade": "grounded",
            "conversation_history": conversation_history or [],
            "transformed_docs_summary": transformed_docs_summary,
            "repo_map": None,
        }

        final_state = self.graph.invoke(initial_state)
        return cast(AgentState, final_state)

    def stream_invoke(
        self,
        query: str,
        repo_url: str = "",
        collection_name: str = "",
        transformed_docs_summary: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Execute the LangGraph Agentic RAG pipeline with real-time SSE streaming.
        Yields events:
        - {"type": "status", "stage": str, "message": str}
        - {"type": "token", "token": str}
        - {"type": "done", "answer": str, "rationale": str, "contexts": List[Dict]}
        - {"type": "error", "error": str}
        """
        state: AgentState = {
            "query": query,
            "effective_query": query,
            "repo_url": repo_url,
            "collection_name": collection_name,
            "route": "code_search",
            "retrieved_documents": [],
            "semantic_memories": [],
            "doc_grade": "relevant",
            "retry_count": 0,
            "max_retries": 2,
            "rationale": "",
            "answer": "",
            "hallucination_grade": "grounded",
            "conversation_history": conversation_history or [],
            "transformed_docs_summary": transformed_docs_summary,
            "repo_map": None,
        }

        completed_steps: List[Dict[str, Any]] = []

        try:
            # Routing stage
            step_routing: Dict[str, Any] = {
                "id": "routing",
                "title": "Query Intent Routing",
                "detail": "Analyzing user query intent...",
                "status": "running",
            }
            yield {"type": "step", "step": step_routing}
            yield {"type": "status", "stage": "routing", "message": "Analyzing query intent..."}

            route_update = self.route_query_node(state)
            state.update(route_update)  # type: ignore
            route = state["route"]

            step_routing["detail"] = f"Intent classified as '{route}'. Directing pipeline to {route.replace('_', ' ').title()}."
            step_routing["status"] = "completed"
            completed_steps.append(dict(step_routing))
            yield {"type": "step", "step": step_routing}

            if route == "direct_chat":
                step_direct: Dict[str, Any] = {
                    "id": "direct_chat",
                    "title": "Direct Conversational Response",
                    "detail": "Formulating direct conversational response...",
                    "status": "running",
                }
                yield {"type": "step", "step": step_direct}
                yield {"type": "status", "stage": "generating", "message": "Formulating direct response..."}

                system_prompt = (
                    "You are GithubChat, an intelligent Agentic RAG assistant for analyzing and discussing GitHub repositories. "
                    "Respond naturally, courteously, and concisely to greetings, compliments, or general inquiries. "
                    "Briefly invite the user to ask any questions about the repository's code, architecture, functions, or dependencies."
                )
                user_prompt = f"User says: {query}"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                accumulated_chunks = []
                try:
                    for chunk in self.llm_client.stream_call(api_kwargs={"messages": messages, "temperature": 0.3}):
                        if chunk:
                            accumulated_chunks.append(chunk)
                            yield {"type": "token", "token": chunk}
                except Exception as e:
                    log.warning(f"Streaming failed for direct chat ({e}), falling back to sync call")
                    sync_resp = self._call_llm_text(system_prompt, user_prompt)
                    if not sync_resp:
                        sync_resp = "Hello! I am GithubChat. Ask me anything about this repository's code, structure, or implementation."
                    accumulated_chunks.append(sync_resp)
                    yield {"type": "token", "token": sync_resp}

                answer = "".join(accumulated_chunks).strip()
                rationale = "Direct conversational response without vector database retrieval."
                state["answer"] = answer
                state["rationale"] = rationale
                self.save_memory_node(state)

                step_direct["detail"] = "Generated conversational reply and saved turn to SQL memory."
                step_direct["status"] = "completed"
                completed_steps.append(dict(step_direct))
                yield {"type": "step", "step": step_direct}

                yield {
                    "type": "done",
                    "answer": answer,
                    "rationale": rationale,
                    "contexts": [],
                    "steps": completed_steps,
                }
                return

            elif route == "repo_overview":
                step_overview: Dict[str, Any] = {
                    "id": "overview",
                    "title": "Repository Architecture & Layout",
                    "detail": "Analyzing repository overview, directory tree, and top-level architecture...",
                    "status": "running",
                }
                yield {"type": "step", "step": step_overview}
                yield {"type": "status", "stage": "overview", "message": "Analyzing repository overview and layout..."}

                overview_update = self.repo_overview_node(state)
                state.update(overview_update)  # type: ignore

                step_overview["detail"] = f"Retrieved repository file structure ({len(state.get('retrieved_documents', []))} context files)."
                step_overview["status"] = "completed"
                completed_steps.append(dict(step_overview))
                yield {"type": "step", "step": step_overview}

            else:
                # Qdrant Hybrid Search & CRAG grading loop
                while True:
                    retry_idx = state["retry_count"]
                    attempt_str = f" (attempt {retry_idx + 1})" if retry_idx > 0 else ""
                    attempt_str = f" (Attempt {retry_idx + 1})" if retry_idx > 0 else ""

                    step_ret: Dict[str, Any] = {
                        "id": f"retrieving_{retry_idx}",
                        "title": f"Qdrant Hybrid Search{attempt_str}",
                        "detail": f"Searching dense Gemini embeddings (768-dim) + BM25 sparse keywords with RRF fusion...",
                        "status": "running",
                    }
                    yield {"type": "step", "step": step_ret}
                    yield {
                        "type": "status",
                        "stage": "retrieving",
                        "message": f"Searching Qdrant Hybrid vectors (Dense + BM25 RRF){attempt_str}...",
                    }

                    ret_update = self.retrieve_node(state)
                    state.update(ret_update)  # type: ignore
                    docs_found = len(state.get("retrieved_documents") or [])
                    mem_found = len(state.get("semantic_memories") or [])

                    step_ret["detail"] = f"Retrieved {docs_found} code chunks + {mem_found} semantic memories from Qdrant and SQL."
                    step_ret["status"] = "completed"
                    completed_steps.append(dict(step_ret))
                    yield {"type": "step", "step": step_ret}

                    step_grade: Dict[str, Any] = {
                        "id": f"grading_{retry_idx}",
                        "title": f"Corrective RAG (CRAG) Relevance Grading{attempt_str}",
                        "detail": "Evaluating chunk relevance against query intent...",
                        "status": "running",
                    }
                    yield {"type": "step", "step": step_grade}
                    yield {
                        "type": "status",
                        "stage": "grading",
                        "message": "Grading retrieved code relevance with Corrective RAG (CRAG)...",
                    }

                    grade_update = self.grade_documents_node(state)
                    state.update(grade_update)  # type: ignore
                    doc_grade = state.get("doc_grade", "relevant")
                    step_grade["detail"] = f"CRAG evaluated retrieved documents as '{doc_grade}'."
                    step_grade["status"] = "completed"
                    completed_steps.append(dict(step_grade))
                    yield {"type": "step", "step": step_grade}

                    if doc_grade == "relevant" or state["retry_count"] >= state["max_retries"]:
                        break

                    step_rw: Dict[str, Any] = {
                        "id": f"rewriting_{retry_idx}",
                        "title": f"Self-Correction Query Rewriting{attempt_str}",
                        "detail": "Reformulating technical query terms for higher search recall...",
                        "status": "running",
                    }
                    yield {"type": "step", "step": step_rw}
                    yield {
                        "type": "status",
                        "stage": "rewriting",
                        "message": "Reformulating technical query terms for higher search recall...",
                    }

                    rewrite_update = self.rewrite_query_node(state)
                    state.update(rewrite_update)  # type: ignore

                    step_rw["detail"] = f"Reformulated query: '{state.get('effective_query', query)}'."
                    step_rw["status"] = "completed"
                    completed_steps.append(dict(step_rw))
                    yield {"type": "step", "step": step_rw}

            # Generation stage
            docs = state.get("retrieved_documents") or []
            memories = state.get("semantic_memories") or []

            context_parts = []
            for i, doc in enumerate(docs):
                file_path = doc.meta_data.get("file_path", f"file_{i}") if hasattr(doc, "meta_data") and doc.meta_data else f"file_{i}"
                context_parts.append(f"### [Document {i+1}] Source: {file_path}\n{doc.text}")

            for m in memories:
                context_parts.append(f"### [Semantic Memory / Known Fact]\n{m}")

            contexts_str = "\n\n".join(context_parts) if context_parts else "No direct code context found."

            step_gen: Dict[str, Any] = {
                "id": "generating",
                "title": "Grounded Answer Synthesis",
                "detail": f"Synthesizing response with multi-provider LLM citing {len(docs)} context sources...",
                "status": "running",
            }
            yield {"type": "step", "step": step_gen}
            yield {
                "type": "status",
                "stage": "generating",
                "message": f"Synthesizing answer with multi-provider LLM citing {len(docs)} context sources...",
            }

            system_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                "You are operating as an Agentic RAG system. "
                "Synthesize an accurate, well-grounded response citing source files when available. "
                "Format your answer directly in rich Markdown (using headers, code blocks, lists, and bold text). "
                "Do NOT output raw JSON. Provide the markdown answer directly for the user."
            )
            user_prompt = f"User Question: {query}\n\nRetrieved Repository Context:\n{contexts_str}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            messages = [{"role": "system", "content": system_prompt}]

            history = state.get("conversation_history") or []
            for turn in history[-4:]:
                if isinstance(turn, dict):
                    u_msg = turn.get("user") or turn.get("content")
                    a_msg = turn.get("assistant")
                    if u_msg:
                        messages.append({"role": "user", "content": u_msg})
                    if a_msg:
                        messages.append({"role": "assistant", "content": a_msg})
            messages.append({"role": "user", "content": user_prompt})

            accumulated_chunks = []
            try:
                for chunk in self.llm_client.stream_call(api_kwargs={"messages": messages, "temperature": 0.2}):
                    if chunk:
                        accumulated_chunks.append(chunk)
                        yield {"type": "token", "token": chunk}
            except Exception as e:
                log.warning(f"Streaming generation failed ({e}), falling back to sync generation")
                sync_resp = self._call_llm_text(system_prompt, user_prompt)
                sync_resp = self._call_llm_text(system_prompt, user_prompt, history=history)
                accumulated_chunks.append(sync_resp)
                yield {"type": "token", "token": sync_resp}


            raw_answer = "".join(accumulated_chunks).strip()

            # Safeguard: If model generated JSON anyway, extract clean markdown answer and rationale
            clean_answer = raw_answer
            extracted_rationale = ""
            if clean_answer.startswith("```json") or clean_answer.startswith("{"):
                try:
                    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_answer, re.DOTALL)
                    clean_json = match.group(1) if match else clean_answer
                    json_match = re.search(r"\{.*?\}", clean_json, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group(0))
                        extracted_rationale = parsed.get("rationale", "")
                        clean_answer = parsed.get("answer", clean_answer)
                except (json.JSONDecodeError, AttributeError, ValueError) as e:
                    log.debug(f"AgenticRAG: Streamed answer was not structured JSON: {e}")

            if not extracted_rationale:
                sources_cited = [
                    d.meta_data.get("file_path", "")
                    for d in docs
                    if hasattr(d, "meta_data") and d.meta_data and d.meta_data.get("file_path")
                ]
                sources_str = f" from {len(sources_cited)} files ({', '.join(sources_cited[:3])}{'...' if len(sources_cited) > 3 else ''})" if sources_cited else ""
                extracted_rationale = (
                    f"Agentic RAG workflow: Routed to '{route}'. "
                    f"Retrieved {len(docs)} chunks from Qdrant Hybrid vectors + {len(memories)} semantic memories{sources_str}. "
                    f"CRAG evaluated relevance as '{state.get('doc_grade', 'relevant')}'. "
                    f"Answer synthesized using multi-provider fallback."
                )

            state["answer"] = clean_answer
            state["rationale"] = extracted_rationale

            step_gen["detail"] = f"Synthesized answer citing {len(docs)} code documents."
            step_gen["status"] = "completed"
            completed_steps.append(dict(step_gen))
            yield {"type": "step", "step": step_gen}

            # Self-RAG Grounding evaluation & persist to Memori Labs SQL
            step_eval: Dict[str, Any] = {
                "id": "evaluating",
                "title": "Self-RAG Grounding & SQL Memory",
                "detail": "Evaluating factual grounding and committing turn to Memori SQL storage...",
                "status": "running",
            }
            yield {"type": "step", "step": step_eval}
            yield {
                "type": "status",
                "stage": "evaluating",
                "message": "Evaluating factual grounding with Self-RAG...",
            }

            hall_update = self.evaluate_hallucination_node(state)
            state.update(hall_update)  # type: ignore
            hall_grade = state.get("hallucination_grade", "grounded")
            if hall_grade == "grounded":
                extracted_rationale += " Self-RAG verified factual grounding."
            state["rationale"] = extracted_rationale
            self.save_memory_node(state)

            step_eval["detail"] = f"Factual grounding evaluated as '{hall_grade}'. Persisted to SQL memory."
            step_eval["status"] = "completed"
            completed_steps.append(dict(step_eval))
            yield {"type": "step", "step": step_eval}

            # Format contexts for final payload
            contexts_dto = []
            for doc in docs:
                meta = getattr(doc, "meta_data", {}) or {}
                contexts_dto.append({
                    "text": getattr(doc, "text", ""),
                    "meta_data": {
                        "file_path": meta.get("file_path", ""),
                        "type": meta.get("type", ""),
                        "is_code": meta.get("is_code", False),
                        "is_implementation": meta.get("is_implementation", False),
                        "title": meta.get("title", ""),
                    },
                })

            yield {
                "type": "done",
                "answer": clean_answer,
                "rationale": extracted_rationale,
                "contexts": contexts_dto,
                "steps": completed_steps,
            }

        except Exception as e:
            log.error(f"Error in stream_invoke: {e}")
            yield {"type": "error", "error": str(e)}

