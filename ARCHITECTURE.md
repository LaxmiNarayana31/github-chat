# GithubChat System Architecture

Production architectural specification and component reference for GithubChat, based strictly on the codebase implementation.

---

## System Overview Architecture

GithubChat is a codebase intelligence platform implementing Agentic Retrieval-Augmented Generation (RAG). It provides semantic code search, autonomous agent routing, hybrid vector retrieval, and conversational memory storage.

The diagram below reflects the architecture as implemented in the repository:

```mermaid
graph TB
    subgraph Client_Tier["Client Tier (Two User Interfaces)"]
        ReactUI["React 18 SPA<br/>Vite + TypeScript + Tailwind CSS<br/>Port 5173"]
        StreamlitUI["Streamlit Web UI<br/>Python Dashboard<br/>Port 8501"]
    end

    subgraph API_Tier["FastAPI Web Server Tier (Port 8000)"]
        FastAPIApp["FastAPI ASGI Application (Uvicorn)"]
        HealthRouter["health_routes (/health, /)"]
        RAGRouter["rag_routes (/init, /query, /query-stream, /set-context, /clear-memory)"]
        RAGService["RAGService (Thread-Safe Singleton)"]
    end

    subgraph Core_Engine_Tier["Shared Core Engine Tier (backend/core)"]
        RAGCore["RAG Orchestrator (backend.core.rag.RAG)"]
        AgenticRAG["LangGraph Agentic RAG (StateGraph Engine)"]
        DataPipeline["DataPipeline (Git Ingestion & Chunking)"]
        QdrantMgr["QdrantManager (Hybrid RRF Search)"]
        MemoriMgr["MemoriManager (SQLAlchemy Session)"]
    end

    subgraph Storage_Tier["Storage Tier (Local & Remote)"]
        PostgresDB[("PostgreSQL Database<br/>Managed Aiven / Cloud Postgres via DB_*")]
        QdrantDB[("Qdrant Cloud Cluster<br/>Managed Vector DB via QDRANT_URL")]
        DiskCache[("Local Disk Cache<br/>~/.adalflow/databases/*.pkl & repos/")]
    end

    subgraph External_AI_Tier["External AI & Local ML Providers"]
        GeminiAPI["Google Gemini API<br/>(gemini-embedding-2 / gemini-embedding-001)"]
        GroqAPI["Groq Cloud API<br/>(openai/gpt-oss-120b / llama-3.3-70b-versatile)"]
        LocalFastEmbed["Local FastEmbed (CPU)<br/>Qdrant/bm25 Sparse Lexical Model"]
    end

    ReactUI -->|"HTTP REST & SSE Stream"| FastAPIApp
    FastAPIApp --> HealthRouter
    FastAPIApp --> RAGRouter
    RAGRouter --> RAGService
    RAGService -->|"Calls in-process"| RAGCore

    StreamlitUI -->|"Direct Python import & call"| RAGCore

    RAGCore --> AgenticRAG
    RAGCore --> DataPipeline
    RAGCore --> QdrantMgr
    RAGCore --> MemoriMgr

    DataPipeline -->|"Clone & Split"| DiskCache
    DataPipeline -->|"Generate Dense Embeddings"| GeminiAPI
    DataPipeline -->|"Compute Sparse Vectors"| LocalFastEmbed
    DataPipeline -->|"Upsert Vectors & Payloads"| QdrantDB

    AgenticRAG -->|"Query Dense Vector"| GeminiAPI
    AgenticRAG -->|"Hybrid Search (RRF)"| QdrantDB
    AgenticRAG -->|"Prompt Inference & Evaluation"| GroqAPI
    AgenticRAG -->|"Recall & Commit Dialog Turns"| PostgresDB
```

---

## Agentic RAG Pipeline Architecture (LangGraph StateGraph)

The RAG engine in `backend/core/agentic_rag.py` executes a compiled LangGraph `StateGraph`.

```mermaid
flowchart TD
    StartNode([Start: User Query]) --> RouteNode["Query Intent Router (LLM Evaluator)"]

    RouteNode -->|"direct_chat"| DirectChatNode["Direct Conversational Chat (LLM)"]
    RouteNode -->|"repo_overview"| OverviewNode["Repository Overview (File Structure & Context)"]
    RouteNode -->|"code_search"| RetrieveNode["Retrieve Node (Gemini Dense + FastEmbed BM25 + Qdrant RRF)"]

    RetrieveNode --> GradeNode["CRAG Document Grader (LLM)"]

    GradeNode -->|"Documents Relevant"| GenerateNode["Generate Grounded Answer & Rationale (Groq LLM)"]
    GradeNode -->|"Not Relevant & Retries Remaining"| RewriteNode["Query Rewriter (Technical Expansion)"]

    RewriteNode -->|"Rewritten Query"| RetrieveNode

    GenerateNode --> HallucinationNode["Self-RAG Grounding Evaluator (LLM)"]

    HallucinationNode -->|"Hallucinated & Retries Remaining"| GenerateNode
    HallucinationNode -->|"Grounded / Pass"| SaveMemoryNode["Save Memory Node (PostgreSQL Database)"]

    DirectChatNode --> SaveMemoryNode
    OverviewNode --> SaveMemoryNode
    SaveMemoryNode --> EndNode([End: Yield Answer, Rationale, Citations])
```

---

## End-to-End Sequence Diagram

The diagram below details the two primary workflows: **Repository Ingestion** and **Interactive Streaming Query**.

```mermaid
sequenceDiagram
    autonumber
    actor User as Client (React / Streamlit)
    participant API as FastAPI (rag_routes)
    participant Service as RAGService
    participant Pipeline as DataPipeline
    participant Agent as AgenticRAG (LangGraph)
    participant Qdrant as Qdrant Vector Store
    participant Gemini as Google Gemini API
    participant Groq as Groq Cloud API
    participant SQL as PostgreSQL Database

    Note over User, Qdrant: Repository Ingestion & Vector Indexing
    User->>API: POST /init { repo_url, force_reindex }
    API->>Service: initialize_repository(repo_url)
    Service->>Pipeline: download_github_repo(repo_url)
    Pipeline->>Pipeline: read_all_documents() (Filter binaries, sizes, ignored paths)
    Pipeline->>Pipeline: TextSplitter (chunk_size=200 words, overlap=100)
    Pipeline->>Gemini: Batch embed chunks (768-dim dense vectors)
    Gemini-->>Pipeline: Dense embeddings
    Pipeline->>Qdrant: Create collection with Cosine + BM25 Sparse
    Pipeline->>Qdrant: Upsert chunk vectors and payloads
    Service-->>API: Repository initialized
    API-->>User: 200 OK { status: "success", repo_url }

    Note over User, SQL: Interactive Streaming Query Workflow
    User->>API: POST /query-stream { query, repo_url }
    API->>Service: stream_query(query)
    Service->>Agent: stream_call(query)
    Agent-->>User: SSE event: step (Routing)
    Agent->>Groq: Evaluate intent (direct_chat, repo_overview, code_search)
    Groq-->>Agent: Route decision
    Agent-->>User: SSE event: step (Hybrid Retrieval)
    Agent->>Gemini: Embed effective query
    Gemini-->>Agent: Query dense vector
    Agent->>Qdrant: hybrid_search(dense_vec, bm25_sparse, top_k=4)
    Qdrant-->>Agent: Fused code snippets
    Agent->>SQL: recall_memories(query, limit=3)
    SQL-->>Agent: Semantic memory facts
    Agent-->>User: SSE event: step (CRAG Document Grading)
    Agent->>Groq: Evaluate chunk relevance
    Groq-->>Agent: Relevance score & grade
    Agent-->>User: SSE event: step (Grounded Generation)
    Agent->>Groq: Generate streaming tokens with context & rationale
    Groq-->>Agent: Token chunks
    Agent-->>User: SSE event: token (Live answer stream)
    Agent-->>User: SSE event: step (Self-RAG Grounding Check)
    Agent->>Groq: Verify factual grounding against citations
    Groq-->>Agent: Grounding verified
    Agent->>SQL: Save dialog turn (Entity, Session, Messages)
    Agent-->>User: SSE event: done { answer, rationale, contexts, steps }
```

---

## Storage & State Architecture

```mermaid
graph TD
    subgraph Vector_Storage["Vector Storage Engine"]
        QdrantInstance["Qdrant Client (Connected to QDRANT_URL)"]
        DenseIndex["Dense Vector Index (Cosine Metric, 768 Dimensions)"]
        SparseIndex["Sparse Vector Index (BM25 Lexical Inverted Index)"]
        Payloads["Document Payloads (File Path, Source Text, Code Metadata)"]
        FusionEngine["Reciprocal Rank Fusion (RRF) Scorer"]

        QdrantInstance --> DenseIndex
        QdrantInstance --> SparseIndex
        QdrantInstance --> Payloads
        DenseIndex --> FusionEngine
        SparseIndex --> FusionEngine
    end

    subgraph Relational_Storage["Memori Labs SQL Schema"]
        Engine["SQLAlchemy Engine (PostgreSQL via DB_* / In-Memory)"]
        T_Entity["memori_entity (Entity ID, External UUID)"]
        T_Process["memori_process (Process ID, Runtime Metadata)"]
        T_Session["memori_session (Deterministic DNS UUIDv5)"]
        T_Conv["memori_conversation (Session ID, Summary)"]
        T_Msg["memori_conversation_message (Role, Content, Timestamps)"]

        Engine --> T_Entity
        Engine --> T_Process
        Engine --> T_Session
        Engine --> T_Conv
        Engine --> T_Msg

        T_Entity -.->|1:N| T_Session
        T_Process -.->|1:N| T_Session
        T_Session -.->|1:N| T_Conv
        T_Conv -.->|1:N| T_Msg
    end

    subgraph File_Cache["Local File System Artifacts"]
        ClonedRepos["AdalFlow Root: ~/.adalflow/repos/{repo_name}"]
        PKLCache["AdalFlow Root: ~/.adalflow/databases/{repo_name}.pkl"]
        FastEmbedCache[".fastembed_cache/ (Cached BM25 tokenizers)"]
    end
```

---

## Host Runtime & Network Topology

```mermaid
graph TB
    subgraph Host_Machine["Host Machine (Local Development / Server)"]
        subgraph Client_Processes["Client Applications"]
            ViteDev["Vite Dev Server (Port 5173, Node.js)"]
            StreamlitProcess["Streamlit Runtime (Port 8501, Python 3.12)"]
        end

        subgraph Backend_Process["Backend Core Service"]
            Uvicorn["Uvicorn ASGI Server (Port 8000, Python 3.12)"]
            FastAPIRuntime["FastAPI Application (backend.main)"]
            SharedCore["Shared Core Engines (rag, agentic_rag, data_pipeline)"]
        end
    end

    subgraph External_Cloud_Services["Configured External Services"]
        GoogleGenAI["Google Gemini API (HTTPS 443)<br/>Key: GEMINI_API_KEY"]
        GroqCloud["Groq Cloud API (HTTPS 443)<br/>Key: GROQ_API_KEY"]
        RemoteQdrantCluster["Qdrant Cloud Cluster (HTTPS 6333)<br/>Cluster URL: QDRANT_URL + QDRANT_API_KEY"]
        PostgresCloud["Managed PostgreSQL Database (TCP 5432 / SSL)<br/>Host: DB_HOST, DB_USER, DB_NAME"]
    end

    ViteDev -->|"CORS HTTP REST & SSE (:8000)"| Uvicorn
    Uvicorn --> FastAPIRuntime
    FastAPIRuntime --> SharedCore

    StreamlitProcess -->|"Direct In-Process Python Import"| SharedCore

    SharedCore -->|"Session Turns & Facts (TCP 5432)"| PostgresCloud
    SharedCore -->|"Hybrid Vectors & Payloads (HTTPS)"| RemoteQdrantCluster
    SharedCore -->|"Dense Embeddings (HTTPS)"| GoogleGenAI
    SharedCore -->|"LLM Generation (HTTPS)"| GroqCloud
```

---

## Component Specifications

### Client Interfaces
- **React 18 SPA (`frontend/`)**:
  - Built with React 18, Vite, TypeScript, and Tailwind CSS.
  - Connects to the backend via HTTP REST and Server-Sent Events (`frontend/src/services/api.ts`).
  - Visualizes real-time progress steps, streaming tokens, model connection health, and file citations.
- **Streamlit Dashboard (`streamlit_app/`)**:
  - Built with Python and Streamlit.
  - Directly imports and invokes the core RAG engine in-process (`RAG(entity_id="streamlit_user", process_id="streamlit_app")`).
  - Provides model key testing widgets, chat history management, and transcript downloads.

### API Gateway
- **FastAPI Application (`backend/main.py`)**:
  - Handles CORS middleware for frontend communication.
  - Implements global exception handlers and startup key checks.
  - Routes:
    - `/health`, `/`: Environment validation and API status.
    - `/init`: Repository ingestion and hybrid vectorization.
    - `/query`: Synchronous RAG semantic answer generation.
    - `/query-stream`: Real-time SSE streaming endpoint.
    - `/set-context`, `/clear-memory`: Conversation history synchronization.

### Core Engines & Services
- **RAGService (`backend/services/rag_service.py`)**:
  - Thread-safe Singleton managing the lifecycle of the underlying RAG orchestrator for FastAPI requests.
- **RAG Orchestrator (`backend/core/rag.py`)**:
  - Bridges ingestion preparation, embedding verification, and delegation to the autonomous LangGraph agent pipeline.
- **LangGraph Agentic RAG (`backend/core/agentic_rag.py`)**:
  - Implements a state machine covering query routing, hybrid retrieval, Corrective RAG (CRAG) document grading, query rewriting, grounded generation, and Self-RAG hallucination evaluation.
- **Data Pipeline (`backend/core/data_pipeline.py`)**:
  - Shallow clones Git repositories, filters ignored files, chunks documents, generates dense embeddings in batches of 100, and caches state in `~/.adalflow/databases/*.pkl`.
- **Qdrant Manager (`backend/core/qdrant_manager.py`)**:
  - Manages dual-vector collections (Dense 768-dim + BM25 Sparse).
  - Connects to your managed Qdrant Cloud cluster via `QDRANT_URL` and `QDRANT_API_KEY`.
- **Memori Manager (`backend/core/memory_manager.py`)**:
  - Manages conversation state, session turns, and semantic facts in a managed PostgreSQL database (e.g., Aiven) via SQLAlchemy, using transient in-memory storage during automated testing without creating any disk files.

### External AI Providers
- **Google Gemini API (`GEMINI_API_KEY`)**:
  - Generates 768-dimensional dense vector embeddings using `gemini-embedding-2` with `gemini-embedding-001` fallback.
- **Groq Cloud API (`GROQ_API_KEY`)**:
  - Provides low-latency LLM inference for agent routing, document grading, query rewriting, and answer generation.
- **Local FastEmbed (`fastembed`)**:
  - Generates BM25 sparse vectors locally on CPU without external API latency.

---

## Authentication & Credentials Model

The application relies strictly on environment variables specified in `backend/.env.example`:

| Variable | Required | Purpose |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | Yes | Dense vector embeddings with Google Gemini |
| `GROQ_API_KEY` | Yes | LLM inference with Groq Cloud |
| `QDRANT_URL` | Yes | Managed Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Yes | API key for managed Qdrant Cloud cluster |
| `DB_USER` | Yes | PostgreSQL username (e.g. Aiven) |
| `DB_PASSWORD` | Yes | PostgreSQL password |
| `DB_HOST` | Yes | PostgreSQL host endpoint |
| `DB_PORT` | No | PostgreSQL port (defaults to 5432) |
| `DB_NAME` | No | PostgreSQL database name (defaults to defaultdb) |
| `SSL_MODE` | No | PostgreSQL SSL connection mode (defaults to require) |
| `DATABASE_URL` | Optional | Direct full connection URI alternative |

Client access to the FastAPI server is open over local network / CORS without authentication.

---

## Deployment & Infrastructure Note

Containerization files (`Dockerfile`, `docker-compose.yml`) and distributed message brokers (Celery, Redis) are not implemented in the repository. The platform runs natively across two or three local processes:
- Backend: `uvicorn backend.main:app --port 8000`
- React Frontend: `npm run dev` (Vite, port 5173)
- Streamlit UI: `streamlit run streamlit_app/main.py` (port 8501)
