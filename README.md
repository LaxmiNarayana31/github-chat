# GithubChat

Production-grade Codebase Intelligence and Agentic Retrieval-Augmented Generation (RAG) platform. Clone, semantically index, and interact with any GitHub repository or local codebase in real time with exact file citations, autonomous agent routing, hybrid vector retrieval, and persistent conversational memory.

---

## Core Capabilities

- **Autonomous Agentic RAG (LangGraph)**:
  - **Query Intent Router**: Dynamically routes requests to direct conversational chat, repository structural overviews, or deep code retrieval.
  - **Corrective RAG (CRAG) Document Grader**: Evaluates retrieved code snippets for semantic relevance before generating answers.
  - **Query Reformulation Loop**: Iteratively expands and rewrites queries when retrieved context is insufficient.
  - **Self-RAG Grounding Evaluator**: Validates generated answers against retrieved source snippets to prevent hallucinations.
- **Git Commit SHA Verification & Incremental Ingestion**:
  - Validates remote repository commit hashes via `git ls-remote` prior to ingestion.
  - If the commit hash matches the existing database record, cloning and vector indexing are skipped entirely, instantly serving queries.
  - When new commits are detected, the repository is automatically updated and incrementally re-vectorized.
- **Qdrant Cloud Cluster Hybrid Search (RRF)**:
  - Connects to your managed **Qdrant Cloud cluster** via `QDRANT_URL` and `QDRANT_API_KEY`.
  - Combines 768-dimensional dense vectors from Google Gemini (`gemini-embedding-2` / `gemini-embedding-001`) with FastEmbed BM25 sparse lexical vectors.
  - Fuses dense and sparse rankings using Reciprocal Rank Fusion (RRF) for high recall across exact symbol names and semantic concepts.
- **Upstash Redis & Redis Caching Layer**:
  - Supports Upstash Redis REST API or standard Redis for distributed locking, raw Git blob chunk deduplication, and embedding caching.
  - Streaming checkpoint tracking allows large repository ingestions to resume seamlessly without re-embedding uncached files.
  - Automatic graceful fallback to in-memory caching when Redis credentials are not configured.
- **AST-Aware Chunking & PageRank Repository Map**:
  - Recursively parses source files with syntax awareness and constructs a repository-wide call/import dependency graph.
  - Computes structural importance via NetworkX PageRank to guide context assembly.
- **Memori Labs Relational SQL Memory**:
  - Persists conversational entities, processes, sessions, conversations, and dialog turns in a managed PostgreSQL database (such as Aiven) via SQLAlchemy (with transient in-memory storage for test runs).
  - High-level semantic memory recall enriches prompt context with known facts across sessions.
- **Real-Time Token & Thought Streaming**:
  - Server-Sent Events deliver real-time progress steps, reasoning summaries, live token streams, and source citations.
- **Dual User Interfaces**:
  - **React 18 SPA**: Modern TypeScript + Tailwind CSS application (running on port 3000) with real-time progress indicators, collapsible reasoning cards, and syntax-highlighted citations.
  - **Streamlit Dashboard**: Standalone Python web dashboard with model connection indicators, history controls, and transcript exports.

---

## Architecture Overview

```mermaid
graph TB
    subgraph Client_Layer["Client Layer (User Interfaces)"]
        ReactUI["React 18 SPA (Vite + Tailwind CSS + TypeScript)<br/>Port 3000"]
        StreamlitUI["Streamlit Dashboard (Python)<br/>Port 8501"]
    end

    subgraph API_Layer["API Gateway Layer (FastAPI)"]
        FastAPIApp["FastAPI Application (ASGI Server)<br/>Port 8000"]
        RAGService["RAGService Singleton"]
    end

    subgraph Core_Services["Service & Orchestration Layer"]
        RAGOrchestrator["RAG Core Engine"]
        AgenticRAG["AgenticRAG (LangGraph StateGraph)"]
        DataPipeline["DataPipeline (Git Ingestion & Indexing)"]
    end

    subgraph Storage_Layer["Data & Persistence Layer"]
        QdrantStore["Qdrant Cloud Cluster<br/>Managed Vector Database via QDRANT_URL"]
        MemoriSQL["PostgreSQL Database (Aiven / Cloud)<br/>Managed Relational Storage via DB_*"]
        RedisStore["Upstash Redis / Redis<br/>Distributed Locks & Embedding Cache"]
        LocalFileCache["Local Disk Cache<br/>~/.adalflow/databases/*.pkl & repos/"]
    end

    subgraph External_Providers["External Cloud AI & Local ML Providers"]
        GeminiAPI["Google Gemini API<br/>gemini-embedding-2 / gemini-embedding-001"]
        GroqAPI["Groq Cloud API<br/>openai/gpt-oss-120b / llama-3.3-70b-versatile"]
        FastEmbedModel["FastEmbed QdrantBM25Model<br/>Local In-Memory CPU"]
    end

    ReactUI -->|"HTTP / SSE"| FastAPIApp
    FastAPIApp --> RAGService
    RAGService -->|"Calls in-process"| RAGOrchestrator

    StreamlitUI -->|"Direct Python Import & Call"| RAGOrchestrator

    RAGOrchestrator --> AgenticRAG
    RAGOrchestrator --> DataPipeline

    DataPipeline -->|"Clone & Parse"| LocalFileCache
    DataPipeline -->|"Cache Embeddings & Blobs"| RedisStore
    DataPipeline -->|"Compute Dense Vectors"| GeminiAPI
    DataPipeline -->|"Compute Sparse Vectors"| FastEmbedModel
    DataPipeline -->|"Upsert Vectors & Payloads"| QdrantStore

    AgenticRAG -->|"Hybrid Search (RRF)"| QdrantStore
    AgenticRAG -->|"Query Embeddings"| GeminiAPI
    AgenticRAG -->|"Reasoning, Routing & Generation"| GroqAPI
    AgenticRAG -->|"Recall & Commit Dialog Turns"| MemoriSQL
```

For complete technical specifications, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Agentic RAG Pipeline

```mermaid
flowchart TD
    StartNode([Start: User Query]) --> RouteNode["Query Intent Router (LLM Evaluator)"]

    RouteNode -->|"direct_chat"| DirectChatNode["Direct Conversational Chat (LLM)"]
    RouteNode -->|"repo_overview"| OverviewNode["Repository Overview (File Tree & Summary)"]
    RouteNode -->|"code_search"| RetrieveNode["Retrieve Node (Gemini Dense + FastEmbed BM25 + Qdrant Cloud RRF)"]

    RetrieveNode --> GradeNode["CRAG Document Grader (LLM)"]

    GradeNode -->|"Documents Relevant"| GenerateNode["Generate Grounded Answer & Rationale (Groq LLM)"]
    GradeNode -->|"Not Relevant & Retries Left"| RewriteNode["Query Rewriter (Technical Expansion)"]

    RewriteNode -->|"Rewritten Query"| RetrieveNode

    GenerateNode --> HallucinationNode["Self-RAG Grounding Evaluator (LLM)"]

    HallucinationNode -->|"Hallucinated & Retries Left"| GenerateNode
    HallucinationNode -->|"Grounded / Pass"| SaveMemoryNode["Save Memory Node (Memori Labs SQL Storage)"]

    DirectChatNode --> SaveMemoryNode
    OverviewNode --> SaveMemoryNode
    SaveMemoryNode --> EndNode([End: Yield Answer, Rationale, Citations])
```

## Prerequisites

- **Python**: 3.12 or higher
- **Node.js**: 18 or higher (for the React frontend)
- **Git**: Installed and available in PATH
- **Cloud Service Credentials**:
  - `GEMINI_API_KEY`: Google AI Studio API key (for dense embeddings)
  - `GROQ_API_KEY`: Groq Cloud API key (for fast LLM inference)
  - `QDRANT_URL`: Managed Qdrant Cloud cluster endpoint
  - `QDRANT_API_KEY`: Managed Qdrant Cloud API key
  - Managed PostgreSQL: `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `SSL_MODE` (or `DATABASE_URL`)
  - `REDIS_URL` or `UPSTASH_REDIS_REST_URL` / `UPSTASH_REDIS_REST_TOKEN`: (Optional) Upstash Redis / standard Redis for distributed locks, deduplication, and embedding caching
  - `GITHUB_TOKEN`: (Optional) GitHub Personal Access Token for private repositories or elevated clone limits

---

## Quickstart & Setup Guide

### Clone the Repository

```bash
git clone https://github.com/LaxmiNarayana31/github-chat.git
cd github-chat
```

---

### Configure Environment Variables

Create your `.env` configuration file from the template:

```bash
# On Linux / macOS
cp backend/.env.example backend/.env

# On Windows PowerShell
Copy-Item backend\.env.example backend\.env
```

Open `backend/.env` and insert your credentials:

```env
# Google AI Studio
GEMINI_API_KEY=your_gemini_api_key_here

# Groq Cloud
GROQ_API_KEY=your_groq_api_key_here

# Qdrant Cloud
QDRANT_URL=https://your-cluster-id.eu-west-2-0.aws.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here

# Managed PostgreSQL (e.g. Aiven)
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=your_db_host.aivencloud.com
DB_PORT=5432
DB_NAME=defaultdb
SSL_MODE=require

# Optional: Upstash Redis REST or standard Redis (falls back to local memory if omitted)
# UPSTASH_REDIS_REST_URL=https://your-database.upstash.io
# UPSTASH_REDIS_REST_TOKEN=your_upstash_rest_token
# REDIS_URL=redis://localhost:6379/0

# Optional: GitHub Personal Access Token for private repositories
# GITHUB_TOKEN=your_github_token_here
```

| Key | Where to Get | Purpose |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey) | Dense semantic vector embeddings |
| `GROQ_API_KEY` | [Groq Cloud Console](https://console.groq.com/keys) | Agent routing, CRAG grading, & generation |
| `QDRANT_URL` & `QDRANT_API_KEY` | [Qdrant Cloud Console](https://cloud.qdrant.io) | Managed hybrid vector retrieval cluster |
| `DB_*` / `DATABASE_URL` | [Aiven Console](https://console.aiven.io) | Managed PostgreSQL relational memory storage |
| `REDIS_URL` / `UPSTASH_*` | [Upstash Console](https://console.upstash.com) | (Optional) Distributed locks & embedding cache |
| `GITHUB_TOKEN` | [GitHub Settings](https://github.com/settings/tokens) | (Optional) Access to private repos & higher rate limits |

---

### Backend Setup & Startup (FastAPI)

Navigate to the `backend` directory:

```bash
cd backend
```

Choose your preferred package manager to install dependencies:

#### Using uv (Fastest)

```bash
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Using standard pip & virtualenv

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell:
.venv\Scripts\Activate.ps1

# Activate on Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch FastAPI backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The backend server is now running at `http://localhost:8000`.
- Health Check: `http://localhost:8000/health`
- Interactive Swagger Docs: `http://localhost:8000/docs`

---

### Client Setup & Startup

You can use either the modern React SPA or the interactive Streamlit dashboard:

#### Option A: React 18 SPA (Recommended)

In a new terminal, navigate to the `frontend` folder:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser. Enter any public GitHub repository URL (such as `https://github.com/fastapi/fastapi`), click **Index**, and begin chatting with full citations and streaming thoughts.

#### Option B: Streamlit Web Dashboard

In a new terminal, launch the Streamlit interface from the repository root:

```bash
# With uv
uv run --project backend streamlit run streamlit_app/main.py

# Or with activated virtual environment
streamlit run streamlit_app/main.py
```

Open `http://localhost:8501` in your browser.

---

### Verification & Testing

Verify that all services, pipelines, and agents are functioning properly by running the automated test suite:

```bash
# Run all unit and integration tests
backend/.venv/Scripts/pytest backend/tests

# Or with uv from repo root
uv run --project backend pytest backend/tests
```

All 61 tests validate the document ingestion pipeline, Gemini embedder batching, Qdrant hybrid retrieval, Memori SQL turn saving, API routes, and LangGraph agent execution.
