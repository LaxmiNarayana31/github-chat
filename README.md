# GithubChat

A RAG assistant to allow you to chat with any GitHub repo. Learn fast. The default repo is AdalFlow github repo.

## Features

- **Chat with GitHub Repos**: Ask questions about any repository's codebase
- **RAG-powered Retrieval**: Uses FAISS for semantic search over code
- **Conversation Memory**: Maintains context across dialog turns
- **Dual Interface**: Streamlit app for quick demos, React frontend for production

## Project Structure

```
.
├── app/                    # Core RAG components
│   ├── rag.py              # Main RAG pipeline with Memory component
│   ├── data_pipeline.py    # DatabaseManager for repo indexing
│   ├── gemini_embedder.py  # Gemini embedding model client
│   ├── groq_client.py      # Groq LLM client
│   ├── config.py           # Model configuration
│   └── system_prompt.py    # System prompts and RAG templates
│
├── backend/                # FastAPI server
│   ├── main.py             # API endpoints (/query, /health)
│   ├── dto.py              # Request/Response data models
│   └── utils.py            # Utility functions
│
├── frontend/               # React frontend application
│   └── src/
│       ├── App.tsx         # Main React app
│       └── components/     # UI components
│
├── streamlit_app.py        # Streamlit chat interface
├── pyproject.toml          # Python dependencies (uv/pip)
└── .env                    # Environment variables (API keys)
```

## Requirements

- Python >= 3.12
- Node.js (for frontend)
- API Keys: `GEMINI_API_KEY`, `GROQ_API_KEY`

## Backend Setup

1. Install dependencies using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Or using pip:

```bash
pip install -e .
```

2. Set up environment variables:

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your-gemini-api-key
GROQ_API_KEY=your-groq-api-key
```

## Running the Applications

### Streamlit UI

Run the Streamlit app for quick demos:

```bash
uv run streamlit run streamlit_app.py
```

### FastAPI Backend

Run the API server:

```bash
uv run backend/main.py
```

The API will be available at http://localhost:8000

- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### React Frontend

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Install Node.js dependencies:

```bash
pnpm install
```

3. Start the development server:

```bash
pnpm run dev
```

The frontend will be available at http://localhost:5173 (Vite default)

## API Endpoints

### GET /

Returns API information and available endpoints.

### GET /health

Health check endpoint - returns status and timestamp.

### POST /query

Analyzes a GitHub repository based on a query.

```json
// Request
{
  "repo_url": "https://github.com/username/repo",
  "query": "What does this repository do?"
}

// Response
{
  "rationale": "Analysis rationale...",
  "answer": "Detailed answer...",
  "contexts": [
    {
      "text": "Code snippet...",
      "meta_data": {
        "file_path": "src/main.py",
        "type": "python",
        "is_code": true
      }
    }
  ]
}
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  React Frontend │────▶│  FastAPI Backend │
└─────────────────┘     └────────┬────────┘
                                 │
┌─────────────────┐              ▼
│  Streamlit App  │────▶┌─────────────────┐
└─────────────────┘     │    RAG Engine   │
                        │  (app/rag.py)   │
                        └────────┬────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ DatabaseManager │    │ FAISS Retriever │    │  LLM Generator  │
│ (data_pipeline) │    │   (Semantic)    │    │ (Gemini/Groq)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## ROADMAP

- [x] Clearly structured RAG that can prepare a repo, persist from reloading, and answer questions.
  - `DatabaseManager` in `app/data_pipeline.py` to manage the database.
  - `RAG` class in `app/rag.py` to manage the whole RAG lifecycle.

### On the RAG backend

- [ ] Conditional retrieval - skip retrieval for follow-up clarifications
- [ ] Create an evaluation dataset
- [ ] Evaluate the RAG performance on the dataset
- [ ] Auto-optimize the RAG model

### On the React frontend

- [ ] Support display of full conversation history
- [ ] Support management of multiple conversations
