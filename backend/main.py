import os
import sys
from datetime import datetime, timezone

import uvicorn
import adalflow as adal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Add project root to Python path so 'app' module can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import RAG
from backend.dto import QueryRequest, DocumentMetadata, Document, QueryResponse

load_dotenv(verbose=True)

# Initialize FastAPI app
app = FastAPI(
    title="GithubChat API", 
    description="API for querying GitHub repositories using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG component
try:
    # Set up adalflow environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    adal.setup_env()
    rag = RAG()
    print("Successfully initialized RAG component")
except Exception as e:
    print(f"Error initializing RAG component: {e}")
    raise RuntimeError(f"Failed to initialize RAG component: {e}")

# Root endpoint with API information
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "GithubChat API",
        "description": "API for querying GitHub repositories using RAG",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health"
    }

# Health check endpoint to verify API is running
@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is running"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    }

# Clear memory endpoint for new chat sessions
@app.post("/clear-memory")
async def clear_memory():
    """Clear the RAG conversation memory for a new chat session"""
    try:
        rag.memory.current_conversation.dialog_turns.clear()
        print("Memory cleared for new chat session")
        return {"status": "success", "message": "Memory cleared"}
    except Exception as e:
        print(f"Error clearing memory: {e}")
        return {"status": "error", "message": str(e)}

# Set conversation context when switching between chats
@app.post("/set-context")
async def set_context(messages: list[dict]):
    """Restore conversation context when switching to a different chat"""
    try:
        # Clear existing memory first
        rag.memory.current_conversation.dialog_turns.clear()
        
        # Rebuild memory from provided messages
        for i in range(0, len(messages) - 1, 2):
            if i + 1 < len(messages):
                user_msg = messages[i]
                assistant_msg = messages[i + 1]
                if user_msg.get("role") == "user" and assistant_msg.get("role") == "assistant":
                    rag.memory.add_dialog_turn(
                        uq=user_msg.get("content", ""),
                        ar=assistant_msg.get("content", "")
                    )
        
        print(f"Context restored with {len(rag.memory.current_conversation.dialog_turns)} turns")
        return {"status": "success", "turns": len(rag.memory.current_conversation.dialog_turns)}
    except Exception as e:
        print(f"Error setting context: {e}")
        return {"status": "error", "message": str(e)}

# Query endpoint to query a GitHub repository with RAG
@app.post("/query", response_model=QueryResponse)
async def query_repository(request: QueryRequest):
    """Query a GitHub repository with RAG"""
    try:
        # Prepare retriever for the repository
        rag.prepare_retriever(request.repo_url)
        
        # Get response and retrieved documents
        response, retrieved_documents = rag(request.query)
        
        # Format response
        return QueryResponse(
            rationale=response.rationale if hasattr(response, 'rationale') else "",
            answer=response.answer if hasattr(response, 'answer') else response.raw_response,
            contexts=[
                Document(
                    text=doc.text,
                    meta_data=DocumentMetadata(
                        file_path=doc.meta_data.get('file_path', ''),
                        type=doc.meta_data.get('type', ''),
                        is_code=doc.meta_data.get('is_code', False),
                        is_implementation=doc.meta_data.get('is_implementation', False),
                        title=doc.meta_data.get('title', '')
                    )
                )
                for doc in retrieved_documents[0].documents
            ] if retrieved_documents and retrieved_documents[0].documents else []
        )
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# Run the app
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")