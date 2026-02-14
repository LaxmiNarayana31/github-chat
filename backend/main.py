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

# Run the app
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")