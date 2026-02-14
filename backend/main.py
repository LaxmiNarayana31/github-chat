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

# Run the app
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")