import os
from dotenv import load_dotenv

from adalflow.components.model_client.groq_client import GroqAPIClient

from app.gemini_embedder import GeminiEmbedderClient

load_dotenv(verbose=True)

config = {
    "embedder": {
        "batch_size": 100,
        # Use custom GeminiEmbedderClient for cloud-based embeddings
        "model_client": lambda: GeminiEmbedderClient(api_key=os.getenv("GEMINI_API_KEY")),
        "model_kwargs": {
            "model": "models/text-embedding-004",
        },
        "dimensions": 768,
        "encoding_format": "float",
    },
    "retriever": {"top_k": 7},
    "generator": {
        "model_client": lambda: GroqAPIClient(api_key=os.getenv("GROQ_API_KEY")),
        "model_kwargs": {
            "model": "groq/compound",
            "temperature": 0.3,
            "stream": False,
        },
    },
    "text_splitter": {
        "split_by": "word",
        "chunk_size": 400,
        "chunk_overlap": 100,
    },
}
