import os
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file if available, otherwise use system environment variables."""
    try:
        # Try to load from .env file for local development
        load_dotenv(verbose=True)
        print("Loaded environment variables from .env file")
    except FileNotFoundError:
        # In production, env variables should be set in the environment
        print("No .env file found, using system environment variables")
    except Exception as e:
        print(f"Note: Error loading .env file: {e}")

# Load environment variables
load_environment()

# Check for required environment variables
gemini_api_key = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
if not gemini_api_key or not groq_api_key:
    raise ValueError(
        "GEMINI_API_KEY and GROQ_API_KEY environment variables are required. Either:\n"
        "1. Create a .env file with GEMINI_API_KEY=your-key-here and GROQ_API_KEY=your-key-here (for local development)\n"
        "2. Set the environment variable in your deployment platform (for production)"
    )