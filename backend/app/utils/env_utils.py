import os
import logging
from typing import Dict
from dotenv import load_dotenv

log = logging.getLogger(__name__)


def load_environment() -> None:
    """Safely load environment variables from backend/.env and root/.env."""
    try:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(this_dir)
        backend_env = os.path.join(backend_dir, ".env")
        if os.path.exists(backend_env):
            load_dotenv(backend_env, override=False)

        root_dir = os.path.dirname(backend_dir)
        root_env = os.path.join(root_dir, ".env")
        if os.path.exists(root_env):
            load_dotenv(root_env, override=False)

        load_dotenv(verbose=False)
    except (OSError, UnicodeDecodeError) as e:
        log.warning(f"Error reading .env file: {e}")
    except Exception as e:
        log.warning(f"Unexpected error loading environment: {e}")


def check_api_keys() -> Dict[str, bool]:
    """Check availability of required third-party API keys without raising exceptions."""
    try:
        load_environment()
        return {
            "gemini": bool(os.getenv("GEMINI_API_KEY")),
            "groq": bool(os.getenv("GROQ_API_KEY")),
        }
    except Exception as e:
        log.error(f"Error checking API keys: {e}")
        return {
            "gemini": False,
            "groq": False,
        }


def ensure_api_keys_configured() -> None:
    """Verify that both GEMINI_API_KEY and GROQ_API_KEY are configured.
    
    Raises:
        ValueError: If either key is missing.
    """
    try:
        keys = check_api_keys()
        missing = [k.upper() + "_API_KEY" for k, configured in keys.items() if not configured]
        if missing:
            raise ValueError(
                f"Missing required API key(s): {', '.join(missing)}. "
                "Please configure them in your .env file or environment variables."
            )
    except ValueError:
        raise
    except Exception as e:
        log.error(f"Error verifying API key configuration: {e}")
        raise ValueError(f"Could not verify API keys: {e}") from e


def build_postgres_url() -> str:
    """Build PostgreSQL connection URL from environment variables or direct URI."""
    try:
        load_environment()
        direct_url = os.getenv("DATABASE_URL", "").strip() or os.getenv("POSTGRES_URL", "").strip()
        if direct_url:
            if direct_url.startswith("postgres://"):
                direct_url = "postgresql://" + direct_url[len("postgres://") :]
            return direct_url

        db_host = os.getenv("DB_HOST", "").strip()
        if db_host:
            db_user = os.getenv("DB_USER", "").strip()
            db_password = os.getenv("DB_PASSWORD", "").strip()
            db_port = os.getenv("DB_PORT", "5432").strip() or "5432"
            db_name = os.getenv("DB_NAME", "defaultdb").strip() or "defaultdb"
            ssl_mode = os.getenv("SSL_MODE", "require").strip() or "require"

            auth = f"{db_user}:{db_password}@" if (db_user or db_password) else ""
            return f"postgresql://{auth}{db_host}:{db_port}/{db_name}?sslmode={ssl_mode}"

        return ""
    except Exception as e:
        log.warning(f"Error constructing PostgreSQL connection URL: {e}")
        return ""
