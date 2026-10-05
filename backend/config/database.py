"""PostgreSQL database engine configuration, schema initialization, and session management."""

from contextlib import contextmanager
import logging
import os
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

log = logging.getLogger(__name__)

load_dotenv()

# Quota and rate limit settings
GUEST_MAX_REQUESTS: int = 50
USER_DAILY_REQUEST_LIMIT: int = 50

# Shared SQLAlchemy declarative base
Base = declarative_base()

# Engine and sessionmaker singletons
engine = None
SessionLocal = None


def build_database_url() -> str:
    """Builds PostgreSQL connection string from environment variables."""
    try:
        direct_url = os.getenv("DATABASE_URL")
        if direct_url and direct_url.strip():
            return direct_url.strip()

        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME")
        ssl_mode = os.getenv("SSL_MODE", "require")

        if db_host and db_name:
            auth = f"{db_user}:{db_password}@" if db_user else ""
            port = f":{db_port}" if db_port else ""
            ssl = f"?sslmode={ssl_mode}" if ssl_mode else ""
            return f"postgresql://{auth}{db_host}{port}/{db_name}{ssl}"

        return ""
    except Exception as e:
        log.error(f"Error building PostgreSQL database URL: {e}", exc_info=True)
        return ""


DATABASE_URL: str = build_database_url()


def get_engine():
    """Initializes and returns SQLAlchemy PostgreSQL database engine."""
    global engine, SessionLocal
    try:
        if engine is not None:
            return engine

        db_url = build_database_url()
        if not db_url:
            raise ValueError(
                "PostgreSQL database is not configured. Please set DB_USER, DB_PASSWORD, DB_HOST, and DB_NAME in backend/.env"
            )

        engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=5,
            max_overflow=10,
            connect_args={"connect_timeout": 10},
        )

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log.info("Connected to PostgreSQL database successfully.")

        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return engine
    except Exception as e:
        log.error(f"PostgreSQL database connection error: {e}", exc_info=True)
        raise


def init_db():
    """Initializes database tables on application startup."""
    try:
        active_engine = get_engine()
        Base.metadata.create_all(bind=active_engine)
        log.info("Database tables initialized successfully on PostgreSQL.")
    except Exception as e:
        log.error(f"Database table initialization failed on PostgreSQL: {e}", exc_info=True)
        raise


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for thread-safe PostgreSQL database transactions."""
    if SessionLocal is None:
        get_engine()
    session: Session = SessionLocal()  # type: ignore
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a managed PostgreSQL database session."""
    if SessionLocal is None:
        get_engine()
    session: Session = SessionLocal()  # type: ignore
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
