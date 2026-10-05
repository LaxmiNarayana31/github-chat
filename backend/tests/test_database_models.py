"""Tests for database models, connection engine, and session management."""

import uuid
import pytest
from sqlalchemy import text

from backend.config.database import (
    GUEST_MAX_REQUESTS,
    USER_DAILY_REQUEST_LIMIT,
    get_db,
    get_db_session,
    get_engine,
    init_db,
)
from backend.app.models.conversation_turn import ConversationTurn
from backend.app.models.indexing_job import IndexingJob
from backend.app.models.repository import Repository


def test_database_initialization():
    """Verifies database tables are created successfully without errors."""
    init_db()
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar()
        assert result == 1


def test_database_session_context_manager():
    """Tests inserting and querying records using the get_db_session context manager."""
    suffix = uuid.uuid4().hex[:8]
    slug = f"test-org_test-repo-{suffix}"
    url = f"https://github.com/test-org/test-repo-{suffix}"

    with get_db_session() as session:
        repo = Repository(
            repo_url=url,
            repo_slug=slug,
            collection_name=f"repo_{slug}",
            commit_sha="abc1234",
            is_indexed=True,
            total_chunks=42,
        )
        session.add(repo)

    # Query back in a new session to confirm auto-commit
    with get_db_session() as session:
        fetched = session.query(Repository).filter_by(repo_slug=slug).first()
        assert fetched is not None
        assert fetched.collection_name == f"repo_{slug}"
        assert fetched.total_chunks == 42


def test_database_models_crud():
    """Tests CRUD lifecycle across GithubChat ORM models."""
    suffix = uuid.uuid4().hex[:8]
    job_id = f"job-{suffix}"
    session_id = f"session-{suffix}"

    with get_db_session() as session:
        # IndexingJob model
        job = IndexingJob(
            job_id=job_id,
            repo_url=f"https://github.com/test/repo-{suffix}",
            status="completed",
            progress=100,
            stage="ready",
        )
        session.add(job)

        # ConversationTurn model
        turn = ConversationTurn(
            session_id=session_id,
            repo_slug=f"slug-{suffix}",
            user_query="How does indexing work?",
            assistant_answer="Indexing chunks the files and stores them in Qdrant.",
        )
        session.add(turn)

    # Verify querying across models
    with get_db_session() as session:
        job_res = session.query(IndexingJob).filter_by(job_id=job_id).first()
        assert job_res is not None
        assert job_res.status == "completed"

        turn_res = session.query(ConversationTurn).filter_by(session_id=session_id).first()
        assert turn_res is not None
        assert "indexing" in turn_res.user_query.lower()


def test_database_dependency_generator():
    """Tests FastAPI get_db session dependency generator."""
    db_gen = get_db()
    session = next(db_gen)
    assert session is not None
    try:
        res = session.execute(text("SELECT 1")).scalar()
        assert res == 1
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


def test_quota_constants():
    """Validates rate limit and quota thresholds."""
    assert GUEST_MAX_REQUESTS == 50
    assert USER_DAILY_REQUEST_LIMIT == 50
