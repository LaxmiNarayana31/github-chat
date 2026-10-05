"""Indexing job ORM model for GithubChat."""

from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Integer, String, Text
from backend.config.database import Base


class IndexingJob(Base):
    """Tracks asynchronous background repository ingestion jobs."""

    __tablename__ = "indexing_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(128), unique=True, nullable=False, index=True)
    repo_url = Column(String(512), nullable=False)
    status = Column(String(32), default="pending", nullable=False, index=True)
    progress = Column(Integer, default=0, nullable=False)
    stage = Column(String(64), default="queued", nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
