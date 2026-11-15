"""Indexing job ORM model for GithubChat."""

from datetime import UTC, datetime

from backend.config.database import Base
from sqlalchemy import Column, DateTime, Index, Integer, String, Text


class IndexingJob(Base):
    """Tracks asynchronous background repository ingestion jobs."""

    __tablename__ = "indexing_jobs"
    __table_args__ = (Index("ix_indexing_jobs_status_created", "status", "created_at"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(128), unique=True, nullable=False, index=True)
    repo_url = Column(String(512), nullable=False)
    status = Column(String(32), default="pending", nullable=False, index=True)
    progress = Column(Integer, default=0, nullable=False)
    stage = Column(String(64), default="queued", nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )
