"""Repository ORM model for GithubChat."""

from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Integer, String
from backend.config.database import Base


class Repository(Base):
    """Tracks indexed repositories and their collection status."""

    __tablename__ = "repositories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repo_url = Column(String(512), unique=True, nullable=False, index=True)
    repo_slug = Column(String(256), unique=True, nullable=False, index=True)
    collection_name = Column(String(256), nullable=False)
    commit_sha = Column(String(64), nullable=True)
    is_indexed = Column(Boolean, default=False, nullable=False)
    total_chunks = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
