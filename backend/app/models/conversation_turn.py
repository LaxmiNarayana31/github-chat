"""Conversation turn ORM model for GithubChat."""

from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Integer, String, Text
from backend.config.database import Base


class ConversationTurn(Base):
    """Stores user queries and assistant responses per session."""

    __tablename__ = "conversation_turns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(128), nullable=False, index=True)
    repo_slug = Column(String(256), nullable=False, index=True)
    user_query = Column(Text, nullable=False)
    assistant_answer = Column(Text, nullable=False)
    rationale = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
