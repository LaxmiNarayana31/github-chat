import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple
import uuid

from adalflow.utils import printc
from memori import Memori
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from backend.app.config.config import config
from backend.app.utils.env_utils import build_postgres_url

log = logging.getLogger(__name__)


class MemoriManager:
    """
    Manages Memori Labs SQL-native memory infrastructure:
    - Connects to PostgreSQL database for enterprise persistent stateful memory.
    - Gracefully falls back to local SQLite if PostgreSQL is unconfigured or unreachable.
    - Stores and retrieves conversational history (dialog turns) in SQL.
    - Stores and recalls semantic facts and knowledge graph associations.
    - Supports LLM registration and attribution tracking.
    """

    _instance: Optional['MemoriManager'] = None

    def __init__(
        self,
        connection_string: Optional[str] = None,
        entity_id: Optional[str] = None,
        process_id: Optional[str] = None,
    ):
        mem_cfg = config.get("memori", {})
        self.enabled = mem_cfg.get("enabled", True)
        self.connection_string = connection_string or mem_cfg.get("connection_string", "") or build_postgres_url()
        self.entity_id = entity_id or mem_cfg.get("entity_id", "default_user")
        self.process_id = process_id or mem_cfg.get("process_id", "github-chat")

        self.engine = None
        self.Session = None
        self.memori: Optional[Memori] = None
        self.is_postgres = False

        if self.enabled:
            self._init_storage()

    def _init_storage(self):
        """Initialize database connection (PostgreSQL with graceful SQLite fallback) and build Memori schema."""
        conn_str = self.connection_string.strip() if self.connection_string else ""

        # Check if PostgreSQL connection string provided
        if conn_str and (conn_str.startswith("postgresql://") or conn_str.startswith("postgres://")):
            if conn_str.startswith("postgres://"):
                conn_str = "postgresql://" + conn_str[len("postgres://") :]

            try:
                printc(f"MemoriManager: Connecting to PostgreSQL database...", color="blue")
                test_engine = create_engine(conn_str, pool_pre_ping=True)
                with test_engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                self.engine = test_engine
                self.is_postgres = True
                printc("MemoriManager: Connected to PostgreSQL successfully.", color="green")
            except Exception as e:
                log.warning(
                    f"MemoriManager: Could not connect to PostgreSQL ({e}). "
                    "Falling back to in-memory transient storage (no disk files created)."
                )
                self.engine = None
                self.is_postgres = False

        if self.engine is None:
            # If a custom connection string was provided (e.g. for unit tests), use it
            if conn_str and not (conn_str.startswith("postgresql://") or conn_str.startswith("postgres://")):
                try:
                    self.engine = create_engine(conn_str)
                    self.is_postgres = False
                except Exception as e:
                    log.error(f"MemoriManager: Error creating database engine for {conn_str}: {e}")
                    self.engine = None

            # If still None, use in-memory SQLite to avoid creating any files on disk
            if self.engine is None:
                printc(
                    "MemoriManager: PostgreSQL not connected. Using in-memory transient storage (no disk files created).",
                    color="yellow",
                )
                self.engine = create_engine("sqlite:///:memory:")
                self.is_postgres = False

        self.Session = sessionmaker(bind=self.engine)

        try:
            self.memori = Memori(conn=self.Session)
            self.memori.attribution(entity_id=self.entity_id, process_id=self.process_id)
            if self.memori and getattr(self.memori, "config", None):
                storage_cfg = getattr(self.memori.config, "storage", None)
                if storage_cfg and hasattr(storage_cfg, "build"):
                    storage_cfg.build()
            printc(
                f"MemoriManager: Memori Labs schema built successfully (Backend: {'PostgreSQL' if self.is_postgres else 'SQLite'}).",
                color="green",
            )
        except Exception as e:
            log.error(f"MemoriManager: Failed to build Memori storage: {e}")
            self.memori = None

    def register_llm(self, client: Any) -> Any:
        """Register LLM client (Groq, OpenAI, Google) with Memori for automatic tracking."""
        if not self.enabled or self.memori is None:
            return client

        try:
            self.memori.llm.register(client)
            printc(f"MemoriManager: Registered LLM client ({type(client).__name__}) with Memori.", color="green")
        except Exception as e:
            log.debug(f"MemoriManager: LLM client registration note: {e}")
        return client

    def capture_dialog_turn(
        self,
        user_query: str,
        assistant_response: str,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Record conversation turn (user + assistant) persistently in Memori SQL database."""
        if not self.enabled or self.engine is None:
            return

        active_session = session_id or self.entity_id
        now = datetime.datetime.now(datetime.timezone.utc)

        try:
            with self.engine.begin() as conn:
                # Ensure entity exists
                entity_res = conn.execute(
                    text("SELECT id FROM memori_entity WHERE external_id = :eid"),
                    {"eid": self.entity_id},
                ).fetchone()
                if entity_res is not None:
                    entity_db_id = entity_res[0]
                else:
                    entity_uuid = str(uuid.uuid4())
                    insert_res = conn.execute(
                        text("INSERT INTO memori_entity (uuid, external_id, date_created, date_updated) "
                             "VALUES (:u, :eid, :dc, :du)"),
                        {"u": entity_uuid, "eid": self.entity_id, "dc": now, "du": now},
                    )
                    entity_res = conn.execute(
                        text("SELECT id FROM memori_entity WHERE external_id = :eid"),
                        {"eid": self.entity_id},
                    ).fetchone()
                    entity_db_id = entity_res[0] if entity_res is not None else 1

                # Ensure process exists
                process_res = conn.execute(
                    text("SELECT id FROM memori_process WHERE external_id = :pid"),
                    {"pid": self.process_id},
                ).fetchone()
                if process_res is not None:
                    process_db_id = process_res[0]
                else:
                    proc_uuid = str(uuid.uuid4())
                    conn.execute(
                        text("INSERT INTO memori_process (uuid, external_id, date_created, date_updated) "
                             "VALUES (:u, :pid, :dc, :du)"),
                        {"u": proc_uuid, "pid": self.process_id, "dc": now, "du": now},
                    )
                    process_res = conn.execute(
                        text("SELECT id FROM memori_process WHERE external_id = :pid"),
                        {"pid": self.process_id},
                    ).fetchone()
                    process_db_id = process_res[0] if process_res is not None else 1

                # Ensure session exists (using deterministic UUID from active_session)
                sess_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, active_session))
                session_res = conn.execute(
                    text("SELECT id FROM memori_session WHERE uuid = :sid"),
                    {"sid": sess_uuid},
                ).fetchone()
                if session_res is not None:
                    session_db_id = session_res[0]
                else:
                    conn.execute(
                        text("INSERT INTO memori_session (uuid, entity_id, process_id, date_created, date_updated) "
                             "VALUES (:u, :eid, :pid, :dc, :du)"),
                        {"u": sess_uuid, "eid": entity_db_id, "pid": process_db_id, "dc": now, "du": now},
                    )
                    session_res = conn.execute(
                        text("SELECT id FROM memori_session WHERE uuid = :u"),
                        {"u": sess_uuid},
                    ).fetchone()
                    session_db_id = session_res[0] if session_res is not None else 1

                # Ensure conversation exists for session
                conv_res = conn.execute(
                    text("SELECT id FROM memori_conversation WHERE session_id = :sid ORDER BY id DESC LIMIT 1"),
                    {"sid": session_db_id},
                ).fetchone()
                if conv_res is not None:
                    conv_id = conv_res[0]
                else:
                    conv_uuid = str(uuid.uuid4())
                    conn.execute(
                        text("INSERT INTO memori_conversation (uuid, session_id, summary, date_created, date_updated) "
                             "VALUES (:u, :sid, :summary, :dc, :du)"),
                        {"u": conv_uuid, "sid": session_db_id, "summary": f"Conversation {active_session}", "dc": now, "du": now},
                    )
                    conv_res = conn.execute(
                        text("SELECT id FROM memori_conversation WHERE uuid = :u"),
                        {"u": conv_uuid},
                    ).fetchone()
                    conv_id = conv_res[0] if conv_res is not None else 1

                # Insert user message
                u_uuid = str(uuid.uuid4())
                conn.execute(
                    text("INSERT INTO memori_conversation_message (uuid, conversation_id, role, type, content, date_created, date_updated) "
                         "VALUES (:u, :cid, :role, :type, :content, :dc, :du)"),
                    {"u": u_uuid, "cid": conv_id, "role": "user", "type": "text", "content": user_query, "dc": now, "du": now},
                )

                # Insert assistant message
                a_uuid = str(uuid.uuid4())
                conn.execute(
                    text("INSERT INTO memori_conversation_message (uuid, conversation_id, role, type, content, date_created, date_updated) "
                         "VALUES (:u, :cid, :role, :type, :content, :dc, :du)"),
                    {"u": a_uuid, "cid": conv_id, "role": "assistant", "type": "text", "content": assistant_response, "dc": now, "du": now},
                )

            printc(f"MemoriManager: Saved dialog turn to SQL memory (Session: {active_session})", color="green")
        except Exception as e:
            log.warning(f"MemoriManager: Error capturing turn to SQL: {e}")

    def record_turn(
        self,
        user_query: str,
        assistant_response: str,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Alias for capture_dialog_turn for API compatibility."""
        return self.capture_dialog_turn(user_query, assistant_response, session_id=session_id, model=model)

    def get_conversation_turns(self, session_id: Optional[str] = None, limit: int = 20) -> List[Tuple[str, str]]:
        """Retrieve historical dialog turns [(user_query, assistant_response)] from SQL storage."""
        if not self.enabled or self.engine is None:
            return []

        active_session = session_id or self.entity_id
        sess_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, active_session))
        turns: List[Tuple[str, str]] = []

        try:
            with self.engine.connect() as conn:
                rows = conn.execute(
                    text(
                        "SELECT m.role, m.content FROM memori_conversation_message m "
                        "JOIN memori_conversation c ON m.conversation_id = c.id "
                        "JOIN memori_session s ON c.session_id = s.id "
                        "WHERE s.uuid = :sid OR s.uuid = :raw_sid "
                        "ORDER BY m.id ASC"
                    ),
                    {"sid": sess_uuid, "raw_sid": active_session},
                ).fetchall()

                current_user = None
                for role, content in rows:
                    if role == "user":
                        current_user = content
                    elif role == "assistant" and current_user is not None:
                        turns.append((current_user, content))
                        current_user = None

            return turns[-limit:]
        except Exception as e:
            log.warning(f"MemoriManager: Error getting conversation turns: {e}")
            return []

    def recall_memories(self, query: str, limit: int = 5) -> List[str]:
        """Retrieve relevant semantic memories (facts/summaries) from Memori."""
        if not self.enabled:
            return []

        memories: List[str] = []

        # Query Memori high-level recall if available
        if self.memori is not None:
            try:
                results = self.memori.recall(query=query, limit=limit)
                if results:
                    for item in results:
                        if isinstance(item, str):
                            memories.append(item)
                        elif isinstance(item, dict):
                            if "fact" in item and item["fact"]:
                                memories.append(str(item["fact"]))
                            elif "text" in item and item["text"]:
                                memories.append(str(item["text"]))
                        else:
                            fact_val = getattr(item, "fact", None)
                            text_val = getattr(item, "text", None)
                            if fact_val is not None:
                                memories.append(str(fact_val))
                            elif text_val is not None:
                                memories.append(str(text_val))
            except Exception as e:
                log.debug(f"Memori recall notice: {e}")

        # Query SQL facts table directly
        if self.engine is not None and len(memories) < limit:
            try:
                with self.engine.connect() as conn:
                    # Simple keyword or substring match across facts
                    tokens = [t.strip().lower() for t in query.split() if len(t.strip()) > 3]
                    if tokens:
                        pattern = f"%{tokens[0]}%"
                        rows = conn.execute(
                            text("SELECT content FROM memori_entity_fact WHERE LOWER(content) LIKE :q LIMIT :lim"),
                            {"q": pattern, "lim": limit},
                        ).fetchall()
                        for r in rows:
                            if r[0] not in memories:
                                memories.append(r[0])
            except Exception as e:
                log.debug(f"Memori SQL fact lookup notice: {e}")

        # Query SQL conversation messages for historical contextual match
        if self.engine is not None and len(memories) < limit:
            try:
                with self.engine.connect() as conn:
                    tokens = [t.strip().lower() for t in query.split() if len(t.strip()) > 3]
                    if tokens:
                        pattern = f"%{tokens[0]}%"
                        rows = conn.execute(
                            text(
                                "SELECT content FROM memori_conversation_message "
                                "WHERE role = 'assistant' AND LOWER(content) LIKE :q "
                                "ORDER BY id DESC LIMIT :lim"
                            ),
                            {"q": pattern, "lim": limit - len(memories)},
                        ).fetchall()
                        for r in rows:
                            msg_snippet = str(r[0])[:300].strip()
                            if msg_snippet and msg_snippet not in memories:
                                memories.append(f"[Past Conversation Match]: {msg_snippet}")
            except Exception as e:
                log.debug(f"Memori SQL message history lookup notice: {e}")

        if memories:
            printc(f"MemoriManager: Recalled {len(memories)} semantic memories for query.", color="green")
        return memories

    def search_conversation_history(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, str]]:
        """Search past conversation messages in SQL database by keyword."""
        if not self.enabled or self.engine is None:
            return []
        tokens = [t.strip().lower() for t in query.split() if len(t.strip()) > 2]
        if not tokens:
            return []
        pattern = f"%{tokens[0]}%"
        results: List[Dict[str, str]] = []
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(
                    text(
                        "SELECT m.role, m.content, m.date_created FROM memori_conversation_message m "
                        "WHERE LOWER(m.content) LIKE :q "
                        "ORDER BY m.id DESC LIMIT :lim"
                    ),
                    {"q": pattern, "lim": limit},
                ).fetchall()
                for role, content, date_created in rows:
                    results.append({
                        "role": str(role),
                        "content": str(content),
                        "date_created": str(date_created),
                    })
        except Exception as e:
            log.warning(f"MemoriManager: Error searching conversation history: {e}")
        return results


    def clear_memory(self, session_id: Optional[str] = None) -> bool:
        """Clear conversation and messages for the session/entity."""
        if not self.enabled or self.engine is None:
            return True

        active_session = session_id or self.entity_id
        sess_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, active_session))
        try:
            with self.engine.begin() as conn:
                # Delete messages for conversations in this session
                conn.execute(
                    text(
                        "DELETE FROM memori_conversation_message WHERE conversation_id IN ("
                        "SELECT c.id FROM memori_conversation c JOIN memori_session s ON c.session_id = s.id "
                        "WHERE s.uuid = :sid OR s.uuid = :raw_sid)"
                    ),
                    {"sid": sess_uuid, "raw_sid": active_session},
                )
                conn.execute(
                    text(
                        "DELETE FROM memori_conversation WHERE session_id IN ("
                        "SELECT s.id FROM memori_session s WHERE s.uuid = :sid OR s.uuid = :raw_sid)"
                    ),
                    {"sid": sess_uuid, "raw_sid": active_session},
                )
            printc(f"MemoriManager: Cleared memory for session '{active_session}'.", color="yellow")
            return True
        except Exception as e:
            log.warning(f"MemoriManager: Error clearing memory: {e}")
            return False

    def close(self) -> None:
        """Dispose SQLAlchemy engine and release all pooled database connections."""
        try:
            if self.engine is not None:
                self.engine.dispose()
        except Exception as e:
            log.warning(f"MemoriManager: Error disposing engine: {e}")
