from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with get_conn(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                agent_mode TEXT NOT NULL DEFAULT 'qa',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                evidence_json TEXT,
                model_used TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id)
                    REFERENCES conversations(conversation_id)
                    ON DELETE CASCADE
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
            ON messages(conversation_id, created_at)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversations_updated
            ON conversations(updated_at DESC)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS message_feedback (
                assistant_message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                feedback TEXT NOT NULL DEFAULT '',
                training_selected INTEGER NOT NULL DEFAULT 0
                    CHECK(training_selected IN (0, 1)),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(assistant_message_id)
                    REFERENCES messages(message_id)
                    ON DELETE CASCADE,
                FOREIGN KEY(conversation_id)
                    REFERENCES conversations(conversation_id)
                    ON DELETE CASCADE
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_message_feedback_training
            ON message_feedback(training_selected, rating, updated_at DESC)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_message_feedback_conversation
            ON message_feedback(conversation_id, updated_at DESC)
            """
        )


def create_conversation(
    db_path: Path,
    title: str = "新对话",
    agent_mode: str = "qa",
) -> dict[str, Any]:
    conversation_id = str(uuid.uuid4())
    now = now_iso()

    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO conversations (
                conversation_id, title, agent_mode, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, title, agent_mode, now, now),
        )

    return {
        "conversation_id": conversation_id,
        "title": title,
        "agent_mode": agent_mode,
        "created_at": now,
        "updated_at": now,
    }


def list_conversations(
    db_path: Path,
    limit: int = 100,
) -> list[dict[str, Any]]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT conversation_id, title, agent_mode, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def get_conversation(
    db_path: Path,
    conversation_id: str,
) -> dict[str, Any] | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            """
            SELECT conversation_id, title, agent_mode, created_at, updated_at
            FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()

    return dict(row) if row else None


def get_messages(
    db_path: Path,
    conversation_id: str,
) -> list[dict[str, Any]]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                m.message_id,
                m.conversation_id,
                m.role,
                m.content,
                m.evidence_json,
                m.model_used,
                m.created_at,
                f.rating AS feedback_rating,
                f.feedback AS feedback_text,
                f.training_selected AS feedback_training_selected,
                f.created_at AS feedback_created_at,
                f.updated_at AS feedback_updated_at
            FROM messages AS m
            LEFT JOIN message_feedback AS f
                ON m.message_id = f.assistant_message_id
            WHERE m.conversation_id = ?
            ORDER BY m.created_at ASC
            """,
            (conversation_id,),
        ).fetchall()
    messages: list[dict[str, Any]] = []

    for row in rows:
        item = dict(row)

        try:
            item["evidence"] = (
                json.loads(item["evidence_json"])
                if item["evidence_json"]
                else []
            )
        except json.JSONDecodeError:
            item["evidence"] = []

        item.pop("evidence_json", None)

        if item["role"] == "assistant" and item["feedback_rating"] is not None:
            item["quality_feedback"] = {
                "assistant_message_id": item["message_id"],
                "conversation_id": item["conversation_id"],
                "rating": item["feedback_rating"],
                "feedback": item["feedback_text"] or "",
                "training_selected": bool(
                    item["feedback_training_selected"]
                ),
                "created_at": item["feedback_created_at"],
                "updated_at": item["feedback_updated_at"],
            }
        else:
            item["quality_feedback"] = None

        item.pop("feedback_rating", None)
        item.pop("feedback_text", None)
        item.pop("feedback_training_selected", None)
        item.pop("feedback_created_at", None)
        item.pop("feedback_updated_at", None)

        messages.append(item)

    return messages


def add_message(
    db_path: Path,
    conversation_id: str,
    role: str,
    content: str,
    evidence: list[dict[str, Any]] | None = None,
    model_used: str | None = None,
) -> dict[str, Any]:
    if role not in {"user", "assistant"}:
        raise ValueError("role 必须是 user 或 assistant")

    message_id = str(uuid.uuid4())
    now = now_iso()
    evidence_json = json.dumps(evidence or [], ensure_ascii=False)

    with get_conn(db_path) as conn:
        exists = conn.execute(
            """
            SELECT 1
            FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()

        if not exists:
            raise ValueError(f"会话不存在：{conversation_id}")

        conn.execute(
            """
            INSERT INTO messages (
                message_id,
                conversation_id,
                role,
                content,
                evidence_json,
                model_used,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                conversation_id,
                role,
                content,
                evidence_json,
                model_used,
                now,
            ),
        )

        conn.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE conversation_id = ?
            """,
            (now, conversation_id),
        )

    return {
        "message_id": message_id,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "evidence": evidence or [],
        "model_used": model_used,
        "created_at": now,
    }


def update_conversation(
    db_path: Path,
    conversation_id: str,
    title: str | None = None,
    agent_mode: str | None = None,
) -> dict[str, Any] | None:
    current = get_conversation(db_path, conversation_id)

    if current is None:
        return None

    new_title = title if title is not None else current["title"]
    new_agent_mode = (
        agent_mode if agent_mode is not None else current["agent_mode"]
    )
    now = now_iso()

    with get_conn(db_path) as conn:
        conn.execute(
            """
            UPDATE conversations
            SET title = ?, agent_mode = ?, updated_at = ?
            WHERE conversation_id = ?
            """,
            (new_title, new_agent_mode, now, conversation_id),
        )

    return get_conversation(db_path, conversation_id)


def delete_conversation(
    db_path: Path,
    conversation_id: str,
) -> bool:
    with get_conn(db_path) as conn:
        result = conn.execute(
            """
            DELETE FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        )

    return result.rowcount > 0

def get_message_feedback(
    db_path: Path,
    assistant_message_id: str,
) -> dict[str, Any] | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                assistant_message_id,
                conversation_id,
                rating,
                feedback,
                training_selected,
                created_at,
                updated_at
            FROM message_feedback
            WHERE assistant_message_id = ?
            """,
            (assistant_message_id,),
        ).fetchone()

    if row is None:
        return None

    item = dict(row)
    item["training_selected"] = bool(item["training_selected"])
    return item


def save_message_feedback(
    db_path: Path,
    assistant_message_id: str,
    rating: int,
    feedback: str = "",
    training_selected: bool = False,
) -> dict[str, Any]:
    if not 1 <= rating <= 5:
        raise ValueError("rating 必须在 1 到 5 之间。")

    with get_conn(db_path) as conn:
        message = conn.execute(
            """
            SELECT message_id, conversation_id, role
            FROM messages
            WHERE message_id = ?
            """,
            (assistant_message_id,),
        ).fetchone()

        if message is None:
            raise ValueError("目标消息不存在。")

        if message["role"] != "assistant":
            raise ValueError("只能对 assistant 消息保存评分。")

        now = now_iso()

        conn.execute(
            """
            INSERT INTO message_feedback (
                assistant_message_id,
                conversation_id,
                rating,
                feedback,
                training_selected,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(assistant_message_id) DO UPDATE SET
                rating = excluded.rating,
                feedback = excluded.feedback,
                training_selected = excluded.training_selected,
                updated_at = excluded.updated_at
            """,
            (
                assistant_message_id,
                message["conversation_id"],
                rating,
                feedback.strip(),
                int(training_selected),
                now,
                now,
            ),
        )

    result = get_message_feedback(db_path, assistant_message_id)

    if result is None:
        raise RuntimeError("评分保存后读取失败。")

    return result


def list_selected_training_samples(
    db_path: Path,
    min_rating: int = 4,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                f.assistant_message_id,
                f.conversation_id,
                f.rating,
                f.feedback,
                f.training_selected,
                f.created_at AS feedback_created_at,
                f.updated_at AS feedback_updated_at,
                c.title AS conversation_title,
                c.agent_mode,
                m.content AS assistant_answer,
                m.model_used
            FROM message_feedback AS f
            INNER JOIN messages AS m
                ON f.assistant_message_id = m.message_id
            INNER JOIN conversations AS c
                ON f.conversation_id = c.conversation_id
            WHERE f.training_selected = 1
              AND f.rating >= ?
            ORDER BY f.updated_at DESC
            LIMIT ?
            """,
            (min_rating, limit),
        ).fetchall()

    rows_as_dict = []

    for row in rows:
        item = dict(row)
        item["training_selected"] = bool(item["training_selected"])
        rows_as_dict.append(item)

    return rows_as_dict
