from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    create_engine,
    delete,
    func,
    insert,
    select,
    text,
)
from sqlalchemy.engine import Engine

from models.workspace import WorkspaceState
from services.paper_identity_service import normalize_reader_lookup_url

DOCUMENT_KEY_GRAPH = "graph"
DOCUMENT_KEY_WORKSPACE = "workspace"

metadata = MetaData()

storage_documents = Table(
    "storage_documents",
    metadata,
    Column("document_key", String(64), primary_key=True),
    Column("payload", JSON, nullable=False),
    Column("payload_hash", String(64), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

papers = Table(
    "papers",
    metadata,
    Column("paper_id", String(128), primary_key=True),
    Column("paper_identity_key", String(255), nullable=False, unique=True),
    Column("paper_title", Text, nullable=False),
    Column("semantic_scholar_paper_id", String(255), nullable=True),
    Column("normalized_title", Text, nullable=False, default=""),
    Column("canonical_source_url", Text, nullable=False, default=""),
    Column("selected_primary_source", Text, nullable=True),
    Column("status", String(32), nullable=False, default="unread"),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

paper_links = Table(
    "paper_links",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("paper_id", String(128), ForeignKey("papers.paper_id", ondelete="CASCADE"), nullable=False),
    Column("raw_url", Text, nullable=False),
    Column("normalized_url", Text, nullable=False),
    Column("is_primary", Boolean, nullable=False, default=False),
)

paper_notes = Table(
    "paper_notes",
    metadata,
    Column("paper_id", String(128), ForeignKey("papers.paper_id", ondelete="CASCADE"), primary_key=True),
    Column("notes_markdown", Text, nullable=False, default=""),
    Column("topic_links", JSON, nullable=False),
    Column("status", String(32), nullable=False, default="unread"),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


@dataclass(frozen=True)
class PostgresParityReport:
    reading_items: int
    theme_notes: int
    annotations_v1: int
    annotations_v2: int
    annotation_archive: int
    graph_nodes: int
    graph_edges: int
    payload_hash_workspace: str
    payload_hash_graph: str


_engine: Engine | None = None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _json_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(canonical.encode("utf-8")).hexdigest()


def _is_postgres_url(url: str) -> bool:
    lowered = url.lower()
    return lowered.startswith("postgresql://") or lowered.startswith("postgres://")


def _database_url() -> str:
    raw = (os.environ.get("DATABASE_URL", "") or "").strip()
    if raw.startswith("postgresql://"):
        return raw.replace("postgresql://", "postgresql+psycopg://", 1)
    if raw.startswith("postgres://"):
        return raw.replace("postgres://", "postgresql+psycopg://", 1)
    return raw


def postgres_enabled() -> bool:
    return bool(_database_url())


def get_engine() -> Engine:
    global _engine
    if _engine is not None:
        return _engine
    url = _database_url()
    if not url:
        raise RuntimeError("DATABASE_URL is required for Postgres storage backend")
    _engine = create_engine(url, future=True, pool_pre_ping=True)
    return _engine


def reset_engine_for_tests() -> None:
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


def ensure_schema() -> None:
    engine = get_engine()
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_papers_semantic_scholar "
                "ON papers (semantic_scholar_paper_id) "
                "WHERE semantic_scholar_paper_id IS NOT NULL"
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_paper_links_normalized_url "
                "ON paper_links (normalized_url)"
            )
        )
        if _is_postgres_url(str(engine.url)):
            conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_paper_links_single_primary "
                    "ON paper_links (paper_id) WHERE is_primary = true"
                )
            )


def load_document(document_key: str) -> dict[str, Any] | None:
    ensure_schema()
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            select(storage_documents.c.payload).where(
                storage_documents.c.document_key == document_key
            )
        ).first()
        if row is None:
            return None
        return row[0]


def save_document(document_key: str, payload: dict[str, Any]) -> None:
    ensure_schema()
    engine = get_engine()
    payload_hash = _json_hash(payload)
    now = _now_utc()
    with engine.begin() as conn:
        existing = conn.execute(
            select(storage_documents.c.document_key).where(
                storage_documents.c.document_key == document_key
            )
        ).first()
        if existing:
            conn.execute(
                storage_documents.update()
                .where(storage_documents.c.document_key == document_key)
                .values(payload=payload, payload_hash=payload_hash, updated_at=now)
            )
        else:
            conn.execute(
                insert(storage_documents).values(
                    document_key=document_key,
                    payload=payload,
                    payload_hash=payload_hash,
                    updated_at=now,
                )
            )


def sync_workspace_projection(payload: dict[str, Any]) -> None:
    ensure_schema()
    state = WorkspaceState.model_validate(payload)
    now = _now_utc()
    claimed_normalized_urls: set[str] = set()
    with get_engine().begin() as conn:
        conn.execute(delete(paper_notes))
        conn.execute(delete(paper_links))
        conn.execute(delete(papers))

        for identity_key, annotation in state.paperAnnotationsV2.items():
            paper_id = identity_key
            created_at = annotation.createdAt or annotation.updatedAt or now.isoformat()
            updated_at = annotation.updatedAt or annotation.createdAt or now.isoformat()
            conn.execute(
                insert(papers).values(
                    paper_id=paper_id,
                    paper_identity_key=identity_key,
                    paper_title=(annotation.paperTitle or identity_key),
                    semantic_scholar_paper_id=annotation.semanticScholarPaperId,
                    normalized_title=annotation.normalizedTitle or "",
                    canonical_source_url=annotation.canonicalSourceUrl or "",
                    selected_primary_source=annotation.selectedPrimarySource,
                    status=annotation.status or "unread",
                    created_at=_parse_iso_dt(created_at, now),
                    updated_at=_parse_iso_dt(updated_at, now),
                )
            )
            conn.execute(
                insert(paper_notes).values(
                    paper_id=paper_id,
                    notes_markdown=annotation.notesMarkdown or "",
                    topic_links=annotation.topicLinks or [],
                    status=annotation.status or "unread",
                    updated_at=_parse_iso_dt(updated_at, now),
                )
            )
            aliases = list(annotation.sourceUrlAliases or [])
            canonical = normalize_reader_lookup_url(annotation.canonicalSourceUrl)
            if canonical and canonical not in aliases:
                aliases.insert(0, canonical)
            for index, alias in enumerate(aliases):
                normalized = normalize_reader_lookup_url(alias)
                if not normalized:
                    continue
                if normalized in claimed_normalized_urls:
                    continue
                conn.execute(
                    insert(paper_links).values(
                        paper_id=paper_id,
                        raw_url=alias,
                        normalized_url=normalized,
                        is_primary=index == 0,
                    )
                )
                claimed_normalized_urls.add(normalized)


def build_parity_report(
    workspace_payload: dict[str, Any] | None,
    graph_payload: dict[str, Any] | None,
) -> PostgresParityReport:
    workspace = workspace_payload or {}
    graph = graph_payload or {}
    return PostgresParityReport(
        reading_items=len(workspace.get("readingItems") or []),
        theme_notes=len(workspace.get("themeNotes") or []),
        annotations_v1=len(workspace.get("paperAnnotations") or {}),
        annotations_v2=len(workspace.get("paperAnnotationsV2") or {}),
        annotation_archive=len(workspace.get("annotationMigrationArchive") or []),
        graph_nodes=len(graph.get("nodes") or []),
        graph_edges=len(graph.get("edges") or []),
        payload_hash_workspace=_json_hash(workspace),
        payload_hash_graph=_json_hash(graph),
    )


def get_workspace_projection_counts() -> dict[str, int]:
    ensure_schema()
    with get_engine().begin() as conn:
        paper_count = conn.execute(select(func.count()).select_from(papers)).scalar_one()
        link_count = conn.execute(select(func.count()).select_from(paper_links)).scalar_one()
        note_count = conn.execute(select(func.count()).select_from(paper_notes)).scalar_one()
    return {
        "papers": int(paper_count),
        "paper_links": int(link_count),
        "paper_notes": int(note_count),
    }


def _parse_iso_dt(raw_value: str, fallback: datetime) -> datetime:
    try:
        return datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except Exception:
        return fallback
