from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

from models.graph import PaperGraph
from models.paper import Paper
from scripts.migrate_to_postgres import run_migration
from services.postgres_storage_service import (
    get_workspace_projection_counts,
    reset_engine_for_tests,
    sync_workspace_projection,
)
from services.storage_service import (
    load_graph,
    load_workspace_state,
    save_graph,
    save_graph_legacy,
    save_workspace_state,
    save_workspace_state_legacy,
)


def _workspace_payload() -> dict:
    return {
        "workspaceSchemaVersion": 2,
        "readingItems": [],
        "themeNotes": [],
        "paperAnnotations": {
            "Paper A": {
                "paperTitle": "Paper A",
                "notesMarkdown": "legacy note",
                "sourceUrl": "https://example.org/paper-a",
                "topicLinks": [],
                "status": "unread",
                "updatedAt": "2025-01-01T00:00:00Z",
            }
        },
        "paperAnnotationsV2": {
            "ssid:paper-a": {
                "paperIdentityKey": "ssid:paper-a",
                "paperTitle": "Paper A",
                "semanticScholarPaperId": "paper-a",
                "normalizedTitle": "paper a",
                "canonicalSourceUrl": "https://example.org/paper-a",
                "sourceUrlAliases": ["https://example.org/paper-a"],
                "notesMarkdown": "canonical note",
                "topicLinks": [],
                "status": "unread",
                "migratedFromKeys": ["Paper A"],
                "selectedPrimarySource": "Paper A",
                "createdAt": "2025-01-01T00:00:00Z",
                "updatedAt": "2025-01-01T00:00:00Z",
            }
        },
        "annotationMigrationArchive": [],
    }


@pytest.fixture
def postgres_sqlite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    db_path = tmp_path / "workspace.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    reset_engine_for_tests()
    yield db_path
    reset_engine_for_tests()


def test_storage_service_uses_sql_backend_for_workspace_and_graph(postgres_sqlite):
    state = _workspace_payload()
    save_workspace_state(state)
    loaded_state = load_workspace_state()
    assert loaded_state == state

    graph = PaperGraph()
    graph.add_paper(Paper(title="Paper A", file_path="a.pdf", topics=["Topic 1"]))
    save_graph(graph)
    loaded_graph = load_graph()
    assert loaded_graph is not None
    assert "Paper A" in loaded_graph.graph.nodes

    projection_counts = get_workspace_projection_counts()
    assert projection_counts["papers"] == 1
    assert projection_counts["paper_links"] == 1
    assert projection_counts["paper_notes"] == 1


def test_projection_rejects_duplicate_normalized_links(postgres_sqlite):
    payload = _workspace_payload()
    payload["paperAnnotationsV2"]["ssid:paper-b"] = {
        **payload["paperAnnotationsV2"]["ssid:paper-a"],
        "paperIdentityKey": "ssid:paper-b",
        "paperTitle": "Paper B",
        "semanticScholarPaperId": "paper-b",
        "sourceUrlAliases": ["https://example.org/paper-a#section"],
        "canonicalSourceUrl": "https://example.org/paper-a#section",
    }
    with pytest.raises(IntegrityError):
        sync_workspace_projection(payload)


def test_one_shot_migration_imports_legacy_payloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    legacy_data_dir = tmp_path / "legacy_data"
    monkeypatch.setenv("LOCAL_DATA_DIR", str(legacy_data_dir))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_engine_for_tests()

    legacy_workspace = _workspace_payload()
    save_workspace_state_legacy(legacy_workspace)
    graph = PaperGraph()
    graph.add_paper(Paper(title="Legacy Paper", file_path="legacy.pdf", topics=["T"]))
    save_graph_legacy(graph)

    sqlite_db = tmp_path / "target.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{sqlite_db}")
    reset_engine_for_tests()

    result = run_migration()
    assert result["status"] == "ok"
    assert result["source"]["payload_hash_workspace"] == result["target"]["payload_hash_workspace"]
    assert result["source"]["payload_hash_graph"] == result["target"]["payload_hash_graph"]
