from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.graph_serialization import serialize_graph
from services.postgres_storage_service import (
    DOCUMENT_KEY_GRAPH,
    DOCUMENT_KEY_WORKSPACE,
    build_parity_report,
    get_workspace_projection_counts,
    load_document,
    save_document,
    sync_workspace_projection,
)
from services.storage_service import load_graph_legacy, load_workspace_state_legacy


def _load_legacy_payloads() -> tuple[dict, dict]:
    workspace_payload = load_workspace_state_legacy() or {}
    graph_obj = load_graph_legacy()
    graph_payload = serialize_graph(graph_obj) if graph_obj is not None else {}
    return workspace_payload, graph_payload


def _assert_hash_match(before_hash: str, after_hash: str, name: str) -> None:
    if before_hash != after_hash:
        raise RuntimeError(
            f"{name} hash mismatch after migration: source={before_hash} target={after_hash}"
        )


def run_migration(output_path: str | None = None) -> dict:
    if not os.environ.get("DATABASE_URL", "").strip():
        raise RuntimeError("DATABASE_URL must be set for Postgres migration")

    workspace_payload, graph_payload = _load_legacy_payloads()
    source_report = build_parity_report(workspace_payload, graph_payload)

    save_document(DOCUMENT_KEY_WORKSPACE, workspace_payload)
    sync_workspace_projection(workspace_payload)
    save_document(DOCUMENT_KEY_GRAPH, graph_payload)

    migrated_workspace = load_document(DOCUMENT_KEY_WORKSPACE) or {}
    migrated_graph = load_document(DOCUMENT_KEY_GRAPH) or {}
    target_report = build_parity_report(migrated_workspace, migrated_graph)
    projection_counts = get_workspace_projection_counts()

    _assert_hash_match(
        source_report.payload_hash_workspace,
        target_report.payload_hash_workspace,
        "workspace",
    )
    _assert_hash_match(
        source_report.payload_hash_graph,
        target_report.payload_hash_graph,
        "graph",
    )

    result = {
        "status": "ok",
        "source": source_report.__dict__,
        "target": target_report.__dict__,
        "projection_counts": projection_counts,
    }
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot migration from legacy JSON storage to Postgres."
    )
    parser.add_argument(
        "--report-path",
        default="",
        help="Optional path for JSON migration report output.",
    )
    args = parser.parse_args()
    result = run_migration(output_path=args.report_path or None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
