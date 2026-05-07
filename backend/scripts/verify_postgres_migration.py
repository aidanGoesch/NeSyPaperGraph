from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.postgres_storage_service import (
    DOCUMENT_KEY_GRAPH,
    DOCUMENT_KEY_WORKSPACE,
    build_parity_report,
    get_workspace_projection_counts,
    load_document,
)
from services.storage_service import load_graph_legacy, load_workspace_state_legacy
from services.graph_serialization import serialize_graph


def main() -> None:
    if not os.environ.get("DATABASE_URL", "").strip():
        raise RuntimeError("DATABASE_URL must be set to verify Postgres migration")

    source_workspace = load_workspace_state_legacy() or {}
    source_graph_obj = load_graph_legacy()
    source_graph = serialize_graph(source_graph_obj) if source_graph_obj is not None else {}

    target_workspace = load_document(DOCUMENT_KEY_WORKSPACE) or {}
    target_graph = load_document(DOCUMENT_KEY_GRAPH) or {}

    source_report = build_parity_report(source_workspace, source_graph)
    target_report = build_parity_report(target_workspace, target_graph)

    verdict = "ok"
    mismatches = []
    if source_report.payload_hash_workspace != target_report.payload_hash_workspace:
        verdict = "mismatch"
        mismatches.append("workspace_payload_hash")
    if source_report.payload_hash_graph != target_report.payload_hash_graph:
        verdict = "mismatch"
        mismatches.append("graph_payload_hash")

    output = {
        "status": verdict,
        "mismatches": mismatches,
        "source": source_report.__dict__,
        "target": target_report.__dict__,
        "projection_counts": get_workspace_projection_counts(),
    }
    print(json.dumps(output, indent=2))
    if verdict != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
