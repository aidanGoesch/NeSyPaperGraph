import asyncio
import io
import os
import sys
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture(autouse=True)
def stable_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("APP_ACCESS_KEY", "")
    monkeypatch.setenv("DESKTOP_APP_MODE", "false")
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path / "local_data"))
    monkeypatch.setenv("UPLOAD_MAX_FILE_MB", "64")
    monkeypatch.setenv("UPLOAD_MAX_TOTAL_MB", "256")
    monkeypatch.setenv("UPLOAD_QUEUE_MAX_JOBS", "2")
    monkeypatch.setenv("UPLOAD_QUEUE_MAX_BYTES_MB", "512")
    monkeypatch.setenv("INGEST_SNAPSHOT_EVERY_PAPERS", "0")
    monkeypatch.setenv("SSE_SUBSCRIBER_QUEUE_MAX_EVENTS", "50")
    monkeypatch.setenv("STORAGE_COMPRESS_JSON", "true")


@pytest.fixture
def app_module(monkeypatch: pytest.MonkeyPatch):
    import main

    async def _noop_warmup() -> None:
        return

    async def _noop_stop_queue_worker(_app) -> None:
        return

    def _noop_start_queue_worker(_app) -> None:
        return

    monkeypatch.setattr(main, "_warmup_docling_once", _noop_warmup)
    monkeypatch.setattr(main, "start_queue_worker", _noop_start_queue_worker)
    monkeypatch.setattr(main, "stop_queue_worker", _noop_stop_queue_worker)
    return main


@pytest.fixture
def client(app_module) -> Generator[TestClient, None, None]:
    with TestClient(app_module.app) as test_client:
        yield test_client


@pytest.fixture
def minimal_pdf_bytes() -> bytes:
    return b"""%PDF-1.1
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 12 >>
stream
Hello World!
endstream
endobj
xref
0 5
0000000000 65535 f
0000000010 00000 n
0000000061 00000 n
0000000118 00000 n
0000000205 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
270
%%EOF
"""


@pytest.fixture
def upload_pdf_file(minimal_pdf_bytes: bytes):
    return ("paper.pdf", io.BytesIO(minimal_pdf_bytes), "application/pdf")


@pytest.fixture(autouse=True)
def reset_graph_builder_globals() -> None:
    from services import graph_builder

    graph_builder.clear_all_topics_seen()
    graph_builder.clear_topic_synonyms_cache()
