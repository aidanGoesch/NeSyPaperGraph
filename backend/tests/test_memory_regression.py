import asyncio
import time

import pytest

from models.graph import PaperGraph
from models.paper import Paper
from services.observability import rss_mb


class _FastGraphBuilder:
    def build_graph(
        self,
        files_data=None,
        file_path=None,
        existing_graph=None,
        on_paper_processed=None,
        total_papers=None,
    ):
        _ = file_path
        graph = existing_graph or PaperGraph()
        papers = files_data or []
        total = total_papers or len(papers)
        for index, (filename, _temp_path, _hash, _bytes) in enumerate(papers, start=1):
            graph.add_paper(
                Paper(
                    title=f"Mem::{filename}",
                    file_path=filename,
                    topics=["MemoryTopic"],
                    summary="summary",
                    embedding=[0.1, 0.2, 0.3],
                )
            )
            if on_paper_processed:
                on_paper_processed(
                    {
                        "status": "processed",
                        "paper_title": filename,
                        "paper_index": index,
                        "paper_total": total,
                        "graph": graph,
                    }
                )
        return graph


def test_coalesce_heavy_snapshot_events():
    import api.graph as graph_api

    queue = asyncio.Queue()
    queue.put_nowait(("paper_processed", {"graph": {"a": 1}}))
    queue.put_nowait(("paper_processed", {"graph": {"a": 2}}))
    queue.put_nowait(("job_done", {"status": "done"}))

    graph_api._coalesce_heavy_events(queue, "paper_processed", {"graph": {"a": 3}})
    queued = list(queue._queue)
    heavy = [item for item in queued if item[0] == "paper_processed" and "graph" in item[1]]
    assert len(heavy) == 0


def test_prune_jobs_respects_ttl_and_cap(monkeypatch):
    import api.graph as graph_api

    now = time.time()
    monkeypatch.setenv("MAX_JOB_HISTORY", "1")
    monkeypatch.setenv("JOB_TTL_SECONDS", "1")
    jobs = {
        "expired": {"finished_at": now - 10, "started_at": now - 20},
        "older": {"finished_at": now, "started_at": now - 5},
        "newer": {"finished_at": now, "started_at": now},
    }
    graph_api.prune_jobs(jobs)
    assert "expired" not in jobs
    assert len(jobs) == 1
    assert "newer" in jobs


@pytest.mark.asyncio
async def test_upload_queue_bytes_decremented_after_processing(monkeypatch):
    import api.graph as graph_api

    class _AppState:
        def __init__(self):
            self.upload_queue = asyncio.Queue()
            self.upload_queue_bytes = 100
            self.jobs = {"job-1": {"status": "pending"}}

    class _App:
        def __init__(self):
            self.state = _AppState()

    async def _fake_process(app, job_id, files_data):
        _ = app
        _ = job_id
        _ = files_data

    app = _App()
    app.state.upload_queue.put_nowait(("job-1", [], 100))
    app.state.upload_queue.put_nowait(None)
    monkeypatch.setattr(graph_api, "process_pdf_job", _fake_process)

    await graph_api.process_upload_queue(app)
    assert app.state.upload_queue_bytes == 0


def test_repeated_upload_cycles_have_bounded_rss(client, upload_pdf_file, monkeypatch):
    import api.graph as graph_api

    monkeypatch.setattr(graph_api, "GraphBuilder", _FastGraphBuilder)
    monkeypatch.setattr(graph_api, "load_graph", lambda: None)
    monkeypatch.setattr(graph_api, "save_graph", lambda _g: None)
    monkeypatch.setattr("services.verification.verify_bipartite", lambda *args, **kwargs: True)

    rss_start = rss_mb()
    for _ in range(5):
        response = client.post("/api/graph/upload", files=[("files", upload_pdf_file)])
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        queued_job_id, files_data, queued_bytes = client.app.state.upload_queue.get_nowait()
        client.app.state.upload_queue.task_done()
        assert queued_job_id == job_id
        client.app.state.upload_queue_bytes = max(
            0, client.app.state.upload_queue_bytes - queued_bytes
        )
        asyncio.run(graph_api.process_pdf_job(client.app, queued_job_id, files_data))
        status = client.get(f"/api/jobs/{job_id}").json()["status"]
        assert status == "done"
    rss_end = rss_mb()
    assert rss_end - rss_start < 512
