import os
import asyncio

from models.graph import PaperGraph
from models.paper import Paper


class _SlowMockBuilder:
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
            # Simulate realistic staged work so peak sampling sees active processing windows.
            time.sleep(0.03)
            graph.add_paper(
                Paper(
                    title=f"Peak::{filename}",
                    file_path=filename,
                    topics=["PeakTopic"],
                    summary="summary",
                    embedding=[0.2, 0.3, 0.4],
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


def test_peak_rss_ceiling_with_mocked_stress(client, upload_pdf_file, monkeypatch):
    import api.graph as graph_api

    monkeypatch.setattr(graph_api, "GraphBuilder", _SlowMockBuilder)
    monkeypatch.setattr(graph_api, "load_graph", lambda: None)
    monkeypatch.setattr(graph_api, "save_graph", lambda _g: None)
    monkeypatch.setattr("services.verification.verify_bipartite", lambda *args, **kwargs: True)

    threshold_mb = float(os.getenv("MEMORY_TEST_PEAK_RSS_MB", "2048"))
    peak_mb = 0.0

    for _ in range(8):
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
        memory_payload = client.get("/api/runtime/memory").json()
        peak_mb = max(peak_mb, float(memory_payload.get("rss_mb", 0.0)))

    assert peak_mb <= threshold_mb
