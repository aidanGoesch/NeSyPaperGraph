from models.graph import PaperGraph
from models.paper import Paper


class _FakeGraphBuilder:
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
                    title=f"Paper::{filename}",
                    file_path=filename,
                    topics=["MockTopic"],
                    summary="mock summary",
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


def test_upload_to_queue_to_persist_pipeline(client, upload_pdf_file, monkeypatch):
    import api.graph as graph_api

    saved = {"count": 0}
    monkeypatch.setattr(graph_api, "GraphBuilder", _FakeGraphBuilder)
    monkeypatch.setattr(graph_api, "load_graph", lambda: None)
    monkeypatch.setattr(graph_api, "save_graph", lambda _g: saved.__setitem__("count", saved["count"] + 1))
    monkeypatch.setattr("services.verification.verify_bipartite", lambda *args, **kwargs: True)
    monkeypatch.setenv("OPENAI_API_KEY", "mock-key")

    response = client.post("/api/graph/upload", files=[("files", upload_pdf_file)])
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    queued_job_id, files_data, queued_bytes = client.app.state.upload_queue.get_nowait()
    client.app.state.upload_queue.task_done()
    assert queued_job_id == job_id
    client.app.state.upload_queue_bytes = max(
        0, client.app.state.upload_queue_bytes - queued_bytes
    )
    import asyncio

    asyncio.run(graph_api.process_pdf_job(client.app, queued_job_id, files_data))
    status_response = client.get(f"/api/jobs/{job_id}")
    assert status_response.status_code == 200
    final_payload = status_response.json()

    assert final_payload is not None
    assert final_payload["status"] == "done"
    assert saved["count"] >= 1
    assert final_payload.get("temp_spooled_bytes") == 0

    load_response = client.get("/api/graph/load")
    assert load_response.status_code == 200
    body = load_response.json()
    assert any(paper["title"].startswith("Paper::") for paper in body["papers"])
