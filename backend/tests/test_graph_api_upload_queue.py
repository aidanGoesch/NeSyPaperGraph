import io


def test_upload_rejects_no_valid_pdf_files(client):
    files = [("files", ("notes.txt", io.BytesIO(b"text"), "text/plain"))]
    response = client.post("/api/graph/upload", files=files)
    assert response.status_code == 400
    assert "No valid PDF files found" in response.json()["detail"]


def test_upload_rejects_oversized_file(client, monkeypatch):
    import api.graph as graph_api

    monkeypatch.setattr(graph_api, "UPLOAD_MAX_FILE_MB", 1)
    monkeypatch.setattr(graph_api, "UPLOAD_MAX_FILE_BYTES", 1)
    files = [("files", ("large.pdf", io.BytesIO(b"ab"), "application/pdf"))]
    response = client.post("/api/graph/upload", files=files)
    assert response.status_code == 413
    assert "too large" in response.json()["detail"]


def test_upload_rejects_when_queue_full(client, upload_pdf_file, monkeypatch):
    import asyncio
    import api.graph as graph_api

    queue = asyncio.Queue(maxsize=1)
    queue.put_nowait(("existing-job", [], 0))
    client.app.state.upload_queue = queue
    client.app.state.upload_queue_bytes = 0
    monkeypatch.setattr(graph_api, "UPLOAD_QUEUE_MAX_BYTES", 1024 * 1024)

    files = [("files", upload_pdf_file)]
    response = client.post("/api/graph/upload", files=files)
    assert response.status_code == 429
    assert "queue is currently full" in response.json()["detail"]


def test_upload_rejects_when_queue_bytes_exceeded(client, upload_pdf_file, monkeypatch):
    import api.graph as graph_api

    client.app.state.upload_queue_bytes = 1024
    monkeypatch.setattr(graph_api, "UPLOAD_QUEUE_MAX_BYTES", 1)

    files = [("files", upload_pdf_file)]
    response = client.post("/api/graph/upload", files=files)
    assert response.status_code == 429


def test_upload_success_enqueues_job(client, upload_pdf_file, monkeypatch):
    import api.graph as graph_api

    monkeypatch.setattr(graph_api, "UPLOAD_QUEUE_MAX_BYTES", 10_000_000)
    client.app.state.upload_queue_bytes = 0
    files = [("files", upload_pdf_file)]
    response = client.post("/api/graph/upload", files=files)

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "pending"
    assert "job_id" in payload
    assert payload["queue_position"] >= 1
