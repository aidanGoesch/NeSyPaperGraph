from services.graph_builder import create_dummy_graph


def test_dummy_graph_endpoint(client):
    response = client.get("/api/graph/dummy")
    assert response.status_code == 200
    payload = response.json()
    assert "papers" in payload
    assert "topics" in payload
    assert "edges" in payload


def test_load_graph_not_found(client, monkeypatch):
    monkeypatch.setattr("api.graph.load_graph", lambda: None)
    client.app.state.graph = None
    response = client.get("/api/graph/load")
    assert response.status_code == 404


def test_load_graph_success(client, monkeypatch):
    graph = create_dummy_graph()
    monkeypatch.setattr("api.graph.load_graph", lambda: graph)
    client.app.state.graph = None
    response = client.get("/api/graph/load")
    assert response.status_code == 200
    payload = response.json()
    assert "papers" in payload
    assert "all_topics_seen" in payload


def test_serialize_sse_event_format():
    import api.graph as graph_api

    body = graph_api._serialize_sse_event("graph_snapshot", {"ok": True})
    assert body.startswith("event: graph_snapshot")
    assert "data:" in body
    assert body.endswith("\n\n")
