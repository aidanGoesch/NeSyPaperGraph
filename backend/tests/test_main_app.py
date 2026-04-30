from services.graph_builder import create_dummy_graph


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_runtime_endpoints(client):
    diagnostics = client.get("/api/runtime/diagnostics")
    assert diagnostics.status_code == 200
    assert diagnostics.json()["status"] == "ok"

    memory = client.get("/api/runtime/memory")
    assert memory.status_code == 200
    payload = memory.json()
    assert payload["status"] == "ok"
    assert "rss_mb" in payload


def test_access_key_middleware(client, monkeypatch):
    monkeypatch.setenv("APP_ACCESS_KEY", "secret")
    unauthorized = client.get("/api/workspace/state")
    assert unauthorized.status_code == 401

    authorized = client.get("/api/workspace/state", headers={"X-Access-Key": "secret"})
    assert authorized.status_code == 200


def test_jobs_endpoint_not_found(client):
    response = client.get("/api/jobs/missing-job")
    assert response.status_code == 404


def test_jobs_endpoint_with_queued_position(client):
    import asyncio

    client.app.state.jobs["job-1"] = {"status": "pending", "queued": True}
    queue = asyncio.Queue()
    queue.put_nowait(("job-1", [], 0))
    client.app.state.upload_queue = queue

    response = client.get("/api/jobs/job-1")
    assert response.status_code == 200
    assert response.json()["queue_position"] == 1


def test_agent_architecture_endpoint(client, monkeypatch):
    class FakeAgent:
        @staticmethod
        def get_agent_architecture_diagram():
            return "graph TD;A-->B;"

    monkeypatch.setattr("services.question_agent.QuestionAgent", FakeAgent)
    response = client.get("/api/agent/architecture")
    assert response.status_code == 200
    assert "mermaid" in response.json()


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_search_endpoint_success(client, monkeypatch):
    init_calls = {"count": 0}

    class FakeAgent:
        def __init__(self, *_args, **_kwargs):
            init_calls["count"] += 1
            self._last_state = {"sources_used": ["Paper A"]}
            self._last_path = {"nodes": ["Paper A", "Topic X"]}

        async def answer_question(self, _query):
            return "Answer text"

        def get_mermaid_diagram(self):
            return "graph TD;A-->B;"

    monkeypatch.setattr("services.question_agent.QuestionAgent", FakeAgent)
    client.app.state.graph = create_dummy_graph()
    payload = {"query": "What is in this graph?"}

    first = client.post("/api/search", json=payload)
    second = client.post("/api/search", json=payload)

    assert first.status_code == 200
    assert first.json()["status"] == "success"
    assert first.json()["answer"] == "Answer text"
    assert init_calls["count"] == 1

    # Force graph identity change; cached agent should rebuild.
    client.app.state.graph = create_dummy_graph()
    third = client.post("/api/search", json=payload)
    assert third.status_code == 200
    assert init_calls["count"] == 2
