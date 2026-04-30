from services.graph_builder import create_dummy_graph


def test_search_results_response_shape(client, monkeypatch):
    class FakeAgent:
        def __init__(self, *_args, **_kwargs):
            self._last_state = {
                "search_results": [{"title": "Paper A"}],
                "sources_used": ["Paper A"],
            }
            self._last_path = {"nodes": ["Paper A", "Topic X"]}

        async def answer_question(self, _query):
            return "SEARCH_RESULTS"

        def get_mermaid_diagram(self):
            return "graph TD;A-->B;"

    monkeypatch.setattr("services.question_agent.QuestionAgent", FakeAgent)
    client.app.state.graph = create_dummy_graph()
    response = client.post("/api/search", json={"query": "find papers"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "search_results"
    assert payload["search_results"][0]["title"] == "Paper A"


def test_search_error_response_shape(client, monkeypatch):
    class FakeAgent:
        def __init__(self, *_args, **_kwargs):
            self._last_state = {}

        async def answer_question(self, _query):
            raise RuntimeError("search failed")

        def get_mermaid_diagram(self):
            return None

    monkeypatch.setattr("services.question_agent.QuestionAgent", FakeAgent)
    client.app.state.graph = create_dummy_graph()
    response = client.post("/api/search", json={"query": "fail please"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert "search failed" in payload["error"]
