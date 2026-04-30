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


def test_topic_search_response_shape(client, monkeypatch):
    class FakeSearchService:
        def __init__(self, *_args, **_kwargs):
            pass

        def search_papers(self, _query, top_k=10):
            assert top_k == 7
            return [
                {
                    "title": "Paper A",
                    "authors": ["Ada Lovelace"],
                    "publication_date": "2024",
                    "topics": ["Neurosymbolic AI"],
                    "summary": "Summary",
                    "score": 1.0,
                    "score_breakdown": {
                        "author_score": 0.2,
                        "title_score": 0.4,
                        "topic_score": 0.2,
                        "text_score": 0.0,
                        "semantic_score": 0.2,
                        "year_boost": 0.0,
                    },
                }
            ]

    monkeypatch.setattr("services.paper_search_service.PaperSearchService", FakeSearchService)
    client.app.state.graph = create_dummy_graph()
    response = client.post("/api/topic-search", json={"query": "neurosymbolic", "top_k": 7})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["query"] == "neurosymbolic"
    assert payload["results"][0]["title"] == "Paper A"


def test_topic_search_error_response_shape(client, monkeypatch):
    class FakeSearchService:
        def __init__(self, *_args, **_kwargs):
            pass

        def search_papers(self, _query, top_k=10):
            raise RuntimeError("topic search failed")

    monkeypatch.setattr("services.paper_search_service.PaperSearchService", FakeSearchService)
    client.app.state.graph = create_dummy_graph()
    response = client.post("/api/topic-search", json={"query": "fail"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert "topic search failed" in payload["error"]
