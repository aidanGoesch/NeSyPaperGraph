from services.graph_builder import create_dummy_graph


def test_search_results_response_shape(client, monkeypatch):
    class FakeAgent:
        def __init__(self, *_args, **_kwargs):
            self._last_state = {
                "search_results": [{"title": "Paper A"}],
                "sources_used": ["Paper A"],
                "answer_structured": {
                    "segments": [{"text": "Paper A is relevant.", "claim_id": None}],
                    "claims": [],
                    "warnings": [],
                },
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
    assert "answer_structured" in payload


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


def test_search_activation_mode_response_shape(client, monkeypatch):
    class FakeAgent:
        def __init__(self, *_args, **_kwargs):
            self._last_state = {}

        async def answer_question(self, _query):
            return "legacy"

        async def answer_question_with_activation(self, _query, conversation_history=None):
            assert isinstance(conversation_history, list)
            return {
                "final_answer": "Activation answer",
                "answer_structured": {
                    "segments": [{"text": "Activation answer", "claim_id": None}],
                    "claims": [],
                    "warnings": [],
                },
                "confidence": 0.77,
                "needs_more_context": False,
                "rounds": [
                    {
                        "round_index": 1,
                        "query_used": "q1",
                        "seed_nodes": [{"node_id": "Paper A", "score": 0.9}],
                        "activated_nodes": [{"node_id": "Paper A", "score": 1.0}],
                        "step_trace": [{"step": 0, "node_id": "Paper A", "score_after_step": 1.0}],
                        "sources_used": ["Paper A"],
                        "answer": "Activation answer",
                        "confidence": 0.77,
                    }
                ],
                "sources_used": ["Paper A"],
            }

        def get_mermaid_diagram(self):
            return None

    monkeypatch.setattr("services.question_agent.QuestionAgent", FakeAgent)
    client.app.state.graph = create_dummy_graph()
    response = client.post(
        "/api/search",
        json={
            "query": "activation question",
            "activation_mode": True,
            "conversation_history": [{"question": "q0", "answer": "a0"}],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["final_answer"] == "Activation answer"
    assert payload["answer"] == "Activation answer"
    assert payload["answer_structured"]["segments"][0]["text"] == "Activation answer"
    assert payload["confidence"] == 0.77
    assert payload["needs_more_context"] is False
    assert len(payload["rounds"]) == 1


def test_search_activation_mode_can_return_second_round(client, monkeypatch):
    class FakeAgent:
        def __init__(self, *_args, **_kwargs):
            self._last_state = {}

        async def answer_question(self, _query):
            return "legacy"

        async def answer_question_with_activation(self, _query, conversation_history=None):
            return {
                "final_answer": "Round 2 answer",
                "answer_structured": {
                    "segments": [{"text": "Round 2 answer", "claim_id": None}],
                    "claims": [],
                    "warnings": [],
                },
                "confidence": 0.49,
                "needs_more_context": True,
                "rounds": [
                    {"round_index": 1, "seed_nodes": [], "activated_nodes": [], "step_trace": []},
                    {"round_index": 2, "seed_nodes": [], "activated_nodes": [], "step_trace": []},
                ],
                "sources_used": [],
            }

        def get_mermaid_diagram(self):
            return None

    monkeypatch.setattr("services.question_agent.QuestionAgent", FakeAgent)
    client.app.state.graph = create_dummy_graph()
    response = client.post(
        "/api/search",
        json={"query": "needs second round", "activation_mode": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["needs_more_context"] is True
    assert len(payload["rounds"]) == 2
