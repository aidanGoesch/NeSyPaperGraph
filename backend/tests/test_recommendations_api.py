def _workspace_payload():
    return {
        "readingItems": [
            {
                "id": "read-1",
                "sourceType": "url",
                "status": "done",
                "topicHints": ["neurosymbolic", "reasoning"],
                "linkedPaperTitle": "Neuro-Symbolic Program Synthesis",
                "linkedThemeId": "theme-1",
                "title": "Neuro-Symbolic Program Synthesis",
                "url": "https://example.com/p1",
                "semanticScholarPaperId": "seed-paper-1",
                "authors": ["Alice Author"],
                "year": 2022,
                "venue": "NeurIPS",
                "quickNote": "",
                "createdAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:00:00Z",
            },
            {
                "id": "read-2",
                "sourceType": "url",
                "status": "reading",
                "topicHints": ["knowledge graphs"],
                "linkedPaperTitle": "Symbolic Priors for Neural Systems",
                "linkedThemeId": "theme-1",
                "title": "Symbolic Priors for Neural Systems",
                "url": "https://example.com/p2",
                "semanticScholarPaperId": "seed-paper-2",
                "authors": ["Bob Author"],
                "year": 2023,
                "venue": "ICLR",
                "quickNote": "",
                "createdAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:00:00Z",
            },
            {
                "id": "read-3",
                "sourceType": "url",
                "status": "queued",
                "topicHints": ["causal inference"],
                "linkedPaperTitle": None,
                "linkedThemeId": None,
                "title": "Queued Symbolic Reasoning",
                "url": "https://example.com/p3",
                "semanticScholarPaperId": "seed-paper-3",
                "authors": ["Cara Author"],
                "year": 2021,
                "venue": "AAAI",
                "quickNote": "",
                "createdAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:00:00Z",
            },
        ],
        "themeNotes": [
            {
                "id": "theme-1",
                "themeTitle": "Neurosymbolic Methods",
                "linkedPaperTitles": [
                    "Neuro-Symbolic Program Synthesis",
                    "Symbolic Priors for Neural Systems",
                ],
                "sections": {
                    "notes": "Integrate logic and learning",
                    "toRead": "- Queued Symbolic Reasoning [queued]",
                },
                "createdAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:00:00Z",
            }
        ],
        "paperAnnotations": {},
    }


def test_paper_recommendations_endpoint_returns_similar_papers(client, monkeypatch):
    import api.recommendations as recommendations_api

    class _FakeSemanticScholarService:
        def find_similar_papers_from_seed(self, seed_paper, limit: int = 10):
            assert seed_paper.get("semanticScholarPaperId") == "seed-paper-1"
            assert limit == 3
            return [
                {
                    "paperId": "rec-1",
                    "title": "Composable Neuro-Symbolic Inference",
                    "year": 2024,
                    "venue": "NeurIPS",
                    "authors": ["A. Author"],
                    "url": "https://example.com/rec-1",
                }
            ]

    monkeypatch.setattr(
        recommendations_api, "SemanticScholarService", _FakeSemanticScholarService
    )

    response = client.post(
        "/api/recommendations/paper",
        json={"semanticScholarPaperId": "seed-paper-1", "limit": 3},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["results"][0]["paperId"] == "rec-1"


def test_paper_recommendations_can_resolve_by_title_when_id_missing(client, monkeypatch):
    import api.recommendations as recommendations_api

    class _FakeSemanticScholarService:
        def find_similar_papers_from_seed(self, seed_paper, limit: int = 10):
            assert not seed_paper.get("semanticScholarPaperId")
            assert seed_paper.get("title") == "Queued Symbolic Reasoning"
            assert limit == 2
            return [{"paperId": "rec-2", "title": "Resolved by title"}]

    monkeypatch.setattr(
        recommendations_api, "SemanticScholarService", _FakeSemanticScholarService
    )

    response = client.post(
        "/api/recommendations/paper",
        json={"title": "Queued Symbolic Reasoning", "limit": 2},
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["paperId"] == "rec-2"


def test_theme_recommendations_uses_workspace_state(client, monkeypatch):
    import api.recommendations as recommendations_api

    monkeypatch.setattr(
        recommendations_api, "load_workspace_state", lambda: _workspace_payload()
    )

    class _FakeSemanticScholarService:
        def recommend_for_theme(self, *, theme_title, seed_papers, limit, candidate_pool_size):
            assert theme_title == "Neurosymbolic Methods"
            assert len(seed_papers) == 3
            assert limit == 2
            assert candidate_pool_size == 12
            return [
                {
                    "paperId": "theme-rec-1",
                    "title": "Themed Candidate",
                    "score": 0.99,
                    "reason": "High centroid similarity",
                }
            ]

    monkeypatch.setattr(
        recommendations_api, "SemanticScholarService", _FakeSemanticScholarService
    )

    response = client.post(
        "/api/recommendations/theme",
        json={"themeId": "theme-1", "limit": 2, "candidatePoolSize": 12},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["paperId"] == "theme-rec-1"


def test_theme_recommendations_returns_404_when_theme_missing(client, monkeypatch):
    import api.recommendations as recommendations_api

    monkeypatch.setattr(
        recommendations_api, "load_workspace_state", lambda: _workspace_payload()
    )

    response = client.post(
        "/api/recommendations/theme",
        json={"themeId": "missing-theme", "limit": 3},
    )
    assert response.status_code == 404
    assert "theme" in response.json()["detail"].lower()


def test_theme_recommendations_accepts_workspace_state_payload(client, monkeypatch):
    import api.recommendations as recommendations_api

    monkeypatch.setattr(
        recommendations_api,
        "load_workspace_state",
        lambda: (_ for _ in ()).throw(RuntimeError("should not read storage")),
    )

    class _FakeSemanticScholarService:
        def recommend_for_theme(self, *, theme_title, seed_papers, limit, candidate_pool_size):
            assert theme_title == "Neurosymbolic Methods"
            assert len(seed_papers) == 3
            assert limit == 2
            assert candidate_pool_size == 12
            return [{"paperId": "theme-rec-2", "title": "From request payload"}]

    monkeypatch.setattr(
        recommendations_api, "SemanticScholarService", _FakeSemanticScholarService
    )

    response = client.post(
        "/api/recommendations/theme",
        json={
            "themeId": "theme-1",
            "limit": 2,
            "candidatePoolSize": 12,
            "workspaceState": _workspace_payload(),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["paperId"] == "theme-rec-2"


def test_theme_recommendations_includes_to_read_seed_without_semantic_scholar_id(
    client, monkeypatch
):
    import api.recommendations as recommendations_api

    payload = _workspace_payload()
    payload["readingItems"] = []
    payload["themeNotes"][0]["sections"][
        "toRead"
    ] = "- Causal Representation Learning in Agents [queued]"
    monkeypatch.setattr(recommendations_api, "load_workspace_state", lambda: payload)

    class _FakeSemanticScholarService:
        def recommend_for_theme(self, *, theme_title, seed_papers, limit, candidate_pool_size):
            assert theme_title == "Neurosymbolic Methods"
            assert len(seed_papers) == 1
            assert seed_papers[0]["title"] == "Causal Representation Learning in Agents"
            assert not seed_papers[0]["paperId"]
            return [{"paperId": "theme-rec-3", "title": "Recovered from toRead seed"}]

    monkeypatch.setattr(
        recommendations_api, "SemanticScholarService", _FakeSemanticScholarService
    )

    response = client.post(
        "/api/recommendations/theme",
        json={"themeId": "theme-1", "limit": 2, "candidatePoolSize": 12},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["results"][0]["paperId"] == "theme-rec-3"


def test_theme_recommendations_filters_out_existing_seed_papers(client, monkeypatch):
    import api.recommendations as recommendations_api

    monkeypatch.setattr(
        recommendations_api, "load_workspace_state", lambda: _workspace_payload()
    )

    class _FakeSemanticScholarService:
        def recommend_for_theme(self, *, theme_title, seed_papers, limit, candidate_pool_size):
            _ = theme_title, seed_papers, limit, candidate_pool_size
            return [
                {
                    "paperId": "seed-paper-1",
                    "title": "Neuro-Symbolic Program Synthesis",
                },
                {
                    "paperId": "new-rec-1",
                    "title": "Composable Neuro-Symbolic Abstractions",
                },
            ]

    monkeypatch.setattr(
        recommendations_api, "SemanticScholarService", _FakeSemanticScholarService
    )

    response = client.post(
        "/api/recommendations/theme",
        json={"themeId": "theme-1", "limit": 4, "candidatePoolSize": 12},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["paperId"] == "new-rec-1"


def test_topic_recommendations_uses_reading_history(client, monkeypatch):
    import api.recommendations as recommendations_api

    monkeypatch.setattr(
        recommendations_api, "load_workspace_state", lambda: _workspace_payload()
    )

    class _FakeSemanticScholarService:
        def recommend_for_topic_profile(self, *, topic_query, seed_papers, limit, candidate_pool_size):
            assert topic_query == "causal reasoning"
            assert len(seed_papers) == 3
            assert limit == 5
            assert candidate_pool_size == 20
            return [
                {
                    "paperId": "topic-rec-1",
                    "title": "Topic Candidate",
                    "score": 0.88,
                    "reason": "Matches neurosymbolic profile",
                }
            ]

    monkeypatch.setattr(
        recommendations_api, "SemanticScholarService", _FakeSemanticScholarService
    )

    response = client.post(
        "/api/recommendations/topic",
        json={"query": "causal reasoning", "limit": 5, "candidatePoolSize": 20},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["paperId"] == "topic-rec-1"


def test_topic_recommendations_rejects_empty_seed_history(client, monkeypatch):
    import api.recommendations as recommendations_api

    monkeypatch.setattr(
        recommendations_api,
        "load_workspace_state",
        lambda: {"readingItems": [], "themeNotes": [], "paperAnnotations": {}},
    )

    response = client.post(
        "/api/recommendations/topic",
        json={"query": "causal reasoning", "limit": 5},
    )
    assert response.status_code == 422
    assert "reading history" in response.json()["detail"].lower()


def test_topic_recommendations_requires_nonempty_query(client, monkeypatch):
    import api.recommendations as recommendations_api

    monkeypatch.setattr(
        recommendations_api, "load_workspace_state", lambda: _workspace_payload()
    )

    response = client.post(
        "/api/recommendations/topic",
        json={"query": "   ", "limit": 5},
    )
    assert response.status_code == 422
    assert "required" in response.json()["detail"].lower()
