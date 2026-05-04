def test_resolve_paper_url_returns_core_biblio_fields(client, monkeypatch):
    class _FakeSemanticScholarService:
        def resolve_url_metadata(self, url: str):
            assert url == "https://arxiv.org/abs/1706.03762"
            return {
                "url": url,
                "semanticScholarPaperId": "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
                "title": "Attention Is All You Need",
                "authors": ["Ashish Vaswani", "Noam Shazeer"],
                "year": 2017,
                "venue": "NeurIPS",
            }

    monkeypatch.setattr(
        "api.workspace.SemanticScholarService", _FakeSemanticScholarService
    )

    response = client.post(
        "/api/workspace/resolve-paper-url",
        json={"url": "https://arxiv.org/abs/1706.03762"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "url": "https://arxiv.org/abs/1706.03762",
        "semanticScholarPaperId": "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "year": 2017,
        "venue": "NeurIPS",
    }


def test_resolve_paper_url_returns_404_for_unresolvable_link(client, monkeypatch):
    class _FakeSemanticScholarService:
        def resolve_url_metadata(self, _url: str):
            return None

    monkeypatch.setattr(
        "api.workspace.SemanticScholarService", _FakeSemanticScholarService
    )

    response = client.post(
        "/api/workspace/resolve-paper-url",
        json={"url": "https://example.com/not-a-paper"},
    )
    assert response.status_code == 404
    assert "Unable to resolve paper metadata" in response.json()["detail"]


def test_resolve_paper_url_surfaces_rate_limit_as_429(client, monkeypatch):
    class _FakeSemanticScholarService:
        def resolve_url_metadata(self, _url: str):
            raise RuntimeError("Semantic Scholar rate limit exceeded")

    monkeypatch.setattr(
        "api.workspace.SemanticScholarService", _FakeSemanticScholarService
    )

    response = client.post(
        "/api/workspace/resolve-paper-url",
        json={"url": "https://doi.org/10.1000/xyz"},
    )
    assert response.status_code == 429
    assert "rate limit" in response.json()["detail"].lower()


def test_resolve_paper_returns_normalized_fields(client, monkeypatch):
    class _FakeSemanticScholarService:
        def resolve_seed_paper_details(self, seed):
            assert seed["title"] == "Attention Is All You Need"
            assert seed["authors"] == ["Ashish Vaswani"]
            return {
                "paperId": "paper-123",
                "title": "Attention Is All You Need",
                "authors": ["Ashish Vaswani", "Noam Shazeer"],
                "year": 2017,
                "venue": "NeurIPS",
                "url": "https://arxiv.org/abs/1706.03762",
            }

    monkeypatch.setattr("api.workspace.SemanticScholarService", _FakeSemanticScholarService)

    response = client.post(
        "/api/workspace/resolve-paper",
        json={
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani"],
            "year": 2017,
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "semanticScholarPaperId": "paper-123",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "year": 2017,
        "venue": "NeurIPS",
        "url": "https://arxiv.org/abs/1706.03762",
    }


def test_resolve_paper_returns_404_when_unresolved(client, monkeypatch):
    class _FakeSemanticScholarService:
        def resolve_seed_paper_details(self, _seed):
            return None

    monkeypatch.setattr("api.workspace.SemanticScholarService", _FakeSemanticScholarService)

    response = client.post(
        "/api/workspace/resolve-paper",
        json={"title": "Some Unresolved Paper"},
    )
    assert response.status_code == 404
    assert "Unable to resolve the paper" in response.json()["detail"]


def test_resolve_paper_requires_seed_identifier(client):
    response = client.post("/api/workspace/resolve-paper", json={"authors": ["A"]})
    assert response.status_code == 422
    assert "Provide semanticScholarPaperId, title, or url" in response.json()["detail"]


def test_workspace_theme_recommendations_route_available(client, monkeypatch):
    monkeypatch.setattr(
        "api.workspace.load_workspace_state",
        lambda: {"readingItems": [], "themeNotes": [], "paperAnnotations": {}},
    )
    monkeypatch.setattr(
        "api.workspace.build_theme_recommendations_payload",
        lambda _workspace, _request: {
            "status": "success",
            "themeId": "theme-1",
            "results": [{"paperId": "p-1", "title": "Test Recommendation"}],
        },
    )

    response = client.post(
        "/api/workspace/recommendations/theme",
        json={"themeId": "theme-1", "limit": 3, "candidatePoolSize": 10},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["results"][0]["paperId"] == "p-1"


def test_workspace_paper_recommendations_route_available(client, monkeypatch):
    class _FakeSemanticScholarService:
        def find_similar_papers_from_seed(self, seed_paper, limit=10):
            assert seed_paper["title"] == "Neuro-Symbolic Program Synthesis"
            assert limit == 4
            return [{"paperId": "p-2", "title": "Paper Route Works"}]

    monkeypatch.setattr("api.workspace.SemanticScholarService", _FakeSemanticScholarService)

    response = client.post(
        "/api/workspace/recommendations/paper",
        json={"title": "Neuro-Symbolic Program Synthesis", "limit": 4},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["results"][0]["paperId"] == "p-2"
