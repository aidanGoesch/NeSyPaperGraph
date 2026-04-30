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
