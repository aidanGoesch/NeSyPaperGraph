from models.graph import PaperGraph
from models.paper import Paper


def test_ingest_url_adds_paper_from_semantic_scholar_abstract(client, monkeypatch):
    import api.graph as graph_api

    class _FakeSemanticScholarService:
        def hydrate_paper(self, url: str, paper_id: str | None = None):
            assert url == "https://arxiv.org/abs/1706.03762"
            assert paper_id == "paper-123"
            return {
                "url": url,
                "semanticScholarPaperId": "paper-123",
                "title": "Attention Is All You Need",
                "authors": ["Ashish Vaswani"],
                "year": 2017,
                "venue": "NeurIPS",
                "abstract": "We propose the Transformer architecture based on attention.",
            }

    class _FakeClient:
        def generate_embeddings(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        def generate_topic_synonyms(self, topics):
            return {topic: [] for topic in topics}

    class _FakeTopicExtractor:
        def __init__(self, _client):
            pass

        def extract_topics(self, text, current_topics=None, max_chars=8000):
            assert "Transformer architecture" in text
            _ = current_topics
            _ = max_chars
            return ["Attention Mechanisms", "Neural Machine Translation"]

    monkeypatch.setattr(graph_api, "SemanticScholarService", _FakeSemanticScholarService)
    monkeypatch.setattr(graph_api, "OpenAILLMClient", _FakeClient)
    monkeypatch.setattr(graph_api, "TopicExtractor", _FakeTopicExtractor)
    monkeypatch.setattr("services.verification.verify_bipartite", lambda *args, **kwargs: True)
    monkeypatch.setattr("services.verification.find_optimal_topic_merge", lambda *_: {})

    client.app.state.graph = PaperGraph()
    response = client.post(
        "/api/graph/ingest-url",
        json={
            "url": "https://arxiv.org/abs/1706.03762",
            "semanticScholarPaperId": "paper-123",
            "title": "ignored by server",
            "authors": ["ignored"],
            "year": 1999,
            "venue": "ignored",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "processed"
    assert payload["paper_title"] == "Attention Is All You Need"
    paper_titles = {paper["title"] for paper in payload["graph"]["papers"]}
    assert "Attention Is All You Need" in paper_titles

    ingested_paper = next(
        paper for paper in payload["graph"]["papers"] if paper["title"] == "Attention Is All You Need"
    )
    assert ingested_paper["abstract"] == "We propose the Transformer architecture based on attention."
    assert set(ingested_paper["topics"]) == {
        "Attention Mechanisms",
        "Neural Machine Translation",
    }


def test_ingest_url_reports_duplicate_and_keeps_graph_unchanged(client, monkeypatch):
    import api.graph as graph_api

    class _FakeSemanticScholarService:
        def hydrate_paper(self, url: str, paper_id: str | None = None):
            _ = paper_id
            return {
                "url": url,
                "semanticScholarPaperId": "paper-123",
                "title": "Attention Is All You Need",
                "authors": ["Ashish Vaswani"],
                "year": 2017,
                "venue": "NeurIPS",
                "abstract": "Transformer abstract",
            }

    class _FakeClient:
        def generate_embeddings(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        def generate_topic_synonyms(self, topics):
            return {topic: [] for topic in topics}

    class _FakeTopicExtractor:
        def __init__(self, _client):
            pass

        def extract_topics(self, text, current_topics=None, max_chars=8000):
            _ = text
            _ = current_topics
            _ = max_chars
            return ["Attention Mechanisms"]

    monkeypatch.setattr(graph_api, "SemanticScholarService", _FakeSemanticScholarService)
    monkeypatch.setattr(graph_api, "OpenAILLMClient", _FakeClient)
    monkeypatch.setattr(graph_api, "TopicExtractor", _FakeTopicExtractor)
    monkeypatch.setattr("services.verification.verify_bipartite", lambda *args, **kwargs: True)
    monkeypatch.setattr("services.verification.find_optimal_topic_merge", lambda *_: {})

    graph = PaperGraph()
    graph.add_paper(
        Paper(
            title="Attention Is All You Need",
            file_path="https://arxiv.org/abs/1706.03762",
            summary="Transformer abstract",
            authors=["Ashish Vaswani"],
            publication_date="2017",
            topics=["Attention Mechanisms"],
            embedding=[0.1, 0.2, 0.3],
        )
    )
    client.app.state.graph = graph

    response = client.post(
        "/api/graph/ingest-url",
        json={"url": "https://arxiv.org/abs/1706.03762", "semanticScholarPaperId": "paper-123"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "skipped"
    assert payload["reason"] == "duplicate_title"
    assert len(payload["graph"]["papers"]) == 1
