from pathlib import Path

from models.graph import PaperGraph
from models.paper import Paper
from services.graph_builder import GraphBuilder


class _FakeDoclingService:
    def parse_pdf(self, _pdf_bytes):
        return {
            "ok": True,
            "text": "Transformer model with attention and benchmarks.",
            "title": "Attention Paper",
            "authors": ["A. Author"],
            "publication_date": "2020",
        }


class _FakeTopicExtractor:
    def __init__(self, _client):
        self._client = _client
        self.llm_client = _client

    def extract_topics(self, _text, current_topics=None):
        _ = current_topics
        return ["Transformers", "Attention"]

    def heuristic_metadata(self, _text):
        return {"title": "Fallback", "authors": ["F. Author"], "publication_date": "2019"}

    def extract_paper_metadata(self, _text):
        return {
            "title": "Attention Paper",
            "authors": ["A. Author"],
            "publication_date": "2020",
        }


class _FakeClient:
    def generate_summary(self, _text):
        return "short summary"

    def generate_embeddings(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def generate_topic_synonyms(self, topics):
        return {topic: [] for topic in topics}


def test_get_papers_from_data_supports_temp_path(tmp_path, monkeypatch):
    from services import graph_builder

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.1 test")
    monkeypatch.setattr(graph_builder, "get_docling_service", lambda: _FakeDoclingService())
    monkeypatch.setattr(graph_builder, "TopicExtractor", _FakeTopicExtractor)

    builder = GraphBuilder()
    papers = list(builder.get_papers_from_data([("paper.pdf", str(pdf_path), "hash-1", 12)]))
    assert len(papers) == 1
    assert papers[0].title == "Attention Paper"
    assert papers[0].content_hash == "hash-1"


def test_build_graph_with_mocks(monkeypatch):
    from services import graph_builder

    monkeypatch.setattr(graph_builder, "get_docling_service", lambda: _FakeDoclingService())
    monkeypatch.setattr(graph_builder, "TopicExtractor", _FakeTopicExtractor)
    monkeypatch.setattr(graph_builder, "OpenAILLMClient", _FakeClient)
    monkeypatch.setattr("services.verification.verify_bipartite", lambda *args, **kwargs: True)
    monkeypatch.setattr("services.verification.find_optimal_topic_merge", lambda *_: {})

    builder = GraphBuilder()
    graph = builder.build_graph(files_data=[("paper.pdf", b"pdf", "hash-1")], total_papers=1)

    assert isinstance(graph, PaperGraph)
    assert "Attention Paper" in graph.graph.nodes
    paper_node = graph.graph.nodes["Attention Paper"]["data"]
    assert paper_node.summary == "short summary"
    assert paper_node.embedding == [0.1, 0.2, 0.3]


def test_ingest_semantic_paper_uses_abstract_for_topics(monkeypatch):
    from services import graph_builder

    class _FakeTopicExtractor:
        def __init__(self, _client):
            self.llm_client = _client

        def extract_topics(self, text, current_topics=None, max_chars=8000):
            assert "abstract source text" in text
            _ = current_topics
            _ = max_chars
            return ["Topic A", "Topic B"]

    class _FakeClient:
        def generate_embeddings(self, texts):
            return [[0.3, 0.2, 0.1] for _ in texts]

        def generate_topic_synonyms(self, topics):
            return {topic: [] for topic in topics}

    monkeypatch.setattr(graph_builder, "TopicExtractor", _FakeTopicExtractor)
    monkeypatch.setattr(graph_builder, "OpenAILLMClient", _FakeClient)
    monkeypatch.setattr("services.verification.verify_bipartite", lambda *args, **kwargs: True)
    monkeypatch.setattr("services.verification.find_optimal_topic_merge", lambda *_: {})

    builder = GraphBuilder()
    graph = builder.ingest_semantic_paper(
        {
            "url": "https://example.org/paper",
            "semanticScholarPaperId": "paper-1",
            "title": "Semantic Paper",
            "authors": ["A. Author"],
            "year": 2025,
            "venue": "Conference X",
            "abstract": "This is abstract source text for topic extraction.",
        }
    )

    assert isinstance(graph, PaperGraph)
    assert "Semantic Paper" in graph.graph.nodes
    node = graph.graph.nodes["Semantic Paper"]["data"]
    assert node.summary == "This is abstract source text for topic extraction."
    assert node.topics == ["Topic A", "Topic B"]
    assert node.embedding == [0.3, 0.2, 0.1]


def test_ingest_semantic_paper_skips_duplicate_title(monkeypatch):
    from services import graph_builder

    class _FakeTopicExtractor:
        def __init__(self, _client):
            self.llm_client = _client

        def extract_topics(self, text, current_topics=None, max_chars=8000):
            _ = text
            _ = current_topics
            _ = max_chars
            return ["Topic A"]

    class _FakeClient:
        def generate_embeddings(self, texts):
            return [[0.3, 0.2, 0.1] for _ in texts]

        def generate_topic_synonyms(self, topics):
            return {topic: [] for topic in topics}

    monkeypatch.setattr(graph_builder, "TopicExtractor", _FakeTopicExtractor)
    monkeypatch.setattr(graph_builder, "OpenAILLMClient", _FakeClient)
    monkeypatch.setattr("services.verification.verify_bipartite", lambda *args, **kwargs: True)
    monkeypatch.setattr("services.verification.find_optimal_topic_merge", lambda *_: {})

    existing_graph = PaperGraph()
    existing_graph.add_paper(
        Paper(
            title="Semantic Paper",
            file_path="https://example.org/paper",
            summary="already there",
            topics=["Topic A"],
            embedding=[0.4, 0.5, 0.6],
        )
    )
    pre_count = len([n for n, d in existing_graph.graph.nodes(data=True) if d.get("type") == "paper"])

    builder = GraphBuilder()
    graph = builder.ingest_semantic_paper(
        {
            "url": "https://example.org/paper",
            "semanticScholarPaperId": "paper-1",
            "title": "Semantic Paper",
            "authors": ["A. Author"],
            "year": 2025,
            "venue": "Conference X",
            "abstract": "new abstract",
        },
        existing_graph=existing_graph,
    )
    post_count = len([n for n, d in graph.graph.nodes(data=True) if d.get("type") == "paper"])
    assert post_count == pre_count
