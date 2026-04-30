from models.graph import PaperGraph, cosine_similarity
from models.paper import Paper


def test_add_paper_creates_bipartite_nodes_and_edges():
    graph = PaperGraph()
    graph.add_paper(
        Paper(title="Paper A", file_path="a.pdf", topics=["Topic A", "Topic B"])
    )
    assert graph.graph.nodes["Paper A"]["type"] == "paper"
    assert graph.graph.nodes["Topic A"]["type"] == "topic"
    assert graph.graph.has_edge("Paper A", "Topic A")
    assert graph.graph.has_edge("Paper A", "Topic B")


def test_merge_topics_moves_edges_to_representative():
    graph = PaperGraph()
    graph.add_paper(Paper(title="Paper A", file_path="a.pdf", topics=["T1"]))
    graph.add_paper(Paper(title="Paper B", file_path="b.pdf", topics=["T2"]))
    graph.merge_topics({0: ["T1", "T2"]})
    assert "T2" not in graph.graph.nodes
    assert graph.graph.has_edge("Paper A", "T1")
    assert graph.graph.has_edge("Paper B", "T1")


def test_add_semantic_edges_respects_pairwise_limit(monkeypatch):
    monkeypatch.setenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.1")
    monkeypatch.setenv("SEMANTIC_MAX_NEIGHBORS_PER_PAPER", "50")
    monkeypatch.setenv("SEMANTIC_MAX_PAIRWISE_CHECKS", "1")
    graph = PaperGraph()
    for idx in range(3):
        graph.add_paper(
            Paper(
                title=f"Paper {idx}",
                file_path=f"{idx}.pdf",
                topics=[f"Topic {idx}"],
                embedding=[0.9, 0.1 + idx],
            )
        )
    graph.add_semantic_edges()
    semantic_edges = [
        e for e in graph.graph.edges(data=True) if e[2].get("type") == "semantic"
    ]
    assert len(semantic_edges) <= 1


def test_cosine_similarity_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0
