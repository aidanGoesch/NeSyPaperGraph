from models.graph import PaperGraph
from models.paper import Paper
from services.verification import verify_bipartite


def _make_graph_with_two_papers() -> PaperGraph:
    graph = PaperGraph()
    graph.add_paper(
        Paper(
            title="Paper A",
            file_path="a.pdf",
            summary="Summary A",
            topics=["Topic X"],
            authors=["Author A"],
            publication_date="2024",
            embedding=[0.1, 0.2],
        )
    )
    graph.add_paper(
        Paper(
            title="Paper B",
            file_path="b.pdf",
            summary="Summary B",
            topics=["Topic Y"],
            authors=["Author B"],
            publication_date="2024",
            embedding=[0.2, 0.1],
        )
    )
    graph.clear_incremental_tracking()
    return graph


def test_verify_bipartite_ignores_semantic_paper_edges():
    graph = _make_graph_with_two_papers()
    graph.graph.add_edge("Paper A", "Paper B", type="semantic", weight=0.9)
    assert verify_bipartite(graph) is True


def test_verify_bipartite_rejects_nonsemantic_same_type_edges():
    graph = _make_graph_with_two_papers()
    graph.graph.add_edge("Paper A", "Paper B", type="topic")
    assert verify_bipartite(graph) is False
