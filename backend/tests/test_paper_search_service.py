from models.graph import PaperGraph
from models.paper import Paper
from services.paper_search_service import PaperSearchService


def _build_graph_for_search():
    graph = PaperGraph()
    graph.add_paper(
        Paper(
            title="Neurosymbolic Reasoning with Logic Programs",
            file_path="p1.pdf",
            text="A neurosymbolic AI approach combining neural perception and symbolic logic reasoning.",
            summary="Introduces a neurosymbolic framework for reasoning.",
            topics=["Neurosymbolic AI", "Logic Programming"],
            authors=["Alice Chen", "David Bornstein"],
            publication_date="2024",
            embedding=[0.98, 0.02, 0.0],
        )
    )
    graph.add_paper(
        Paper(
            title="Neural Program Repair for Code",
            file_path="p2.pdf",
            text="Applies neural language models to code repair.",
            summary="Program repair using neural methods.",
            topics=["Program Repair", "Neural Networks"],
            authors=["Alice Chen"],
            publication_date="2023",
            embedding=[0.2, 0.8, 0.0],
        )
    )
    graph.add_paper(
        Paper(
            title="Statistical Learning for Vision",
            file_path="p3.pdf",
            text="Classic statistical approaches for computer vision.",
            summary="Vision models based on statistics.",
            topics=["Computer Vision", "Machine Learning"],
            authors=["Bob Lee"],
            publication_date="2022",
            embedding=[0.0, 1.0, 0.0],
        )
    )
    return graph


def test_author_query_matches_authors_field():
    graph = _build_graph_for_search()
    service = PaperSearchService(graph_obj=graph, embed_query_fn=lambda _q: [0.0, 0.0, 0.0])

    results = service.search_papers("chen", top_k=5)
    titles = [entry["title"] for entry in results]

    assert "Neurosymbolic Reasoning with Logic Programs" in titles
    assert "Neural Program Repair for Code" in titles


def test_title_query_ranks_exact_partial_match_first():
    graph = _build_graph_for_search()
    service = PaperSearchService(graph_obj=graph, embed_query_fn=lambda _q: [0.0, 0.0, 0.0])

    results = service.search_papers("reasoning logic programs", top_k=3)

    assert results[0]["title"] == "Neurosymbolic Reasoning with Logic Programs"


def test_semantic_query_can_retrieve_without_strong_lexical_overlap():
    graph = _build_graph_for_search()
    service = PaperSearchService(graph_obj=graph, embed_query_fn=lambda _q: [1.0, 0.0, 0.0])

    results = service.search_papers("neurosymbolic ai", top_k=3)

    assert results[0]["title"] == "Neurosymbolic Reasoning with Logic Programs"
    assert results[0]["score_breakdown"]["semantic_score"] > 0


def test_mixed_author_year_query_prioritizes_matching_paper():
    graph = _build_graph_for_search()
    service = PaperSearchService(graph_obj=graph, embed_query_fn=lambda _q: [1.0, 0.0, 0.0])

    results = service.search_papers("chen and bornstein 2024", top_k=3)

    assert results[0]["title"] == "Neurosymbolic Reasoning with Logic Programs"


def test_embedding_failure_falls_back_to_lexical():
    graph = _build_graph_for_search()

    def _raise_embed(_query):
        raise RuntimeError("embedding unavailable")

    service = PaperSearchService(graph_obj=graph, embed_query_fn=_raise_embed)
    results = service.search_papers("logic programs", top_k=3)

    assert results
    assert results[0]["title"] == "Neurosymbolic Reasoning with Logic Programs"
    assert results[0]["score_breakdown"]["semantic_score"] == 0


def test_payload_includes_expected_contract_fields():
    graph = _build_graph_for_search()
    service = PaperSearchService(graph_obj=graph, embed_query_fn=lambda _q: [1.0, 0.0, 0.0])

    result = service.search_papers("neurosymbolic ai", top_k=1)[0]

    assert set(result.keys()) >= {
        "title",
        "authors",
        "publication_date",
        "topics",
        "summary",
        "score",
        "score_breakdown",
    }
    assert set(result["score_breakdown"].keys()) >= {
        "author_score",
        "title_score",
        "topic_score",
        "text_score",
        "semantic_score",
        "year_boost",
    }
