from models.graph import PaperGraph
from models.paper import Paper
from services.graph_activation_service import ActivationConfig, GraphActivationService


def make_graph() -> PaperGraph:
    graph = PaperGraph()
    paper_a = Paper(
        title="Paper A",
        file_path="a.pdf",
        summary="A paper about neurosymbolic AI.",
        topics=["Neurosymbolic AI", "Reasoning"],
        embedding=[1.0, 0.0, 0.0],
        authors=["Ada"],
    )
    paper_b = Paper(
        title="Paper B",
        file_path="b.pdf",
        summary="A paper about program repair.",
        topics=["Program Repair"],
        embedding=[0.2, 0.9, 0.0],
        authors=["Grace"],
    )
    paper_c = Paper(
        title="Paper C",
        file_path="c.pdf",
        summary="A paper about theorem proving.",
        topics=["Reasoning", "Theorem Proving"],
        embedding=[0.8, 0.1, 0.0],
        authors=["Ada"],
    )
    graph.add_paper(paper_a)
    graph.add_paper(paper_b)
    graph.add_paper(paper_c)
    graph.graph.add_edge("Paper A", "Paper C", type="semantic", weight=0.7)
    return graph


def test_select_seed_nodes_prefers_closest_embedding():
    graph = make_graph()
    service = GraphActivationService(graph, embed_fn=lambda _text: [1.0, 0.0, 0.0])
    seeds = service.select_seed_nodes([1.0, 0.0, 0.0], seed_count=3)
    assert seeds
    assert seeds[0]["node_id"] == "Paper A"
    assert seeds[0]["score"] >= seeds[-1]["score"]


def test_random_surfer_is_deterministic_with_seed():
    graph = make_graph()
    service = GraphActivationService(graph, embed_fn=lambda _text: [1.0, 0.0, 0.0])
    seeds = [{"node_id": "Paper A", "node_type": "paper", "score": 1.0}]
    activated_a, trace_a = service.random_surfer(
        seed_nodes=seeds,
        surfer_steps=40,
        restart_probability=0.2,
        rng_seed=11,
        max_activated_nodes=20,
    )
    activated_b, trace_b = service.random_surfer(
        seed_nodes=seeds,
        surfer_steps=40,
        restart_probability=0.2,
        rng_seed=11,
        max_activated_nodes=20,
    )
    assert activated_a == activated_b
    assert trace_a == trace_b


def test_activate_returns_normalized_scores_and_steps():
    graph = make_graph()
    service = GraphActivationService(graph, embed_fn=lambda _text: [0.9, 0.1, 0.0])
    payload = service.activate(
        question="How does neurosymbolic reasoning connect to theorem proving?",
        conversation_history=[{"question": "Earlier question", "answer": "Earlier answer"}],
        config=ActivationConfig(
            seed_count=4,
            surfer_steps=35,
            restart_probability=0.18,
            rng_seed=3,
            max_activated_nodes=12,
        ),
    )
    assert payload["seed_nodes"]
    assert payload["activated_nodes"]
    assert payload["step_trace"]
    scores = [node["score"] for node in payload["activated_nodes"]]
    assert max(scores) <= 1.0
    assert min(scores) >= 0.0
    assert payload["step_trace"][0]["step"] == 0


def test_build_query_text_uses_questions_only_and_recency_weighting():
    graph = make_graph()
    service = GraphActivationService(graph, embed_fn=lambda _text: [0.9, 0.1, 0.0])
    history = [
        {"question": "oldest question", "answer": "oldest answer should not be embedded"},
        {"question": "middle question", "answer": "middle answer should not be embedded"},
        {"question": "newest history question", "answer": "newest answer should not be embedded"},
    ]

    query_text = service._build_query_text(
        question="current question",
        conversation_history=history,
    )

    assert "A:" not in query_text
    assert "oldest answer should not be embedded" not in query_text
    assert "newest answer should not be embedded" not in query_text
    assert query_text.count("Q: oldest question") == 0
    assert query_text.count("Q: middle question") == 1
    assert query_text.count("Q: newest history question") == 3
    assert query_text.count("Q: current question") == 8


def test_activate_uses_retrieval_query_override():
    graph = make_graph()
    captured = {}

    def fake_embed(text):
        captured["text"] = text
        return [0.9, 0.1, 0.0]

    service = GraphActivationService(graph, embed_fn=fake_embed)
    payload = service.activate(
        question="ignored question for embedding",
        conversation_history=[{"question": "old q", "answer": "old a"}],
        retrieval_query_text="override retrieval query",
    )
    assert payload["seed_nodes"]
    assert captured["text"] == "override retrieval query"
