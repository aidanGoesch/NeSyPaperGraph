from models.graph import PaperGraph
from models.paper import Paper
from services.graph_serialization import deserialize_graph, serialize_graph


def test_graph_serialization_roundtrip_preserves_content_hashes():
    graph = PaperGraph()
    paper = Paper(
        title="Paper A",
        file_path="a.pdf",
        topics=["Topic A"],
        summary="Summary A",
        content_hash="hash-a",
    )
    graph.add_paper(paper)
    graph.paper_content_hashes.add("hash-a")
    graph.topic_synonyms = {"Topic A": ["Topic Alpha"]}
    graph.topic_merge_groups = {0: ["Topic A", "Topic Alpha"]}

    serialized = serialize_graph(graph)
    restored = deserialize_graph(serialized)

    assert "Paper A" in restored.graph.nodes
    assert restored.paper_content_hashes == {"hash-a"}
    assert restored.topic_synonyms["Topic A"] == ["Topic Alpha"]
    assert restored.topic_merge_groups["0"] == ["Topic A", "Topic Alpha"]
