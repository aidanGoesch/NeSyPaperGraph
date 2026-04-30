from typing import Any

from models.graph import PaperGraph
from models.paper import Paper


def _to_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json_value(inner) for key, inner in value.items()}
    if isinstance(value, set):
        return [_to_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_value(item) for item in value]
    if isinstance(value, list):
        return [_to_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def serialize_graph(graph_obj: PaperGraph) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    for node_id, attrs in graph_obj.graph.nodes(data=True):
        node_payload: dict[str, Any] = {
            "id": node_id,
            "type": attrs.get("type"),
        }
        if attrs.get("type") == "paper" and attrs.get("data") is not None:
            paper_data = attrs["data"]
            if isinstance(paper_data, Paper):
                node_payload["paper"] = paper_data.model_dump()
            else:
                node_payload["paper"] = _to_json_value(paper_data)
        if "merged_topics" in attrs:
            node_payload["merged_topics"] = _to_json_value(attrs["merged_topics"])
        nodes.append(node_payload)

    edges: list[dict[str, Any]] = []
    for source, target, edge_attrs in graph_obj.graph.edges(data=True):
        edges.append(
            {
                "source": source,
                "target": target,
                "attributes": _to_json_value(edge_attrs or {}),
            }
        )

    return {
        "version": 1,
        "nodes": nodes,
        "edges": edges,
        "topic_synonyms": _to_json_value(getattr(graph_obj, "topic_synonyms", {})),
        "topic_merge_groups": _to_json_value(
            getattr(graph_obj, "topic_merge_groups", {})
        ),
        "paper_content_hashes": _to_json_value(
            getattr(graph_obj, "paper_content_hashes", set())
        ),
    }


def deserialize_graph(payload: dict[str, Any]) -> PaperGraph:
    graph_obj = PaperGraph()
    graph_obj.graph.clear()

    for node in payload.get("nodes", []):
        node_id = node.get("id")
        node_type = node.get("type")
        if not node_id or not node_type:
            continue
        if node_type == "paper":
            paper_payload = node.get("paper") or {}
            paper = Paper(**paper_payload)
            graph_obj.graph.add_node(node_id, type="paper", data=paper)
        else:
            graph_obj.graph.add_node(node_id, type=node_type)

        merged_topics = node.get("merged_topics")
        if merged_topics is not None:
            graph_obj.graph.nodes[node_id]["merged_topics"] = list(merged_topics)

    for edge in payload.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        if not source or not target:
            continue
        edge_attrs = edge.get("attributes") or {}
        graph_obj.graph.add_edge(source, target, **edge_attrs)

    graph_obj.topic_synonyms = dict(payload.get("topic_synonyms") or {})
    graph_obj.topic_merge_groups = dict(payload.get("topic_merge_groups") or {})
    graph_obj.paper_content_hashes = set(payload.get("paper_content_hashes") or [])
    graph_obj.clear_incremental_tracking()
    return graph_obj
