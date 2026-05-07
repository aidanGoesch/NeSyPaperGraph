import random
from dataclasses import dataclass
from typing import Any, Callable

from models.graph import cosine_similarity


EmbeddingFn = Callable[[str], list[float]]


@dataclass
class ActivationConfig:
    seed_count: int = 6
    surfer_steps: int = 100
    restart_probability: float = 0.2
    rng_seed: int = 7
    max_activated_nodes: int = 60


class GraphActivationService:
    """Deterministic memory activation over graph nodes for chat turns."""

    def __init__(self, graph_obj: Any, embed_fn: EmbeddingFn):
        self.graph_obj = graph_obj
        self.embed_fn = embed_fn

    @staticmethod
    def _node_text(node_id: str, node_data: dict[str, Any]) -> str:
        node_type = node_data.get("type")
        if node_type == "paper":
            paper = node_data.get("data")
            title = str(getattr(paper, "title", node_id) or node_id)
            summary = str(getattr(paper, "summary", "") or "")
            topics = ", ".join(getattr(paper, "topics", []) or [])
            return f"{title}\n{summary}\nTopics: {topics}"
        return str(node_id)

    @staticmethod
    def _topic_embedding(topic_id: str, graph_obj: Any) -> list[float]:
        neighbors = list(graph_obj.graph.neighbors(topic_id))
        vectors: list[list[float]] = []
        for neighbor in neighbors:
            neighbor_data = graph_obj.graph.nodes.get(neighbor, {})
            if neighbor_data.get("type") != "paper":
                continue
            paper = neighbor_data.get("data")
            vector = getattr(paper, "embedding", None)
            if isinstance(vector, list) and vector:
                vectors.append(vector)
        if not vectors:
            return []
        width = len(vectors[0])
        sums = [0.0] * width
        for vector in vectors:
            if len(vector) != width:
                continue
            for idx, value in enumerate(vector):
                sums[idx] += value
        count = max(1, len(vectors))
        return [total / count for total in sums]

    def _node_embedding(self, node_id: str, node_data: dict[str, Any]) -> list[float]:
        node_type = node_data.get("type")
        if node_type == "paper":
            paper = node_data.get("data")
            vector = getattr(paper, "embedding", None)
            if isinstance(vector, list):
                return vector
            return []
        if node_type == "topic":
            return self._topic_embedding(node_id, self.graph_obj)
        return []

    def _build_query_text(self, question: str, conversation_history: list[dict[str, str]]) -> str:
        # Question-only context with recency weighting.
        # Aggressive recency decay: current question dominates, only the most
        # recent history gets meaningful weight.
        recent = conversation_history[-4:] if conversation_history else []
        lines = []
        question_texts: list[str] = []
        for turn in recent:
            q = str(turn.get("question", "") or "").strip()
            if not q:
                continue
            question_texts.append(q[:280])

        weighted_questions: list[str] = []
        # Traverse from newest to oldest with steep decay.
        for recency_index, q in enumerate(reversed(question_texts)):
            if recency_index == 0:
                repeats = 3
            elif recency_index == 1:
                repeats = 1
            else:
                repeats = 0
            if repeats > 0:
                weighted_questions.extend([q] * repeats)

        current_question = str(question or "").strip()[:320]
        if current_question:
            weighted_questions.extend([current_question] * 8)
        for item in weighted_questions:
            lines.append(f"Q: {item}")
        return "\n".join(lines)

    def select_seed_nodes(
        self, query_embedding: list[float], seed_count: int
    ) -> list[dict[str, Any]]:
        if not self.graph_obj or not query_embedding:
            return []
        scored_nodes: list[dict[str, Any]] = []
        for node_id, node_data in self.graph_obj.graph.nodes(data=True):
            vector = self._node_embedding(node_id, node_data)
            if not vector:
                continue
            score = cosine_similarity(query_embedding, vector)
            if score <= 0:
                continue
            scored_nodes.append(
                {
                    "node_id": str(node_id),
                    "node_type": node_data.get("type", "unknown"),
                    "score": round(float(score), 6),
                }
            )
        scored_nodes.sort(key=lambda item: item["score"], reverse=True)
        return scored_nodes[: max(1, seed_count)]

    def random_surfer(
        self,
        seed_nodes: list[dict[str, Any]],
        surfer_steps: int,
        restart_probability: float,
        rng_seed: int,
        max_activated_nodes: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not self.graph_obj or not seed_nodes:
            return [], []

        seed_ids = [item["node_id"] for item in seed_nodes]
        rng = random.Random(rng_seed)
        visit_counts: dict[str, int] = {node_id: 0 for node_id in seed_ids}
        first_seen_step: dict[str, int] = {}
        step_trace: list[dict[str, Any]] = []
        current = rng.choice(seed_ids)

        for step in range(max(1, surfer_steps)):
            visit_counts[current] = visit_counts.get(current, 0) + 1
            first_seen_step.setdefault(current, step)

            max_visits = max(1, max(visit_counts.values()))
            step_trace.append(
                {
                    "step": step,
                    "node_id": current,
                    "score_after_step": round(visit_counts[current] / max_visits, 6),
                }
            )

            if rng.random() < restart_probability:
                current = rng.choice(seed_ids)
                continue

            neighbors = list(self.graph_obj.graph.neighbors(current))
            if not neighbors:
                current = rng.choice(seed_ids)
                continue

            # Probability is proportional to already-visited neighborhoods.
            weights = [(visit_counts.get(str(neighbor), 0) + 1) for neighbor in neighbors]
            current = str(rng.choices(neighbors, weights=weights, k=1)[0])

        max_visits = max(1, max(visit_counts.values()))
        activated_nodes: list[dict[str, Any]] = []
        for node_id, visits in visit_counts.items():
            node_data = self.graph_obj.graph.nodes.get(node_id, {})
            activated_nodes.append(
                {
                    "node_id": node_id,
                    "node_type": node_data.get("type", "unknown"),
                    "visits": int(visits),
                    "first_step": int(first_seen_step.get(node_id, 0)),
                    "score": round(visits / max_visits, 6),
                    "is_seed": node_id in seed_ids,
                    "text": self._node_text(node_id, node_data),
                }
            )

        activated_nodes.sort(
            key=lambda item: (item["score"], item["visits"]), reverse=True
        )
        return activated_nodes[: max(1, max_activated_nodes)], step_trace

    def activate(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        config: ActivationConfig | None = None,
        retrieval_query_text: str | None = None,
    ) -> dict[str, Any]:
        run_config = config or ActivationConfig()
        history = conversation_history or []
        query_text = (
            str(retrieval_query_text or "").strip()
            or self._build_query_text(question, history)
        )
        query_embedding = self.embed_fn(query_text) if self.embed_fn else []
        seed_nodes = self.select_seed_nodes(query_embedding, run_config.seed_count)
        activated_nodes, step_trace = self.random_surfer(
            seed_nodes=seed_nodes,
            surfer_steps=run_config.surfer_steps,
            restart_probability=run_config.restart_probability,
            rng_seed=run_config.rng_seed,
            max_activated_nodes=run_config.max_activated_nodes,
        )
        return {
            "query_text": query_text,
            "seed_nodes": seed_nodes,
            "activated_nodes": activated_nodes,
            "step_trace": step_trace,
        }
