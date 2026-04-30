import os
import re
from typing import Callable

from models.graph import cosine_similarity


STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "about",
    "paper",
    "papers",
    "find",
    "show",
    "me",
}

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def _normalize_tokens(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall((text or "").lower()) if token]


class PaperSearchService:
    def __init__(
        self,
        graph_obj,
        embed_query_fn: Callable[[str], list[float]] | None = None,
    ):
        self.graph_obj = graph_obj
        self._embed_query_fn = embed_query_fn
        self._llm_client = None

    def _query_embedding(self, query: str) -> list[float] | None:
        if self._embed_query_fn is not None:
            return self._embed_query_fn(query)

        if not os.getenv("OPENAI_API_KEY"):
            return None

        if self._llm_client is None:
            from services.llm_service import OpenAILLMClient

            self._llm_client = OpenAILLMClient()
        return self._llm_client.generate_embedding(query)

    def search_papers(self, query: str, top_k: int = 10) -> list[dict]:
        query = (query or "").strip()
        if not query:
            return []
        if not self.graph_obj:
            return []

        query_tokens = [token for token in _normalize_tokens(query) if token not in STOP_WORDS]
        year_hint_match = YEAR_PATTERN.search(query.lower())
        year_hint = year_hint_match.group(0) if year_hint_match else None

        semantic_vector = None
        try:
            semantic_vector = self._query_embedding(query)
        except Exception:
            semantic_vector = None

        has_author_like_intent = any(
            token not in STOP_WORDS and not token.isdigit() and len(token) > 2
            for token in query_tokens
        )

        scored_results = []
        for node, data in self.graph_obj.graph.nodes(data=True):
            if data.get("type") != "paper" or "data" not in data:
                continue
            paper = data["data"]
            title = paper.title or str(node)
            authors = paper.authors or []
            topics = paper.topics or []
            summary = paper.summary or ""
            paper_text = paper.text or ""
            publication_date = paper.publication_date

            title_text = title.lower()
            authors_text = " ".join(authors).lower()
            topics_text = " ".join(topics).lower()
            body_text = f"{summary} {paper_text}".lower()

            token_count = max(1, len(query_tokens))
            author_matches = sum(1 for token in query_tokens if token in authors_text)
            title_matches = sum(1 for token in query_tokens if token in title_text)
            topic_matches = sum(1 for token in query_tokens if token in topics_text)
            text_matches = sum(1 for token in query_tokens if token in body_text)

            author_score = author_matches / token_count
            title_score = title_matches / token_count
            topic_score = topic_matches / token_count
            text_score = text_matches / token_count

            semantic_score = 0.0
            if semantic_vector and getattr(paper, "embedding", None):
                semantic_score = max(0.0, cosine_similarity(semantic_vector, paper.embedding))

            year_boost = 0.0
            if year_hint and publication_date and year_hint in publication_date:
                year_boost = 1.0

            intent_author_boost = 0.06 * author_score if has_author_like_intent else 0.0
            score = (
                0.28 * title_score
                + 0.26 * author_score
                + 0.18 * topic_score
                + 0.12 * text_score
                + 0.28 * semantic_score
                + 0.1 * year_boost
                + intent_author_boost
            )

            if score <= 0:
                continue

            scored_results.append(
                {
                    "title": title,
                    "authors": authors,
                    "publication_date": publication_date,
                    "topics": topics,
                    "summary": summary,
                    "score": round(float(score), 6),
                    "score_breakdown": {
                        "author_score": round(float(author_score), 6),
                        "title_score": round(float(title_score), 6),
                        "topic_score": round(float(topic_score), 6),
                        "text_score": round(float(text_score), 6),
                        "semantic_score": round(float(semantic_score), 6),
                        "year_boost": round(float(year_boost), 6),
                    },
                }
            )

        scored_results.sort(key=lambda item: item["score"], reverse=True)
        return scored_results[: max(1, top_k)]
