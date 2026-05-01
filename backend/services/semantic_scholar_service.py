import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError


SEMANTIC_SCHOLAR_BASE_URL = os.getenv(
    "SEMANTIC_SCHOLAR_BASE_URL", "https://api.semanticscholar.org/graph/v1"
)
SEMANTIC_SCHOLAR_RECOMMENDATIONS_BASE_URL = os.getenv(
    "SEMANTIC_SCHOLAR_RECOMMENDATIONS_BASE_URL",
    "https://api.semanticscholar.org/recommendations/v1",
)
SEMANTIC_SCHOLAR_TIMEOUT_SECONDS = float(
    os.getenv("SEMANTIC_SCHOLAR_TIMEOUT_SECONDS", "10") or "10"
)
SEMANTIC_SCHOLAR_CACHE_TTL_SECONDS = int(
    os.getenv("SEMANTIC_SCHOLAR_CACHE_TTL_SECONDS", "21600") or "21600"
)
SEMANTIC_SCHOLAR_MAX_RETRIES = int(
    os.getenv("SEMANTIC_SCHOLAR_MAX_RETRIES", "2") or "2"
)
SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS = float(
    os.getenv("SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS", "1.0") or "1.0"
)
SEMANTIC_SCHOLAR_BACKOFF_BASE_SECONDS = float(
    os.getenv("SEMANTIC_SCHOLAR_BACKOFF_BASE_SECONDS", "1.0") or "1.0"
)
SEMANTIC_SCHOLAR_BACKOFF_MAX_SECONDS = float(
    os.getenv("SEMANTIC_SCHOLAR_BACKOFF_MAX_SECONDS", "8.0") or "8.0"
)


class SemanticScholarError(Exception):
    pass


class SemanticScholarRateLimitError(SemanticScholarError):
    pass


@dataclass
class ResolvedReference:
    identifiers: list[str]
    source_url: str


class SemanticScholarService:
    _metadata_cache: dict[str, tuple[float, dict[str, Any]]] = {}
    _request_lock = threading.Lock()
    _last_request_monotonic = 0.0

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()

    def resolve_url_metadata(self, url: str) -> dict[str, Any] | None:
        hydrated = self.hydrate_paper(url=url, paper_id=None)
        if hydrated is None:
            return None
        return {
            "url": hydrated.get("url") or url,
            "semanticScholarPaperId": hydrated.get("semanticScholarPaperId"),
            "title": hydrated.get("title"),
            "authors": hydrated.get("authors") or [],
            "year": hydrated.get("year"),
            "venue": hydrated.get("venue"),
        }

    def hydrate_paper(self, url: str, paper_id: str | None = None) -> dict[str, Any] | None:
        cache_key = self._cache_key(url, paper_id)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        resolved = self._resolve_reference(url=url, paper_id=paper_id)
        if resolved is None:
            return None

        payload = None
        for identifier in resolved.identifiers:
            payload = self._request_json(
                f"/paper/{quote(identifier, safe=':')}",
                query="fields=paperId,title,authors,year,venue,abstract,url,externalIds",
                suppress_not_found=True,
            )
            if payload:
                break
        if not payload:
            return None
        normalized = self._normalize_paper_payload(payload, fallback_url=resolved.source_url)
        self._set_cached(cache_key, normalized)
        if normalized.get("semanticScholarPaperId"):
            self._set_cached(
                self._cache_key(url=resolved.source_url, paper_id=normalized["semanticScholarPaperId"]),
                normalized,
            )
        return normalized

    def find_similar_papers(self, paper_id: str, limit: int = 10) -> list[dict[str, Any]]:
        paper_id = (paper_id or "").strip()
        if not paper_id:
            raise ValueError("semanticScholarPaperId is required")
        source = self.fetch_paper_details(paper_id)
        if not source:
            return []
        candidate_pool = max(limit * 4, 20)
        try:
            recommendations = self.fetch_recommended_papers_for_paper(
                paper_id=paper_id,
                limit=candidate_pool,
            )
        except SemanticScholarError:
            recommendations = []
        filtered = self._exclude_source_paper(
            recommendations, source_paper_id=paper_id
        )

        if not filtered:
            candidates = self.search_papers(
                query=self._build_query_from_paper(source), limit=candidate_pool
            )
            filtered = self._exclude_source_paper(candidates, source_paper_id=paper_id)

        # Some papers only return themselves for strict title+abstract queries.
        # Fall back to broader query variants so users still get recommendations.
        if not filtered:
            for fallback_query in self._build_fallback_queries_from_paper(source):
                fallback_candidates = self.search_papers(
                    query=fallback_query, limit=candidate_pool
                )
                filtered = self._merge_unique_candidates(
                    filtered,
                    self._exclude_source_paper(
                        fallback_candidates, source_paper_id=paper_id
                    ),
                )
                if len(filtered) >= limit:
                    break
        source_embedding = self.extract_embedding_vector(source.get("embedding"))
        if source_embedding:
            return self.rank_candidates_by_embedding_similarity(
                profile_embedding=source_embedding,
                candidates=filtered,
                limit=limit,
                reason_prefix="Similar to selected paper",
            )
        return filtered[: max(1, limit)]

    def find_similar_papers_from_seed(
        self, seed_paper: dict[str, Any], limit: int = 10
    ) -> list[dict[str, Any]]:
        source = self.resolve_seed_paper_details(seed_paper)
        if not source:
            raise ValueError("Unable to resolve the selected paper in Semantic Scholar.")
        paper_id = (source.get("paperId") or "").strip()
        if not paper_id:
            raise ValueError("Unable to resolve a Semantic Scholar paper id.")
        return self.find_similar_papers(paper_id=paper_id, limit=limit)

    def recommend_for_theme(
        self,
        *,
        theme_title: str,
        seed_papers: list[dict[str, Any]],
        limit: int = 10,
        candidate_pool_size: int = 40,
    ) -> list[dict[str, Any]]:
        profile_embedding = self._build_profile_embedding(seed_papers)
        candidate_pool = max(limit, candidate_pool_size)
        candidates: list[dict[str, Any]] = []

        # Prefer seed-driven neighborhoods first so recommendations are anchored to
        # to-read/theme papers, not just the theme title query.
        seed_candidates = self._collect_seed_neighborhood_candidates(
            seed_papers=seed_papers,
            candidate_pool_size=candidate_pool,
        )
        candidates = self._merge_unique_candidates(candidates, seed_candidates)

        query = self._build_theme_query(theme_title=theme_title, seed_papers=seed_papers)
        if len(candidates) < candidate_pool:
            candidates = self._merge_unique_candidates(
                candidates,
                self.search_papers(query=query, limit=candidate_pool),
            )
        if len(candidates) < max(1, limit):
            for fallback_query in self._build_theme_fallback_queries(
                theme_title=theme_title, seed_papers=seed_papers
            ):
                fallback_candidates = self.search_papers(
                    query=fallback_query, limit=candidate_pool
                )
                candidates = self._merge_unique_candidates(candidates, fallback_candidates)
                if len(candidates) >= candidate_pool:
                    break
        if not profile_embedding:
            return candidates[: max(1, limit)]
        return self.rank_candidates_by_embedding_similarity(
            profile_embedding=profile_embedding,
            candidates=candidates,
            limit=limit,
            reason_prefix="Relevant to selected theme",
        )

    def _collect_seed_neighborhood_candidates(
        self, *, seed_papers: list[dict[str, Any]], candidate_pool_size: int
    ) -> list[dict[str, Any]]:
        if not seed_papers:
            return []
        merged: list[dict[str, Any]] = []
        seed_count = max(1, min(len(seed_papers), 6))
        per_seed_limit = max(4, candidate_pool_size // seed_count)

        for seed in seed_papers[:6]:
            details = None
            try:
                details = self.resolve_seed_paper_details(seed)
            except SemanticScholarError:
                details = None
            if not details:
                continue

            seed_paper_id = (details.get("paperId") or "").strip()
            if seed_paper_id:
                try:
                    recommended = self.fetch_recommended_papers_for_paper(
                        paper_id=seed_paper_id, limit=per_seed_limit
                    )
                except SemanticScholarError:
                    recommended = []
                merged = self._merge_unique_candidates(
                    merged,
                    self._exclude_source_paper(
                        recommended, source_paper_id=seed_paper_id
                    ),
                )

            seed_title = (details.get("title") or seed.get("title") or "").strip()
            if seed_title:
                merged = self._merge_unique_candidates(
                    merged,
                    self.search_papers(query=seed_title, limit=max(4, per_seed_limit // 2)),
                )

            if len(merged) >= candidate_pool_size:
                break
        return merged[:candidate_pool_size]

    def recommend_for_topic_profile(
        self,
        *,
        topic_query: str,
        seed_papers: list[dict[str, Any]],
        limit: int = 10,
        candidate_pool_size: int = 40,
    ) -> list[dict[str, Any]]:
        profile_embedding = self._build_profile_embedding(seed_papers)
        expanded_query = self._expand_topic_query(topic_query, seed_papers)
        candidates = self.search_papers(
            query=expanded_query,
            limit=max(limit, candidate_pool_size),
        )
        if not profile_embedding:
            return candidates[: max(1, limit)]
        return self.rank_candidates_by_embedding_similarity(
            profile_embedding=profile_embedding,
            candidates=candidates,
            limit=limit,
            reason_prefix="Matches reading profile",
        )

    def fetch_paper_details(self, paper_id: str) -> dict[str, Any] | None:
        paper_id = (paper_id or "").strip()
        if not paper_id:
            return None
        payload = self._request_json(
            f"/paper/{quote(paper_id, safe=':')}",
            query="fields=paperId,title,authors,year,venue,abstract,url,embedding",
            suppress_not_found=True,
        )
        if not payload:
            return None
        normalized = self._normalize_candidate(payload)
        normalized["embedding"] = payload.get("embedding")
        return normalized

    def fetch_recommended_papers_for_paper(
        self, *, paper_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        normalized_paper_id = (paper_id or "").strip()
        if not normalized_paper_id:
            return []
        payload = self._request_json(
            f"/papers/forpaper/{quote(normalized_paper_id, safe=':')}",
            query=(
                f"limit={max(1, min(limit, 100))}"
                "&fields=paperId,title,authors,year,venue,abstract,url,embedding"
            ),
            suppress_not_found=True,
            base_url=SEMANTIC_SCHOLAR_RECOMMENDATIONS_BASE_URL,
        )
        if not payload:
            return []
        data = payload.get("recommendedPapers") or []
        if not isinstance(data, list):
            return []
        return [self._normalize_candidate(item) for item in data if isinstance(item, dict)]

    def search_papers(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return []
        payload = self._request_json(
            "/paper/search",
            query=(
                f"query={quote(cleaned_query)}"
                f"&limit={max(1, min(limit, 100))}"
                "&fields=paperId,title,authors,year,venue,abstract,url,embedding"
            ),
            suppress_not_found=True,
        )
        if not payload:
            return []
        data = payload.get("data") or []
        if not isinstance(data, list):
            return []
        return [self._normalize_candidate(item) for item in data if isinstance(item, dict)]

    def resolve_seed_paper_details(self, seed: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(seed, dict):
            return None
        paper_id = (seed.get("paperId") or seed.get("semanticScholarPaperId") or "").strip()
        if paper_id:
            details = self.fetch_paper_details(paper_id)
            if details:
                return details

        url = (seed.get("url") or "").strip()
        if url:
            hydrated = self.hydrate_paper(url=url, paper_id=None)
            hydrated_id = (hydrated or {}).get("semanticScholarPaperId")
            if hydrated_id:
                details = self.fetch_paper_details(hydrated_id)
                if details:
                    return details

        title = (seed.get("title") or "").strip()
        if title:
            year = seed.get("year")
            authors = seed.get("authors")
            return self._resolve_paper_from_title(title=title, year=year, authors=authors)
        return None

    def compute_centroid_embedding(self, embeddings: list[list[float]]) -> list[float]:
        if not embeddings:
            return []
        width = len(embeddings[0])
        if width == 0:
            return []
        totals = [0.0] * width
        count = 0
        for vector in embeddings:
            if not isinstance(vector, list) or len(vector) != width:
                continue
            try:
                numeric_vector = [float(value) for value in vector]
            except (TypeError, ValueError):
                continue
            totals = [left + right for left, right in zip(totals, numeric_vector)]
            count += 1
        if count == 0:
            return []
        return [value / count for value in totals]

    def rank_candidates_by_embedding_similarity(
        self,
        *,
        profile_embedding: list[float],
        candidates: list[dict[str, Any]],
        limit: int = 10,
        reason_prefix: str = "Embedding similarity",
    ) -> list[dict[str, Any]]:
        if not profile_embedding:
            return candidates[: max(1, limit)]
        scored_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_vector = self.extract_embedding_vector(candidate.get("embedding"))
            if not candidate_vector:
                continue
            score = self._cosine_similarity(profile_embedding, candidate_vector)
            enriched = dict(candidate)
            enriched["score"] = round(score, 6)
            enriched["reason"] = f"{reason_prefix} ({enriched['score']})"
            scored_candidates.append(enriched)

        if not scored_candidates:
            return candidates[: max(1, limit)]

        scored_candidates.sort(key=lambda item: item.get("score", float("-inf")), reverse=True)
        return scored_candidates[: max(1, limit)]

    def extract_embedding_vector(self, embedding_payload: Any) -> list[float] | None:
        if not embedding_payload:
            return None
        vector = None
        if isinstance(embedding_payload, dict):
            vector = embedding_payload.get("vector")
        elif isinstance(embedding_payload, list):
            vector = embedding_payload
        if not isinstance(vector, list) or not vector:
            return None
        try:
            return [float(value) for value in vector]
        except (TypeError, ValueError):
            return None

    def _cache_key(self, url: str, paper_id: str | None) -> str:
        normalized_url = (url or "").strip().lower()
        normalized_paper_id = (paper_id or "").strip().lower()
        return f"{normalized_paper_id}::{normalized_url}"

    def _build_profile_embedding(self, seed_papers: list[dict[str, Any]]) -> list[float]:
        embeddings: list[list[float]] = []
        for seed in seed_papers:
            try:
                details = self.resolve_seed_paper_details(seed)
            except SemanticScholarError:
                continue
            if not details:
                continue
            vector = self.extract_embedding_vector(details.get("embedding"))
            if vector:
                embeddings.append(vector)
        return self.compute_centroid_embedding(embeddings)

    def _build_query_from_paper(self, paper: dict[str, Any]) -> str:
        title = (paper.get("title") or "").strip()
        abstract = (paper.get("abstract") or "").strip()
        if not title and not abstract:
            return "neurosymbolic ai"
        if not abstract:
            return title
        abstract_tokens = abstract.split()
        excerpt = " ".join(abstract_tokens[:20])
        return f"{title} {excerpt}".strip()

    def _build_fallback_queries_from_paper(self, paper: dict[str, Any]) -> list[str]:
        title = (paper.get("title") or "").strip()
        authors = paper.get("authors") or []
        year = str(paper.get("year") or "").strip()
        fallback_queries: list[str] = []
        if title:
            fallback_queries.append(title)
            title_tokens = title.split()
            if len(title_tokens) > 6:
                fallback_queries.append(" ".join(title_tokens[:6]))
        if authors:
            primary_author = ""
            for author in authors:
                if isinstance(author, str) and author.strip():
                    primary_author = author.strip()
                    break
            if primary_author and title:
                author_query = f"{title} {primary_author}"
                if year:
                    author_query = f"{author_query} {year}"
                fallback_queries.append(author_query)
        # Preserve order, drop duplicates/empties.
        return [query for query in dict.fromkeys(fallback_queries) if query]

    def _exclude_source_paper(
        self, candidates: list[dict[str, Any]], *, source_paper_id: str
    ) -> list[dict[str, Any]]:
        return [
            candidate
            for candidate in candidates
            if (candidate.get("paperId") or "").strip()
            and candidate.get("paperId") != source_paper_id
        ]

    def _merge_unique_candidates(
        self, existing: list[dict[str, Any]], incoming: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        merged = list(existing)
        seen_ids = {
            (candidate.get("paperId") or "").strip().lower()
            for candidate in merged
            if (candidate.get("paperId") or "").strip()
        }
        for candidate in incoming:
            candidate_id = (candidate.get("paperId") or "").strip().lower()
            if not candidate_id or candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)
            merged.append(candidate)
        return merged

    def _build_theme_query(self, theme_title: str, seed_papers: list[dict[str, Any]]) -> str:
        title = (theme_title or "").strip()
        seed_terms = []
        for seed in seed_papers[:5]:
            seed_title = (seed.get("title") or "").strip()
            if seed_title:
                seed_terms.append(seed_title)
        if seed_terms:
            return f"{title} {' '.join(seed_terms)}".strip()
        return title or "neurosymbolic ai"

    def _build_theme_fallback_queries(
        self, *, theme_title: str, seed_papers: list[dict[str, Any]]
    ) -> list[str]:
        fallback_queries: list[str] = []
        title = (theme_title or "").strip()
        if title:
            fallback_queries.append(title)

        seed_titles = [
            (seed.get("title") or "").strip()
            for seed in seed_papers[:8]
            if (seed.get("title") or "").strip()
        ]
        fallback_queries.extend(seed_titles[:4])
        if title and seed_titles:
            fallback_queries.append(f"{title} {' '.join(seed_titles[:2])}".strip())

        topic_hints: list[str] = []
        for seed in seed_papers[:10]:
            for hint in seed.get("topicHints") or []:
                if isinstance(hint, str) and hint.strip():
                    topic_hints.append(hint.strip())
        unique_hints = list(dict.fromkeys(topic_hints))[:4]
        if unique_hints:
            hint_query = " ".join(unique_hints)
            fallback_queries.append(
                f"{title} {hint_query}".strip() if title else hint_query
            )

        return [query for query in dict.fromkeys(fallback_queries) if query]

    def _expand_topic_query(self, topic_query: str, seed_papers: list[dict[str, Any]]) -> str:
        base_query = (topic_query or "").strip() or "neurosymbolic ai"
        topic_hints = []
        for seed in seed_papers:
            for hint in seed.get("topicHints") or []:
                if isinstance(hint, str) and hint.strip():
                    topic_hints.append(hint.strip())
        uniq_hints = list(dict.fromkeys(topic_hints))[:10]
        if not uniq_hints:
            return base_query
        return f"{base_query} {' '.join(uniq_hints)}".strip()

    def _normalize_candidate(self, payload: dict[str, Any]) -> dict[str, Any]:
        authors = payload.get("authors") or []
        normalized_authors = []
        for author in authors:
            if isinstance(author, dict):
                name = (author.get("name") or "").strip()
                if name:
                    normalized_authors.append(name)
            elif isinstance(author, str) and author.strip():
                normalized_authors.append(author.strip())
        return {
            "paperId": payload.get("paperId"),
            "title": payload.get("title"),
            "authors": normalized_authors,
            "year": payload.get("year"),
            "venue": payload.get("venue"),
            "abstract": payload.get("abstract") or "",
            "url": payload.get("url"),
            "embedding": payload.get("embedding"),
        }

    def _resolve_paper_from_title(
        self, *, title: str, year: Any = None, authors: Any = None
    ) -> dict[str, Any] | None:
        candidates = self.search_papers(query=title, limit=8)
        if not candidates:
            return None
        normalized_title = self._normalize_title(title)
        normalized_authors = {
            self._normalize_title(author)
            for author in (authors or [])
            if isinstance(author, str) and author.strip()
        }
        best_candidate = None
        best_score = float("-inf")
        for candidate in candidates:
            score = 0.0
            candidate_title = self._normalize_title(candidate.get("title") or "")
            if candidate_title == normalized_title:
                score += 4.0
            elif normalized_title and normalized_title in candidate_title:
                score += 2.0

            candidate_year = candidate.get("year")
            if year and candidate_year and str(year) == str(candidate_year):
                score += 1.0

            candidate_authors = {
                self._normalize_title(author)
                for author in (candidate.get("authors") or [])
                if isinstance(author, str)
            }
            if normalized_authors and candidate_authors.intersection(normalized_authors):
                score += 1.0

            if score > best_score:
                best_score = score
                best_candidate = candidate

        if not best_candidate:
            return None
        # Keep the candidate directly to avoid an extra API round-trip for each
        # to-read seed title. Search results already include embeddings when present.
        candidate_id = (best_candidate.get("paperId") or "").strip()
        if not candidate_id:
            return None
        return best_candidate

    def _normalize_title(self, value: str) -> str:
        return re.sub(r"\s+", " ", (value or "").strip().lower())

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return -1.0
        dot_product = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for left_val, right_val in zip(left, right):
            dot_product += left_val * right_val
            left_norm += left_val * left_val
            right_norm += right_val * right_val
        if left_norm <= 0.0 or right_norm <= 0.0:
            return -1.0
        return dot_product / (math.sqrt(left_norm) * math.sqrt(right_norm))

    def _get_cached(self, cache_key: str) -> dict[str, Any] | None:
        item = self._metadata_cache.get(cache_key)
        if not item:
            return None
        expires_at, payload = item
        if expires_at < time.time():
            self._metadata_cache.pop(cache_key, None)
            return None
        return payload

    def _set_cached(self, cache_key: str, payload: dict[str, Any]) -> None:
        if not payload:
            return
        self._metadata_cache[cache_key] = (
            time.time() + max(1, SEMANTIC_SCHOLAR_CACHE_TTL_SECONDS),
            payload,
        )

    def _resolve_reference(self, url: str, paper_id: str | None) -> ResolvedReference | None:
        if paper_id and str(paper_id).strip():
            return ResolvedReference(
                identifiers=[str(paper_id).strip()],
                source_url=url,
            )

        pmc_match = re.search(
            r"pmc\.ncbi\.nlm\.nih\.gov/articles/(PMC[0-9]+)",
            url,
            re.IGNORECASE,
        )
        if pmc_match:
            pmc_id = pmc_match.group(1).upper()
            pmc_identifiers = self._resolve_pmc_identifiers(pmc_id)
            if pmc_identifiers:
                return ResolvedReference(
                    identifiers=[*pmc_identifiers, f"URL:{url}"],
                    source_url=url,
                )

        doi_match = re.search(r"doi\.org/(10\.\d{4,9}/[^\s?#]+)", url, re.IGNORECASE)
        if doi_match:
            doi_identifier = f"DOI:{doi_match.group(1)}"
            identifiers = [doi_identifier]
            # Common DOI variant for arXiv entries: 10.48550/arXiv.xxxxx
            arxiv_in_doi = re.search(
                r"10\.48550/arxiv\.([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)",
                doi_match.group(1),
                re.IGNORECASE,
            )
            if arxiv_in_doi:
                arxiv_id = arxiv_in_doi.group(1)
                identifiers.append(f"ARXIV:{arxiv_id}")
                stripped = re.sub(r"v[0-9]+$", "", arxiv_id, flags=re.IGNORECASE)
                if stripped != arxiv_id:
                    identifiers.append(f"ARXIV:{stripped}")
            return ResolvedReference(
                identifiers=identifiers,
                source_url=url,
            )

        arxiv_match = re.search(r"arxiv\.org/(abs|pdf)/([^/?#]+)", url, re.IGNORECASE)
        if arxiv_match:
            arxiv_id = arxiv_match.group(2).replace(".pdf", "")
            identifiers = [f"ARXIV:{arxiv_id}"]
            arxiv_id_without_version = re.sub(
                r"v[0-9]+$",
                "",
                arxiv_id,
                flags=re.IGNORECASE,
            )
            if arxiv_id_without_version != arxiv_id:
                identifiers.append(f"ARXIV:{arxiv_id_without_version}")
            identifiers.append(f"URL:{url}")
            return ResolvedReference(
                identifiers=identifiers,
                source_url=url,
            )

        paper_id_match = re.search(r"/paper/[^/]+/([a-f0-9]{20,})", url, re.IGNORECASE)
        if paper_id_match:
            return ResolvedReference(
                identifiers=[paper_id_match.group(1)],
                source_url=url,
            )

        url_encoded_ref = f"URL:{url}"
        direct_url_payload = self._request_json(
            f"/paper/{quote(url_encoded_ref, safe=':')}",
            query="fields=paperId",
            suppress_not_found=True,
        )
        if direct_url_payload and direct_url_payload.get("paperId"):
            return ResolvedReference(
                identifiers=[direct_url_payload["paperId"], url_encoded_ref],
                source_url=url,
            )

        host = urlparse(url).netloc.lower()
        if "arxiv.org" in host:
            # arXiv URL pattern existed but failed to parse into id.
            return None

        return None

    def _resolve_pmc_identifiers(self, pmc_id: str) -> list[str]:
        """
        Resolve PMC IDs (e.g. PMC11058347) to identifiers that Semantic Scholar
        can resolve more reliably, such as DOI/PMID.
        """
        lookup_url = (
            "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
            f"?ids={quote(pmc_id)}&format=json"
        )
        request = Request(
            url=lookup_url,
            headers={"User-Agent": "NeSyPaperGraph/1.0"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=SEMANTIC_SCHOLAR_TIMEOUT_SECONDS) as response:
                raw = response.read().decode("utf-8")
                payload = json.loads(raw) if raw else {}
        except Exception:
            return []

        records = payload.get("records") or []
        if not records:
            return []
        record = records[0] if isinstance(records[0], dict) else {}
        identifiers: list[str] = []

        doi = (record.get("doi") or "").strip()
        if doi:
            identifiers.append(f"DOI:{doi}")

        pmid = str(record.get("pmid") or "").strip()
        if pmid and pmid.isdigit():
            identifiers.append(f"PMID:{pmid}")

        return identifiers

    def _normalize_paper_payload(
        self, payload: dict[str, Any], fallback_url: str
    ) -> dict[str, Any]:
        authors = payload.get("authors") or []
        normalized_authors = []
        for author in authors:
            if isinstance(author, dict):
                name = (author.get("name") or "").strip()
                if name:
                    normalized_authors.append(name)
            elif isinstance(author, str) and author.strip():
                normalized_authors.append(author.strip())

        year = payload.get("year")
        if isinstance(year, str) and year.isdigit():
            year = int(year)

        return {
            "url": payload.get("url") or fallback_url,
            "semanticScholarPaperId": payload.get("paperId"),
            "title": payload.get("title"),
            "authors": normalized_authors,
            "year": year,
            "venue": payload.get("venue"),
            "abstract": payload.get("abstract") or "",
        }

    def _request_json(
        self,
        path: str,
        query: str | None = None,
        suppress_not_found: bool = False,
        base_url: str | None = None,
    ) -> dict[str, Any] | None:
        root_url = (base_url or SEMANTIC_SCHOLAR_BASE_URL).rstrip("/")
        url = f"{root_url}{path}"
        if query:
            url = f"{url}?{query}"
        headers = {"User-Agent": "NeSyPaperGraph/1.0"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        request = Request(url=url, headers=headers, method="GET")
        last_error: Exception | None = None
        for attempt in range(max(0, SEMANTIC_SCHOLAR_MAX_RETRIES) + 1):
            try:
                self._wait_for_rate_limit_slot()
                with urlopen(request, timeout=SEMANTIC_SCHOLAR_TIMEOUT_SECONDS) as response:
                    raw = response.read().decode("utf-8")
                    if not raw:
                        return None
                    return json.loads(raw)
            except HTTPError as exc:
                if exc.code == 404:
                    if suppress_not_found:
                        return None
                    return None

                body = ""
                try:
                    body = exc.read().decode("utf-8", "ignore")
                except Exception:
                    body = ""

                if exc.code == 429:
                    if attempt < SEMANTIC_SCHOLAR_MAX_RETRIES:
                        retry_after_header = exc.headers.get("Retry-After")
                        retry_after_seconds = self._compute_backoff_seconds(attempt)
                        if retry_after_header:
                            try:
                                retry_after_seconds = max(
                                    retry_after_seconds, float(retry_after_header)
                                )
                            except ValueError:
                                pass
                        time.sleep(retry_after_seconds)
                        continue
                    guidance = ""
                    if not self.api_key:
                        guidance = (
                            " Configure SEMANTIC_SCHOLAR_API_KEY in backend/.env and restart "
                            "the backend to avoid anonymous quota limits."
                        )
                    raise SemanticScholarRateLimitError(
                        "Semantic Scholar rate limit exceeded."
                        + guidance
                        + (f" Upstream response: {body}" if body else "")
                    ) from exc

                raise SemanticScholarError(
                    f"Semantic Scholar request failed with HTTP {exc.code}: {body or exc.reason}"
                ) from exc
            except Exception as exc:
                last_error = exc
                if attempt < SEMANTIC_SCHOLAR_MAX_RETRIES:
                    time.sleep(self._compute_backoff_seconds(attempt))
                    continue
                break

        raise SemanticScholarError(
            f"Semantic Scholar request failed: {last_error}"
        ) from last_error

    def _wait_for_rate_limit_slot(self) -> None:
        min_interval = max(0.0, SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS)
        if min_interval <= 0:
            return
        with self._request_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_monotonic
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_request_monotonic = time.monotonic()

    def _compute_backoff_seconds(self, attempt: int) -> float:
        base = max(0.1, SEMANTIC_SCHOLAR_BACKOFF_BASE_SECONDS)
        cap = max(base, SEMANTIC_SCHOLAR_BACKOFF_MAX_SECONDS)
        return min(cap, base * (2**max(0, attempt)))
