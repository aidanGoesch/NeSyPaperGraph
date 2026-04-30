import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError


SEMANTIC_SCHOLAR_BASE_URL = os.getenv(
    "SEMANTIC_SCHOLAR_BASE_URL", "https://api.semanticscholar.org/graph/v1"
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

    def _cache_key(self, url: str, paper_id: str | None) -> str:
        normalized_url = (url or "").strip().lower()
        normalized_paper_id = (paper_id or "").strip().lower()
        return f"{normalized_paper_id}::{normalized_url}"

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
    ) -> dict[str, Any] | None:
        url = f"{SEMANTIC_SCHOLAR_BASE_URL}{path}"
        if query:
            url = f"{url}?{query}"
        headers = {"User-Agent": "NeSyPaperGraph/1.0"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        request = Request(url=url, headers=headers, method="GET")
        last_error: Exception | None = None
        for attempt in range(max(0, SEMANTIC_SCHOLAR_MAX_RETRIES) + 1):
            try:
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
                        retry_after_seconds = 1.0
                        if retry_after_header and retry_after_header.isdigit():
                            retry_after_seconds = max(0.2, float(retry_after_header))
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
                    time.sleep(0.5 * (attempt + 1))
                    continue
                break

        raise SemanticScholarError(
            f"Semantic Scholar request failed: {last_error}"
        ) from last_error
