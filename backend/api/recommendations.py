from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import re
from typing import Any

from models.workspace import WorkspaceState, default_workspace_state
from services.semantic_scholar_service import (
    SemanticScholarError,
    SemanticScholarRateLimitError,
    SemanticScholarService,
)
from services.storage_service import load_workspace_state

router = APIRouter()


class PaperRecommendationsRequest(BaseModel):
    semanticScholarPaperId: str | None = None
    title: str | None = None
    url: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str | None = None
    limit: int = Field(default=10, ge=1, le=50)


class ThemeRecommendationsRequest(BaseModel):
    themeId: str
    limit: int = Field(default=10, ge=1, le=50)
    candidatePoolSize: int = Field(default=40, ge=5, le=200)
    workspaceState: dict[str, Any] | None = None


class TopicRecommendationsRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=50)
    candidatePoolSize: int = Field(default=40, ge=5, le=200)


def _load_workspace() -> WorkspaceState:
    payload = load_workspace_state()
    if payload is None:
        return default_workspace_state()
    return WorkspaceState.model_validate(payload)


def _seed_from_item(item) -> dict:
    return {
        "paperId": item.semanticScholarPaperId,
        "url": item.url,
        "title": item.title or item.linkedPaperTitle or "",
        "authors": item.authors or [],
        "year": item.year,
        "topicHints": item.topicHints or [],
    }


def _normalize_lookup_value(value: str | None) -> str:
    return (value or "").strip().lower()


def _extract_to_read_mentions(to_read_text: str | None) -> set[str]:
    mentions: set[str] = set()
    if not to_read_text:
        return mentions
    for raw_line in to_read_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*]\s*", "", line)
        line = re.sub(r"^\d+\.\s*", "", line)
        line = re.sub(r"\s*\[[^\]]+\]\s*$", "", line).strip()
        normalized = _normalize_lookup_value(line)
        if normalized:
            mentions.add(normalized)
    return mentions


def _extract_to_read_entries(to_read_text: str | None) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    if not to_read_text:
        return entries
    for raw_line in to_read_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*]\s*", "", line)
        line = re.sub(r"^\d+\.\s*", "", line)
        line = re.sub(r"\s*\[[^\]]+\]\s*$", "", line).strip()
        normalized = _normalize_lookup_value(line)
        if not normalized:
            continue
        entries.append({"raw": line, "normalized": normalized})
    return entries


def _filter_out_existing_theme_seeds(
    results: list[dict], *, seeds: list[dict], linked_titles: set[str], to_read_mentions: set[str]
) -> list[dict]:
    seed_ids = {
        (seed.get("paperId") or "").strip().lower()
        for seed in seeds
        if (seed.get("paperId") or "").strip()
    }
    seed_titles = {
        _normalize_lookup_value(seed.get("title") or "")
        for seed in seeds
        if _normalize_lookup_value(seed.get("title") or "")
    }
    seed_urls = {
        _normalize_lookup_value(seed.get("url") or "")
        for seed in seeds
        if _normalize_lookup_value(seed.get("url") or "")
    }

    filtered = []
    for result in results:
        if not isinstance(result, dict):
            continue
        result_id = (result.get("paperId") or "").strip().lower()
        result_title = _normalize_lookup_value(result.get("title") or "")
        result_url = _normalize_lookup_value(result.get("url") or "")
        if result_id and result_id in seed_ids:
            continue
        if result_title and (
            result_title in seed_titles
            or result_title in linked_titles
            or result_title in to_read_mentions
        ):
            continue
        if result_url and (result_url in seed_urls or result_url in to_read_mentions):
            continue
        filtered.append(result)
    return filtered


def build_theme_recommendations_payload(
    workspace: WorkspaceState, request: ThemeRecommendationsRequest
) -> dict:
    selected_theme = next(
        (theme for theme in workspace.themeNotes if theme.id == request.themeId.strip()),
        None,
    )
    if selected_theme is None:
        raise HTTPException(status_code=404, detail="Theme not found.")

    linked_titles = {
        _normalize_lookup_value(title) for title in selected_theme.linkedPaperTitles if title.strip()
    }
    to_read_mentions = _extract_to_read_mentions(selected_theme.sections.toRead)
    to_read_entries = _extract_to_read_entries(selected_theme.sections.toRead)
    seeds = []
    matched_to_read_mentions: set[str] = set()
    for item in workspace.readingItems:
        title = _normalize_lookup_value(item.title or item.linkedPaperTitle or "")
        url = _normalize_lookup_value(item.url)
        is_linked_to_theme = item.linkedThemeId == selected_theme.id
        has_linked_title = bool(title and title in linked_titles)
        is_mentioned_in_to_read = bool(
            (title and title in to_read_mentions)
            or (url and url in to_read_mentions)
        )
        if is_linked_to_theme or has_linked_title or is_mentioned_in_to_read:
            seeds.append(_seed_from_item(item))
            if is_mentioned_in_to_read:
                if title and title in to_read_mentions:
                    matched_to_read_mentions.add(title)
                if url and url in to_read_mentions:
                    matched_to_read_mentions.add(url)

    for entry in to_read_entries:
        if entry["normalized"] in matched_to_read_mentions:
            continue
        raw_value = entry["raw"]
        is_url = bool(re.match(r"^https?://", raw_value.strip(), flags=re.IGNORECASE))
        seeds.append(
            {
                "paperId": None,
                "url": raw_value if is_url else "",
                "title": "" if is_url else raw_value,
                "authors": [],
                "year": None,
                "topicHints": [],
            }
        )

    if not seeds:
        raise HTTPException(
            status_code=422,
            detail=(
                "No theme-linked papers were found in linked papers or to-read entries."
            ),
        )

    results = SemanticScholarService().recommend_for_theme(
        theme_title=selected_theme.themeTitle,
        seed_papers=seeds,
        limit=request.limit,
        candidate_pool_size=request.candidatePoolSize,
    )
    results = _filter_out_existing_theme_seeds(
        results,
        seeds=seeds,
        linked_titles=linked_titles,
        to_read_mentions=to_read_mentions,
    )
    return {"status": "success", "themeId": selected_theme.id, "results": results}


@router.post("/recommendations/paper")
def get_paper_recommendations(request: PaperRecommendationsRequest):
    try:
        seed = {
            "paperId": request.semanticScholarPaperId,
            "semanticScholarPaperId": request.semanticScholarPaperId,
            "title": request.title,
            "url": request.url,
            "authors": request.authors,
            "year": request.year,
            "abstract": request.abstract,
        }
        has_any_seed = any(
            [
                (request.semanticScholarPaperId or "").strip(),
                (request.title or "").strip(),
                (request.url or "").strip(),
            ]
        )
        if not has_any_seed:
            raise ValueError(
                "Provide semanticScholarPaperId, title, or url to resolve the paper."
            )
        results = SemanticScholarService().find_similar_papers_from_seed(
            seed_paper=seed,
            limit=request.limit,
        )
        return {"status": "success", "results": results}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SemanticScholarRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except SemanticScholarError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/recommendations/theme")
def get_theme_recommendations(request: ThemeRecommendationsRequest):
    try:
        if request.workspaceState is not None:
            workspace = WorkspaceState.model_validate(request.workspaceState)
        else:
            workspace = _load_workspace()
        return build_theme_recommendations_payload(workspace, request)
    except HTTPException:
        raise
    except SemanticScholarRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except SemanticScholarError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/recommendations/topic")
def get_topic_recommendations(request: TopicRecommendationsRequest):
    try:
        cleaned_query = request.query.strip()
        if not cleaned_query:
            raise HTTPException(
                status_code=422,
                detail="Recommendation topic query is required.",
            )
        workspace = _load_workspace()
        seeds = []
        for item in workspace.readingItems:
            if item.semanticScholarPaperId:
                seeds.append(_seed_from_item(item))

        if not seeds:
            raise HTTPException(
                status_code=422,
                detail="No reading history with Semantic Scholar IDs is available.",
            )

        results = SemanticScholarService().recommend_for_topic_profile(
            topic_query=cleaned_query,
            seed_papers=seeds,
            limit=request.limit,
            candidate_pool_size=request.candidatePoolSize,
        )
        return {"status": "success", "query": cleaned_query, "results": results}
    except HTTPException:
        raise
    except SemanticScholarRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except SemanticScholarError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
