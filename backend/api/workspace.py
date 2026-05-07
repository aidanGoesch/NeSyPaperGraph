from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError

from models.workspace import (
    WORKSPACE_SCHEMA_VERSION,
    WorkspaceState,
    default_workspace_state,
    utc_now_iso,
)
from api.recommendations import (
    PaperRecommendationsRequest,
    ThemeRecommendationsRequest,
    build_theme_recommendations_payload,
)
from services.paper_identity_service import (
    merge_workspace_annotations_v2,
    project_v2_to_legacy_annotations,
    upgrade_workspace_state,
)
from services.semantic_scholar_service import (
    SemanticScholarError,
    SemanticScholarRateLimitError,
    SemanticScholarService,
)
from services.storage_service import load_workspace_state, save_workspace_state

router = APIRouter()


class ResolvePaperUrlRequest(BaseModel):
    url: str


class ResolvePaperRequest(BaseModel):
    semanticScholarPaperId: str | None = None
    url: str | None = None
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None


def _timestamps_equal(left: dict, right: dict) -> bool:
    left_copy = {k: v for k, v in left.items() if k not in {"createdAt", "updatedAt"}}
    right_copy = {k: v for k, v in right.items() if k not in {"createdAt", "updatedAt"}}
    return left_copy == right_copy


def _canonicalize_workspace_state(
    incoming: WorkspaceState, previous: WorkspaceState
) -> WorkspaceState:
    now = utc_now_iso()
    previous_reading_by_id = {item.id: item for item in previous.readingItems}
    reading_items = []
    for item in incoming.readingItems:
        current_data = item.model_dump()
        previous_item = previous_reading_by_id.get(item.id)
        if previous_item is None:
            current_data["createdAt"] = current_data.get("createdAt") or now
            current_data["updatedAt"] = now
        else:
            previous_data = previous_item.model_dump()
            current_data["createdAt"] = previous_data.get("createdAt") or current_data.get(
                "createdAt"
            )
            current_data["updatedAt"] = (
                previous_data.get("updatedAt")
                if _timestamps_equal(current_data, previous_data)
                else now
            )
        reading_items.append(current_data)

    previous_theme_by_id = {item.id: item for item in previous.themeNotes}
    theme_notes = []
    for note in incoming.themeNotes:
        current_data = note.model_dump()
        previous_note = previous_theme_by_id.get(note.id)
        if previous_note is None:
            current_data["createdAt"] = current_data.get("createdAt") or now
            current_data["updatedAt"] = now
        else:
            previous_data = previous_note.model_dump()
            current_data["createdAt"] = previous_data.get("createdAt") or current_data.get(
                "createdAt"
            )
            current_data["updatedAt"] = (
                previous_data.get("updatedAt")
                if _timestamps_equal(current_data, previous_data)
                else now
            )
        theme_notes.append(current_data)

    return WorkspaceState(
        workspaceSchemaVersion=WORKSPACE_SCHEMA_VERSION,
        readingItems=reading_items,
        themeNotes=theme_notes,
        paperAnnotations=incoming.paperAnnotations,
        paperAnnotationsV2=incoming.paperAnnotationsV2,
        annotationMigrationArchive=incoming.annotationMigrationArchive,
    )


def _validate_v2_identity_integrity(state: WorkspaceState) -> None:
    for identity_key, annotation in state.paperAnnotationsV2.items():
        if identity_key != annotation.paperIdentityKey:
            raise ValueError(
                f"paperAnnotationsV2 key '{identity_key}' does not match "
                f"annotation.paperIdentityKey '{annotation.paperIdentityKey}'"
            )

    alias_to_identity: dict[str, str] = {}
    for identity_key, annotation in state.paperAnnotationsV2.items():
        for alias in annotation.sourceUrlAliases:
            normalized = (alias or "").strip()
            if not normalized:
                continue
            existing = alias_to_identity.get(normalized)
            if existing and existing != identity_key:
                raise ValueError(
                    f"sourceUrl alias collision detected for '{normalized}' "
                    f"between '{existing}' and '{identity_key}'"
                )
            alias_to_identity[normalized] = identity_key


@router.get("/workspace/state")
def get_workspace_state():
    try:
        state_payload = load_workspace_state()
        if state_payload is None:
            return default_workspace_state().model_dump()
        state = WorkspaceState.model_validate(state_payload)
        upgraded_state = upgrade_workspace_state(state)
        if upgraded_state.model_dump() != state.model_dump():
            save_workspace_state(upgraded_state.model_dump())
        return upgraded_state.model_dump()
    except ValidationError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Persisted workspace state is invalid: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load workspace state: {exc}",
        ) from exc


@router.put("/workspace/state")
def put_workspace_state(state: WorkspaceState):
    try:
        existing_payload = load_workspace_state()
        previous_state_raw = (
            WorkspaceState.model_validate(existing_payload)
            if existing_payload
            else default_workspace_state()
        )
        previous_state = upgrade_workspace_state(previous_state_raw)
        incoming_state = upgrade_workspace_state(state, project_legacy_annotations=False)
        _validate_v2_identity_integrity(incoming_state)

        merged_v2_annotations, merged_archive = merge_workspace_annotations_v2(
            previous_state=previous_state,
            incoming_state=incoming_state,
            incoming_v2_annotations=state.paperAnnotationsV2,
            now_iso=utc_now_iso(),
        )
        canonical_state = _canonicalize_workspace_state(
            WorkspaceState(
                workspaceSchemaVersion=WORKSPACE_SCHEMA_VERSION,
                readingItems=incoming_state.readingItems,
                themeNotes=incoming_state.themeNotes,
                paperAnnotations=project_v2_to_legacy_annotations(merged_v2_annotations),
                paperAnnotationsV2=merged_v2_annotations,
                annotationMigrationArchive=merged_archive,
            ),
            previous_state,
        )
        save_workspace_state(canonical_state.model_dump())
        return canonical_state.model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save workspace state: {exc}",
        ) from exc


@router.post("/workspace/resolve-paper-url")
def resolve_paper_url(request: ResolvePaperUrlRequest):
    try:
        metadata = SemanticScholarService().resolve_url_metadata(request.url.strip())
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail="Unable to resolve paper metadata from the provided URL.",
            )
        return metadata
    except HTTPException:
        raise
    except SemanticScholarRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except SemanticScholarError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        detail = str(exc)
        if "rate limit" in detail.lower():
            raise HTTPException(status_code=429, detail=detail) from exc
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resolve URL metadata: {exc}",
        ) from exc


@router.post("/workspace/resolve-paper")
def resolve_paper(request: ResolvePaperRequest):
    try:
        seed = {
            "semanticScholarPaperId": (request.semanticScholarPaperId or "").strip(),
            "url": (request.url or "").strip(),
            "title": (request.title or "").strip(),
            "authors": request.authors or [],
            "year": request.year,
        }
        if not any([seed["semanticScholarPaperId"], seed["url"], seed["title"]]):
            raise HTTPException(
                status_code=422,
                detail="Provide semanticScholarPaperId, title, or url to resolve the paper.",
            )

        details = SemanticScholarService().resolve_seed_paper_details(seed)
        if not details:
            raise HTTPException(
                status_code=404,
                detail="Unable to resolve the paper in Semantic Scholar.",
            )

        return {
            "semanticScholarPaperId": details.get("paperId")
            or details.get("semanticScholarPaperId"),
            "url": details.get("url") or "",
            "title": details.get("title") or seed["title"] or "",
            "authors": details.get("authors") or [],
            "year": details.get("year"),
            "venue": details.get("venue"),
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SemanticScholarRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except SemanticScholarError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        detail = str(exc)
        if "rate limit" in detail.lower():
            raise HTTPException(status_code=429, detail=detail) from exc
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resolve paper metadata: {exc}",
        ) from exc


@router.post("/workspace/recommendations/theme")
def workspace_theme_recommendations(request: ThemeRecommendationsRequest):
    try:
        if request.workspaceState is not None:
            workspace = WorkspaceState.model_validate(request.workspaceState)
        else:
            payload = load_workspace_state()
            workspace = (
                WorkspaceState.model_validate(payload)
                if payload is not None
                else default_workspace_state()
            )
        return build_theme_recommendations_payload(workspace, request)
    except HTTPException:
        raise
    except SemanticScholarRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except SemanticScholarError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/workspace/recommendations/paper")
def workspace_paper_recommendations(request: PaperRecommendationsRequest):
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
            raise HTTPException(
                status_code=422,
                detail="Provide semanticScholarPaperId, title, or url to resolve the paper.",
            )
        results = SemanticScholarService().find_similar_papers_from_seed(
            seed_paper=seed,
            limit=request.limit,
        )
        return {"status": "success", "results": results}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SemanticScholarRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except SemanticScholarError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
