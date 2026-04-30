from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pydantic import ValidationError

from models.workspace import WorkspaceState, default_workspace_state, utc_now_iso
from services.semantic_scholar_service import (
    SemanticScholarError,
    SemanticScholarRateLimitError,
    SemanticScholarService,
)
from services.storage_service import load_workspace_state, save_workspace_state

router = APIRouter()


class ResolvePaperUrlRequest(BaseModel):
    url: str


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

    previous_annotations = previous.paperAnnotations
    annotations = {}
    for paper_title, annotation in incoming.paperAnnotations.items():
        current_data = annotation.model_dump()
        canonical_title = current_data.get("paperTitle") or paper_title
        current_data["paperTitle"] = canonical_title
        previous_annotation = previous_annotations.get(canonical_title)
        if previous_annotation is None:
            current_data["updatedAt"] = now
        else:
            previous_data = previous_annotation.model_dump()
            current_data["updatedAt"] = (
                previous_data.get("updatedAt")
                if _timestamps_equal(current_data, previous_data)
                else now
            )
        annotations[canonical_title] = current_data

    return WorkspaceState(
        readingItems=reading_items,
        themeNotes=theme_notes,
        paperAnnotations=annotations,
    )


@router.get("/workspace/state")
def get_workspace_state():
    try:
        state_payload = load_workspace_state()
        if state_payload is None:
            return default_workspace_state().model_dump()
        return WorkspaceState.model_validate(state_payload).model_dump()
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
        previous_state = (
            WorkspaceState.model_validate(existing_payload)
            if existing_payload
            else default_workspace_state()
        )
        canonical_state = _canonicalize_workspace_state(state, previous_state)
        save_workspace_state(canonical_state.model_dump())
        return canonical_state.model_dump()
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
