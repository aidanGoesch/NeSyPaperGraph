from __future__ import annotations

import hashlib
import re
from datetime import datetime
from urllib.parse import urlparse

from models.workspace import (
    AnnotationMigrationArchiveEntry,
    PaperAnnotation,
    PaperAnnotationV2,
    WORKSPACE_SCHEMA_VERSION,
    WorkspaceState,
    utc_now_iso,
)


def normalize_paper_title(value: str | None) -> str:
    cleaned = (value or "").strip().lower()
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_semantic_scholar_id(value: str | None) -> str:
    return (value or "").strip().lower()


def normalize_reader_lookup_url(url: str | None) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        if not parsed.scheme:
            return (url or "").strip().lower()
        path = parsed.path.rstrip("/") or "/"
        return (
            f"{parsed.scheme.lower()}://{parsed.hostname.lower()}"
            f"{path}{f'?{parsed.query}' if parsed.query else ''}"
        )
    except Exception:
        return (url or "").strip().lower()


def is_url_like(value: str | None) -> bool:
    text = (value or "").strip()
    if not text:
        return False
    try:
        parsed = urlparse(text)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def extract_semantic_scholar_id_from_url(url: str | None) -> str:
    text = (url or "").strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
    except Exception:
        return ""
    host = (parsed.hostname or "").lower()
    if "semanticscholar.org" not in host:
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    for idx, part in enumerate(parts):
        if part.lower() == "paper" and idx + 2 < len(parts):
            return normalize_semantic_scholar_id(parts[idx + 2])
    return ""


def build_paper_identity_key(
    semantic_scholar_paper_id: str | None,
    paper_title: str | None,
    source_url: str | None,
) -> str:
    normalized_ssid = normalize_semantic_scholar_id(semantic_scholar_paper_id)
    if normalized_ssid:
        return f"ssid:{normalized_ssid}"

    normalized_title = normalize_paper_title(paper_title)
    normalized_url = normalize_reader_lookup_url(source_url)
    digest = hashlib.sha256(f"{normalized_title}|{normalized_url}".encode("utf-8")).hexdigest()[
        :20
    ]
    return f"fallback:{digest}"


def resolve_identity_hints(
    *,
    annotation_key: str | None = None,
    paper_title: str | None = None,
    source_url: str | None = None,
    semantic_scholar_paper_id: str | None = None,
) -> tuple[str, str | None, str, str]:
    normalized_url = normalize_reader_lookup_url(source_url)
    extracted_ssid = extract_semantic_scholar_id_from_url(source_url)
    ssid = normalize_semantic_scholar_id(semantic_scholar_paper_id) or extracted_ssid

    key_fallback = (annotation_key or "").strip()
    resolved_title = (paper_title or "").strip()
    if not resolved_title and key_fallback and not is_url_like(key_fallback):
        resolved_title = key_fallback
    if not normalized_url and is_url_like(key_fallback):
        normalized_url = normalize_reader_lookup_url(key_fallback)
    normalized_title = normalize_paper_title(resolved_title)
    identity_key = build_paper_identity_key(ssid, resolved_title, normalized_url)
    return identity_key, (ssid or None), normalized_title, normalized_url


def _parse_updated_at(value: str | None) -> datetime:
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = (value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _build_primary_annotation_v2(
    identity_key: str,
    primary_source: str,
    primary_annotation: PaperAnnotation,
    *,
    normalized_title: str,
    normalized_url: str,
    semantic_scholar_paper_id: str | None,
    migrated_keys: list[str],
    source_url_aliases: list[str],
    now_iso: str,
    previous_annotation: PaperAnnotationV2 | None = None,
) -> PaperAnnotationV2:
    paper_title = (primary_annotation.paperTitle or "").strip()
    if not paper_title:
        paper_title = (
            previous_annotation.paperTitle
            if previous_annotation and previous_annotation.paperTitle
            else identity_key
        )
    canonical_source_url = (
        normalize_reader_lookup_url(primary_annotation.sourceUrl)
        or normalized_url
        or (previous_annotation.canonicalSourceUrl if previous_annotation else "")
    )
    aliases = _dedupe(
        source_url_aliases
        + ([canonical_source_url] if canonical_source_url else [])
        + (previous_annotation.sourceUrlAliases if previous_annotation else [])
    )
    return PaperAnnotationV2(
        paperIdentityKey=identity_key,
        paperTitle=paper_title,
        semanticScholarPaperId=semantic_scholar_paper_id
        or (previous_annotation.semanticScholarPaperId if previous_annotation else None),
        normalizedTitle=normalized_title
        or (previous_annotation.normalizedTitle if previous_annotation else ""),
        canonicalSourceUrl=canonical_source_url,
        sourceUrlAliases=aliases,
        notesMarkdown=primary_annotation.notesMarkdown,
        topicLinks=primary_annotation.topicLinks,
        status=primary_annotation.status or "unread",
        migratedFromKeys=_dedupe(
            migrated_keys + (previous_annotation.migratedFromKeys if previous_annotation else [])
        ),
        selectedPrimarySource=primary_source,
        createdAt=(previous_annotation.createdAt if previous_annotation else None) or now_iso,
        updatedAt=primary_annotation.updatedAt or now_iso,
    )


def migrate_legacy_annotations(
    *,
    legacy_annotations: dict[str, PaperAnnotation],
    existing_v2_annotations: dict[str, PaperAnnotationV2] | None = None,
    existing_archive: list[AnnotationMigrationArchiveEntry] | None = None,
    now_iso: str | None = None,
) -> tuple[dict[str, PaperAnnotationV2], list[AnnotationMigrationArchiveEntry]]:
    now = now_iso or utc_now_iso()
    v2_annotations = dict(existing_v2_annotations or {})
    archives = list(existing_archive or [])
    archive_ids = {entry.archiveId for entry in archives}
    grouped: dict[str, list[tuple[str, PaperAnnotation, str | None, str, str]]] = {}
    pending_without_ssid: list[tuple[str, PaperAnnotation, str, str, str]] = []
    title_to_ssid_groups: dict[str, set[str]] = {}
    pending_title_counts: dict[str, int] = {}

    for identity_key, annotation in v2_annotations.items():
        if not normalize_semantic_scholar_id(annotation.semanticScholarPaperId):
            continue
        normalized_title = normalize_paper_title(annotation.paperTitle) or annotation.normalizedTitle
        if normalized_title:
            title_to_ssid_groups.setdefault(normalized_title, set()).add(identity_key)

    for original_key, annotation in legacy_annotations.items():
        key, ssid, normalized_title, normalized_url = resolve_identity_hints(
            annotation_key=original_key,
            paper_title=annotation.paperTitle,
            source_url=annotation.sourceUrl,
            semantic_scholar_paper_id=None,
        )
        if ssid:
            grouped.setdefault(key, []).append(
                (original_key, annotation, ssid, normalized_title, normalized_url)
            )
            if normalized_title:
                title_to_ssid_groups.setdefault(normalized_title, set()).add(key)
            continue
        pending_without_ssid.append(
            (original_key, annotation, key, normalized_title, normalized_url)
        )
        if normalized_title:
            pending_title_counts[normalized_title] = pending_title_counts.get(normalized_title, 0) + 1

    for original_key, annotation, fallback_key, normalized_title, normalized_url in pending_without_ssid:
        target_key = fallback_key
        if normalized_title:
            candidate_ssid_groups = title_to_ssid_groups.get(normalized_title, set())
            if len(candidate_ssid_groups) == 1:
                target_key = next(iter(candidate_ssid_groups))
            elif len(candidate_ssid_groups) == 0 and pending_title_counts.get(normalized_title, 0) > 1:
                target_key = f"title:{normalized_title}"
        grouped.setdefault(target_key, []).append(
            (original_key, annotation, None, normalized_title, normalized_url)
        )

    for identity_key, entries in grouped.items():
        previous_annotation = v2_annotations.get(identity_key)
        primary_source, primary_annotation, ssid, normalized_title, normalized_url = max(
            entries,
            key=lambda item: (_parse_updated_at(item[1].updatedAt), item[0]),
        )
        migrated_keys = [item[0] for item in entries]
        url_aliases = [
            normalize_reader_lookup_url(item[1].sourceUrl)
            or (normalize_reader_lookup_url(item[0]) if is_url_like(item[0]) else "")
            for item in entries
        ]
        v2_annotations[identity_key] = _build_primary_annotation_v2(
            identity_key,
            primary_source,
            primary_annotation,
            normalized_title=normalized_title,
            normalized_url=normalized_url,
            semantic_scholar_paper_id=ssid,
            migrated_keys=migrated_keys,
            source_url_aliases=[alias for alias in url_aliases if alias],
            now_iso=now,
            previous_annotation=previous_annotation,
        )
        if previous_annotation is not None:
            previous_as_legacy = PaperAnnotation(
                paperTitle=previous_annotation.paperTitle or identity_key,
                notesMarkdown=previous_annotation.notesMarkdown,
                sourceUrl=previous_annotation.canonicalSourceUrl,
                topicLinks=previous_annotation.topicLinks,
                status=previous_annotation.status,
                updatedAt=previous_annotation.updatedAt,
            )
            selected_annotation = v2_annotations[identity_key]
            has_changed = (
                previous_as_legacy.notesMarkdown != selected_annotation.notesMarkdown
                or normalize_reader_lookup_url(previous_as_legacy.sourceUrl)
                != normalize_reader_lookup_url(selected_annotation.canonicalSourceUrl)
                or previous_as_legacy.status != selected_annotation.status
            )
            if has_changed:
                archive_id = hashlib.sha256(
                    (
                        f"{identity_key}|__previous_primary__|"
                        f"{previous_as_legacy.updatedAt}|{previous_as_legacy.notesMarkdown}"
                    ).encode("utf-8")
                ).hexdigest()[:24]
                if archive_id not in archive_ids:
                    archives.append(
                        AnnotationMigrationArchiveEntry(
                            archiveId=archive_id,
                            paperIdentityKey=identity_key,
                            originalKey="__previous_primary__",
                            archivedAt=now,
                            annotation=previous_as_legacy,
                        )
                    )
                    archive_ids.add(archive_id)

        for original_key, annotation, *_ in entries:
            if original_key == primary_source:
                continue
            archive_id = hashlib.sha256(
                f"{identity_key}|{original_key}|{annotation.updatedAt}|{annotation.notesMarkdown}".encode(
                    "utf-8"
                )
            ).hexdigest()[:24]
            if archive_id in archive_ids:
                continue
            archives.append(
                AnnotationMigrationArchiveEntry(
                    archiveId=archive_id,
                    paperIdentityKey=identity_key,
                    originalKey=original_key,
                    archivedAt=now,
                    annotation=annotation,
                )
            )
            archive_ids.add(archive_id)

    return v2_annotations, archives


def project_v2_to_legacy_annotations(
    annotations_v2: dict[str, PaperAnnotationV2],
) -> dict[str, PaperAnnotation]:
    projected: dict[str, PaperAnnotation] = {}
    for identity_key, annotation in annotations_v2.items():
        key = (annotation.paperTitle or "").strip() or identity_key
        if key in projected:
            key = identity_key
        source_url = annotation.canonicalSourceUrl or (
            annotation.sourceUrlAliases[0] if annotation.sourceUrlAliases else ""
        )
        projected[key] = PaperAnnotation(
            paperTitle=(annotation.paperTitle or key),
            notesMarkdown=annotation.notesMarkdown,
            sourceUrl=source_url,
            topicLinks=annotation.topicLinks,
            status=annotation.status,
            updatedAt=annotation.updatedAt,
        )
    return projected


def upgrade_workspace_state(
    state: WorkspaceState,
    now_iso: str | None = None,
    *,
    project_legacy_annotations: bool = True,
) -> WorkspaceState:
    now = now_iso or utc_now_iso()
    migrated_annotations, migrated_archive = migrate_legacy_annotations(
        legacy_annotations=state.paperAnnotations,
        existing_v2_annotations=state.paperAnnotationsV2,
        existing_archive=state.annotationMigrationArchive,
        now_iso=now,
    )
    return WorkspaceState(
        workspaceSchemaVersion=WORKSPACE_SCHEMA_VERSION,
        readingItems=state.readingItems,
        themeNotes=state.themeNotes,
        paperAnnotations=(
            project_v2_to_legacy_annotations(migrated_annotations)
            if project_legacy_annotations
            else state.paperAnnotations
        ),
        paperAnnotationsV2=migrated_annotations,
        annotationMigrationArchive=migrated_archive,
    )


def merge_workspace_annotations_v2(
    *,
    previous_state: WorkspaceState,
    incoming_state: WorkspaceState,
    incoming_v2_annotations: dict[str, PaperAnnotationV2] | None = None,
    now_iso: str | None = None,
) -> tuple[dict[str, PaperAnnotationV2], list[AnnotationMigrationArchiveEntry]]:
    now = now_iso or utc_now_iso()
    existing_v2 = dict(previous_state.paperAnnotationsV2)
    existing_v2.update(incoming_v2_annotations or {})
    merged_v2, merged_archive = migrate_legacy_annotations(
        legacy_annotations=incoming_state.paperAnnotations,
        existing_v2_annotations=existing_v2,
        existing_archive=previous_state.annotationMigrationArchive,
        now_iso=now,
    )
    return merged_v2, merged_archive
