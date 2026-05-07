from datetime import datetime
from typing import Dict, List, Literal

from pydantic import BaseModel, Field, field_validator

WORKSPACE_SCHEMA_VERSION = 2


class ThemeSections(BaseModel):
    notes: str = ""
    toRead: str = ""


class ReadingItem(BaseModel):
    id: str
    sourceType: Literal["url", "pdf"]
    status: Literal["inbox", "queued", "reading", "done"]
    topicHints: List[str] = Field(default_factory=list)
    linkedPaperTitle: str | None = None
    linkedThemeId: str | None = None
    title: str | None = None
    url: str | None = None
    semanticScholarPaperId: str | None = None
    authors: List[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    quickNote: str | None = None
    createdAt: str
    updatedAt: str


class ThemeNote(BaseModel):
    id: str
    themeTitle: str
    linkedPaperTitles: List[str] = Field(default_factory=list)
    sections: ThemeSections = Field(default_factory=ThemeSections)
    createdAt: str
    updatedAt: str

    @field_validator("themeTitle")
    @classmethod
    def validate_theme_title(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("themeTitle must be non-empty")
        return cleaned


class PaperAnnotation(BaseModel):
    paperTitle: str
    notesMarkdown: str = ""
    sourceUrl: str = ""
    topicLinks: List[str] = Field(default_factory=list)
    status: str = "unread"
    updatedAt: str | None = None


class PaperAnnotationV2(BaseModel):
    paperIdentityKey: str
    paperTitle: str = ""
    semanticScholarPaperId: str | None = None
    normalizedTitle: str = ""
    canonicalSourceUrl: str = ""
    sourceUrlAliases: List[str] = Field(default_factory=list)
    notesMarkdown: str = ""
    topicLinks: List[str] = Field(default_factory=list)
    status: str = "unread"
    migratedFromKeys: List[str] = Field(default_factory=list)
    selectedPrimarySource: str | None = None
    createdAt: str | None = None
    updatedAt: str | None = None

    @field_validator("paperIdentityKey")
    @classmethod
    def validate_paper_identity_key(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("paperIdentityKey must be non-empty")
        return cleaned


class AnnotationMigrationArchiveEntry(BaseModel):
    archiveId: str
    paperIdentityKey: str
    originalKey: str
    archivedAt: str
    reason: str = "migration_conflict_non_primary"
    annotation: PaperAnnotation


class WorkspaceState(BaseModel):
    readingItems: List[ReadingItem] = Field(default_factory=list)
    themeNotes: List[ThemeNote] = Field(default_factory=list)
    workspaceSchemaVersion: int = WORKSPACE_SCHEMA_VERSION
    paperAnnotations: Dict[str, PaperAnnotation] = Field(default_factory=dict)
    paperAnnotationsV2: Dict[str, PaperAnnotationV2] = Field(default_factory=dict)
    annotationMigrationArchive: List[AnnotationMigrationArchiveEntry] = Field(
        default_factory=list
    )


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_workspace_state() -> WorkspaceState:
    return WorkspaceState(workspaceSchemaVersion=WORKSPACE_SCHEMA_VERSION)
