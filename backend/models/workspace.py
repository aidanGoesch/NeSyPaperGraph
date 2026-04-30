from datetime import datetime
from typing import Dict, List, Literal

from pydantic import BaseModel, Field, field_validator


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
    topicLinks: List[str] = Field(default_factory=list)
    status: str = "unread"
    updatedAt: str | None = None


class WorkspaceState(BaseModel):
    readingItems: List[ReadingItem] = Field(default_factory=list)
    themeNotes: List[ThemeNote] = Field(default_factory=list)
    paperAnnotations: Dict[str, PaperAnnotation] = Field(default_factory=dict)


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_workspace_state() -> WorkspaceState:
    return WorkspaceState()
