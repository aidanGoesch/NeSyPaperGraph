import React, { useEffect, useMemo, useRef, useState } from "react";

function ensureSections(sections) {
    return {
        notes:
            sections?.notes ||
            [sections?.summary, sections?.evidence, sections?.questions]
                .filter(Boolean)
                .join("\n\n"),
        toRead: sections?.toRead || sections?.nextReads || "",
    };
}

export default function ThemeNotebook({
    themeNotes,
    selectedThemeId,
    themeQueueItems,
    onSelectTheme,
    onUpsertTheme,
    onReorderReadingItem,
    onSelectThemePaper,
    onRequestThemeRecommendations,
    onAddRecommendationToReadingList,
}) {
    const selectedTheme = useMemo(
        () => themeNotes.find((note) => note.id === selectedThemeId) || null,
        [themeNotes, selectedThemeId]
    );
    const [activeDraft, setActiveDraft] = useState(null);
    const [isAutoSaving, setIsAutoSaving] = useState(false);
    const [draggingItemId, setDraggingItemId] = useState(null);
    const [dragOverItemId, setDragOverItemId] = useState(null);
    const dragPreviewRef = useRef(null);
    const [themeRecommendations, setThemeRecommendations] = useState([]);
    const [themeRecommendationState, setThemeRecommendationState] = useState("idle");
    const [themeRecommendationError, setThemeRecommendationError] = useState("");
    const [expandedRecommendationKey, setExpandedRecommendationKey] = useState(null);

    useEffect(() => {
        if (!selectedTheme) {
            setActiveDraft((prev) => (prev?.isNew ? prev : null));
            return;
        }
        setActiveDraft({
            id: selectedTheme.id,
            themeTitle: selectedTheme.themeTitle || "",
            linkedPaperTitles: selectedTheme.linkedPaperTitles || [],
            sections: ensureSections(selectedTheme.sections),
            isNew: false,
        });
        setThemeRecommendations([]);
        setThemeRecommendationState("idle");
        setThemeRecommendationError("");
        setExpandedRecommendationKey(null);
    }, [selectedTheme]);

    const isShowingEditor = Boolean(activeDraft);
    const hasValidTitle = Boolean(activeDraft?.themeTitle?.trim());
    const syncedToReadText = useMemo(() => {
        if (!themeQueueItems || themeQueueItems.length === 0) {
            return activeDraft?.sections?.toRead || "";
        }
        return themeQueueItems
            .map((item) => {
                const label = item.title || item.url || "Untitled item";
                return `- ${label} [${item.status}]`;
            })
            .join("\n");
    }, [activeDraft?.sections?.toRead, themeQueueItems]);

    const saveThemeDraft = ({
        draft = activeDraft,
        selectAfterSave = true,
        markAsExisting = true,
    } = {}) => {
        if (!draft?.themeTitle?.trim()) return null;
        const savedTheme = onUpsertTheme({
            id: draft.id,
            themeTitle: draft.themeTitle.trim(),
            linkedPaperTitles: draft.linkedPaperTitles,
            sections: {
                ...draft.sections,
                toRead: syncedToReadText,
            },
        });
        if (markAsExisting) {
            setActiveDraft((prev) =>
                prev
                    ? {
                          ...prev,
                          id: savedTheme.id,
                          isNew: false,
                      }
                    : prev
            );
        }
        if (selectAfterSave) {
            onSelectTheme(savedTheme.id);
        }
        return savedTheme;
    };

    useEffect(() => {
        if (!activeDraft || activeDraft.isNew || !hasValidTitle) return;
        setIsAutoSaving(true);
        const timer = setTimeout(() => {
            saveThemeDraft({
                draft: activeDraft,
                selectAfterSave: false,
                markAsExisting: false,
            });
            setIsAutoSaving(false);
        }, 500);
        return () => {
            clearTimeout(timer);
            setIsAutoSaving(false);
        };
    }, [
        activeDraft?.id,
        activeDraft?.isNew,
        activeDraft?.themeTitle,
        activeDraft?.sections?.notes,
        activeDraft?.linkedPaperTitles,
        hasValidTitle,
        syncedToReadText,
    ]);

    useEffect(() => {
        return () => {
            if (dragPreviewRef.current) {
                dragPreviewRef.current.remove();
                dragPreviewRef.current = null;
            }
        };
    }, []);

    const handleDragStart = (event, itemId) => {
        setDraggingItemId(itemId);
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", itemId);

        const sourceItem = event.currentTarget.closest(".theme-queue-item");
        if (!sourceItem) return;
        const previewNode = sourceItem.cloneNode(true);
        previewNode.classList.add("theme-queue-drag-preview");
        previewNode.style.width = `${sourceItem.getBoundingClientRect().width}px`;
        previewNode.style.position = "fixed";
        previewNode.style.top = "-9999px";
        previewNode.style.left = "-9999px";
        document.body.appendChild(previewNode);
        dragPreviewRef.current = previewNode;
        event.dataTransfer.setDragImage(previewNode, 28, 20);
    };

    const handleDragEnd = () => {
        setDraggingItemId(null);
        setDragOverItemId(null);
        if (dragPreviewRef.current) {
            dragPreviewRef.current.remove();
            dragPreviewRef.current = null;
        }
    };

    return (
        <section className="workspace-panel workspace-panel-right">
            <div className="workspace-panel-header">
                <h3>Theme Notebook</h3>
                <span>{themeNotes.length} themes</span>
            </div>
            <div className="theme-selector-row">
                <button
                    className="new-theme-button"
                    type="button"
                    onClick={() => {
                        if (activeDraft?.isNew) {
                            saveThemeDraft();
                            return;
                        }
                        onSelectTheme(null);
                        setActiveDraft({
                            id: null,
                            themeTitle: "",
                            linkedPaperTitles: [],
                            sections: ensureSections(null),
                            isNew: true,
                        });
                    }}
                    disabled={Boolean(activeDraft?.isNew) && !hasValidTitle}
                >
                    {activeDraft?.isNew ? "Save New Theme" : "New Theme"}
                </button>
                <div className="theme-tabs theme-tabs-horizontal">
                    {themeNotes.map((note) => (
                        <button
                            key={note.id}
                            type="button"
                            className={selectedThemeId === note.id ? "active" : ""}
                            onClick={() => onSelectTheme(note.id)}
                        >
                            {note.themeTitle}
                        </button>
                    ))}
                </div>
            </div>
            {isShowingEditor ? (
                <div className="theme-editor">
                    <label htmlFor="theme-title">Theme title</label>
                    <input
                        id="theme-title"
                        value={activeDraft.themeTitle}
                        onChange={(event) =>
                            setActiveDraft((prev) => ({
                                ...prev,
                                themeTitle: event.target.value,
                            }))
                        }
                        placeholder="Theme title (required)"
                        onKeyDown={(event) => {
                            if (event.key === "Enter" && activeDraft?.isNew) {
                                event.preventDefault();
                                saveThemeDraft();
                            }
                        }}
                    />
                    <label htmlFor="theme-notes">Notes</label>
                    <textarea
                        id="theme-notes"
                        value={activeDraft.sections.notes}
                        onChange={(event) =>
                            setActiveDraft((prev) => ({
                                ...prev,
                                sections: {
                                    ...prev.sections,
                                    notes: event.target.value,
                                },
                            }))
                        }
                    />
                    <label htmlFor="theme-to-read">To-read</label>
                    <div className="theme-sync-hint">Synced from queue for this theme</div>
                    {themeQueueItems && themeQueueItems.length > 0 ? (
                        <div
                            className={`theme-queue-list ${
                                draggingItemId ? "theme-queue-list-dragging" : ""
                            }`}
                            id="theme-to-read"
                        >
                            {themeQueueItems.map((item) => (
                                <div
                                    key={item.id}
                                    className={`theme-queue-item ${
                                        draggingItemId === item.id
                                            ? "theme-queue-item-dragging"
                                            : ""
                                    } ${
                                        dragOverItemId === item.id &&
                                        draggingItemId &&
                                        draggingItemId !== item.id
                                            ? "theme-queue-item-drop-target"
                                            : ""
                                    }`}
                                    onDragEnter={() => {
                                        if (
                                            !draggingItemId ||
                                            draggingItemId === item.id
                                        ) {
                                            return;
                                        }
                                        setDragOverItemId(item.id);
                                        if (onReorderReadingItem) {
                                            onReorderReadingItem(
                                                draggingItemId,
                                                item.id
                                            );
                                        }
                                    }}
                                    onDragOver={(event) => {
                                        if (
                                            !draggingItemId ||
                                            draggingItemId === item.id
                                        ) {
                                            return;
                                        }
                                        event.preventDefault();
                                    }}
                                    onDrop={(event) => {
                                        event.preventDefault();
                                        handleDragEnd();
                                    }}
                                >
                                    <button
                                        type="button"
                                        className="theme-queue-drag-handle"
                                        title="Drag to reorder"
                                        aria-label="Drag to reorder"
                                        draggable
                                        onDragStart={(event) =>
                                            handleDragStart(event, item.id)
                                        }
                                        onDragEnd={handleDragEnd}
                                    >
                                        :::
                                    </button>
                                    <span className="theme-queue-title">
                                        {item.title || "Untitled item"}
                                    </span>
                                    {item.url && (
                                        <a
                                            href={item.url}
                                            target="_blank"
                                            rel="noreferrer"
                                            className="theme-queue-open-button"
                                            title={item.url}
                                        >
                                            Open
                                        </a>
                                    )}
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="theme-queue-list" id="theme-to-read">
                            <div className="theme-queue-item theme-queue-item-empty">
                                <span className="theme-queue-title">
                                    No papers queued for this theme yet.
                                </span>
                            </div>
                        </div>
                    )}
                    {!hasValidTitle && (
                        <p className="validation-error">
                            Title is required before saving this theme.
                        </p>
                    )}
                    {!activeDraft?.isNew && (
                        <p className="theme-sync-hint">
                            {isAutoSaving ? "Saving..." : "Changes save automatically."}
                        </p>
                    )}
                    {!activeDraft?.isNew && (
                        <div className="paper-actions">
                            <button
                                type="button"
                                onClick={async () => {
                                    if (!onRequestThemeRecommendations || !activeDraft?.id) return;
                                    setThemeRecommendationState("loading");
                                    setThemeRecommendationError("");
                                    setThemeRecommendations([]);
                                    try {
                                        const results = await onRequestThemeRecommendations(
                                            activeDraft.id
                                        );
                                        setThemeRecommendations(
                                            Array.isArray(results) ? results : []
                                        );
                                        setThemeRecommendationState("success");
                                        setExpandedRecommendationKey(null);
                                    } catch (error) {
                                        setThemeRecommendationError(
                                            `Theme recommendations failed: ${
                                                error?.message || "Unknown error"
                                            }`
                                        );
                                        setThemeRecommendationState("error");
                                    }
                                }}
                            >
                                Recommend papers for theme
                            </button>
                        </div>
                    )}
                    {themeRecommendationState === "loading" && (
                        <p className="theme-sync-hint">Finding recommendations...</p>
                    )}
                    {themeRecommendationState === "error" && (
                        <p className="validation-error">{themeRecommendationError}</p>
                    )}
                    {themeRecommendationState === "success" &&
                        themeRecommendations.length > 0 && (
                            <div className="linked-papers">
                                <strong>Recommended papers</strong>
                                <ul className="theme-linked-paper-list">
                                    {themeRecommendations.map((paper, index) => (
                                        <li key={paper.paperId || `${paper.title}-${index}`}>
                                            <div className="theme-linked-paper-card">
                                                <button
                                                    type="button"
                                                    className="paper-list-item"
                                                    onClick={() => {
                                                        const key =
                                                            paper.paperId ||
                                                            paper.title ||
                                                            String(index);
                                                        setExpandedRecommendationKey(
                                                            (previous) =>
                                                                previous === key ? null : key
                                                        );
                                                    }}
                                                >
                                                    <strong>
                                                        {paper.title || "Untitled paper"}
                                                    </strong>
                                                    {paper.source && (
                                                        <small>{paper.source}</small>
                                                    )}
                                                </button>
                                                {expandedRecommendationKey ===
                                                    (paper.paperId ||
                                                        paper.title ||
                                                        String(index)) && (
                                                    <div className="theme-linked-paper-meta">
                                                        <p>
                                                            {(paper.authors || []).length
                                                                ? paper.authors.join(", ")
                                                                : "Unknown authors"}{" "}
                                                            |{" "}
                                                            {paper.year || "Unknown year"}
                                                        </p>
                                                        <p>
                                                            {paper.abstract ||
                                                                "No summary available."}
                                                        </p>
                                                        <div className="paper-actions">
                                                            {paper.title &&
                                                                onAddRecommendationToReadingList && (
                                                                <button
                                                                    type="button"
                                                                    className="recommendation-action-button"
                                                                    onClick={() =>
                                                                        onAddRecommendationToReadingList(
                                                                            paper,
                                                                            activeDraft?.id ||
                                                                                selectedThemeId ||
                                                                                null
                                                                        )
                                                                    }
                                                                >
                                                                    Add to reading list
                                                                </button>
                                                            )}
                                                            {paper.url && (
                                                                <a
                                                                    className="recommendation-action-button recommendation-action-link"
                                                                    href={paper.url}
                                                                    target="_blank"
                                                                    rel="noreferrer"
                                                                >
                                                                    Open in browser
                                                                </a>
                                                            )}
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    {themeRecommendationState === "success" &&
                        themeRecommendations.length === 0 && (
                            <p className="theme-sync-hint">
                                No recommendations found for this theme yet. Try linking
                                more seed papers with Semantic Scholar IDs.
                            </p>
                        )}
                    <div className="linked-papers">
                        <strong>Papers in this theme</strong>
                        {(activeDraft.linkedPaperTitles || []).length === 0 ? (
                            <p className="empty-panel-copy">
                                No papers assigned yet. Use “Send to Theme” from a paper.
                            </p>
                        ) : (
                            <ul className="theme-linked-paper-list">
                                {activeDraft.linkedPaperTitles.map((paperTitle) => (
                                    <li key={paperTitle}>
                                        <div className="theme-linked-paper-card">
                                            <div className="theme-linked-paper-row">
                                                <button
                                                    type="button"
                                                    className="theme-linked-paper-title-button"
                                                    onClick={() => onSelectThemePaper(paperTitle)}
                                                >
                                                    <strong>{paperTitle}</strong>
                                                </button>
                                            </div>
                                        </div>
                                    </li>
                                ))}
                            </ul>
                        )}
                    </div>
                </div>
            ) : (
                <p className="empty-panel-copy">
                    Select a theme to open its Notes and To-read boxes.
                </p>
            )}
        </section>
    );
}
