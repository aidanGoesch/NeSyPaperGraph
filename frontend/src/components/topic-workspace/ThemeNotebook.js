import React, { useEffect, useMemo, useState } from "react";

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
    onSelectThemePaper,
}) {
    const selectedTheme = useMemo(
        () => themeNotes.find((note) => note.id === selectedThemeId) || null,
        [themeNotes, selectedThemeId]
    );
    const [activeDraft, setActiveDraft] = useState(null);

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
                        onSelectTheme(null);
                        setActiveDraft({
                            id: null,
                            themeTitle: "",
                            linkedPaperTitles: [],
                            sections: ensureSections(null),
                            isNew: true,
                        });
                    }}
                >
                    New Theme
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
                        <div className="theme-queue-list" id="theme-to-read">
                            {themeQueueItems.map((item) => (
                                <div key={item.id} className="theme-queue-item">
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
                                    <span className="theme-queue-status">
                                        {item.status}
                                    </span>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <textarea
                            id="theme-to-read"
                            value={syncedToReadText}
                            readOnly
                        />
                    )}
                    <button
                        type="button"
                        disabled={!hasValidTitle}
                        onClick={() => {
                            const savedTheme = onUpsertTheme({
                                id: activeDraft.id,
                                themeTitle: activeDraft.themeTitle.trim(),
                                linkedPaperTitles: activeDraft.linkedPaperTitles,
                                sections: {
                                    ...activeDraft.sections,
                                    toRead: syncedToReadText,
                                },
                            });
                            setActiveDraft((prev) => ({
                                ...prev,
                                id: savedTheme.id,
                                isNew: false,
                            }));
                            onSelectTheme(savedTheme.id);
                        }}
                    >
                        Save Theme Notes
                    </button>
                    {!hasValidTitle && (
                        <p className="validation-error">
                            Title is required before saving this theme.
                        </p>
                    )}
                    <div className="linked-papers">
                        <strong>Papers in this theme</strong>
                        {(activeDraft.linkedPaperTitles || []).length === 0 ? (
                            <p className="empty-panel-copy">
                                No papers assigned yet. Use “Send to Theme” from a paper.
                            </p>
                        ) : (
                            <ul>
                                {activeDraft.linkedPaperTitles.map((paperTitle) => (
                                    <li key={paperTitle}>
                                        <button
                                            type="button"
                                            className="text-button"
                                            onClick={() => onSelectThemePaper(paperTitle)}
                                        >
                                            {paperTitle}
                                        </button>
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
