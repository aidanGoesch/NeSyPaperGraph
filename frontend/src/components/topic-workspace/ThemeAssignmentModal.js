import React, { useMemo, useState } from "react";

export default function ThemeAssignmentModal({
    paperTitle,
    themeNotes,
    onClose,
    onSave,
}) {
    const initiallySelectedIds = useMemo(
        () =>
            themeNotes
                .filter((theme) =>
                    (theme.linkedPaperTitles || []).includes(paperTitle)
                )
                .map((theme) => theme.id),
        [paperTitle, themeNotes]
    );
    const [selectedThemeIds, setSelectedThemeIds] = useState(initiallySelectedIds);

    const toggleThemeSelection = (themeId) => {
        setSelectedThemeIds((prev) =>
            prev.includes(themeId)
                ? prev.filter((id) => id !== themeId)
                : [...prev, themeId]
        );
    };

    return (
        <div className="theme-modal-overlay" role="presentation" onClick={onClose}>
            <div
                className="theme-modal"
                role="dialog"
                aria-modal="true"
                aria-label="Assign paper to themes"
                onClick={(event) => event.stopPropagation()}
            >
                <h3>Assign Paper to Themes</h3>
                <p className="theme-modal-subtitle">{paperTitle}</p>
                <div className="theme-modal-list">
                    {themeNotes.length === 0 ? (
                        <p className="empty-panel-copy">
                            No themes exist yet. Create one first.
                        </p>
                    ) : (
                        themeNotes.map((theme) => (
                            <label key={theme.id} className="theme-modal-item">
                                <input
                                    type="checkbox"
                                    checked={selectedThemeIds.includes(theme.id)}
                                    onChange={() => toggleThemeSelection(theme.id)}
                                />
                                <span>{theme.themeTitle}</span>
                            </label>
                        ))
                    )}
                </div>
                <div className="theme-modal-actions">
                    <button type="button" onClick={onClose}>
                        Cancel
                    </button>
                    <button
                        type="button"
                        onClick={() => onSave(selectedThemeIds)}
                    >
                        Save
                    </button>
                </div>
            </div>
        </div>
    );
}
