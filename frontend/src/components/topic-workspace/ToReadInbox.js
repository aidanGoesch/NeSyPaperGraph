import React, { useMemo, useState } from "react";

const READING_STATES = ["inbox", "queued", "reading", "done"];

function deriveTitleFromUrl(url) {
    try {
        const parsed = new URL(url);
        const lastPath = parsed.pathname.split("/").filter(Boolean).pop();
        if (lastPath && lastPath.length > 6) {
            return decodeURIComponent(lastPath).slice(0, 80);
        }
        return parsed.hostname.replace("www.", "");
    } catch {
        return "";
    }
}

function formatUrlPreview(url) {
    if (!url) return "";
    try {
        const parsed = new URL(url);
        const preview = `${parsed.hostname}${parsed.pathname}`;
        return preview.length > 90 ? `${preview.slice(0, 90)}...` : preview;
    } catch {
        return url.length > 90 ? `${url.slice(0, 90)}...` : url;
    }
}

export default function ToReadInbox({
    readingItems,
    topics,
    themeNotes,
    onAddReadingItem,
    onUpdateReadingItem,
    onRemoveReadingItem,
    onFocusPaper,
}) {
    const [urlInput, setUrlInput] = useState("");
    const [titleInput, setTitleInput] = useState("");
    const [themeInput, setThemeInput] = useState("");
    const [statusInput, setStatusInput] = useState("inbox");
    const [themeFilter, setThemeFilter] = useState("");

    const sortedItems = useMemo(
        () =>
            [...readingItems].sort(
                (a, b) =>
                    new Date(b.updatedAt || b.createdAt).getTime() -
                    new Date(a.updatedAt || a.createdAt).getTime()
            ),
        [readingItems]
    );
    const visibleItems = useMemo(
        () =>
            themeFilter
                ? sortedItems.filter((item) => item.linkedThemeId === themeFilter)
                : sortedItems,
        [sortedItems, themeFilter]
    );

    const handleAddPaperLink = () => {
        if (!urlInput.trim()) return;
        onAddReadingItem({
            sourceType: "url",
            url: urlInput.trim(),
            title: titleInput.trim() || deriveTitleFromUrl(urlInput.trim()),
            status: statusInput,
            linkedThemeId: themeInput || null,
        });
        setUrlInput("");
        setTitleInput("");
        setThemeInput("");
        setStatusInput("inbox");
    };

    return (
        <section className="workspace-panel workspace-panel-inbox">
            <div className="workspace-panel-header">
                <h3>To-Read Inbox</h3>
                <div className="inbox-header-controls">
                    <select
                        value={themeFilter}
                        onChange={(event) => setThemeFilter(event.target.value)}
                    >
                        <option value="">All themes</option>
                        {themeNotes.map((theme) => (
                            <option key={theme.id} value={theme.id}>
                                {theme.themeTitle}
                            </option>
                        ))}
                    </select>
                    <span>{visibleItems.length} items</span>
                </div>
            </div>
            <div className="inbox-add-grid">
                <input
                    className="inbox-title-input"
                    value={titleInput}
                    onChange={(event) => setTitleInput(event.target.value)}
                    placeholder="Paper title (optional)"
                    onKeyDown={(event) => {
                        if (event.key === "Enter") {
                            event.preventDefault();
                            handleAddPaperLink();
                        }
                    }}
                />
                <input
                    className="inbox-url-input"
                    value={urlInput}
                    onChange={(event) => setUrlInput(event.target.value)}
                    placeholder="Paste paper URL"
                    onKeyDown={(event) => {
                        if (event.key === "Enter") {
                            event.preventDefault();
                            handleAddPaperLink();
                        }
                    }}
                />
                <select
                    value={themeInput}
                    onChange={(event) => setThemeInput(event.target.value)}
                >
                    <option value="">Optional: assign theme now</option>
                    {themeNotes.map((theme) => (
                        <option key={theme.id} value={theme.id}>
                            {theme.themeTitle}
                        </option>
                    ))}
                </select>
                <select
                    className="inbox-start-select"
                    value={statusInput}
                    onChange={(event) => setStatusInput(event.target.value)}
                >
                    {READING_STATES.map((state) => (
                        <option key={state} value={state}>
                            Start as: {state}
                        </option>
                    ))}
                </select>
                <div className="inbox-add-actions">
                    <button
                        className="inbox-primary-add"
                        type="button"
                        onClick={handleAddPaperLink}
                    >
                        Add
                    </button>
                </div>
            </div>
            <div className="inbox-items">
                {visibleItems.map((item) => (
                    <div key={item.id} className="inbox-item">
                        <div className="inbox-item-main">
                            <div className="inbox-item-text">
                                <strong>{item.title || "Untitled item"}</strong>
                                {item.url && (
                                    <a
                                        href={item.url}
                                        target="_blank"
                                        rel="noreferrer"
                                        className="inbox-url-preview"
                                        title={item.url}
                                    >
                                        {formatUrlPreview(item.url)}
                                    </a>
                                )}
                            </div>
                            {item.url && (
                                <a
                                    href={item.url}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="open-link-button"
                                >
                                    Open
                                </a>
                            )}
                            <div className="inbox-controls inbox-controls-inline">
                                <select
                                    className="inbox-inline-select"
                                    value={item.status}
                                    onChange={(event) =>
                                        event.target.value === "done"
                                            ? onRemoveReadingItem(item.id)
                                            : onUpdateReadingItem(item.id, {
                                                  status: event.target.value,
                                              })
                                    }
                                >
                                    {READING_STATES.map((state) => (
                                        <option key={state} value={state}>
                                            {state}
                                        </option>
                                    ))}
                                </select>
                                <select
                                    className="inbox-inline-select"
                                    value={item.topicHints?.[0] || ""}
                                    onChange={(event) =>
                                        onUpdateReadingItem(item.id, {
                                            topicHints: event.target.value
                                                ? [event.target.value]
                                                : [],
                                        })
                                    }
                                >
                                    <option value="">Topic hint</option>
                                    {topics.map((topic) => (
                                        <option key={topic} value={topic}>
                                            {topic}
                                        </option>
                                    ))}
                                </select>
                                <select
                                    className="inbox-inline-select"
                                    value={item.linkedThemeId || ""}
                                    onChange={(event) =>
                                        onUpdateReadingItem(item.id, {
                                            linkedThemeId: event.target.value || null,
                                        })
                                    }
                                >
                                    <option value="">Link theme</option>
                                    {themeNotes.map((theme) => (
                                        <option key={theme.id} value={theme.id}>
                                            {theme.themeTitle}
                                        </option>
                                    ))}
                                </select>
                            </div>
                            <button
                                type="button"
                                className="inbox-delete-button"
                                onClick={() => onRemoveReadingItem(item.id)}
                                aria-label="Delete queued paper"
                                title="Delete paper from queue"
                            >
                                ×
                            </button>
                        </div>
                        {item.linkedPaperTitle && (
                            <button
                                type="button"
                                className="text-button"
                                onClick={() => onFocusPaper(item.linkedPaperTitle)}
                            >
                                Open linked paper
                            </button>
                        )}
                    </div>
                ))}
            </div>
        </section>
    );
}
