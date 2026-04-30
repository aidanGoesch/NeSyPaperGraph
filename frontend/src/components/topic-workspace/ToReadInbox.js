import React, { useMemo, useRef, useState } from "react";

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
    onReorderReadingItem,
    onFocusPaper,
    onResolveReadingUrl,
    onMarkReadingItemDone,
}) {
    const [urlInput, setUrlInput] = useState("");
    const [titleInput, setTitleInput] = useState("");
    const [themeInput, setThemeInput] = useState("");
    const [statusInput, setStatusInput] = useState("inbox");
    const [themeFilter, setThemeFilter] = useState("");
    const [isResolvingAdd, setIsResolvingAdd] = useState(false);
    const [addError, setAddError] = useState("");
    const [itemBusyState, setItemBusyState] = useState({});
    const [itemErrors, setItemErrors] = useState({});
    const [draggingItemId, setDraggingItemId] = useState(null);
    const [dragOverItemId, setDragOverItemId] = useState(null);
    const dragPreviewRef = useRef(null);

    const visibleItems = useMemo(
        () =>
            themeFilter
                ? readingItems.filter((item) => item.linkedThemeId === themeFilter)
                : readingItems,
        [readingItems, themeFilter]
    );

    const setItemBusy = (itemId, isBusy) => {
        setItemBusyState((prev) => ({ ...prev, [itemId]: isBusy }));
    };

    const setItemError = (itemId, message) => {
        setItemErrors((prev) => ({ ...prev, [itemId]: message }));
    };

    const clearItemError = (itemId) => {
        setItemErrors((prev) => {
            if (!prev[itemId]) return prev;
            const next = { ...prev };
            delete next[itemId];
            return next;
        });
    };

    const handleAddPaperLink = async () => {
        if (!urlInput.trim()) return;
        setIsResolvingAdd(true);
        setAddError("");

        const trimmedUrl = urlInput.trim();
        const fallbackTitle = titleInput.trim() || deriveTitleFromUrl(trimmedUrl);
        let resolvedMetadata = null;
        if (onResolveReadingUrl) {
            try {
                resolvedMetadata = await onResolveReadingUrl(trimmedUrl);
            } catch (error) {
                setAddError(error.message || "Semantic Scholar metadata lookup failed.");
                setIsResolvingAdd(false);
                return;
            }
        }

        onAddReadingItem({
            sourceType: "url",
            url: trimmedUrl,
            title: titleInput.trim() || resolvedMetadata?.title || fallbackTitle,
            status: statusInput,
            linkedThemeId: themeInput || null,
            semanticScholarPaperId: resolvedMetadata?.semanticScholarPaperId || null,
            authors: resolvedMetadata?.authors || [],
            year: resolvedMetadata?.year ?? null,
            venue: resolvedMetadata?.venue || null,
        });
        setUrlInput("");
        setTitleInput("");
        setThemeInput("");
        setStatusInput("inbox");
        setIsResolvingAdd(false);
    };

    const handleDragStart = (event, itemId) => {
        setDraggingItemId(itemId);
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", itemId);

        const sourceItem = event.currentTarget.closest(".inbox-item");
        if (!sourceItem) return;
        const previewNode = sourceItem.cloneNode(true);
        previewNode.classList.add("inbox-drag-preview");
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
                        disabled={isResolvingAdd}
                        onClick={handleAddPaperLink}
                    >
                        {isResolvingAdd ? "Resolving..." : "Add"}
                    </button>
                </div>
            </div>
            {addError && <p className="inbox-error-text">{addError}</p>}
            <div
                className={`inbox-items ${
                    draggingItemId ? "inbox-items-dragging" : ""
                }`}
            >
                {visibleItems.map((item) => (
                    <div
                        key={item.id}
                        className={`inbox-item ${
                            draggingItemId === item.id ? "inbox-item-dragging" : ""
                        } ${
                            dragOverItemId === item.id &&
                            draggingItemId &&
                            draggingItemId !== item.id
                                ? "inbox-item-drop-target"
                                : ""
                        }`}
                        onDragEnter={() => {
                            if (!draggingItemId || draggingItemId === item.id) return;
                            setDragOverItemId(item.id);
                            if (onReorderReadingItem) {
                                onReorderReadingItem(draggingItemId, item.id);
                            }
                        }}
                        onDragOver={(event) => {
                            if (!draggingItemId || draggingItemId === item.id) return;
                            event.preventDefault();
                        }}
                        onDrop={(event) => {
                            event.preventDefault();
                            handleDragEnd();
                        }}
                    >
                        <div className="inbox-item-main">
                            <button
                                type="button"
                                className="inbox-drag-handle"
                                title="Drag to reorder"
                                aria-label="Drag to reorder"
                                draggable
                                onDragStart={(event) => {
                                    handleDragStart(event, item.id);
                                }}
                                onDragEnd={handleDragEnd}
                            >
                                :::
                            </button>
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
                                {(item.authors?.length || item.year || item.venue) && (
                                    <small className="inbox-metadata">
                                        {item.authors?.length
                                            ? item.authors.slice(0, 2).join(", ")
                                            : "Unknown authors"}
                                        {item.year ? ` • ${item.year}` : ""}
                                        {item.venue ? ` • ${item.venue}` : ""}
                                    </small>
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
                                    disabled={Boolean(itemBusyState[item.id])}
                                    onChange={async (event) => {
                                        const nextStatus = event.target.value;
                                        if (nextStatus !== "done") {
                                            clearItemError(item.id);
                                            onUpdateReadingItem(item.id, {
                                                status: nextStatus,
                                            });
                                            return;
                                        }
                                        if (!onMarkReadingItemDone) return;
                                        setItemBusy(item.id, true);
                                        clearItemError(item.id);
                                        try {
                                            await onMarkReadingItemDone(item);
                                        } catch (error) {
                                            setItemError(
                                                item.id,
                                                error.message ||
                                                    "Failed to add paper to graph."
                                            );
                                            onUpdateReadingItem(item.id, {
                                                status: item.status || "reading",
                                            });
                                        } finally {
                                            setItemBusy(item.id, false);
                                        }
                                    }}
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
                        {itemErrors[item.id] && (
                            <p className="inbox-error-text">{itemErrors[item.id]}</p>
                        )}
                    </div>
                ))}
            </div>
        </section>
    );
}
