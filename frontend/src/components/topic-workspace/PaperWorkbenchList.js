import React, { useEffect, useMemo, useRef, useState } from "react";

function normalizeAuthor(authors) {
    if (!authors) return "Unknown";
    if (Array.isArray(authors)) return authors.join(", ");
    return String(authors);
}

function recommendationSourceLabel(source) {
    return source === "graph" ? "In graph" : "Semantic Scholar";
}

function recommendationBrowserUrl(paper) {
    if (paper?.url) return paper.url;
    if (paper?.paperId) {
        return `https://www.semanticscholar.org/paper/${paper.paperId}`;
    }
    return "";
}

function normalizePaperTitle(title) {
    return String(title || "").trim().toLowerCase();
}

function formatReaderUrl(url) {
    if (!url) return "";
    try {
        const parsed = new URL(url);
        const preview = `${parsed.hostname}${parsed.pathname}`;
        return preview.length > 78 ? `${preview.slice(0, 78)}...` : preview;
    } catch {
        return url.length > 78 ? `${url.slice(0, 78)}...` : url;
    }
}

const MAX_PAPER_NOTE_CHARS = 12000;

function insertAtSelection(sourceText, selectionStart, selectionEnd, insertText) {
    const before = sourceText.slice(0, selectionStart);
    const after = sourceText.slice(selectionEnd);
    return {
        text: `${before}${insertText}${after}`,
        start: before.length + insertText.length,
        end: before.length + insertText.length,
    };
}

function updateSelectionLines(sourceText, selectionStart, selectionEnd, mapper) {
    const start = Math.max(0, sourceText.lastIndexOf("\n", selectionStart - 1) + 1);
    let end = sourceText.indexOf("\n", selectionEnd);
    if (end < 0) end = sourceText.length;
    const selectedBlock = sourceText.slice(start, end);
    const lines = selectedBlock.split("\n");
    const mapped = lines.map(mapper);
    const nextBlock = mapped.join("\n");
    const nextText = `${sourceText.slice(0, start)}${nextBlock}${sourceText.slice(end)}`;
    const delta = nextBlock.length - selectedBlock.length;
    return {
        text: nextText,
        start: selectionStart + delta,
        end: selectionEnd + delta,
    };
}

export default function PaperWorkbenchList({
    papers,
    totalPaperCount,
    hasMorePapers,
    onLoadMorePapers,
    selectedTopic,
    selectedTopicLabel,
    hasActiveFilter,
    onClearFilters,
    onOpenThemeAssignmentModal,
    getPaperAnnotation,
    onUpdatePaperAnnotation,
    requestedPaperTitle,
    requestedPaperNonce = 0,
    requestedReaderItem = null,
    requestedReaderNonce = 0,
    onRequestSimilarPapers,
    onAddRecommendationToReadingList,
}) {
    const [selectedPaperTitle, setSelectedPaperTitle] = useState(null);
    const [similarPapers, setSimilarPapers] = useState([]);
    const [similarState, setSimilarState] = useState("idle");
    const [similarError, setSimilarError] = useState("");
    const [expandedSimilarKey, setExpandedSimilarKey] = useState(null);
    const [isPaperReaderOpen, setIsPaperReaderOpen] = useState(false);
    const [isPreviewOpen, setIsPreviewOpen] = useState(false);
    const [noteEditorError, setNoteEditorError] = useState("");
    const [flashPaperTitle, setFlashPaperTitle] = useState(null);
    const [externalReaderItem, setExternalReaderItem] = useState(null);
    const [isReaderFrameLoaded, setIsReaderFrameLoaded] = useState(false);
    const [readerFrameError, setReaderFrameError] = useState("");
    const noteTextareaRef = useRef(null);
    const flashTimerRef = useRef(null);

    useEffect(() => {
        if (!requestedPaperTitle) return;
        const requestedKey = normalizePaperTitle(requestedPaperTitle);
        const matchedPaper = papers.find(
            (paper) =>
                paper.title === requestedPaperTitle ||
                normalizePaperTitle(paper.title) === requestedKey
        );
        if (matchedPaper?.title) {
            setSelectedPaperTitle(matchedPaper.title);
            if (requestedPaperNonce > 0) {
                setFlashPaperTitle(matchedPaper.title);
                if (flashTimerRef.current) {
                    clearTimeout(flashTimerRef.current);
                }
                flashTimerRef.current = setTimeout(() => {
                    setFlashPaperTitle((current) =>
                        current === matchedPaper.title ? null : current
                    );
                }, 1000);
            }
        }
        return () => {
            if (flashTimerRef.current) {
                clearTimeout(flashTimerRef.current);
                flashTimerRef.current = null;
            }
        };
    }, [requestedPaperTitle, requestedPaperNonce, papers]);

    const selectedPaper = useMemo(
        () => papers.find((paper) => paper.title === selectedPaperTitle) || null,
        [papers, selectedPaperTitle]
    );

    const selectedAnnotation = selectedPaper
        ? getPaperAnnotation(selectedPaper.title)
        : null;

    useEffect(() => {
        if (!requestedReaderItem || requestedReaderNonce <= 0) return;
        setExternalReaderItem({
            title: requestedReaderItem.title || "Untitled paper",
            annotationKey:
                requestedReaderItem.annotationKey ||
                requestedReaderItem.title ||
                requestedReaderItem.url ||
                null,
            url: requestedReaderItem.url || "",
            authors: requestedReaderItem.authors || [],
            publication_date: requestedReaderItem.publication_date || "",
            venue: requestedReaderItem.venue || "",
            status: requestedReaderItem.status || "inbox",
        });
        setIsPaperReaderOpen(true);
    }, [requestedReaderItem, requestedReaderNonce]);

    const activeReader = externalReaderItem || selectedPaper;
    const activeReaderAnnotationKey =
        externalReaderItem?.annotationKey || selectedPaper?.title || null;
    const activeReaderAnnotation = activeReaderAnnotationKey
        ? getPaperAnnotation(activeReaderAnnotationKey)
        : null;

    useEffect(() => {
        setIsReaderFrameLoaded(false);
        setReaderFrameError("");
    }, [activeReader?.url]);

    useEffect(() => {
        setSimilarPapers([]);
        setSimilarState("idle");
        setSimilarError("");
        setExpandedSimilarKey(null);
        setNoteEditorError("");
        setIsPreviewOpen(false);
    }, [selectedPaperTitle]);

    useEffect(() => {
        if (!isPaperReaderOpen) return undefined;
        const handleEscape = (event) => {
            if (event.key === "Escape") {
                setIsPaperReaderOpen(false);
            }
        };
        window.addEventListener("keydown", handleEscape);
        return () => window.removeEventListener("keydown", handleEscape);
    }, [isPaperReaderOpen]);

    const setPaperNotes = (
        paperTitle,
        nextValue,
        { rejectMessage = `Notes are limited to ${MAX_PAPER_NOTE_CHARS.toLocaleString()} characters.` } = {}
    ) => {
        if (!paperTitle) return false;
        if (nextValue.length > MAX_PAPER_NOTE_CHARS) {
            setNoteEditorError(rejectMessage);
            return false;
        }
        onUpdatePaperAnnotation(paperTitle, {
            notesMarkdown: nextValue,
        });
        setNoteEditorError("");
        return true;
    };

    const handleNoteKeyDown = (event) => {
        if (!activeReaderAnnotationKey) return;
        const textarea = event.currentTarget;
        const value = textarea.value;
        const selectionStart = textarea.selectionStart;
        const selectionEnd = textarea.selectionEnd;

        if (event.key === "Tab") {
            event.preventDefault();
            const result = updateSelectionLines(
                value,
                selectionStart,
                selectionEnd,
                (line) => {
                    if (event.shiftKey) {
                        if (line.startsWith("  ")) return line.slice(2);
                        if (line.startsWith("\t")) return line.slice(1);
                        return line;
                    }
                    return `  ${line}`;
                }
            );
            if (!setPaperNotes(activeReaderAnnotationKey, result.text)) return;
            requestAnimationFrame(() => {
                textarea.setSelectionRange(result.start, result.end);
            });
            return;
        }

        if (event.key !== "Enter" || event.shiftKey) return;

        const lineStart = value.lastIndexOf("\n", selectionStart - 1) + 1;
        const lineEndIndex = value.indexOf("\n", selectionStart);
        const lineEnd = lineEndIndex < 0 ? value.length : lineEndIndex;
        const currentLine = value.slice(lineStart, lineEnd);
        const bulletMatch = currentLine.match(/^(\s*)([-*+]|\d+\.)\s(.*)$/);
        if (!bulletMatch) return;

        event.preventDefault();
        const [, indent, marker, tail] = bulletMatch;
        const lineBody = tail.trim();
        const shouldExitList = lineBody.length === 0;
        const nextLine = shouldExitList ? `${indent}` : `${indent}${marker} `;
        const insertion = `\n${nextLine}`;
        const result = insertAtSelection(value, selectionStart, selectionEnd, insertion);
        if (!setPaperNotes(activeReaderAnnotationKey, result.text)) return;
        requestAnimationFrame(() => {
            textarea.setSelectionRange(result.start, result.end);
        });
    };

    const handleNotePaste = async (event) => {
        if (!activeReaderAnnotationKey) return;
        const clipboardItems = Array.from(event.clipboardData?.items || []);
        const imageItem = clipboardItems.find(
            (item) => item.kind === "file" && item.type.startsWith("image/")
        );
        if (!imageItem) return;
        const file = imageItem.getAsFile();
        if (!file) return;
        event.preventDefault();
        const readAsDataUrl = () =>
            new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(String(reader.result || ""));
                reader.onerror = () => reject(new Error("Failed to read pasted image."));
                reader.readAsDataURL(file);
            });
        try {
            const dataUrl = await readAsDataUrl();
            if (!dataUrl) return;
            const textarea = noteTextareaRef.current;
            const sourceValue = activeReaderAnnotation?.notesMarkdown || "";
            const selectionStart = textarea?.selectionStart ?? sourceValue.length;
            const selectionEnd = textarea?.selectionEnd ?? sourceValue.length;
            const timestamp = new Date()
                .toISOString()
                .replace("T", " ")
                .replace(/\.\d+Z$/, " UTC");
            const imageMarkdown = `\n![Screenshot ${timestamp}](${dataUrl})\n`;
            const result = insertAtSelection(
                sourceValue,
                selectionStart,
                selectionEnd,
                imageMarkdown
            );
            const accepted = setPaperNotes(activeReaderAnnotationKey, result.text, {
                rejectMessage:
                    "Pasted image is too large for the current note size limit. Try a smaller screenshot.",
            });
            if (!accepted || !textarea) return;
            requestAnimationFrame(() => {
                textarea.setSelectionRange(result.start, result.end);
            });
        } catch (error) {
            setNoteEditorError(error?.message || "Unable to paste screenshot.");
        }
    };

    return (
        <section className="workspace-panel workspace-panel-center">
            <div className="workspace-panel-header">
                <h3>
                    Papers{" "}
                    {selectedTopicLabel
                        ? `for "${selectedTopicLabel}"`
                        : selectedTopic
                          ? `for "${selectedTopic}"`
                          : "(cluster scope)"}
                </h3>
                <div className="paper-header-actions">
                    <span>
                        {papers.length} / {totalPaperCount || papers.length} items
                    </span>
                    {hasActiveFilter && (
                        <button
                            type="button"
                            className="text-button"
                            onClick={onClearFilters}
                        >
                            Show all
                        </button>
                    )}
                </div>
            </div>
            <div className="paper-workbench-layout">
                <div className="paper-list">
                    {papers.map((paper) => (
                        <button
                            key={paper.title}
                            type="button"
                            className={`paper-list-item ${
                                selectedPaperTitle === paper.title ? "active" : ""
                            } ${
                                flashPaperTitle === paper.title ? "paper-list-item-flash" : ""
                            }`}
                            onClick={() => {
                                setExternalReaderItem(null);
                                setSelectedPaperTitle(paper.title);
                            }}
                        >
                            <strong>{paper.title}</strong>
                            <small>{normalizeAuthor(paper.authors)}</small>
                            <span>{(paper.topics || []).slice(0, 3).join(" • ")}</span>
                        </button>
                    ))}
                    {hasMorePapers && (
                        <button
                            type="button"
                            className="paper-load-more-button"
                            onClick={onLoadMorePapers}
                        >
                            Load more papers
                        </button>
                    )}
                </div>
                <div
                    className={`paper-details ${
                        selectedPaper?.title &&
                        flashPaperTitle === selectedPaper.title
                            ? "paper-details-flash"
                            : ""
                    }`}
                >
                    {selectedPaper ? (
                        <>
                            <h4>{selectedPaper.title}</h4>
                            <p className="paper-details-meta">
                                {normalizeAuthor(selectedPaper.authors)} |{" "}
                                {selectedPaper.publication_date || "Unknown year"}
                            </p>
                            <p className="paper-details-abstract">
                                {selectedPaper.abstract || "No summary available."}
                            </p>
                            <div className="paper-actions">
                                <button
                                    type="button"
                                    onClick={() =>
                                        onOpenThemeAssignmentModal(selectedPaper.title)
                                    }
                                >
                                    Send to Theme
                                </button>
                                <button
                                    type="button"
                                    onClick={async () => {
                                        if (!onRequestSimilarPapers) return;
                                        setSimilarState("loading");
                                        setSimilarError("");
                                        setSimilarPapers([]);
                                        try {
                                            const results =
                                                (await onRequestSimilarPapers(selectedPaper)) || [];
                                            setSimilarPapers(
                                                Array.isArray(results) ? results : []
                                            );
                                            setSimilarState("success");
                                            setExpandedSimilarKey(null);
                                        } catch (error) {
                                            setSimilarError(
                                                `Failed to load recommendations: ${
                                                    error?.message || "Unknown error"
                                                }`
                                            );
                                            setSimilarState("error");
                                        }
                                    }}
                                >
                                    See similar papers
                                </button>
                            </div>
                            {similarState === "loading" && (
                                <p className="theme-sync-hint">Loading recommendations...</p>
                            )}
                            {similarState === "error" && (
                                <p className="validation-error">{similarError}</p>
                            )}
                            {similarState === "success" && similarPapers.length === 0 && (
                                <p className="theme-sync-hint">
                                    No similar papers found for this paper.
                                </p>
                            )}
                            {similarState === "success" && similarPapers.length > 0 && (
                                <div className="linked-papers">
                                    <strong>Similar papers</strong>
                                    <ul className="theme-linked-paper-list">
                                        {similarPapers.map((paper, index) => (
                                            <li key={paper.paperId || `${paper.title}-${index}`}>
                                                <div className="theme-linked-paper-card">
                                                    <button
                                                        type="button"
                                                        className="paper-list-item"
                                                        onClick={() => {
                                                            const cardKey =
                                                                paper.paperId ||
                                                                paper.title ||
                                                                String(index);
                                                            setExpandedSimilarKey((previous) =>
                                                                previous === cardKey
                                                                    ? null
                                                                    : cardKey
                                                            );
                                                        }}
                                                    >
                                                        <strong>
                                                            {paper.title || "Untitled paper"}
                                                        </strong>
                                                        <small>
                                                            <span
                                                                className={`recommendation-source-badge ${
                                                                    paper.source === "graph"
                                                                        ? "recommendation-source-graph"
                                                                        : "recommendation-source-semantic"
                                                                }`}
                                                            >
                                                                {recommendationSourceLabel(
                                                                    paper.source
                                                                )}
                                                            </span>
                                                        </small>
                                                    </button>
                                                    {expandedSimilarKey ===
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
                                                                {paper.source !== "graph" &&
                                                                    onAddRecommendationToReadingList && (
                                                                        <button
                                                                            type="button"
                                                                            onClick={() =>
                                                                                onAddRecommendationToReadingList(
                                                                                    paper
                                                                                )
                                                                            }
                                                                        >
                                                                            Add to reading list
                                                                        </button>
                                                                    )}
                                                                {paper.source !== "graph" &&
                                                                    recommendationBrowserUrl(
                                                                        paper
                                                                    ) && (
                                                                        <button
                                                                            type="button"
                                                                            className="theme-queue-open-button"
                                                                            onClick={() =>
                                                                                window.open(
                                                                                    recommendationBrowserUrl(
                                                                                        paper
                                                                                    ),
                                                                                    "_blank",
                                                                                    "noopener,noreferrer"
                                                                                )
                                                                            }
                                                                        >
                                                                            Open in browser
                                                                        </button>
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
                            <label className="annotation-label" htmlFor="annotation-input">
                                Paper Note
                            </label>
                            <button
                                type="button"
                                className="annotation-preview-card"
                                onClick={() => {
                                    setExternalReaderItem(null);
                                    setIsPaperReaderOpen(true);
                                }}
                            >
                                {selectedAnnotation?.notesMarkdown?.trim() ? (
                                    selectedAnnotation.notesMarkdown
                                ) : (
                                    <span className="annotation-preview-empty">
                                        No note yet. Click to open the split paper reader popup.
                                    </span>
                                )}
                            </button>
                            <button
                                type="button"
                                className="topic-search-open-button"
                                onClick={() => {
                                    setExternalReaderItem(null);
                                    setIsPaperReaderOpen(true);
                                }}
                            >
                                Open paper
                            </button>
                        </>
                    ) : (
                        <p className="empty-panel-copy">
                            Select a paper to review details and add an annotation.
                        </p>
                    )}
                </div>
            </div>
            {activeReader && isPaperReaderOpen && (
                <div
                    className="paper-note-modal-overlay"
                    role="dialog"
                    aria-modal="true"
                    aria-label="Paper reader and notes"
                    onMouseDown={(event) => {
                        if (event.target === event.currentTarget) {
                            setIsPaperReaderOpen(false);
                        }
                    }}
                >
                    <div className="paper-note-modal paper-reader-modal">
                        <div className="paper-note-modal-header">
                            <h3>Paper reader</h3>
                            <button
                                type="button"
                                className="topic-search-close-button"
                                onClick={() => setIsPaperReaderOpen(false)}
                            >
                                Done
                            </button>
                        </div>
                        <div className="paper-reader-modal-body">
                            <section className="paper-reader-pane">
                                <p className="paper-note-modal-subtitle">{activeReader.title}</p>
                                <p className="paper-details-meta">
                                    {normalizeAuthor(activeReader.authors)} |{" "}
                                    {activeReader.publication_date || "Unknown year"}
                                    {activeReader.venue ? ` • ${activeReader.venue}` : ""}
                                </p>
                                {activeReader.url ? (
                                    <>
                                        <div className="paper-reader-frame-toolbar">
                                            <a
                                                href={activeReader.url}
                                                target="_blank"
                                                rel="noreferrer"
                                                className="paper-reader-source-link"
                                                title={activeReader.url}
                                            >
                                                {formatReaderUrl(activeReader.url)}
                                            </a>
                                            <a
                                                href={activeReader.url}
                                                target="_blank"
                                                rel="noreferrer"
                                                className="open-link-button"
                                            >
                                                Open in browser
                                            </a>
                                        </div>
                                        {!isReaderFrameLoaded && (
                                            <p className="theme-sync-hint">
                                                Loading paper page...
                                            </p>
                                        )}
                                        {readerFrameError && (
                                            <p className="validation-error">{readerFrameError}</p>
                                        )}
                                        <iframe
                                            key={activeReader.url}
                                            className="paper-reader-frame"
                                            src={activeReader.url}
                                            title={`Paper content: ${activeReader.title}`}
                                            onLoad={() => setIsReaderFrameLoaded(true)}
                                            onError={() => {
                                                setReaderFrameError(
                                                    "This site blocked in-app embedding. Use 'Open in browser' to read it."
                                                );
                                            }}
                                        />
                                        <p className="theme-sync-hint">
                                            If this pane appears blank, the publisher likely blocks
                                            embedding. Use "Open in browser".
                                        </p>
                                    </>
                                ) : (
                                    <p className="paper-details-abstract">
                                        {activeReader.abstract || "No summary available."}
                                    </p>
                                )}
                                {selectedPaper && (
                                    <>
                                        <div className="paper-actions">
                                            <button
                                                type="button"
                                                onClick={() =>
                                                    onOpenThemeAssignmentModal(selectedPaper.title)
                                                }
                                            >
                                                Send to Theme
                                            </button>
                                            <button
                                                type="button"
                                                onClick={async () => {
                                                    if (!onRequestSimilarPapers) return;
                                                    setSimilarState("loading");
                                                    setSimilarError("");
                                                    setSimilarPapers([]);
                                                    try {
                                                        const results =
                                                            (await onRequestSimilarPapers(
                                                                selectedPaper
                                                            )) || [];
                                                        setSimilarPapers(
                                                            Array.isArray(results) ? results : []
                                                        );
                                                        setSimilarState("success");
                                                        setExpandedSimilarKey(null);
                                                    } catch (error) {
                                                        setSimilarError(
                                                            `Failed to load recommendations: ${
                                                                error?.message || "Unknown error"
                                                            }`
                                                        );
                                                        setSimilarState("error");
                                                    }
                                                }}
                                            >
                                                See similar papers
                                            </button>
                                        </div>
                                        {similarState === "loading" && (
                                            <p className="theme-sync-hint">
                                                Loading recommendations...
                                            </p>
                                        )}
                                        {similarState === "error" && (
                                            <p className="validation-error">{similarError}</p>
                                        )}
                                        {similarState === "success" &&
                                            similarPapers.length === 0 && (
                                                <p className="theme-sync-hint">
                                                    No similar papers found for this paper.
                                                </p>
                                            )}
                                        {similarState === "success" &&
                                            similarPapers.length > 0 && (
                                                <div className="linked-papers">
                                                    <strong>Similar papers</strong>
                                                    <ul className="theme-linked-paper-list">
                                                        {similarPapers.map((paper, index) => (
                                                            <li
                                                                key={
                                                                    paper.paperId ||
                                                                    `${paper.title}-${index}`
                                                                }
                                                            >
                                                                <div className="theme-linked-paper-card">
                                                                    <button
                                                                        type="button"
                                                                        className="paper-list-item"
                                                                        onClick={() => {
                                                                            const cardKey =
                                                                                paper.paperId ||
                                                                                paper.title ||
                                                                                String(index);
                                                                            setExpandedSimilarKey(
                                                                                (previous) =>
                                                                                    previous ===
                                                                                    cardKey
                                                                                        ? null
                                                                                        : cardKey
                                                                            );
                                                                        }}
                                                                    >
                                                                        <strong>
                                                                            {paper.title ||
                                                                                "Untitled paper"}
                                                                        </strong>
                                                                        <small>
                                                                            <span
                                                                                className={`recommendation-source-badge ${
                                                                                    paper.source ===
                                                                                    "graph"
                                                                                        ? "recommendation-source-graph"
                                                                                        : "recommendation-source-semantic"
                                                                                }`}
                                                                            >
                                                                                {recommendationSourceLabel(
                                                                                    paper.source
                                                                                )}
                                                                            </span>
                                                                        </small>
                                                                    </button>
                                                                    {expandedSimilarKey ===
                                                                        (paper.paperId ||
                                                                            paper.title ||
                                                                            String(index)) && (
                                                                        <div className="theme-linked-paper-meta">
                                                                            <p>
                                                                                {(paper.authors ||
                                                                                    []).length
                                                                                    ? paper.authors.join(
                                                                                          ", "
                                                                                      )
                                                                                    : "Unknown authors"}{" "}
                                                                                |{" "}
                                                                                {paper.year ||
                                                                                    "Unknown year"}
                                                                            </p>
                                                                            <p>
                                                                                {paper.abstract ||
                                                                                    "No summary available."}
                                                                            </p>
                                                                            <div className="paper-actions">
                                                                                {paper.source !==
                                                                                    "graph" &&
                                                                                    onAddRecommendationToReadingList && (
                                                                                        <button
                                                                                            type="button"
                                                                                            onClick={() =>
                                                                                                onAddRecommendationToReadingList(
                                                                                                    paper
                                                                                                )
                                                                                            }
                                                                                        >
                                                                                            Add to reading
                                                                                            list
                                                                                        </button>
                                                                                    )}
                                                                                {paper.source !==
                                                                                    "graph" &&
                                                                                    recommendationBrowserUrl(
                                                                                        paper
                                                                                    ) && (
                                                                                        <button
                                                                                            type="button"
                                                                                            className="theme-queue-open-button"
                                                                                            onClick={() =>
                                                                                                window.open(
                                                                                                    recommendationBrowserUrl(
                                                                                                        paper
                                                                                                    ),
                                                                                                    "_blank",
                                                                                                    "noopener,noreferrer"
                                                                                                )
                                                                                            }
                                                                                        >
                                                                                            Open in browser
                                                                                        </button>
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
                                    </>
                                )}
                            </section>
                            <section className="paper-notes-pane">
                                <div className="paper-notes-pane-header">
                                    <strong>Paper notes</strong>
                                    <div className="paper-notes-pane-tools">
                                        <span className="paper-note-char-count">
                                            {(activeReaderAnnotation?.notesMarkdown || "").length} /{" "}
                                            {MAX_PAPER_NOTE_CHARS}
                                        </span>
                                        <button
                                            type="button"
                                            className="topic-search-open-button"
                                            onClick={() =>
                                                setIsPreviewOpen((previous) => !previous)
                                            }
                                        >
                                            {isPreviewOpen ? "Hide preview" : "Show preview"}
                                        </button>
                                    </div>
                                </div>
                                <textarea
                                    ref={noteTextareaRef}
                                    id="annotation-modal-input"
                                    className="paper-note-modal-textarea"
                                    value={activeReaderAnnotation?.notesMarkdown || ""}
                                    onChange={(event) =>
                                        setPaperNotes(activeReaderAnnotationKey, event.target.value)
                                    }
                                    onKeyDown={handleNoteKeyDown}
                                    onPaste={handleNotePaste}
                                    placeholder="Capture paper-specific insights. Use Tab/Shift+Tab for nested bullets, and paste screenshots directly."
                                />
                                {noteEditorError && (
                                    <p className="validation-error">{noteEditorError}</p>
                                )}
                                {isPreviewOpen && (
                                    <div className="paper-note-markdown-preview">
                                        <pre>
                                            {activeReaderAnnotation?.notesMarkdown ||
                                                "No notes yet."}
                                        </pre>
                                    </div>
                                )}
                            </section>
                        </div>
                    </div>
                </div>
            )}
        </section>
    );
}
