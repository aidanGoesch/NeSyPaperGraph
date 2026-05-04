import React, { useEffect, useMemo, useRef, useState } from "react";
import DesktopPaperWebview from "./DesktopPaperWebview";

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

const MAX_PAPER_NOTE_CHARS = 250000;

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
    onResolvePaperMetadata,
    desktopConfig = {},
}) {
    const [selectedPaperTitle, setSelectedPaperTitle] = useState(null);
    const [similarPapers, setSimilarPapers] = useState([]);
    const [similarState, setSimilarState] = useState("idle");
    const [similarError, setSimilarError] = useState("");
    const [expandedSimilarKey, setExpandedSimilarKey] = useState(null);
    const [isPaperReaderOpen, setIsPaperReaderOpen] = useState(false);
    const [noteEditorError, setNoteEditorError] = useState("");
    const [flashPaperTitle, setFlashPaperTitle] = useState(null);
    const [externalReaderItem, setExternalReaderItem] = useState(null);
    const [isReaderFrameLoaded, setIsReaderFrameLoaded] = useState(false);
    const [readerFrameError, setReaderFrameError] = useState("");
    const [isResolvingReaderUrl, setIsResolvingReaderUrl] = useState(false);
    const [readerResolveError, setReaderResolveError] = useState("");
    const [manualReaderUrl, setManualReaderUrl] = useState("");
    const [manualReaderUrlError, setManualReaderUrlError] = useState("");
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
    const readerUrl = (activeReader?.url || activeReaderAnnotation?.sourceUrl || "").trim();
    const useDesktopInAppBrowser = Boolean(
        desktopConfig?.isDesktop && desktopConfig?.supportsInAppBrowser
    );

    useEffect(() => {
        setIsReaderFrameLoaded(false);
        setReaderFrameError("");
    }, [readerUrl]);

    useEffect(() => {
        setSimilarPapers([]);
        setSimilarState("idle");
        setSimilarError("");
        setExpandedSimilarKey(null);
        setNoteEditorError("");
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

    useEffect(() => {
        if (!isPaperReaderOpen || externalReaderItem || !selectedPaper) return;
        if (readerUrl || !onResolvePaperMetadata) return;
        const selectedTitle = (selectedPaper.title || "").trim();
        if (!selectedTitle) return;

        const parsedYear = Number.parseInt(
            String(selectedPaper.publication_date || "").slice(0, 4),
            10
        );
        let isCancelled = false;
        setIsResolvingReaderUrl(true);
        setReaderResolveError("");
        setManualReaderUrlError("");

        onResolvePaperMetadata({
            semanticScholarPaperId: selectedPaper.semanticScholarPaperId || "",
            url: selectedPaper.url || "",
            title: selectedTitle,
            authors: Array.isArray(selectedPaper.authors) ? selectedPaper.authors : [],
            year: Number.isFinite(parsedYear) ? parsedYear : null,
        })
            .then((resolved) => {
                if (isCancelled) return;
                const resolvedUrl = String(resolved?.url || "").trim();
                if (!resolvedUrl) {
                    setReaderResolveError(
                        "Unable to auto-resolve this paper. Enter a link manually."
                    );
                    return;
                }
                onUpdatePaperAnnotation(activeReaderAnnotationKey, {
                    sourceUrl: resolvedUrl,
                });
                setExternalReaderItem({
                    title: resolved?.title || selectedPaper.title || "Untitled paper",
                    annotationKey: activeReaderAnnotationKey,
                    url: resolvedUrl,
                    authors:
                        Array.isArray(resolved?.authors) && resolved.authors.length > 0
                            ? resolved.authors
                            : selectedPaper.authors || [],
                    publication_date:
                        resolved?.year != null
                            ? String(resolved.year)
                            : selectedPaper.publication_date || "",
                    venue: resolved?.venue || selectedPaper.venue || "",
                    status: "reading",
                });
                setManualReaderUrl("");
            })
            .catch((error) => {
                if (isCancelled) return;
                setReaderResolveError(
                    error?.message ||
                        "Unable to auto-resolve this paper. Enter a link manually."
                );
            })
            .finally(() => {
                if (!isCancelled) {
                    setIsResolvingReaderUrl(false);
                }
            });

        return () => {
            isCancelled = true;
        };
    }, [
        activeReaderAnnotationKey,
        externalReaderItem,
        isPaperReaderOpen,
        onResolvePaperMetadata,
        onUpdatePaperAnnotation,
        readerUrl,
        selectedPaper,
    ]);

    useEffect(() => {
        if (!isPaperReaderOpen || !activeReaderAnnotationKey) return;
        const resolvedReaderUrl = String(readerUrl || "").trim();
        if (!resolvedReaderUrl) return;
        const storedSourceUrl = String(activeReaderAnnotation?.sourceUrl || "").trim();
        if (storedSourceUrl === resolvedReaderUrl) return;
        // Persist the canonical reader URL so every paper annotation keeps an associated link.
        onUpdatePaperAnnotation(activeReaderAnnotationKey, {
            sourceUrl: resolvedReaderUrl,
        });
    }, [
        activeReaderAnnotation?.sourceUrl,
        activeReaderAnnotationKey,
        isPaperReaderOpen,
        onUpdatePaperAnnotation,
        readerUrl,
    ]);

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

    const persistReaderUrl = (inputUrl) => {
        if (!activeReaderAnnotationKey) return false;
        const trimmed = String(inputUrl || "").trim();
        if (!trimmed) {
            setManualReaderUrlError("Enter a paper URL.");
            return false;
        }
        let parsed;
        try {
            parsed = new URL(trimmed);
        } catch {
            setManualReaderUrlError("Enter a valid URL, including https://");
            return false;
        }
        if (!["http:", "https:"].includes(parsed.protocol)) {
            setManualReaderUrlError("Only http(s) links are supported.");
            return false;
        }
        const normalizedUrl = parsed.toString();
        onUpdatePaperAnnotation(activeReaderAnnotationKey, {
            sourceUrl: normalizedUrl,
        });
        setExternalReaderItem((previous) => ({
            title: previous?.title || selectedPaper?.title || activeReader?.title || "Untitled paper",
            annotationKey: activeReaderAnnotationKey,
            url: normalizedUrl,
            authors:
                previous?.authors ||
                selectedPaper?.authors ||
                activeReader?.authors ||
                [],
            publication_date:
                previous?.publication_date ||
                selectedPaper?.publication_date ||
                activeReader?.publication_date ||
                "",
            venue: previous?.venue || selectedPaper?.venue || activeReader?.venue || "",
            status: previous?.status || activeReader?.status || "reading",
        }));
        setReaderResolveError("");
        setManualReaderUrlError("");
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

    const handleNotePaste = (event) => {
        const clipboardItems = Array.from(event.clipboardData?.items || []);
        const hasImage = clipboardItems.some(
            (item) => item.kind === "file" && item.type.startsWith("image/")
        );
        if (!hasImage) return;
        event.preventDefault();
        setNoteEditorError(
            "Image pasting is disabled in notes. Please paste text only."
        );
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
                                    setManualReaderUrl("");
                                    setManualReaderUrlError("");
                                    setReaderResolveError("");
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
                                {readerUrl ? (
                                    useDesktopInAppBrowser ? (
                                        <DesktopPaperWebview
                                            url={readerUrl}
                                            title={activeReader.title}
                                        />
                                    ) : (
                                        <>
                                            {!isReaderFrameLoaded && (
                                                <p className="theme-sync-hint">
                                                    Loading paper page...
                                                </p>
                                            )}
                                            {readerFrameError && (
                                                <p className="validation-error">
                                                    {readerFrameError}
                                                </p>
                                            )}
                                            <iframe
                                                key={readerUrl}
                                                className="paper-reader-frame"
                                                src={readerUrl}
                                                title={`Paper content: ${activeReader.title}`}
                                                onLoad={() => setIsReaderFrameLoaded(true)}
                                                onError={() => {
                                                    setReaderFrameError(
                                                        "This site blocked in-app embedding. Use 'Open in browser' to read it."
                                                    );
                                                }}
                                            />
                                            <div className="paper-reader-frame-toolbar">
                                                <a
                                                    href={readerUrl}
                                                    target="_blank"
                                                    rel="noreferrer"
                                                    className="open-link-button"
                                                >
                                                    Open in browser
                                                </a>
                                            </div>
                                            <p className="theme-sync-hint">
                                                If this pane appears blank, the publisher likely
                                                blocks embedding. Use "Open in browser".
                                            </p>
                                        </>
                                    )
                                ) : (
                                    <div className="paper-reader-link-fallback">
                                        {isResolvingReaderUrl ? (
                                            <p className="theme-sync-hint">
                                                Resolving paper page from Semantic Scholar...
                                            </p>
                                        ) : (
                                            <p className="theme-sync-hint">
                                                No paper URL is saved yet. Add a link to open the
                                                reader pane.
                                            </p>
                                        )}
                                        {readerResolveError && (
                                            <p className="validation-error">{readerResolveError}</p>
                                        )}
                                        <div className="paper-reader-manual-link-row">
                                            <input
                                                type="url"
                                                value={manualReaderUrl}
                                                onChange={(event) =>
                                                    setManualReaderUrl(event.target.value)
                                                }
                                                placeholder="https://arxiv.org/abs/..."
                                                className="paper-reader-manual-link-input"
                                                disabled={isResolvingReaderUrl}
                                            />
                                            <button
                                                type="button"
                                                className="topic-search-open-button"
                                                disabled={isResolvingReaderUrl}
                                                onClick={() => {
                                                    persistReaderUrl(manualReaderUrl);
                                                }}
                                            >
                                                Save link
                                            </button>
                                        </div>
                                        {manualReaderUrlError && (
                                            <p className="validation-error">
                                                {manualReaderUrlError}
                                            </p>
                                        )}
                                    </div>
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
                                    </div>
                                </div>
                                <textarea
                                    id="annotation-modal-input"
                                    className="paper-note-modal-textarea"
                                    value={activeReaderAnnotation?.notesMarkdown || ""}
                                    onChange={(event) =>
                                        setPaperNotes(
                                            activeReaderAnnotationKey,
                                            event.target.value
                                        )
                                    }
                                    onKeyDown={handleNoteKeyDown}
                                    onPaste={handleNotePaste}
                                    placeholder="Capture paper-specific insights. Use Tab/Shift+Tab for nested bullets."
                                />
                                {noteEditorError && (
                                    <p className="validation-error">{noteEditorError}</p>
                                )}
                            </section>
                        </div>
                    </div>
                </div>
            )}
        </section>
    );
}
