import React, { useEffect, useMemo, useState } from "react";

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

export default function PaperWorkbenchList({
    papers,
    totalPaperCount,
    hasMorePapers,
    onLoadMorePapers,
    selectedTopic,
    selectedTopicLabel,
    hasActiveFilter,
    onClearFilters,
    onFocusPaper,
    onOpenThemeAssignmentModal,
    getPaperAnnotation,
    onUpdatePaperAnnotation,
    requestedPaperTitle,
    onRequestSimilarPapers,
    onAddRecommendationToReadingList,
}) {
    const [selectedPaperTitle, setSelectedPaperTitle] = useState(null);
    const [similarPapers, setSimilarPapers] = useState([]);
    const [similarState, setSimilarState] = useState("idle");
    const [similarError, setSimilarError] = useState("");
    const [expandedSimilarKey, setExpandedSimilarKey] = useState(null);
    const [isNoteModalOpen, setIsNoteModalOpen] = useState(false);

    useEffect(() => {
        if (!requestedPaperTitle) return;
        const existsInList = papers.some(
            (paper) => paper.title === requestedPaperTitle
        );
        if (existsInList) {
            setSelectedPaperTitle(requestedPaperTitle);
        }
    }, [requestedPaperTitle, papers]);

    const selectedPaper = useMemo(
        () => papers.find((paper) => paper.title === selectedPaperTitle) || null,
        [papers, selectedPaperTitle]
    );

    const selectedAnnotation = selectedPaper
        ? getPaperAnnotation(selectedPaper.title)
        : null;

    useEffect(() => {
        setSimilarPapers([]);
        setSimilarState("idle");
        setSimilarError("");
        setExpandedSimilarKey(null);
        setIsNoteModalOpen(false);
    }, [selectedPaperTitle]);

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
                            }`}
                            onClick={() => setSelectedPaperTitle(paper.title)}
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
                <div className="paper-details">
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
                                    onClick={() => onFocusPaper(selectedPaper.title)}
                                >
                                    Focus in Graph
                                </button>
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
                                                                {paper.source === "graph" &&
                                                                    paper.title && (
                                                                    <button
                                                                        type="button"
                                                                        onClick={() =>
                                                                            onFocusPaper(
                                                                                paper.title
                                                                            )
                                                                        }
                                                                    >
                                                                        Focus in Graph
                                                                    </button>
                                                                )}
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
                                onClick={() => setIsNoteModalOpen(true)}
                            >
                                {selectedAnnotation?.notesMarkdown?.trim() ? (
                                    selectedAnnotation.notesMarkdown
                                ) : (
                                    <span className="annotation-preview-empty">
                                        No note yet. Click to open a larger note editor.
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
            {selectedPaper && isNoteModalOpen && (
                <div
                    className="paper-note-modal-overlay"
                    role="dialog"
                    aria-modal="true"
                    aria-label="Paper note editor"
                >
                    <div className="paper-note-modal">
                        <div className="paper-note-modal-header">
                            <h3>Paper Note</h3>
                            <button
                                type="button"
                                className="topic-search-close-button"
                                onClick={() => setIsNoteModalOpen(false)}
                            >
                                Done
                            </button>
                        </div>
                        <p className="paper-note-modal-subtitle">{selectedPaper.title}</p>
                        <textarea
                            id="annotation-modal-input"
                            className="paper-note-modal-textarea"
                            value={selectedAnnotation?.notesMarkdown || ""}
                            onChange={(event) =>
                                onUpdatePaperAnnotation(selectedPaper.title, {
                                    notesMarkdown: event.target.value,
                                })
                            }
                            placeholder="Capture paper-specific insights and how they connect to topics."
                        />
                    </div>
                </div>
            )}
        </section>
    );
}
