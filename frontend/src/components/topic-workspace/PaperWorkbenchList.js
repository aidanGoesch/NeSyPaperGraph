import React, { useEffect, useMemo, useState } from "react";

function normalizeAuthor(authors) {
    if (!authors) return "Unknown";
    if (Array.isArray(authors)) return authors.join(", ");
    return String(authors);
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
}) {
    const [selectedPaperTitle, setSelectedPaperTitle] = useState(null);

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
                            </div>
                            <label className="annotation-label" htmlFor="annotation-input">
                                Paper Note
                            </label>
                            <textarea
                                id="annotation-input"
                                value={selectedAnnotation?.notesMarkdown || ""}
                                onChange={(event) =>
                                    onUpdatePaperAnnotation(selectedPaper.title, {
                                        notesMarkdown: event.target.value,
                                    })
                                }
                                placeholder="Capture paper-specific insights and how they connect to topics."
                            />
                        </>
                    ) : (
                        <p className="empty-panel-copy">
                            Select a paper to review details and add an annotation.
                        </p>
                    )}
                </div>
            </div>
        </section>
    );
}
