import React, {
    useDeferredValue,
    useEffect,
    useMemo,
    useRef,
    useState,
} from "react";
import ClusterTree from "./ClusterTree";
import PaperWorkbenchList from "./PaperWorkbenchList";
import ThemeNotebook from "./ThemeNotebook";
import ToReadInbox from "./ToReadInbox";
import ThemeAssignmentModal from "./ThemeAssignmentModal";
import { buildTopicClusters } from "../../utils/clustering";

const INITIAL_PAPER_RENDER_LIMIT = 150;
const PAPER_RENDER_BATCH_SIZE = 150;
const MAX_HIGHLIGHT_PAPERS = 600;
const SPLITTER_SIZE_PX = 8;
const MIN_LEFT_PANEL_PX = 220;
const MIN_CENTER_PANEL_PX = 320;
const MIN_RIGHT_PANEL_PX = 280;
const MIN_TO_READ_HEIGHT_PX = 150;
const DEFAULT_TO_READ_HEIGHT_PX = 220;

function filterPapers(graphData, selectedCluster, selectedTreeNode) {
    const papers = graphData?.papers || [];
    if (selectedTreeNode?.topics?.length) {
        return papers.filter((paper) =>
            (paper.topics || []).some((topic) =>
                selectedTreeNode.topics.includes(topic)
            )
        );
    }
    if (!selectedCluster) return papers;
    return papers.filter((paper) =>
        (paper.topics || []).some((topic) => selectedCluster.topics.includes(topic))
    );
}

function formatAuthorsCompact(authors) {
    if (!Array.isArray(authors) || authors.length === 0) return "Unknown authors";
    if (authors.length <= 2) return authors.join(", ");
    return `${authors.slice(0, 2).join(", ")} + ${authors.length - 2} more`;
}

async function parseResponseError(response) {
    let detail = "";
    try {
        const payload = await response.json();
        detail =
            payload?.detail ||
            payload?.error ||
            (typeof payload === "string" ? payload : "");
    } catch {
        detail = "";
    }
    if (detail) return detail;
    return `HTTP ${response.status}`;
}

export default function TopicWorkspace({
    graphData,
    workspaceStore,
    onFocusPaper,
    onSetGraphHighlight,
    onResolveReadingUrl,
    onIngestReadingItem,
    apiBase,
    apiFetch,
}) {
    const { state, actions, selectors } = workspaceStore;
    const deferredGraphData = useDeferredValue(graphData);
    const { clusters } = useMemo(
        () => buildTopicClusters(deferredGraphData),
        [deferredGraphData]
    );

    const [selectedClusterId, setSelectedClusterId] = useState(null);
    const [selectedTreeNode, setSelectedTreeNode] = useState(null);
    const [hasAutoSelectedCluster, setHasAutoSelectedCluster] = useState(false);
    const [selectedThemeId, setSelectedThemeId] = useState(null);
    const [requestedPaperTitle, setRequestedPaperTitle] = useState(null);
    const [isThemeModalOpen, setIsThemeModalOpen] = useState(false);
    const [themeModalPaperTitle, setThemeModalPaperTitle] = useState(null);
    const [paperRenderLimit, setPaperRenderLimit] = useState(
        INITIAL_PAPER_RENDER_LIMIT
    );
    const [panelWidthsPx, setPanelWidthsPx] = useState(null);
    const [toReadHeightPx, setToReadHeightPx] = useState(DEFAULT_TO_READ_HEIGHT_PX);
    const [topicSearchQuery, setTopicSearchQuery] = useState("");
    const [topicSearchResults, setTopicSearchResults] = useState([]);
    const [topicSearchState, setTopicSearchState] = useState("idle");
    const [topicSearchError, setTopicSearchError] = useState("");
    const [isResultOverlayOpen, setIsResultOverlayOpen] = useState(false);
    const workspaceRef = useRef(null);
    const topGridRef = useRef(null);
    const lastTopGridWidthRef = useRef(0);
    const topicSearchTimerRef = useRef(null);

    useEffect(() => {
        if (!hasAutoSelectedCluster && !selectedClusterId && clusters.length > 0) {
            setSelectedClusterId(clusters[0].id);
            setHasAutoSelectedCluster(true);
        }
    }, [clusters, hasAutoSelectedCluster, selectedClusterId]);

    const selectedCluster =
        clusters.find((cluster) => cluster.id === selectedClusterId) || null;
    const filteredPapers = filterPapers(graphData, selectedCluster, selectedTreeNode);
    const visiblePapers = useMemo(
        () => filteredPapers.slice(0, paperRenderLimit),
        [filteredPapers, paperRenderLimit]
    );
    const hasMorePapers = visiblePapers.length < filteredPapers.length;
    const themeQueueItems = selectedThemeId
        ? state.readingItems.filter((item) => item.linkedThemeId === selectedThemeId)
        : [];
    const hasActiveFilter = Boolean(selectedClusterId || selectedTreeNode);

    useEffect(() => {
        setPaperRenderLimit(INITIAL_PAPER_RENDER_LIMIT);
    }, [selectedClusterId, selectedTreeNode?.id]);

    useEffect(() => {
        setTopicSearchQuery("");
        setTopicSearchResults([]);
        setTopicSearchState("idle");
        setTopicSearchError("");
        setIsResultOverlayOpen(false);
    }, [selectedClusterId, selectedTreeNode?.id]);

    useEffect(() => {
        return () => {
            if (topicSearchTimerRef.current) {
                clearTimeout(topicSearchTimerRef.current);
            }
        };
    }, []);

    useEffect(() => {
        if (!requestedPaperTitle) return;
        const requestedIndex = filteredPapers.findIndex(
            (paper) => paper.title === requestedPaperTitle
        );
        if (requestedIndex >= 0 && requestedIndex + 1 > paperRenderLimit) {
            setPaperRenderLimit(
                Math.min(filteredPapers.length, requestedIndex + PAPER_RENDER_BATCH_SIZE)
            );
        }
    }, [requestedPaperTitle, filteredPapers, paperRenderLimit]);

    useEffect(() => {
        const papersForHighlight = filteredPapers.slice(0, MAX_HIGHLIGHT_PAPERS);
        const highlightNodes = selectedTreeNode
            ? [
                  ...selectedTreeNode.topics,
                  ...papersForHighlight.map((paper) => paper.title),
              ]
            : selectedCluster
              ? [
                    ...selectedCluster.topics,
                    ...papersForHighlight.map((paper) => paper.title),
                ]
              : papersForHighlight.map((paper) => paper.title);
        onSetGraphHighlight({
            nodes: Array.from(new Set(highlightNodes)),
        });
    }, [filteredPapers, onSetGraphHighlight, selectedCluster, selectedTreeNode]);

    useEffect(() => {
        const topGridEl = topGridRef.current;
        if (!topGridEl) return;
        const updateWidths = () => {
            const currentWidth = Math.max(
                0,
                topGridEl.getBoundingClientRect().width - SPLITTER_SIZE_PX * 2
            );
            if (!currentWidth) return;
            setPanelWidthsPx((previous) => {
                if (!previous) {
                    const initial = [
                        currentWidth * (1 / 3.6),
                        currentWidth * (1.4 / 3.6),
                        currentWidth * (1.2 / 3.6),
                    ];
                    lastTopGridWidthRef.current = currentWidth;
                    return initial;
                }
                const previousWidth = lastTopGridWidthRef.current || currentWidth;
                if (!previousWidth || Math.abs(previousWidth - currentWidth) < 2) {
                    lastTopGridWidthRef.current = currentWidth;
                    return previous;
                }
                const scale = currentWidth / previousWidth;
                const scaled = previous.map((width) => width * scale);
                lastTopGridWidthRef.current = currentWidth;
                return scaled;
            });
        };
        updateWidths();
        window.addEventListener("resize", updateWidths);
        return () => window.removeEventListener("resize", updateWidths);
    }, []);

    const handleHorizontalResizeStart = (dividerIndex, event) => {
        event.preventDefault();
        const topGridEl = topGridRef.current;
        if (!topGridEl) return;
        const totalWidth = Math.max(
            0,
            topGridEl.getBoundingClientRect().width - SPLITTER_SIZE_PX * 2
        );
        const baselineWidths =
            panelWidthsPx && panelWidthsPx.length === 3
                ? panelWidthsPx
                : [
                      totalWidth * (1 / 3.6),
                      totalWidth * (1.4 / 3.6),
                      totalWidth * (1.2 / 3.6),
                  ];
        const [startLeft, startCenter, startRight] = baselineWidths;
        const startX = event.clientX;

        const onMove = (moveEvent) => {
            const dx = moveEvent.clientX - startX;
            if (dividerIndex === 0) {
                const minDx = MIN_LEFT_PANEL_PX - startLeft;
                const maxDx = startCenter - MIN_CENTER_PANEL_PX;
                const clampedDx = Math.max(minDx, Math.min(maxDx, dx));
                setPanelWidthsPx([
                    startLeft + clampedDx,
                    startCenter - clampedDx,
                    startRight,
                ]);
                return;
            }
            const minDx = MIN_CENTER_PANEL_PX - startCenter;
            const maxDx = startRight - MIN_RIGHT_PANEL_PX;
            const clampedDx = Math.max(minDx, Math.min(maxDx, dx));
            setPanelWidthsPx([
                startLeft,
                startCenter + clampedDx,
                startRight - clampedDx,
            ]);
        };

        const onUp = () => {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onUp);
        };

        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
    };

    const handleVerticalResizeStart = (event) => {
        event.preventDefault();
        const workspaceEl = workspaceRef.current;
        if (!workspaceEl) return;
        const startY = event.clientY;
        const startHeight = toReadHeightPx;
        const workspaceHeight = workspaceEl.getBoundingClientRect().height;
        const maxToReadHeight = Math.max(
            MIN_TO_READ_HEIGHT_PX,
            workspaceHeight - 260
        );

        const onMove = (moveEvent) => {
            const dy = moveEvent.clientY - startY;
            const nextHeight = startHeight - dy;
            const clamped = Math.max(
                MIN_TO_READ_HEIGHT_PX,
                Math.min(maxToReadHeight, nextHeight)
            );
            setToReadHeightPx(clamped);
        };

        const onUp = () => {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onUp);
        };

        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
    };

    const topGridTemplateColumns = panelWidthsPx
        ? `${panelWidthsPx[0]}px ${SPLITTER_SIZE_PX}px ${panelWidthsPx[1]}px ${SPLITTER_SIZE_PX}px ${panelWidthsPx[2]}px`
        : "minmax(220px, 1fr) 8px minmax(360px, 1.4fr) 8px minmax(320px, 1.2fr)";

    const runTopicSearch = () => {
        const query = topicSearchQuery.trim();
        if (!query || !apiBase || !apiFetch) return;
        setTopicSearchState("loading");
        setTopicSearchError("");
        setTopicSearchResults([]);

        apiFetch(`${apiBase}/api/topic-search`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, top_k: 10 }),
        })
            .then(async (response) => {
                if (!response.ok) {
                    const detail = await parseResponseError(response);
                    throw new Error(detail);
                }
                const payload = await response.json();
                if (payload.status !== "success") {
                    throw new Error(payload.error || "search failed");
                }
                const results = Array.isArray(payload.results) ? payload.results : [];
                setTopicSearchResults(results);
                setTopicSearchState("success");
                setIsResultOverlayOpen(true);
            })
            .catch((error) => {
                const detail = String(error?.message || "").trim();
                if (detail.includes("404")) {
                    setTopicSearchError(
                        "Topic search endpoint not found. Restart backend and try again."
                    );
                } else if (detail) {
                    setTopicSearchError(`Topic search failed: ${detail}`);
                } else {
                    setTopicSearchError("Topic search failed. Please retry.");
                }
                setTopicSearchState("error");
                setIsResultOverlayOpen(true);
            });
    };

    const scheduleTopicSearch = () => {
        if (!topicSearchQuery.trim() || !apiBase || !apiFetch) return;
        setTopicSearchState("loading");
        setTopicSearchError("");
        setTopicSearchResults([]);
        setIsResultOverlayOpen(true);
        if (topicSearchTimerRef.current) {
            clearTimeout(topicSearchTimerRef.current);
        }
        topicSearchTimerRef.current = setTimeout(runTopicSearch, 300);
    };

    return (
        <div
            className="topic-workspace"
            ref={workspaceRef}
            style={{
                gridTemplateRows: `minmax(0, 1fr) ${SPLITTER_SIZE_PX}px ${toReadHeightPx}px`,
            }}
        >
            <div className="topic-workspace-main">
                <section className="workspace-panel topic-search-panel">
                    <div className="workspace-panel-header">
                        <h3>Search</h3>
                        <span>Author, title, topic, semantic intent</span>
                    </div>
                    <div className="topic-search-controls">
                        <input
                            type="text"
                            value={topicSearchQuery}
                            placeholder="Search papers, authors, or topics"
                            onChange={(event) => setTopicSearchQuery(event.target.value)}
                            onKeyDown={(event) => {
                                if (event.key === "Enter") {
                                    event.preventDefault();
                                    scheduleTopicSearch();
                                }
                            }}
                        />
                        <button
                            type="button"
                            className="topic-search-submit-button"
                            onClick={scheduleTopicSearch}
                            disabled={!topicSearchQuery.trim() || !apiBase || !apiFetch}
                        >
                            Search
                        </button>
                    </div>
                    {isResultOverlayOpen && (
                        <div className="topic-search-overlay" role="region" aria-label="Search results">
                            <div className="topic-search-overlay-header">
                                <strong>
                                    {topicSearchState === "success"
                                        ? `${topicSearchResults.length} results`
                                        : "Search status"}
                                </strong>
                                <button
                                    type="button"
                                    className="topic-search-close-button"
                                    onClick={() => setIsResultOverlayOpen(false)}
                                >
                                    Close
                                </button>
                            </div>
                            {topicSearchState === "loading" && (
                                <p className="topic-search-status">Searching topic workspace...</p>
                            )}
                            {topicSearchState === "error" && (
                                <p className="topic-search-status topic-search-error">{topicSearchError}</p>
                            )}
                            {topicSearchState === "success" && topicSearchResults.length === 0 && (
                                <p className="topic-search-status">No matching papers found.</p>
                            )}
                            {topicSearchResults.length > 0 && (
                                <div className="topic-search-results">
                                    {topicSearchResults.map((result) => (
                                        <article key={result.title} className="topic-search-result-card">
                                            <strong>{result.title}</strong>
                                            <small>
                                                {formatAuthorsCompact(result.authors)} |{" "}
                                                {result.publication_date || "Unknown year"}
                                            </small>
                                            <span>
                                                {(result.topics || []).slice(0, 3).join(" • ")}
                                            </span>
                                            <div className="topic-search-result-actions">
                                                <button
                                                    type="button"
                                                    className="topic-search-open-button"
                                                    onClick={() => {
                                                        setSelectedClusterId(null);
                                                        setSelectedTreeNode(null);
                                                        setRequestedPaperTitle(result.title);
                                                        setIsResultOverlayOpen(false);
                                                    }}
                                                >
                                                    Open paper
                                                </button>
                                            </div>
                                        </article>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
                </section>
                <div
                    className="topic-workspace-grid"
                    ref={topGridRef}
                    style={{ gridTemplateColumns: topGridTemplateColumns }}
                >
                    <ClusterTree
                        clusters={clusters}
                        selectedClusterId={selectedClusterId}
                        selectedNodeId={selectedTreeNode?.id || null}
                        onSelectCluster={(clusterId) => {
                            setSelectedClusterId(clusterId);
                            setSelectedTreeNode(null);
                        }}
                        onSelectNode={(node) => {
                            setSelectedTreeNode(node);
                        }}
                    />
                    <div
                        className="topic-resizer topic-resizer-vertical"
                        onMouseDown={(event) => handleHorizontalResizeStart(0, event)}
                        role="separator"
                        aria-orientation="vertical"
                        aria-label="Resize left and center panels"
                    />
                    <PaperWorkbenchList
                        papers={visiblePapers}
                        totalPaperCount={filteredPapers.length}
                        hasMorePapers={hasMorePapers}
                        onLoadMorePapers={() =>
                            setPaperRenderLimit((prev) =>
                                Math.min(filteredPapers.length, prev + PAPER_RENDER_BATCH_SIZE)
                            )
                        }
                        selectedTopic={
                            selectedTreeNode?.topics?.length === 1
                                ? selectedTreeNode.topics[0]
                                : null
                        }
                        selectedTopicLabel={
                            selectedTreeNode
                                ? selectedTreeNode.topics?.length === 1
                                    ? selectedTreeNode.topics[0]
                                    : `${selectedTreeNode.topics.length} selected topics`
                                : null
                        }
                        hasActiveFilter={hasActiveFilter}
                        onClearFilters={() => {
                            setSelectedClusterId(null);
                            setSelectedTreeNode(null);
                        }}
                        onFocusPaper={onFocusPaper}
                        requestedPaperTitle={requestedPaperTitle}
                        onOpenThemeAssignmentModal={(paperTitle) => {
                            setThemeModalPaperTitle(paperTitle);
                            setIsThemeModalOpen(true);
                        }}
                        getPaperAnnotation={selectors.getPaperAnnotation}
                        onUpdatePaperAnnotation={actions.upsertPaperAnnotation}
                    />
                    <div
                        className="topic-resizer topic-resizer-vertical"
                        onMouseDown={(event) => handleHorizontalResizeStart(1, event)}
                        role="separator"
                        aria-orientation="vertical"
                        aria-label="Resize center and right panels"
                    />
                    <ThemeNotebook
                        themeNotes={state.themeNotes}
                        selectedThemeId={selectedThemeId}
                        themeQueueItems={themeQueueItems}
                        onSelectTheme={setSelectedThemeId}
                        onUpsertTheme={actions.upsertThemeNote}
                        onReorderReadingItem={actions.reorderReadingItem}
                        onSelectThemePaper={(paperTitle) => {
                            setSelectedClusterId(null);
                            setSelectedTreeNode(null);
                            setRequestedPaperTitle(paperTitle);
                        }}
                    />
                </div>
            </div>
            <div
                className="topic-resizer topic-resizer-horizontal"
                onMouseDown={handleVerticalResizeStart}
                role="separator"
                aria-orientation="horizontal"
                aria-label="Resize to-read panel height"
            />
            <ToReadInbox
                readingItems={state.readingItems}
                topics={graphData?.topics || []}
                themeNotes={state.themeNotes}
                onAddReadingItem={actions.addReadingItem}
                onUpdateReadingItem={actions.updateReadingItem}
                onRemoveReadingItem={actions.removeReadingItem}
                onReorderReadingItem={actions.reorderReadingItem}
                onFocusPaper={onFocusPaper}
                onResolveReadingUrl={onResolveReadingUrl}
                onMarkReadingItemDone={async (item) => {
                    const result = await onIngestReadingItem(item);
                    if (item.linkedThemeId && result?.paper_title) {
                        actions.linkPaperToTheme(item.linkedThemeId, result.paper_title);
                    }
                    actions.removeReadingItem(item.id);
                    if (result?.paper_title) {
                        setSelectedClusterId(null);
                        setSelectedTreeNode(null);
                        setRequestedPaperTitle(result.paper_title);
                    }
                }}
            />
            {isThemeModalOpen && themeModalPaperTitle && (
                <ThemeAssignmentModal
                    paperTitle={themeModalPaperTitle}
                    themeNotes={state.themeNotes}
                    onClose={() => {
                        setIsThemeModalOpen(false);
                        setThemeModalPaperTitle(null);
                    }}
                    onSave={(themeIds) => {
                        actions.setPaperThemeMembership(themeModalPaperTitle, themeIds);
                        setIsThemeModalOpen(false);
                        setThemeModalPaperTitle(null);
                    }}
                />
            )}
        </div>
    );
}
