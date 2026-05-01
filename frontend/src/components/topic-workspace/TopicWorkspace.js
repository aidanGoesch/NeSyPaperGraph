import React, {
    forwardRef,
    useDeferredValue,
    useEffect,
    useImperativeHandle,
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
const DEFAULT_LEFT_PANEL_RATIO = 0.9;
const DEFAULT_CENTER_PANEL_RATIO = 1.5;
const DEFAULT_RIGHT_PANEL_RATIO = 1.2;
const DEFAULT_PANEL_RATIO_TOTAL =
    DEFAULT_LEFT_PANEL_RATIO +
    DEFAULT_CENTER_PANEL_RATIO +
    DEFAULT_RIGHT_PANEL_RATIO;

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

function normalizeToken(value) {
    return String(value || "").trim().toLowerCase();
}

function scoreLocalSimilarity(seedPaper, candidatePaper) {
    if (!seedPaper || !candidatePaper) return 0;
    const seedTopics = new Set((seedPaper.topics || []).map(normalizeToken).filter(Boolean));
    const candidateTopics = new Set(
        (candidatePaper.topics || []).map(normalizeToken).filter(Boolean)
    );
    let topicOverlap = 0;
    seedTopics.forEach((topic) => {
        if (candidateTopics.has(topic)) topicOverlap += 1;
    });

    const seedAuthors = new Set((seedPaper.authors || []).map(normalizeToken).filter(Boolean));
    const candidateAuthors = new Set(
        (candidatePaper.authors || []).map(normalizeToken).filter(Boolean)
    );
    let authorOverlap = 0;
    seedAuthors.forEach((author) => {
        if (candidateAuthors.has(author)) authorOverlap += 1;
    });

    const seedYear =
        Number(seedPaper.year || seedPaper.publication_date || 0) || null;
    const candidateYear =
        Number(candidatePaper.year || candidatePaper.publication_date || 0) || null;
    const yearScore =
        seedYear && candidateYear ? Math.max(0, 1 - Math.min(Math.abs(seedYear - candidateYear), 8) / 8) : 0;

    return topicOverlap * 3 + authorOverlap * 2 + yearScore;
}

function buildLocalGraphRecommendations(seedPaper, graphPapers, limit = 8) {
    if (!seedPaper || !Array.isArray(graphPapers)) return [];
    const seedTitle = normalizeToken(seedPaper.title);
    const scored = graphPapers
        .filter((paper) => normalizeToken(paper.title) && normalizeToken(paper.title) !== seedTitle)
        .map((paper) => ({
            ...paper,
            _localScore: scoreLocalSimilarity(seedPaper, paper),
        }))
        .filter((paper) => paper._localScore > 0)
        .sort((left, right) => right._localScore - left._localScore)
        .slice(0, Math.max(1, limit))
        .map((paper) => ({
            paperId: paper.paperId || null,
            title: paper.title,
            authors: paper.authors || [],
            year: paper.year || paper.publication_date || null,
            venue: paper.venue || null,
            abstract: paper.abstract || "",
            source: "graph",
            reason: "Similar to selected paper in your graph",
        }));
    return scored;
}

function interleaveRecommendations(localItems, externalItems, limit = 8) {
    const merged = [];
    const seenTitles = new Set();
    const locals = Array.isArray(localItems) ? localItems : [];
    const externals = Array.isArray(externalItems) ? externalItems : [];
    let localIdx = 0;
    let externalIdx = 0;

    while (merged.length < limit && (localIdx < locals.length || externalIdx < externals.length)) {
        if (localIdx < locals.length) {
            const nextLocal = locals[localIdx++];
            const key = normalizeToken(nextLocal?.title);
            if (key && !seenTitles.has(key)) {
                seenTitles.add(key);
                merged.push(nextLocal);
                if (merged.length >= limit) break;
            }
        }
        if (externalIdx < externals.length) {
            const nextExternal = externals[externalIdx++];
            const key = normalizeToken(nextExternal?.title);
            if (key && !seenTitles.has(key)) {
                seenTitles.add(key);
                merged.push({ ...nextExternal, source: nextExternal?.source || "semanticScholar" });
            }
        }
    }
    return merged.slice(0, limit);
}

function recommendationBrowserUrl(paper) {
    if (paper?.url) return paper.url;
    if (paper?.semanticScholarPaperId) {
        return `https://www.semanticscholar.org/paper/${paper.semanticScholarPaperId}`;
    }
    if (paper?.paperId) {
        return `https://www.semanticscholar.org/paper/${paper.paperId}`;
    }
    return "";
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

const TopicWorkspace = forwardRef(function TopicWorkspace({
    graphData,
    workspaceStore,
    onFocusPaper,
    onSetGraphHighlight,
    onResolveReadingUrl,
    onIngestReadingItem,
    apiBase,
    apiFetch,
    showSearchPanel = true,
}, ref) {
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
    const [topicRecommendations, setTopicRecommendations] = useState([]);
    const [topicRecommendationState, setTopicRecommendationState] = useState("idle");
    const [topicRecommendationError, setTopicRecommendationError] = useState("");
    const [topicActionMode, setTopicActionMode] = useState("search");
    const [isResultOverlayOpen, setIsResultOverlayOpen] = useState(false);
    const [selectedSearchResult, setSelectedSearchResult] = useState(null);
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
        setSelectedSearchResult(null);
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
                        currentWidth * (DEFAULT_LEFT_PANEL_RATIO / DEFAULT_PANEL_RATIO_TOTAL),
                        currentWidth * (DEFAULT_CENTER_PANEL_RATIO / DEFAULT_PANEL_RATIO_TOTAL),
                        currentWidth * (DEFAULT_RIGHT_PANEL_RATIO / DEFAULT_PANEL_RATIO_TOTAL),
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
                      totalWidth *
                          (DEFAULT_LEFT_PANEL_RATIO / DEFAULT_PANEL_RATIO_TOTAL),
                      totalWidth *
                          (DEFAULT_CENTER_PANEL_RATIO / DEFAULT_PANEL_RATIO_TOTAL),
                      totalWidth *
                          (DEFAULT_RIGHT_PANEL_RATIO / DEFAULT_PANEL_RATIO_TOTAL),
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
        : "minmax(220px, 0.9fr) 8px minmax(360px, 1.5fr) 8px minmax(320px, 1.2fr)";

    const runTopicSearch = (queryOverride = null) => {
        const query = (queryOverride ?? topicSearchQuery).trim();
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

    const requestPaperRecommendations = async (paper) => {
        if (!apiBase || !apiFetch) {
            throw new Error("API unavailable");
        }
        const matchedReadingItem = state.readingItems.find(
            (item) =>
                item.title === paper?.title || item.linkedPaperTitle === paper?.title
        );
        const paperId =
            paper?.semanticScholarPaperId || matchedReadingItem?.semanticScholarPaperId;
        const requestBody = JSON.stringify({
            semanticScholarPaperId: paperId || null,
            title: paper?.title || matchedReadingItem?.title || null,
            url: paper?.url || matchedReadingItem?.url || null,
            authors: paper?.authors || matchedReadingItem?.authors || [],
            year:
                paper?.year ||
                (paper?.publication_date ? Number(paper.publication_date) : null) ||
                matchedReadingItem?.year ||
                null,
            abstract: paper?.abstract || null,
            limit: 8,
        });
        let response = await apiFetch(`${apiBase}/api/recommendations/paper`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: requestBody,
        });
        if (response.status === 404) {
            response = await apiFetch(`${apiBase}/api/workspace/recommendations/paper`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: requestBody,
            });
        }
        if (!response.ok) {
            throw new Error(await parseResponseError(response));
        }
        const payload = await response.json();
        const external = Array.isArray(payload?.results) ? payload.results : [];
        const local = buildLocalGraphRecommendations(paper, graphData?.papers || [], 8);
        return interleaveRecommendations(local, external, 8);
    };

    const addRecommendationToReadingList = (paper) => {
        if (!paper || !paper.title) return;
        actions.addReadingItem({
            sourceType: paper.url ? "url" : "semantic_scholar",
            status: "inbox",
            title: paper.title,
            url: paper.url || "",
            semanticScholarPaperId: paper.paperId || null,
            authors: Array.isArray(paper.authors) ? paper.authors : [],
            year:
                typeof paper.year === "number" && Number.isFinite(paper.year)
                    ? paper.year
                    : null,
            venue: paper.venue || null,
            quickNote:
                paper.source === "graph"
                    ? "Added from graph recommendation."
                    : "Added from Semantic Scholar recommendation.",
        });
    };

    const requestThemeRecommendations = async (themeId) => {
        if (!apiBase || !apiFetch) {
            throw new Error("API unavailable");
        }
        const requestBody = JSON.stringify({
            themeId,
            limit: 8,
            candidatePoolSize: 40,
            workspaceState: state,
        });
        let response = await apiFetch(`${apiBase}/api/recommendations/theme`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: requestBody,
        });
        if (response.status === 404) {
            response = await apiFetch(`${apiBase}/api/workspace/recommendations/theme`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: requestBody,
            });
        }
        if (!response.ok) {
            throw new Error(await parseResponseError(response));
        }
        const payload = await response.json();
        return Array.isArray(payload?.results) ? payload.results : [];
    };

    const addThemeRecommendationToReadingList = (paper, themeId = null) => {
        if (!paper || !paper.title) return;
        actions.addReadingItem({
            sourceType: paper.url ? "url" : "semantic_scholar",
            status: "inbox",
            linkedThemeId: themeId || null,
            title: paper.title,
            url: paper.url || "",
            semanticScholarPaperId: paper.paperId || null,
            authors: Array.isArray(paper.authors) ? paper.authors : [],
            year:
                typeof paper.year === "number" && Number.isFinite(paper.year)
                    ? paper.year
                    : null,
            venue: paper.venue || null,
            quickNote: "Added from theme recommendation.",
        });
    };

    const runTopicRecommendations = async (queryOverride = null) => {
        if (!apiBase || !apiFetch) return;
        const query = (queryOverride ?? topicSearchQuery).trim();
        if (!query) {
            setTopicRecommendationState("error");
            setTopicRecommendationError(
                "Enter a broad topic to get recommendations (for example: causal reasoning)."
            );
            setTopicRecommendations([]);
            return;
        }
        setTopicRecommendationState("loading");
        setTopicRecommendationError("");
        setTopicRecommendations([]);
        setIsResultOverlayOpen(false);
        setSelectedSearchResult(null);
        try {
            const response = await apiFetch(`${apiBase}/api/recommendations/topic`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query,
                    limit: 10,
                    candidatePoolSize: 40,
                }),
            });
            if (!response.ok) {
                throw new Error(await parseResponseError(response));
            }
            const payload = await response.json();
            setTopicRecommendations(
                Array.isArray(payload?.results) ? payload.results : []
            );
            setTopicRecommendationState("success");
        } catch (error) {
            setTopicRecommendationError(
                `Topic recommendations failed: ${error?.message || "Unknown error"}`
            );
            setTopicRecommendationState("error");
        }
    };

    const scheduleTopicSearch = (queryOverride = null) => {
        const query = (queryOverride ?? topicSearchQuery).trim();
        if (!query || !apiBase || !apiFetch) return;
        setTopicSearchState("loading");
        setTopicSearchError("");
        setTopicSearchResults([]);
        setIsResultOverlayOpen(true);
        if (topicSearchTimerRef.current) {
            clearTimeout(topicSearchTimerRef.current);
        }
        topicSearchTimerRef.current = setTimeout(() => runTopicSearch(query), 300);
    };

    const runActiveTopicAction = () => {
        if (topicActionMode === "recommend") {
            runTopicRecommendations();
            return;
        }
        scheduleTopicSearch();
    };

    useImperativeHandle(
        ref,
        () => ({
            runTopicAction: ({ query, mode = "search" }) => {
                const normalizedQuery = String(query || "").trim();
                if (!normalizedQuery) return;
                setTopicSearchQuery(normalizedQuery);
                setTopicActionMode(mode === "recommend" ? "recommend" : "search");
                if (mode === "recommend") {
                    runTopicRecommendations(normalizedQuery);
                } else {
                    scheduleTopicSearch(normalizedQuery);
                }
            },
        }),
        [apiBase, apiFetch, topicSearchQuery]
    );

    const openSearchResultInPaperTab = (result) => {
        if (!result?.title) return;
        const existsInGraph = (graphData?.papers || []).some(
            (paper) => paper?.title === result.title
        );
        if (!existsInGraph) {
            const browserUrl = recommendationBrowserUrl(result);
            if (browserUrl) {
                window.open(browserUrl, "_blank", "noopener,noreferrer");
            }
            return;
        }
        setSelectedClusterId(null);
        setSelectedTreeNode(null);
        setRequestedPaperTitle(result.title);
        setIsResultOverlayOpen(false);
        setSelectedSearchResult(null);
    };

    const addSearchResultToReadingList = (result) => {
        if (!result?.title) return;
        const publicationYear = Number(result.publication_date || result.year || 0);
        actions.addReadingItem({
            sourceType: result.url ? "url" : "semantic_scholar",
            status: "inbox",
            title: result.title,
            url: result.url || "",
            semanticScholarPaperId:
                result.semanticScholarPaperId || result.paperId || null,
            authors: Array.isArray(result.authors) ? result.authors : [],
            year:
                Number.isFinite(publicationYear) && publicationYear > 0
                    ? publicationYear
                    : null,
            venue: result.venue || null,
            quickNote: "Added from topic search result.",
        });
    };

    const isRecommendationOverlayOpen =
        topicRecommendationState === "loading" ||
        topicRecommendationState === "error" ||
        (topicRecommendationState === "success" && topicRecommendations.length > 0);

    return (
        <div
            className="topic-workspace"
            ref={workspaceRef}
            style={{
                gridTemplateRows: `minmax(0, 1fr) ${SPLITTER_SIZE_PX}px ${toReadHeightPx}px`,
            }}
        >
            <div className="topic-workspace-main">
                <section
                    className={`topic-search-panel ${
                        showSearchPanel ? "" : "topic-search-panel-host-only"
                    }`}
                >
                    {showSearchPanel && (
                        <div className="topic-search-controls">
                            <input
                                type="text"
                                value={topicSearchQuery}
                                placeholder={
                                    topicActionMode === "recommend"
                                        ? "Recommendation topic (e.g., causal reasoning)"
                                        : "Search papers, authors, or topics"
                                }
                                onChange={(event) => setTopicSearchQuery(event.target.value)}
                                onKeyDown={(event) => {
                                    if (event.key === "Enter") {
                                        event.preventDefault();
                                        runActiveTopicAction();
                                    }
                                }}
                            />
                            <button
                                type="button"
                                className={`topic-search-submit-button topic-mode-toggle ${
                                    topicActionMode === "recommend"
                                        ? "is-recommend"
                                        : "is-search"
                                }`}
                                onClick={() =>
                                    setTopicActionMode((previous) =>
                                        previous === "search" ? "recommend" : "search"
                                    )
                                }
                                aria-label="Toggle topic action mode"
                            >
                                <span
                                    className="topic-mode-toggle-spacer"
                                    aria-hidden="true"
                                >
                                    Recommend
                                </span>
                                <span className="topic-mode-toggle-label topic-mode-toggle-search">
                                    Search
                                </span>
                                <span className="topic-mode-toggle-label topic-mode-toggle-recommend">
                                    Recommend
                                </span>
                            </button>
                        </div>
                    )}
                    {isRecommendationOverlayOpen && (
                        <div
                            className="topic-search-overlay topic-recommendation-overlay"
                            role="region"
                            aria-label="Recommendation results"
                        >
                            <div className="topic-search-overlay-header">
                                <strong>
                                    {topicRecommendationState === "success"
                                        ? `${topicRecommendations.length} recommendations`
                                        : "Recommendation status"}
                                </strong>
                                <button
                                    type="button"
                                    className="topic-search-close-button"
                                    onClick={() => {
                                        setTopicRecommendationState("idle");
                                        setTopicRecommendationError("");
                                        setTopicRecommendations([]);
                                    }}
                                >
                                    Close
                                </button>
                            </div>
                            {topicRecommendationState === "loading" && (
                                <p className="topic-search-status">Finding recommendations...</p>
                            )}
                            {topicRecommendationState === "error" && (
                                <p className="topic-search-status topic-search-error">
                                    {topicRecommendationError}
                                </p>
                            )}
                            {topicRecommendationState === "success" &&
                                topicRecommendations.length > 0 && (
                                    <div className="topic-search-results">
                                        {topicRecommendations.map((result, index) => (
                                            <article
                                                key={result.paperId || `${result.title}-${index}`}
                                                className="topic-search-result-card topic-search-result-clickable"
                                                onClick={() => setSelectedSearchResult(result)}
                                                role="button"
                                                tabIndex={0}
                                                onKeyDown={(event) => {
                                                    if (
                                                        event.key === "Enter" ||
                                                        event.key === " "
                                                    ) {
                                                        event.preventDefault();
                                                        setSelectedSearchResult(result);
                                                    }
                                                }}
                                            >
                                                <strong>{result.title || "Untitled paper"}</strong>
                                                <small>
                                                    {formatAuthorsCompact(result.authors)} |{" "}
                                                    {result.year || "Unknown year"}
                                                </small>
                                                {(result.topics || []).length > 0 && (
                                                    <span>
                                                        {(result.topics || [])
                                                            .slice(0, 3)
                                                            .join(" • ")}
                                                    </span>
                                                )}
                                                <div className="topic-search-result-actions">
                                                    <button
                                                        type="button"
                                                        className="topic-search-open-button"
                                                        onClick={(event) => {
                                                            event.stopPropagation();
                                                            openSearchResultInPaperTab(result);
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
                                    onClick={() => {
                                        setIsResultOverlayOpen(false);
                                        setSelectedSearchResult(null);
                                    }}
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
                                        <article
                                            key={result.title}
                                            className="topic-search-result-card topic-search-result-clickable"
                                            onClick={() => setSelectedSearchResult(result)}
                                            role="button"
                                            tabIndex={0}
                                            onKeyDown={(event) => {
                                                if (event.key === "Enter" || event.key === " ") {
                                                    event.preventDefault();
                                                    setSelectedSearchResult(result);
                                                }
                                            }}
                                        >
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
                                                    onClick={(event) => {
                                                        event.stopPropagation();
                                                        openSearchResultInPaperTab(result);
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
                    {selectedSearchResult && (
                        <div className="topic-search-modal-overlay" role="dialog" aria-modal="true">
                            <div className="topic-search-modal">
                                <h3>{selectedSearchResult.title || "Untitled paper"}</h3>
                                <p className="topic-search-modal-meta">
                                    {formatAuthorsCompact(selectedSearchResult.authors)} |{" "}
                                    {selectedSearchResult.publication_date ||
                                        selectedSearchResult.year ||
                                        "Unknown year"}
                                </p>
                                {(selectedSearchResult.topics || []).length > 0 && (
                                    <p className="topic-search-modal-topics">
                                        {(selectedSearchResult.topics || [])
                                            .slice(0, 5)
                                            .join(" • ")}
                                    </p>
                                )}
                                <p className="topic-search-modal-summary">
                                    {selectedSearchResult.summary ||
                                        selectedSearchResult.abstract ||
                                        "No summary available for this paper."}
                                </p>
                                <div className="topic-search-modal-actions">
                                    <button
                                        type="button"
                                        className="topic-search-open-button"
                                        onClick={() =>
                                            openSearchResultInPaperTab(selectedSearchResult)
                                        }
                                    >
                                        Open paper
                                    </button>
                                    <button
                                        type="button"
                                        className="topic-search-open-button"
                                        onClick={() =>
                                            addSearchResultToReadingList(selectedSearchResult)
                                        }
                                    >
                                        Add to reading list
                                    </button>
                                    <button
                                        type="button"
                                        className="topic-search-close-button"
                                        onClick={() => setSelectedSearchResult(null)}
                                    >
                                        Close
                                    </button>
                                </div>
                            </div>
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
                        onRequestSimilarPapers={requestPaperRecommendations}
                        onAddRecommendationToReadingList={addRecommendationToReadingList}
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
                        onRequestThemeRecommendations={requestThemeRecommendations}
                        onAddRecommendationToReadingList={
                            addThemeRecommendationToReadingList
                        }
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
});

export default TopicWorkspace;
