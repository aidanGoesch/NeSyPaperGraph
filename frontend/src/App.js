import React, { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import { marked } from "marked";
import GraphVisualization from "./GraphVisualization";
import TopicWorkspace from "./components/topic-workspace/TopicWorkspace";
import { useWorkspaceStore } from "./state/workspaceStore";
import mermaid from "mermaid";
import "./App.css";
import "./components/topic-workspace/topicWorkspace.css";

const MAX_CHAT_HISTORY = 30;
const MAX_SEARCH_RESULTS_PER_ENTRY = 8;
const MAX_RESULT_SUMMARY_CHARS = 800;
const MAX_ANSWER_CHARS = 12000;
const MAX_UPLOAD_FILES_PER_REQUEST = 25;
const MAX_UPLOAD_BATCH_BYTES = 40 * 1024 * 1024; // 40 MB per request
const MAX_UPLOAD_BATCH_RETRIES = 3;
const UPLOAD_BATCH_RETRY_BASE_MS = 1200;

function capChatHistory(entries) {
    return entries.length > MAX_CHAT_HISTORY
        ? entries.slice(entries.length - MAX_CHAT_HISTORY)
        : entries;
}

function trimSearchResult(result) {
    if (!result || typeof result !== "object") return result;
    if (result.type === "semantic_pair" && Array.isArray(result.papers)) {
        return {
            type: result.type,
            similarity: result.similarity,
            papers: result.papers.slice(0, 2).map((paper) => ({
                title: paper.title,
                abstract: (paper.abstract || "").slice(0, MAX_RESULT_SUMMARY_CHARS),
            })),
        };
    }
    return {
        type: result.type || "keyword",
        title: result.title,
        author: result.author,
        similarity: result.similarity,
        topics: Array.isArray(result.topics) ? result.topics.slice(0, 12) : [],
        summary: (result.summary || "").slice(0, MAX_RESULT_SUMMARY_CHARS),
    };
}

function buildUploadBatches(files) {
    const batches = [];
    let currentBatch = [];
    let currentBatchBytes = 0;

    for (const file of files) {
        const nextFileBytes = Math.max(0, Number(file?.size) || 0);
        const wouldExceedCount = currentBatch.length >= MAX_UPLOAD_FILES_PER_REQUEST;
        const wouldExceedBytes =
            currentBatch.length > 0 &&
            currentBatchBytes + nextFileBytes > MAX_UPLOAD_BATCH_BYTES;

        if (wouldExceedCount || wouldExceedBytes) {
            batches.push(currentBatch);
            currentBatch = [];
            currentBatchBytes = 0;
        }

        currentBatch.push(file);
        currentBatchBytes += nextFileBytes;
    }

    if (currentBatch.length > 0) {
        batches.push(currentBatch);
    }

    return batches;
}

function App() {
    const isDesktopRuntime =
        typeof window !== "undefined" && Boolean(window.desktopBridge);
    const [desktopConfig, setDesktopConfig] = useState(() => ({
        isDesktop: isDesktopRuntime,
        apiBaseUrl: isDesktopRuntime
            ? ""
            : process.env.REACT_APP_API_URL || "http://localhost:8000",
    }));
    const [runtimeConfigLoaded, setRuntimeConfigLoaded] = useState(
        !isDesktopRuntime
    );
    const API_BASE = desktopConfig.apiBaseUrl;
    const FORCE_DUMMY_DATA = process.env.REACT_APP_USE_DUMMY_DATA === "true";
    const ACCESS_KEY_STORAGE_KEY = "nesy_access_key";
    const [accessKey, setAccessKey] = useState(
        () => localStorage.getItem(ACCESS_KEY_STORAGE_KEY) || ""
    );
    const [accessKeyInput, setAccessKeyInput] = useState("");
    const [authError, setAuthError] = useState(null);
    const [isBootingBackend, setIsBootingBackend] = useState(false);
    const [backendBootMessage, setBackendBootMessage] = useState("");
    const [graphData, setGraphData] = useState(null);
    const [isDarkMode, setIsDarkMode] = useState(false);
    const [searchTerm, setSearchTerm] = useState("");
    const [isSearchExpanded, setIsSearchExpanded] = useState(false);
    const [headerTopicActionMode, setHeaderTopicActionMode] = useState("search");
    const [isUploading, setIsUploading] = useState(false);
    const [showUploadMenu, setShowUploadMenu] = useState(false);
    const [uploadError, setUploadError] = useState(null);
    const [showMermaid, setShowMermaid] = useState(false);
    const [agentArchitectureDiagram, setAgentArchitectureDiagram] =
        useState("");
    const [showChatPanel, setShowChatPanel] = useState(false);
    const [chatHistory, setChatHistory] = useState([]);
    const [isSearching, setIsSearching] = useState(false);
    const [followUpQuestion, setFollowUpQuestion] = useState("");
    const [highlightPath, setHighlightPath] = useState(null);
    const [uploadStatus, setUploadStatus] = useState(null);
    const [uploadStatusDetail, setUploadStatusDetail] = useState("");
    const [uploadProgressCurrent, setUploadProgressCurrent] = useState(0);
    const [uploadProgressTotal, setUploadProgressTotal] = useState(0);
    const [recentlyCompletedPapers, setRecentlyCompletedPapers] = useState([]);
    const [activeView, setActiveView] = useState("graph");
    const [pendingFocus, setPendingFocus] = useState(null);
    const [runtimeDiagnostics, setRuntimeDiagnostics] = useState(null);
    const [desktopSecretError, setDesktopSecretError] = useState("");
    const [openAiKeyInput, setOpenAiKeyInput] = useState("");
    const [appAccessKeyInput, setAppAccessKeyInput] = useState("");
    const [isSavingDesktopSecrets, setIsSavingDesktopSecrets] = useState(false);
    const visibleGraphData = graphData;

    // Function to handle paper citation clicks
    const handlePaperCitationClick = (paperTitle) => {
        // Hide chat panel
        setIsFadingOut(true);
        setTimeout(() => {
            setShowChatPanel(false);
            setIsFadingOut(false);
        }, 800);

        if (graphRef.current && graphData) {
            // Find the paper in the graph data
            const paper = graphData.papers.find((p) => p.title === paperTitle);
            if (paper) {
                // Use the graph's focusOnPaper method if it exists, or simulate paper click
                if (graphRef.current.focusOnPaper) {
                    graphRef.current.focusOnPaper(paperTitle);
                } else {
                    // Fallback: trigger paper selection directly
                    console.log("Opening paper info for:", paperTitle);
                }
            }
        }
    };

    // Function to process citations and markdown
    const processTextWithCitations = (text) => {
        if (!text || !graphData) return text;

        // First convert markdown to HTML
        let htmlText = marked(text);

        // Then process citations in the HTML
        const paperTitles = graphData.papers.map((p) => p.title);

        paperTitles.forEach((title) => {
            const bracketPattern = new RegExp(
                `\\[${title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\]`,
                "g"
            );
            htmlText = htmlText.replace(bracketPattern, (match) => {
                return `<span class="paper-citation" data-paper-title="${title}" style="color: #4CAF50; cursor: pointer; text-decoration: underline;">${match}</span>`;
            });
        });

        return htmlText;
    };
    const agentArchitectureMermaidRef = useRef();
    const chatContentRef = useRef();
    const [expandedResult, setExpandedResult] = useState(null);
    const chatInputRef = useRef();
    const [isFadingOut, setIsFadingOut] = useState(false);
    const graphRef = useRef();
    const topicWorkspaceRef = useRef();
    const searchInputRef = useRef();
    const uploadMenuRef = useRef();
    const eventSourceRef = useRef(null);
    const activeUploadJobsRef = useRef(new Set());
    const uploadJobStateRef = useRef(new Map());
    const pendingUploadAckCountRef = useRef(0);
    const graphLoadPromiseRef = useRef(null);
    const architectureLoadedRef = useRef(false);
    const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
    const setUploadProgress = (current, total) => {
        const safeTotal = Math.max(0, Number(total) || 0);
        const safeCurrent = Math.max(0, Number(current) || 0);
        setUploadProgressTotal(safeTotal);
        setUploadProgressCurrent(safeCurrent);
    };
    const getAggregateUploadProgress = useCallback(() => {
        let total = 0;
        let completed = 0;
        uploadJobStateRef.current.forEach((jobState) => {
            const paperTotal = Math.max(0, Number(jobState?.paperTotal) || 0);
            const paperCompleted = Math.max(
                0,
                Math.min(paperTotal, Number(jobState?.paperCompleted) || 0)
            );
            total += paperTotal;
            completed += paperCompleted;
        });
        total += Math.max(0, Number(pendingUploadAckCountRef.current) || 0);
        return { completed, total };
    }, []);
    const hasPendingUploadWork = useCallback(() => {
        return (
            activeUploadJobsRef.current.size > 0 ||
            Math.max(0, Number(pendingUploadAckCountRef.current) || 0) > 0
        );
    }, []);
    const upsertUploadJobState = useCallback(
        (jobId, patch = {}) => {
            const previous = uploadJobStateRef.current.get(jobId) || {
                paperTotal: 0,
                paperCompleted: 0,
                status: "pending",
                queuePosition: 1,
            };
            uploadJobStateRef.current.set(jobId, { ...previous, ...patch });
            const aggregate = getAggregateUploadProgress();
            setUploadProgress(aggregate.completed, aggregate.total);
            return aggregate;
        },
        [getAggregateUploadProgress]
    );
    const removeUploadJobState = useCallback(
        (jobId) => {
            uploadJobStateRef.current.delete(jobId);
            const aggregate = getAggregateUploadProgress();
            setUploadProgress(aggregate.completed, aggregate.total);
            return aggregate;
        },
        [getAggregateUploadProgress]
    );
    const uploadProgressPercent =
        uploadProgressTotal > 0
            ? Math.min(
                  100,
                  Math.max(
                      0,
                      Math.round((uploadProgressCurrent / uploadProgressTotal) * 100)
                  )
              )
            : 0;
    const formatElapsed = (startedAtSeconds) => {
        if (!startedAtSeconds) return "";
        const elapsedSeconds = Math.max(
            0,
            Math.floor(Date.now() / 1000 - startedAtSeconds)
        );
        const minutes = Math.floor(elapsedSeconds / 60);
        const seconds = elapsedSeconds % 60;
        return `${minutes}:${String(seconds).padStart(2, "0")}`;
    };

    const apiFetch = useCallback(
        async (url, options = {}) => {
            const headers = { ...(options.headers || {}) };
            if (accessKey) {
                headers["X-Access-Key"] = accessKey;
            }
            const response = await fetch(url, { ...options, headers });
            if (response.status === 401) {
                setAuthError("Invalid access key. Please try again.");
                setAccessKey("");
                localStorage.removeItem(ACCESS_KEY_STORAGE_KEY);
            }
            return response;
        },
        [accessKey]
    );
    const workspaceStore = useWorkspaceStore({
        apiBase: API_BASE,
        apiFetch,
        isEnabled: Boolean(accessKey && API_BASE),
    });

    const requiresDesktopSetup =
        desktopConfig.isDesktop && runtimeDiagnostics && !runtimeDiagnostics.openai_configured;

    const fetchRuntimeDiagnostics = useCallback(async () => {
        try {
            const response = await apiFetch(`${API_BASE}/api/runtime/diagnostics`);
            if (!response.ok) return;
            const payload = await response.json();
            setRuntimeDiagnostics(payload);
        } catch (error) {
            console.warn("Runtime diagnostics unavailable:", error);
        }
    }, [API_BASE, apiFetch]);

    const probeBackendReachable = async () => {
        try {
            // no-cors probe distinguishes "backend down" from "CORS blocked for API requests"
            await fetch(`${API_BASE}/health`, {
                method: "GET",
                mode: "no-cors",
                cache: "no-store",
            });
            return true;
        } catch {
            return false;
        }
    };

    const fetchGraph = async () => {
        if (graphLoadPromiseRef.current) {
            return graphLoadPromiseRef.current;
        }

        const loadDummyGraph = async () => {
            const fallbackResponse = await apiFetch(`${API_BASE}/api/graph/dummy`);
            if (!fallbackResponse.ok) return false;
            const fallbackData = await fallbackResponse.json();
            setGraphData(fallbackData);
            return true;
        };

        const loadPromise = (async () => {
        if (!accessKey) {
            setGraphData(null);
            return;
        }

        setIsBootingBackend(true);
        setBackendBootMessage("Waking backend...");
        setUploadError(null);

        if (FORCE_DUMMY_DATA) {
            const loadedDummy = await loadDummyGraph();
            if (loadedDummy) {
                setIsBootingBackend(false);
                return;
            }
        }

        const maxAttempts = 8;
        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            setBackendBootMessage(
                `Waking backend... (${attempt}/${maxAttempts})`
            );
            try {
                const response = await apiFetch(`${API_BASE}/api/graph/load`);
                if (response.status === 401) {
                    setIsBootingBackend(false);
                    return;
                }
                if (response.ok) {
                    const data = await response.json();
                    setGraphData(data);
                    setIsBootingBackend(false);
                    return;
                }
                if (response.status === 502 || response.status === 503) {
                    await sleep(Math.min(1000 * 2 ** (attempt - 1), 8000));
                    continue;
                }
                // If loading a persisted graph fails, fall back to dummy graph.
                if (response.status === 404 || response.status >= 500) {
                    const loadedDummy = await loadDummyGraph();
                    if (loadedDummy) {
                        setIsBootingBackend(false);
                        return;
                    }
                }
            } catch (error) {
                console.error("Error loading graph:", error);
                const backendReachable = await probeBackendReachable();
                if (backendReachable) {
                    setUploadError(
                        "Backend is reachable, but browser access is blocked. Check FRONTEND_URL CORS origin and APP_ACCESS_KEY."
                    );
                    setIsBootingBackend(false);
                    return;
                }
            }

            await sleep(Math.min(1500 * 2 ** (attempt - 1), 10000));
        }

        setUploadError("Backend is still waking up. Please refresh in a moment.");
        // Preserve any existing graph snapshot instead of blanking the UI.
        setIsBootingBackend(false);
        })();

        graphLoadPromiseRef.current = loadPromise.finally(() => {
            graphLoadPromiseRef.current = null;
        });
        return graphLoadPromiseRef.current;
    };

    const monitorUploadJob = async (jobId) => {
        const maxPolls = 240; // up to ~20 minutes at 5s max backoff
        for (let attempt = 1; attempt <= maxPolls; attempt++) {
            try {
                const response = await apiFetch(`${API_BASE}/api/jobs/${jobId}`);
                if (response.ok) {
                    const data = await response.json();
                    if (activeUploadJobsRef.current.has(jobId)) {
                        if (data.status === "processing") {
                            const completed = data.paper_index || 0;
                            const total = data.paper_total || 0;
                            upsertUploadJobState(jobId, {
                                status: "processing",
                                paperTotal: total,
                                paperCompleted: completed,
                            });
                            const aggregate = getAggregateUploadProgress();
                            setUploadStatus(
                                `processing paper ${aggregate.completed} / ${aggregate.total}`
                            );
                            const elapsed = formatElapsed(data.started_at);
                            const currentPaper = data.current_paper
                                ? `Current: ${data.current_paper}`
                                : "Current: extracting and analyzing paper content";
                            setUploadStatusDetail(
                                elapsed
                                    ? `${currentPaper} · elapsed ${elapsed}`
                                    : currentPaper
                            );
                        } else if (data.status === "pending") {
                            const total = data.paper_total || 0;
                            const queuePosition = data.queue_position || 1;
                            upsertUploadJobState(jobId, {
                                status: "pending",
                                queuePosition,
                                paperTotal: total,
                                paperCompleted: 0,
                            });
                            const aggregate = getAggregateUploadProgress();
                            setUploadStatus(
                                `queued (#${queuePosition}) - paper ${aggregate.completed} / ${aggregate.total}`
                            );
                            setUploadStatusDetail(
                                queuePosition > 1
                                    ? `Waiting in queue (${queuePosition - 1} job(s) ahead)...`
                                    : "Waiting in upload queue..."
                            );
                        }
                    }
                    if (data.status === "done") {
                        activeUploadJobsRef.current.delete(jobId);
                        removeUploadJobState(jobId);
                        await fetchGraph();
                        if (!hasPendingUploadWork()) {
                            setIsUploading(false);
                            setUploadStatus("done");
                            setUploadStatusDetail("");
                            setUploadProgress(
                                data.paper_total || uploadProgressTotal,
                                data.paper_total || uploadProgressTotal
                            );
                        }
                        return;
                    }
                    if (data.status === "error") {
                        activeUploadJobsRef.current.delete(jobId);
                        removeUploadJobState(jobId);
                        setUploadError(data.error || "Upload processing failed");
                        if (!hasPendingUploadWork()) {
                            setIsUploading(false);
                            setUploadStatus("error");
                            setUploadStatusDetail("");
                        }
                        return;
                    }
                }
            } catch (error) {
                // Keep trying; SSE may still deliver updates.
                console.warn("Upload job monitor error:", error);
            }
            await sleep(Math.min(1000 * 2 ** Math.floor(attempt / 10), 5000));
        }
    };

    // Keyboard shortcut for Cmd+G to focus search
    useEffect(() => {
        const loadDesktopConfig = async () => {
            const bridge = window.desktopBridge;
            if (!bridge?.getConfig) {
                setRuntimeConfigLoaded(true);
                return;
            }
            try {
                const config = await bridge.getConfig();
                if (config?.apiBaseUrl) {
                    setDesktopConfig({
                        isDesktop: Boolean(config.isDesktop),
                        apiBaseUrl: config.apiBaseUrl,
                    });
                }
                const storedAccessKey = await bridge.getSecret("APP_ACCESS_KEY");
                if (storedAccessKey) {
                    setAccessKey(storedAccessKey);
                    localStorage.setItem(ACCESS_KEY_STORAGE_KEY, storedAccessKey);
                }
            } catch (error) {
                console.warn("Failed to load desktop runtime config:", error);
            } finally {
                setRuntimeConfigLoaded(true);
            }
        };
        loadDesktopConfig();
    }, []);

    // Keyboard shortcut for Cmd+G to focus search
    useEffect(() => {
        const handleKeyDown = (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === "g") {
                e.preventDefault();
                if (searchInputRef.current) {
                    searchInputRef.current.focus();
                }
            }
        };

        document.addEventListener("keydown", handleKeyDown);
        return () => document.removeEventListener("keydown", handleKeyDown);
    }, []);

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (
                showUploadMenu &&
                uploadMenuRef.current &&
                !uploadMenuRef.current.contains(event.target)
            ) {
                setShowUploadMenu(false);
            }
        };

        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, [showUploadMenu]);

    useEffect(() => {
        mermaid.initialize({ startOnLoad: true });
    }, []);

    useEffect(() => {
        if (
            showMermaid &&
            agentArchitectureDiagram &&
            agentArchitectureMermaidRef.current
        ) {
            agentArchitectureMermaidRef.current.innerHTML = "";
            // Clean the mermaid code
            const cleanMermaidCode = agentArchitectureDiagram
                .replace(/<pre[^>]*>/gi, "")
                .replace(/<\/pre>/gi, "")
                .replace(/<[^>]+>/g, "")
                .trim();

            // Use mermaid.render() for dynamic content
            mermaid
                .render("mermaid-agent-architecture", cleanMermaidCode)
                .then(({ svg }) => {
                    if (agentArchitectureMermaidRef.current) {
                        agentArchitectureMermaidRef.current.innerHTML = svg;
                    }
                })
                .catch((err) => {
                    console.error("Mermaid rendering error:", err);
                    if (agentArchitectureMermaidRef.current) {
                        agentArchitectureMermaidRef.current.innerHTML = `<p style="color: red; padding: 10px;">Error rendering diagram: ${err.message}</p>`;
                    }
                });
        }
    }, [showMermaid, agentArchitectureDiagram]);

    useEffect(() => {
        // Auto-scroll to bottom when chat history updates
        if (chatContentRef.current) {
            chatContentRef.current.scrollTop =
                chatContentRef.current.scrollHeight;
        }

        // Render mermaid diagrams in chat entries
        // Use a small delay to ensure DOM elements are ready when chat panel opens
        const renderMermaidDiagrams = () => {
            if (chatContentRef.current && showChatPanel) {
                chatHistory.forEach((entry, index) => {
                    if (entry.mermaid && entry.mermaid.trim()) {
                        const mermaidContainer = document.getElementById(
                            `chat-mermaid-${index}`
                        );
                        if (mermaidContainer) {
                            // Check if already rendered by looking for SVG content
                            const hasRenderedContent =
                                mermaidContainer.querySelector("svg");
                            if (!hasRenderedContent) {
                                // Clear container
                                mermaidContainer.innerHTML = "";
                                // Clean the mermaid code (remove any HTML tags that might have been added)
                                const cleanMermaidCode = entry.mermaid
                                    .replace(/<pre[^>]*>/gi, "")
                                    .replace(/<\/pre>/gi, "")
                                    .replace(/<[^>]+>/g, "")
                                    .trim();

                                // Only render if we have valid mermaid code
                                if (cleanMermaidCode) {
                                    // Use mermaid.render() for dynamic content
                                    const mermaidId = `mermaid-chat-${index}`;
                                    mermaid
                                        .render(mermaidId, cleanMermaidCode)
                                        .then(({ svg }) => {
                                            mermaidContainer.innerHTML = svg;
                                        })
                                        .catch((err) => {
                                            console.error(
                                                "Mermaid rendering error:",
                                                err
                                            );
                                            mermaidContainer.innerHTML = `<p style="color: red; padding: 10px;">Error rendering diagram: ${err.message}</p>`;
                                        });
                                }
                            }
                        }
                    }
                });
            }
        };

        // If chat panel is showing, render diagrams (with small delay to ensure DOM is ready)
        if (showChatPanel) {
            setTimeout(renderMermaidDiagrams, 100);
        }
    }, [chatHistory, showChatPanel]);

    // Load graph when access key is available
    useEffect(() => {
        if (accessKey && API_BASE) {
            fetchGraph();
        }
    }, [accessKey, API_BASE]);

    useEffect(() => {
        if (!accessKey) return;
        fetchRuntimeDiagnostics();
    }, [accessKey, fetchRuntimeDiagnostics]);

    useEffect(() => {
        if (!accessKey || !API_BASE) {
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
                eventSourceRef.current = null;
            }
            return;
        }

        const streamUrl = `${API_BASE}/api/graph/stream?access_key=${encodeURIComponent(accessKey)}`;
        const source = new EventSource(streamUrl);
        eventSourceRef.current = source;

        const parsePayload = (event) => {
            try {
                return JSON.parse(event.data);
            } catch (error) {
                console.error("Failed to parse SSE payload:", error);
                return null;
            }
        };

        source.addEventListener("graph_snapshot", (event) => {
            const payload = parsePayload(event);
            if (payload?.graph) {
                setGraphData(payload.graph);
            }
        });

        source.addEventListener("job_queued", (event) => {
            const payload = parsePayload(event);
            if (!payload) return;
            if (activeUploadJobsRef.current.has(payload.job_id)) {
                setIsUploading(true);
                const queuePosition = payload.queue_position || 1;
                upsertUploadJobState(payload.job_id, {
                    status: "pending",
                    queuePosition,
                    paperTotal: payload.paper_total || 0,
                    paperCompleted: 0,
                });
                const aggregate = getAggregateUploadProgress();
                setUploadStatus(
                    `queued (#${queuePosition}) - paper ${aggregate.completed} / ${aggregate.total}`
                );
                setUploadStatusDetail(
                    queuePosition > 1
                        ? `Queued behind ${queuePosition - 1} job(s).`
                        : "Waiting in upload queue..."
                );
            }
        });

        source.addEventListener("job_started", (event) => {
            const payload = parsePayload(event);
            if (!payload) return;
            if (activeUploadJobsRef.current.has(payload.job_id)) {
                setIsUploading(true);
                upsertUploadJobState(payload.job_id, {
                    status: "processing",
                    paperTotal: payload.paper_total || 0,
                    paperCompleted: 0,
                });
                const aggregate = getAggregateUploadProgress();
                setUploadStatus(
                    `processing paper ${aggregate.completed} / ${aggregate.total}`
                );
                setUploadStatusDetail("Current: preparing upload batch");
            }
        });

        source.addEventListener("paper_processed", (event) => {
            const payload = parsePayload(event);
            if (!payload) return;
            if (payload.graph) {
                setGraphData(payload.graph);
            }
            if (activeUploadJobsRef.current.has(payload.job_id)) {
                upsertUploadJobState(payload.job_id, {
                    status: payload.status || "processing",
                    paperTotal: payload.paper_total || 0,
                    paperCompleted: payload.paper_index || 0,
                });
                const aggregate = getAggregateUploadProgress();
                const statusLabel =
                    payload.status === "processed"
                        ? "processed"
                        : `skipped:${payload.reason || "unknown"}`;
                setUploadStatus(
                    `${statusLabel} paper ${aggregate.completed} / ${aggregate.total}`
                );
                setUploadStatusDetail(
                    payload.paper_title
                        ? `Latest: ${payload.paper_title}`
                        : "Latest: paper update received"
                );
                if (payload.status === "processed" && payload.paper_title) {
                    setRecentlyCompletedPapers((prev) => {
                        const next = [payload.paper_title, ...prev.filter((title) => title !== payload.paper_title)];
                        return next.slice(0, 5);
                    });
                }
                setIsUploading(true);
            }
        });

        source.addEventListener("job_done", (event) => {
            const payload = parsePayload(event);
            if (!payload) return;
            if (payload.graph) {
                setGraphData(payload.graph);
            }
            if (activeUploadJobsRef.current.has(payload.job_id)) {
                activeUploadJobsRef.current.delete(payload.job_id);
                removeUploadJobState(payload.job_id);
                if (!hasPendingUploadWork()) {
                    setIsUploading(false);
                    setUploadStatus("done");
                    setUploadStatusDetail("");
                    setUploadProgress(
                        payload.paper_total || uploadProgressTotal,
                        payload.paper_total || uploadProgressTotal
                    );
                }
            }
            // Ensure graph is refreshed from backend even if stream payload is stale.
            fetchGraph();
        });

        source.addEventListener("job_error", (event) => {
            const payload = parsePayload(event);
            if (!payload) return;
            if (activeUploadJobsRef.current.has(payload.job_id)) {
                activeUploadJobsRef.current.delete(payload.job_id);
                removeUploadJobState(payload.job_id);
                setUploadError(payload.error || "Upload processing failed");
                if (!hasPendingUploadWork()) {
                    setIsUploading(false);
                    setUploadStatus("error");
                    setUploadStatusDetail("");
                }
            }
        });

        source.onerror = () => {
            if (activeUploadJobsRef.current.size > 0) {
                setUploadError(
                    "Realtime stream disconnected. Graph may lag until connection resumes."
                );
            }
        };

        return () => {
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
                eventSourceRef.current = null;
            }
        };
    }, [API_BASE, accessKey]);

    useEffect(() => {
        // Topic workspace uses "selection" highlights; clear them whenever graph view is active.
        if (activeView === "graph" && highlightPath?.mode === "selection") {
            setHighlightPath(null);
        }
    }, [activeView, highlightPath]);

    useEffect(() => {
        if (activeView !== "graph" || !pendingFocus || !graphRef.current) return;
        if (pendingFocus.type === "paper" && graphRef.current.focusOnPaper) {
            graphRef.current.focusOnPaper(pendingFocus.value);
        }
        if (pendingFocus.type === "topic" && graphRef.current.focusOnTopic) {
            graphRef.current.focusOnTopic(pendingFocus.value);
        }
        setPendingFocus(null);
    }, [activeView, pendingFocus, visibleGraphData]);

    const showAgentArchitecture = async () => {
        if (architectureLoadedRef.current && agentArchitectureDiagram) {
            setShowMermaid(true);
            return;
        }
        try {
            const response = await apiFetch(
                `${API_BASE}/api/agent/architecture`
            );
            const data = await response.json();
            if (data.mermaid) {
                setAgentArchitectureDiagram(data.mermaid);
                architectureLoadedRef.current = true;
                setShowMermaid(true);
            }
        } catch (error) {
            console.error("Error fetching architecture:", error);
        }
    };

    const resolveReadingUrlMetadata = useCallback(
        async (url) => {
            if (!accessKey || !API_BASE) {
                throw new Error("Backend connection is not ready.");
            }
            const response = await apiFetch(
                `${API_BASE}/api/workspace/resolve-paper-url`,
                {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ url }),
                }
            );
            if (!response.ok) {
                let detail = "Failed to resolve paper metadata.";
                try {
                    const payload = await response.json();
                    if (payload?.detail) {
                        detail = payload.detail;
                    }
                } catch {
                    // Keep default detail fallback.
                }
                throw new Error(detail);
            }
            return response.json();
        },
        [API_BASE, accessKey, apiFetch]
    );

    const ingestReadingItemToGraph = useCallback(
        async (item) => {
            if (!accessKey || !API_BASE) {
                throw new Error("Backend connection is not ready.");
            }
            const response = await apiFetch(`${API_BASE}/api/graph/ingest-url`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    url: item.url,
                    semanticScholarPaperId: item.semanticScholarPaperId || null,
                    title: item.title || null,
                    authors: item.authors || [],
                    year: item.year ?? null,
                    venue: item.venue || null,
                }),
            });
            let payload = null;
            try {
                payload = await response.json();
            } catch {
                payload = null;
            }
            if (!response.ok) {
                throw new Error(payload?.detail || "Failed to ingest paper into graph.");
            }
            if (payload?.graph) {
                setGraphData(payload.graph);
            } else {
                await fetchGraph();
            }
            return payload;
        },
        [API_BASE, accessKey, apiFetch, fetchGraph]
    );

    const handleAccessKeySubmit = async (event) => {
        event.preventDefault();
        const enteredKey = accessKeyInput.trim();
        if (!enteredKey) {
            setAuthError("Please enter your access key.");
            return;
        }

        setAuthError(null);
        setAccessKey(enteredKey);
        localStorage.setItem(ACCESS_KEY_STORAGE_KEY, enteredKey);
        if (desktopConfig.isDesktop && window.desktopBridge?.setSecret) {
            await window.desktopBridge.setSecret("APP_ACCESS_KEY", enteredKey);
            setAppAccessKeyInput(enteredKey);
        }
        setAccessKeyInput("");
    };

    const handleDesktopSecretsSave = async (event) => {
        event.preventDefault();
        const bridge = window.desktopBridge;
        if (!bridge?.setSecret) return;
        setDesktopSecretError("");
        setIsSavingDesktopSecrets(true);
        try {
            const trimmedOpenAi = openAiKeyInput.trim();
            if (!trimmedOpenAi) {
                setDesktopSecretError("OpenAI key is required to process PDFs.");
                return;
            }
            await bridge.setSecret("OPENAI_API_KEY", trimmedOpenAi);
            const maybeAccessKey = appAccessKeyInput.trim();
            if (maybeAccessKey) {
                await bridge.setSecret("APP_ACCESS_KEY", maybeAccessKey);
                setAccessKey(maybeAccessKey);
                localStorage.setItem(ACCESS_KEY_STORAGE_KEY, maybeAccessKey);
            }
            setOpenAiKeyInput("");
            await sleep(1200);
            await fetchRuntimeDiagnostics();
        } catch (error) {
            setDesktopSecretError(
                error?.message || "Failed to store desktop secrets."
            );
        } finally {
            setIsSavingDesktopSecrets(false);
        }
    };

    const handleSearch = async (query) => {
        if (!query.trim() || isSearching) return;

        // Check for special search syntax
        if (query.startsWith(':topic=')) {
            const topicName = query.slice(7).trim();
            if (graphRef.current && graphData) {
                // Find matching topic
                const matchingTopic = graphData.topics?.find(t => 
                    t.toLowerCase().includes(topicName.toLowerCase())
                );
                if (matchingTopic) {
                    graphRef.current.focusOnTopic(matchingTopic);
                    setSearchTerm('');
                    return;
                } else {
                    alert(`Topic not found: ${topicName}`);
                    return;
                }
            }
            return;
        }

        if (query.startsWith(':paper=') || query.startsWith(':title=')) {
            const paperTitle = query.slice(query.indexOf('=') + 1).trim();
            if (graphRef.current && graphData) {
                // Find matching paper
                const matchingPaper = graphData.papers?.find(p => 
                    p.title.toLowerCase().includes(paperTitle.toLowerCase())
                );
                if (matchingPaper) {
                    handlePaperCitationClick(matchingPaper.title);
                    setSearchTerm('');
                    return;
                } else {
                    alert(`Paper not found: ${paperTitle}`);
                    return;
                }
            }
            return;
        }

        // Regular LLM search
        // Reset highlight path when asking a new question
        // setHighlightPath(null);

        setIsSearching(true);
        setShowChatPanel(true);

        // Add question to chat history immediately with loading state
        const questionEntry = {
            question: query,
            answer: null, // null indicates loading
            timestamp: new Date().toLocaleTimeString(),
        };
        setChatHistory((prev) => capChatHistory([...prev, questionEntry]));

        // Auto-scroll to bottom after adding question
        setTimeout(() => {
            if (chatContentRef.current) {
                console.log("Auto-scrolling to bottom");
                const lastChild = chatContentRef.current.lastElementChild;
                if (lastChild) {
                    lastChild.scrollIntoView({
                        behavior: "smooth",
                        block: "end",
                    });
                }
            } else {
                console.log("chatContentRef.current is null");
            }
        }, 200);

        try {
            const response = await apiFetch(`${API_BASE}/api/search`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ query }),
            });

            const data = await response.json();
            console.log("Search response:", data);
            console.log("Search results:", data.search_results);

            // Only set highlight path if we have both path and mermaid diagram
            // If there's no mermaid, don't highlight the old path
            // if (data.mermaid && data.path && data.path.nodes) {
            //     setHighlightPath(data.path);
            // } else {
            //     setHighlightPath(null);
            // }

            // Update the last entry with the answer
            setChatHistory((prev) => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                    ...updated[updated.length - 1],
                    answer:
                        data.status === "search_results"
                            ? "SEARCH_RESULTS"
                            : (data.answer || data.error || "No response").slice(
                                  0,
                                  MAX_ANSWER_CHARS
                              ),
                    search_results: Array.isArray(data.search_results)
                        ? data.search_results
                              .slice(0, MAX_SEARCH_RESULTS_PER_ENTRY)
                              .map(trimSearchResult)
                        : null,
                    mermaid: data.mermaid || null,
                    sources_used: Array.isArray(data.sources_used)
                        ? data.sources_used.slice(0, 20)
                        : null,
                };

                // Auto-scroll after updating chat history
                setTimeout(() => {
                    if (chatContentRef.current) {
                        const lastChild =
                            chatContentRef.current.lastElementChild;
                        if (lastChild) {
                            lastChild.scrollIntoView({
                                behavior: "smooth",
                                block: "end",
                            });
                        }
                    }
                }, 50);

                return capChatHistory(updated);
            });
        } catch (error) {
            console.error("Search error:", error);
            // Update the last entry with error
            setChatHistory((prev) => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                    ...updated[updated.length - 1],
                    answer: "Error: Could not connect to server",
                };
                return capChatHistory(updated);
            });
        } finally {
            setIsSearching(false);
        }
    };

    const runHeaderTopicAction = () => {
        const query = searchTerm.trim();
        if (!query || isSearching) return;
        if (activeView === "workspace") {
            topicWorkspaceRef.current?.runTopicAction?.({
                query,
                mode: headerTopicActionMode,
            });
        } else if (headerTopicActionMode === "recommend") {
            handleSearch(`Recommend papers about: ${query}`);
        } else {
            handleSearch(query);
        }
        setSearchTerm("");
    };

    const handleFileUpload = async (event) => {
        const selectedFiles = Array.from(event.target.files || []);
        const files = selectedFiles.filter(
            (file) =>
                file?.name &&
                !file.name.startsWith(".") &&
                file.name.toLowerCase().endsWith(".pdf")
        );
        if (files.length > 0) {
            const hadPendingJobs = activeUploadJobsRef.current.size > 0 || isUploading;
            pendingUploadAckCountRef.current += files.length;
            const aggregateBeforeUpload = getAggregateUploadProgress();
            setIsUploading(true);
            setUploadError(null);
            if (!hadPendingJobs) {
                setUploadStatus(
                    `paper ${aggregateBeforeUpload.completed} / ${aggregateBeforeUpload.total}`
                );
                setUploadProgress(
                    aggregateBeforeUpload.completed,
                    aggregateBeforeUpload.total
                );
                setUploadStatusDetail("Uploading files to local backend...");
            } else {
                const baseDisplayedTotal = Math.max(
                    0,
                    Number(uploadProgressTotal) || 0
                );
                const baseDisplayedCurrent = Math.max(
                    0,
                    Number(uploadProgressCurrent) || 0
                );
                const baseTotal = Math.max(
                    aggregateBeforeUpload.total,
                    baseDisplayedTotal
                );
                const baseCurrent = Math.max(
                    aggregateBeforeUpload.completed,
                    Math.min(baseDisplayedCurrent, baseTotal)
                );
                const optimisticTotal = baseTotal + files.length;
                setUploadStatus(
                    `paper ${baseCurrent} / ${optimisticTotal}`
                );
                setUploadProgress(baseCurrent, optimisticTotal);
                setUploadStatusDetail(
                    `Appending ${files.length} paper(s) to existing upload queue...`
                );
            }

            let pendingForThisSelection = files.length;
            try {
                const fileBatches = buildUploadBatches(files);
                let failedBatchCount = 0;
                let failedFileCount = 0;
                let firstBatchErrorMessage = "";

                for (let batchIndex = 0; batchIndex < fileBatches.length; batchIndex++) {
                    const batch = fileBatches[batchIndex];
                    const formData = new FormData();
                    for (let i = 0; i < batch.length; i++) {
                        formData.append("files", batch[i]);
                    }

                    setUploadStatusDetail(
                        `Uploading batch ${batchIndex + 1} of ${fileBatches.length} (${batch.length} file(s))...`
                    );
                    let accepted = false;
                    let lastBatchError = null;
                    for (
                        let attempt = 1;
                        attempt <= MAX_UPLOAD_BATCH_RETRIES;
                        attempt++
                    ) {
                        try {
                            const response = await apiFetch(
                                `${API_BASE}/api/graph/upload`,
                                {
                                    method: "POST",
                                    body: formData,
                                }
                            );

                            if (!response.ok) {
                                const errorData = await response.json();
                                const message =
                                    errorData.detail || "Failed to upload files";
                                const shouldRetry =
                                    response.status === 429 ||
                                    response.status >= 500;
                                if (
                                    shouldRetry &&
                                    attempt < MAX_UPLOAD_BATCH_RETRIES
                                ) {
                                    await sleep(
                                        UPLOAD_BATCH_RETRY_BASE_MS * attempt
                                    );
                                    continue;
                                }
                                throw new Error(message);
                            }

                            const data = await response.json();
                            if (!data.job_id) {
                                throw new Error("Upload did not return a job_id");
                            }
                            activeUploadJobsRef.current.add(data.job_id);
                            const queuePosition =
                                data.queue_position ||
                                activeUploadJobsRef.current.size;
                            const paperTotal = data.paper_total || batch.length;
                            pendingUploadAckCountRef.current = Math.max(
                                0,
                                pendingUploadAckCountRef.current - paperTotal
                            );
                            pendingForThisSelection = Math.max(
                                0,
                                pendingForThisSelection - paperTotal
                            );
                            upsertUploadJobState(data.job_id, {
                                status: "pending",
                                queuePosition,
                                paperTotal,
                                paperCompleted: 0,
                            });
                            const aggregate = getAggregateUploadProgress();
                            setUploadStatus(
                                `queued (#${queuePosition}) - paper ${aggregate.completed} / ${aggregate.total}`
                            );
                            setUploadStatusDetail(
                                queuePosition > 1
                                    ? `Upload appended to queue (${queuePosition - 1} job(s) ahead).`
                                    : "Upload accepted. Waiting for processing worker..."
                            );
                            setUploadError(null);
                            monitorUploadJob(data.job_id);
                            accepted = true;
                            break;
                        } catch (batchError) {
                            lastBatchError = batchError;
                            const message =
                                batchError?.message || "Failed to upload files";
                            const looksTransient =
                                message.includes("Failed to fetch") ||
                                message.includes("NetworkError") ||
                                message.includes("network");
                            if (
                                looksTransient &&
                                attempt < MAX_UPLOAD_BATCH_RETRIES
                            ) {
                                await sleep(UPLOAD_BATCH_RETRY_BASE_MS * attempt);
                                continue;
                            }
                            break;
                        }
                    }

                    if (!accepted) {
                        failedBatchCount += 1;
                        failedFileCount += batch.length;
                        pendingUploadAckCountRef.current = Math.max(
                            0,
                            pendingUploadAckCountRef.current - batch.length
                        );
                        pendingForThisSelection = Math.max(
                            0,
                            pendingForThisSelection - batch.length
                        );
                        if (!firstBatchErrorMessage) {
                            firstBatchErrorMessage =
                                lastBatchError?.message || "batch upload failed";
                        }
                        const aggregate = getAggregateUploadProgress();
                        setUploadProgress(aggregate.completed, aggregate.total);
                    }
                }

                if (failedBatchCount > 0) {
                    const suffix = firstBatchErrorMessage
                        ? ` Last error: ${firstBatchErrorMessage}`
                        : "";
                    setUploadError(
                        `Some upload batches failed (${failedFileCount} file(s) across ${failedBatchCount} batch(es)).${suffix}`
                    );
                }
            } catch (error) {
                console.error("Upload error:", error);
                pendingUploadAckCountRef.current = Math.max(
                    0,
                    pendingUploadAckCountRef.current - pendingForThisSelection
                );
                const aggregate = getAggregateUploadProgress();
                setUploadError(
                    error.message || "Failed to upload and process files"
                );
                if (activeUploadJobsRef.current.size === 0) {
                    setUploadStatus("error");
                    setIsUploading(false);
                } else {
                    setUploadStatus(
                        `processing paper ${aggregate.completed} / ${aggregate.total}`
                    );
                }
                setUploadProgress(aggregate.completed, aggregate.total);
                // Keep graph data as null on error
            } finally {
                // Reset file input
                event.target.value = "";
            }
        } else {
            setUploadError(
                "No PDF files found in the selected files/folder."
            );
            event.target.value = "";
        }
    };

    return (
        <div className={`app ${isDarkMode ? "dark" : "light"}`}>
            {!runtimeConfigLoaded && (
                <div className="auth-overlay">
                    <div className="auth-card">
                        <h2>Starting desktop runtime...</h2>
                        <p>Connecting to local backend.</p>
                    </div>
                </div>
            )}
            {runtimeConfigLoaded && !accessKey && (
                <div className="auth-overlay">
                    <form className="auth-card" onSubmit={handleAccessKeySubmit}>
                        <h2>Private Access</h2>
                        <p>Enter your access key to use NeSyPaperGraph.</p>
                        <input
                            type="password"
                            value={accessKeyInput}
                            onChange={(e) => setAccessKeyInput(e.target.value)}
                            placeholder="Access key"
                            autoFocus
                        />
                        {authError && <div className="auth-error">{authError}</div>}
                        <button type="submit">Unlock</button>
                    </form>
                </div>
            )}
            {runtimeConfigLoaded && requiresDesktopSetup && (
                <div className="auth-overlay">
                    <form className="auth-card" onSubmit={handleDesktopSecretsSave}>
                        <h2>Desktop Setup Required</h2>
                        <p>
                            Add your OpenAI API key to continue. Keys are stored in
                            macOS Keychain.
                        </p>
                        <input
                            type="password"
                            value={openAiKeyInput}
                            onChange={(e) => setOpenAiKeyInput(e.target.value)}
                            placeholder="OpenAI API key"
                            autoFocus
                        />
                        <input
                            type="password"
                            value={appAccessKeyInput}
                            onChange={(e) => setAppAccessKeyInput(e.target.value)}
                            placeholder="App access key (optional)"
                        />
                        {desktopSecretError && (
                            <div className="auth-error">{desktopSecretError}</div>
                        )}
                        <button type="submit" disabled={isSavingDesktopSecrets}>
                            {isSavingDesktopSecrets ? "Saving..." : "Save and Restart Backend"}
                        </button>
                    </form>
                </div>
            )}
            <header className="app-header">
                <div className="app-title-label">Paper Graph</div>
                <div className="header-search">
                    <div className="topic-search-controls header-topic-search-controls">
                        <input
                            type="text"
                            className="header-topic-search-input"
                            placeholder={
                                headerTopicActionMode === "recommend"
                                    ? "Recommendation topic (e.g., causal reasoning)"
                                    : "Search papers, authors, or topics"
                            }
                            value={searchTerm}
                            onChange={(e) => {
                                setSearchTerm(e.target.value);
                            }}
                            onKeyDown={(e) => {
                                if (e.key === "Enter") {
                                    e.preventDefault();
                                    runHeaderTopicAction();
                                }
                            }}
                            ref={searchInputRef}
                        />
                        <button
                            type="button"
                            className={`topic-search-submit-button topic-mode-toggle ${
                                headerTopicActionMode === "recommend"
                                    ? "is-recommend"
                                    : "is-search"
                            }`}
                            onClick={() =>
                                setHeaderTopicActionMode((previous) =>
                                    previous === "search" ? "recommend" : "search"
                                )
                            }
                            aria-label="Toggle header action mode"
                        >
                            <span className="topic-mode-toggle-spacer" aria-hidden="true">
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
                </div>
                <div
                    className={`header-view-toggle ${
                        activeView === "workspace" ? "is-workspace" : "is-graph"
                    }`}
                >
                    <button
                        type="button"
                        className={activeView === "graph" ? "active" : ""}
                        onClick={() => {
                            // Clear workspace selection highlight when returning to graph view.
                            setHighlightPath(null);
                            setActiveView("graph");
                        }}
                    >
                        Graph View
                    </button>
                    <button
                        type="button"
                        className={activeView === "workspace" ? "active" : ""}
                        onClick={() => setActiveView("workspace")}
                    >
                        Topic Workspace
                    </button>
                </div>
                <div className="upload-menu" ref={uploadMenuRef}>
                    <button
                        type="button"
                        className="upload-button"
                        onClick={() => setShowUploadMenu((previous) => !previous)}
                        disabled={!accessKey}
                    >
                        <span className="upload-button-icon" aria-hidden="true">
                            ⬆
                        </span>
                        Upload
                    </button>
                    {showUploadMenu && accessKey && (
                        <div className="upload-dropdown">
                            <button
                                type="button"
                                className="upload-dropdown-item"
                                onClick={() => {
                                    document.getElementById("file-upload").click();
                                    setShowUploadMenu(false);
                                }}
                            >
                                {isUploading ? "Upload More Papers" : "Upload Papers"}
                            </button>
                            <button
                                type="button"
                                className="upload-dropdown-item"
                                onClick={() => {
                                    document.getElementById("folder-upload").click();
                                    setShowUploadMenu(false);
                                }}
                            >
                                {isUploading ? "Add Folder" : "Upload Folder"}
                            </button>
                        </div>
                    )}
                </div>
                <div className="theme-toggle">
                    <button
                        type="button"
                        onClick={() => setIsDarkMode(!isDarkMode)}
                        aria-label={
                            isDarkMode ? "Switch to light mode" : "Switch to dark mode"
                        }
                    >
                        <span aria-hidden="true">{isDarkMode ? "☀️" : "🌙"}</span>
                    </button>
                </div>
            </header>
            <main className="app-main">
                {isBootingBackend ? (
                    <div className="skeleton-wrapper">
                        <div className="skeleton-title" />
                        <div className="skeleton-graph" />
                        <div className="skeleton-row">
                            <div className="skeleton-pill" />
                            <div className="skeleton-pill" />
                            <div className="skeleton-pill" />
                        </div>
                        <div className="skeleton-status">{backendBootMessage}</div>
                    </div>
                ) : visibleGraphData ? (
                    <div className="view-stack">
                        <div
                            className={`view-panel ${
                                activeView === "graph"
                                    ? "view-panel-active"
                                    : "view-panel-hidden"
                            }`}
                        >
                            <GraphVisualization
                                ref={graphRef}
                                data={visibleGraphData}
                                isDarkMode={isDarkMode}
                                onShowArchitecture={showAgentArchitecture}
                                highlightPath={highlightPath}
                                apiBase={API_BASE}
                                apiFetch={apiFetch}
                                onAddRecommendationToReadingList={(paper) => {
                                    if (!paper?.title) return;
                                    workspaceStore.actions.addReadingItem({
                                        sourceType: paper.url ? "url" : "semantic_scholar",
                                        status: "inbox",
                                        title: paper.title,
                                        url: paper.url || "",
                                        semanticScholarPaperId: paper.paperId || null,
                                        authors: Array.isArray(paper.authors)
                                            ? paper.authors
                                            : [],
                                        year:
                                            typeof paper.year === "number" &&
                                            Number.isFinite(paper.year)
                                                ? paper.year
                                                : null,
                                        venue: paper.venue || null,
                                        quickNote:
                                            paper.source === "graph"
                                                ? "Added from graph recommendation."
                                                : "Added from Semantic Scholar recommendation.",
                                    });
                                }}
                            />
                        </div>
                        {activeView === "workspace" && (
                            <div className="view-panel view-panel-active">
                                <TopicWorkspace
                                    ref={topicWorkspaceRef}
                                    graphData={visibleGraphData}
                                    workspaceStore={workspaceStore}
                                    apiBase={API_BASE}
                                    apiFetch={apiFetch}
                                    showSearchPanel={false}
                                    onResolveReadingUrl={resolveReadingUrlMetadata}
                                    onIngestReadingItem={ingestReadingItemToGraph}
                                    onFocusPaper={(paperTitle) => {
                                        setHighlightPath(null);
                                        setActiveView("graph");
                                        setPendingFocus({
                                            type: "paper",
                                            value: paperTitle,
                                        });
                                    }}
                                    onSetGraphHighlight={(pathPayload) =>
                                        setHighlightPath({
                                            mode: "selection",
                                            nodes: pathPayload?.nodes || [],
                                        })
                                    }
                                />
                            </div>
                        )}
                    </div>
                ) : (
                    <div className="loading">
                        Upload papers to visualize the graph
                    </div>
                )}
                <input
                    type="file"
                    multiple
                    accept=".pdf"
                    onChange={handleFileUpload}
                    style={{ display: "none" }}
                    id="file-upload"
                />
                <input
                    type="file"
                    multiple
                    webkitdirectory=""
                    directory=""
                    mozdirectory=""
                    onChange={handleFileUpload}
                    style={{ display: "none" }}
                    id="folder-upload"
                />
                {isUploading && (
                    <div
                        style={{
                            position: "fixed",
                            top: "72px",
                            left: "20px",
                            zIndex: 1500,
                            background: isDarkMode ? "#2b2b2b" : "white",
                            color: isDarkMode ? "#f5f5f5" : "#222",
                            border: isDarkMode
                                ? "1px solid #4a4a4a"
                                : "1px solid #d9d9d9",
                            borderRadius: "16px",
                            padding: "10px 14px",
                            boxShadow: "0 6px 16px rgba(0,0,0,0.18)",
                            maxWidth: "320px",
                        }}
                    >
                        <div style={{ fontWeight: 600 }}>
                            Processing papers
                        </div>
                        {(uploadStatusDetail || uploadStatus) && (
                            <div
                                style={{
                                    marginTop: "6px",
                                    fontSize: "12px",
                                    opacity: 0.9,
                                }}
                            >
                                {uploadStatusDetail || uploadStatus}
                                {uploadStatusDetail &&
                                    uploadStatus &&
                                    uploadStatus !== "done" &&
                                    uploadStatus !== "error" &&
                                    ` · ${uploadStatus}`}
                            </div>
                        )}
                        {(uploadProgressTotal > 0 || isUploading) && (
                            <div
                                style={{
                                    marginTop: "8px",
                                    height: "8px",
                                    width: "100%",
                                    background: isDarkMode ? "#3a3a3a" : "#ececec",
                                    borderRadius: "999px",
                                    overflow: "hidden",
                                }}
                            >
                                <div
                                    style={{
                                        height: "100%",
                                        width: `${uploadProgressPercent}%`,
                                        background: "#4CAF50",
                                        transition: "width 240ms ease",
                                    }}
                                />
                            </div>
                        )}
                        {recentlyCompletedPapers.length > 0 && (
                            <div
                                style={{
                                    marginTop: "6px",
                                    fontSize: "12px",
                                    opacity: 0.9,
                                }}
                            >
                                Done: {recentlyCompletedPapers.join(" • ")}
                            </div>
                        )}
                    </div>
                )}
                {uploadError && (
                    <div style={{ color: "red", marginTop: "10px" }}>
                        Error: {uploadError}
                    </div>
                )}
                {workspaceStore.syncWarning && (
                    <div
                        style={{
                            marginTop: "10px",
                            color: "#b65c00",
                            background: "rgba(255, 193, 7, 0.12)",
                            border: "1px solid rgba(182, 92, 0, 0.45)",
                            borderRadius: "8px",
                            padding: "8px 10px",
                        }}
                    >
                        {workspaceStore.syncWarning}
                    </div>
                )}
                {activeView === "graph" && !showChatPanel && (
                    <input
                        type="text"
                        placeholder="Ask a Question... (or :topic=name, :paper=title)"
                        value={searchTerm}
                        onChange={(e) => {
                            setSearchTerm(e.target.value);
                        }}
                        onKeyDown={(e) => {
                            if (e.key === "Enter") {
                                runHeaderTopicAction();
                            }
                        }}
                        onFocus={() => setIsSearchExpanded(true)}
                        onBlur={() => setIsSearchExpanded(false)}
                        onWheel={(e) => {
                            if (e.deltaY > 0 && chatHistory.length > 0) {
                                e.preventDefault();
                                setShowChatPanel(true);
                            }
                        }}
                        className={`search-bar unified-input ${
                            isSearchExpanded ? "expanded" : ""
                        }`}
                    />
                )}
                {activeView === "graph" && showChatPanel && (
                    <div
                        className={`chat-panel-wrapper ${
                            isDarkMode ? "dark" : "light"
                        } ${isFadingOut ? "fading-out" : ""}`}
                    >
                        <div
                            className={`chat-panel ${
                                isDarkMode ? "dark" : "light"
                            }`}
                            onWheel={(e) => {
                                const panel = e.currentTarget;
                                const scrollTop = panel.scrollTop;
                                const isScrollingUp = e.deltaY < 0;

                                console.log(
                                    "React onWheel - scrollTop:",
                                    scrollTop,
                                    "isScrollingUp:",
                                    isScrollingUp
                                );

                                // Only close if we're at the very top AND scrolling up
                                if (scrollTop === 0 && isScrollingUp) {
                                    console.log(
                                        "Closing chat panel from React handler"
                                    );
                                    e.preventDefault();
                                    setIsFadingOut(true);
                                    setTimeout(() => {
                                        setShowChatPanel(false);
                                        setIsFadingOut(false);
                                    }, 800);
                                }
                            }}
                        >
                            {/* X CLOSE BUTTON — now in top right corner */}
                            <button
                                onClick={() => {
                                    setShowChatPanel(false);
                                    setTimeout(() => {
                                        setSearchTerm("");
                                        setIsSearchExpanded(false);
                                    }, 300);
                                }}
                                className="close-button"
                            >
                                ×
                            </button>

                            {/* Scrollable content */}
                            <div className="chat-content" ref={chatContentRef}>
                                {chatHistory.map((entry, index) => (
                                    <div key={index} className="chat-entry">
                                        <div className="question">
                                            <strong>Q:</strong> {entry.question}
                                            <span className="timestamp">
                                                {entry.timestamp}
                                            </span>
                                        </div>
                                        <div className="answer">
                                            <strong>A:</strong>
                                            {entry.answer === null ? (
                                                <div className="loading-dots">
                                                    <span></span>
                                                    <span></span>
                                                    <span></span>
                                                </div>
                                            ) : entry.answer ===
                                                  "SEARCH_RESULTS" &&
                                              entry.search_results ? (
                                                <>
                                                    <div className="search-results">
                                                        {entry.search_results.map(
                                                            (result, idx) => 
                                                                result.type === "semantic_pair" ? (
                                                                    // Semantic pair container
                                                                    <div key={idx} className="semantic-pair-container">
                                                                        <div className="similarity-header">
                                                                            Similarity: {(result.similarity * 100).toFixed(0)}%
                                                                        </div>
                                                                        <div className="paper-pair">
                                                                            {result.papers.map((paper, pIdx) => (
                                                                                <div
                                                                                    key={pIdx}
                                                                                    className="search-result-block"
                                                                                    onClick={() => handlePaperCitationClick(paper.title)}
                                                                                >
                                                                                    <h4>{paper.title}</h4>
                                                                                    {paper.abstract && (
                                                                                        <p className="abstract-preview">
                                                                                            {paper.abstract.substring(0, 150)}...
                                                                                        </p>
                                                                                    )}
                                                                                </div>
                                                                            ))}
                                                                        </div>
                                                                    </div>
                                                                ) : (
                                                                    // Regular keyword search result
                                                                    <div
                                                                        key={idx}
                                                                        className={`search-result-block ${
                                                                            expandedResult ===
                                                                            idx
                                                                                ? "expanded"
                                                                                : ""
                                                                        }`}
                                                                        onClick={() =>
                                                                            setExpandedResult(
                                                                                expandedResult ===
                                                                                    idx
                                                                                    ? null
                                                                                    : idx
                                                                            )
                                                                        }
                                                                    >
                                                                        <h4>
                                                                            {result.title}
                                                                            {result.similarity && (
                                                                                <span className="similarity-badge">
                                                                                    {(result.similarity * 100).toFixed(0)}%
                                                                                </span>
                                                                            )}
                                                                        </h4>
                                                                    <p className="author">
                                                                        {
                                                                            result.author
                                                                        }
                                                                    </p>
                                                                    <div className="topics">
                                                                        {result.topics.map(
                                                                            (
                                                                                topic,
                                                                                topicIdx
                                                                            ) => (
                                                                                <span
                                                                                    key={
                                                                                        topicIdx
                                                                                    }
                                                                                    className="topic-tag"
                                                                                    onClick={(
                                                                                        e
                                                                                    ) => {
                                                                                        e.stopPropagation();
                                                                                        setIsFadingOut(
                                                                                            true
                                                                                        );
                                                                                        setTimeout(
                                                                                            () => {
                                                                                                setShowChatPanel(
                                                                                                    false
                                                                                                );
                                                                                                setIsFadingOut(
                                                                                                    false
                                                                                                );
                                                                                                // Focus on topic in graph
                                                                                                if (
                                                                                                    graphRef.current
                                                                                                ) {
                                                                                                    graphRef.current.focusOnTopic(
                                                                                                        topic
                                                                                                    );
                                                                                                }
                                                                                            },
                                                                                            800
                                                                                        );
                                                                                    }}
                                                                                >
                                                                                    {
                                                                                        topic
                                                                                    }
                                                                                </span>
                                                                            )
                                                                        )}
                                                                    </div>
                                                                    {expandedResult ===
                                                                        idx && (
                                                                        <div className="summary">
                                                                            <ReactMarkdown>
                                                                                {
                                                                                    result.summary
                                                                                }
                                                                            </ReactMarkdown>
                                                                        </div>
                                                                    )}
                                                                </div>
                                                            )
                                                        )}
                                                    </div>
                                                    {/* {entry.mermaid && (
                                                        <div
                                                            style={{
                                                                width: "90%",
                                                                marginTop: "15px",
                                                                marginLeft: "auto",
                                                                marginRight: "auto",
                                                                display: "flex",
                                                                justifyContent: "center"
                                                            }}
                                                        >
                                                            <div
                                                                style={{
                                                                    padding: "15px",
                                                                    backgroundColor:
                                                                        isDarkMode
                                                                            ? "#2d2d2d"
                                                                            : "#f5f5f5",
                                                                    borderRadius: "8px",
                                                                    display: "inline-block"
                                                                }}
                                                            >
                                                                <div 
                                                                    id={`chat-mermaid-${index}`}
                                                                    className="chat-mermaid-diagram"
                                                                    style={{
                                                                        transform: "scale(0.5)"
                                                                    }}
                                                                ></div>
                                                            </div>
                                                        </div>
                                                    )} */}
                                                </>
                                            ) : (
                                                <>
                                                    <div
                                                        className="markdown-content"
                                                        onClick={(e) => {
                                                            if (
                                                                e.target.classList.contains(
                                                                    "paper-citation"
                                                                )
                                                            ) {
                                                                const paperTitle =
                                                                    e.target.getAttribute(
                                                                        "data-paper-title"
                                                                    );
                                                                handlePaperCitationClick(
                                                                    paperTitle
                                                                );
                                                            }
                                                        }}
                                                        dangerouslySetInnerHTML={{
                                                            __html: processTextWithCitations(
                                                                entry.answer
                                                            ),
                                                        }}
                                                    />
                                                    {/* {entry.mermaid && (
                                                        <div
                                                            style={{
                                                                width: "90%",
                                                                marginTop: "15px",
                                                                marginLeft: "auto",
                                                                marginRight: "auto",
                                                                display: "flex",
                                                                justifyContent: "center"
                                                            }}
                                                        >
                                                            <div
                                                                style={{
                                                                    padding: "15px",
                                                                    backgroundColor:
                                                                        isDarkMode
                                                                            ? "#2d2d2d"
                                                                            : "#f5f5f5",
                                                                    borderRadius: "8px",
                                                                    display: "inline-block"
                                                                }}
                                                            >
                                                                <div 
                                                                    id={`chat-mermaid-${index}`}
                                                                    className="chat-mermaid-diagram"
                                                                    style={{
                                                                        transform: "scale(0.5)"
                                                                    }}
                                                                ></div>
                                                            </div>
                                                        </div>
                                                    )} */}
                                                    {entry.sources_used && entry.sources_used.length > 0 && (
                                                        <div className="sources-section" style={{
                                                            marginTop: "15px",
                                                            paddingTop: "15px",
                                                            borderTop: isDarkMode ? "1px solid #444" : "1px solid #ddd"
                                                        }}>
                                                            <strong>Sources:</strong>
                                                            <div style={{ marginTop: "8px" }}>
                                                                {entry.sources_used.map((source, sIdx) => (
                                                                    <div
                                                                        key={sIdx}
                                                                        className="source-item"
                                                                        onClick={() => handlePaperCitationClick(source)}
                                                                        style={{
                                                                            color: "#4CAF50",
                                                                            cursor: "pointer",
                                                                            textDecoration: "underline",
                                                                            marginBottom: "4px"
                                                                        }}
                                                                    >
                                                                        {source}
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    )}
                                                </>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="chat-input-area">
                            <input
                                type="text"
                                placeholder="Ask a follow-up question..."
                                value={followUpQuestion}
                                onChange={(e) =>
                                    setFollowUpQuestion(e.target.value)
                                }
                                onKeyDown={(e) => {
                                    if (
                                        e.key === "Enter" &&
                                        !isSearching &&
                                        followUpQuestion.trim()
                                    ) {
                                        e.preventDefault();
                                        console.log(
                                            "Follow-up question:",
                                            followUpQuestion
                                        );
                                        handleSearch(followUpQuestion);
                                        setFollowUpQuestion("");
                                        // Focus back to this input, not chatInputRef
                                        e.target.focus();
                                    }
                                }}
                                ref={chatInputRef}
                                className={`chat-input unified-input ${
                                    isDarkMode ? "dark" : "light"
                                }`}
                            />
                        </div>
                    </div>
                )}
                {showMermaid && (
                    <div
                        style={{
                            position: "fixed",
                            top: "50%",
                            left: "50%",
                            transform: "translate(-50%, -50%)",
                            backgroundColor: "white",
                            padding: "20px",
                            borderRadius: "8px",
                            boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
                            zIndex: 2000,
                            maxWidth: "80vw",
                            maxHeight: "80vh",
                            overflow: "auto",
                        }}
                    >
                        <button
                            onClick={() => setShowMermaid(false)}
                            style={{
                                position: "absolute",
                                top: "10px",
                                right: "10px",
                                background: "none",
                                border: "none",
                                fontSize: "24px",
                                cursor: "pointer",
                            }}
                        >
                            ×
                        </button>
                        <h3>Agent Architecture</h3>
                        <div ref={agentArchitectureMermaidRef}></div>
                    </div>
                )}
            </main>
        </div>
    );
}

export default App;
