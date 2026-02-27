import React, { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import { marked } from "marked";
import GraphVisualization from "./GraphVisualization";
import mermaid from "mermaid";
import "./App.css";

function App() {
    const [graphData, setGraphData] = useState(null);
    const [isDarkMode, setIsDarkMode] = useState(false);
    const [searchTerm, setSearchTerm] = useState("");
    const [isSearchExpanded, setIsSearchExpanded] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadError, setUploadError] = useState(null);
    const [showMermaid, setShowMermaid] = useState(false);
    const [agentArchitectureDiagram, setAgentArchitectureDiagram] =
        useState("");
    const [showChatPanel, setShowChatPanel] = useState(false);
    const [chatHistory, setChatHistory] = useState([]);
    const [isSearching, setIsSearching] = useState(false);
    const [followUpQuestion, setFollowUpQuestion] = useState("");
    const [highlightPath, setHighlightPath] = useState(null);

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
    const searchInputRef = useRef();

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

    // Load saved graph on startup, fallback to dummy if none exists
    useEffect(() => {
        // Try to load saved graph first
        fetch("http://localhost:8000/api/graph/load")
            .then((response) => {
                if (response.ok) {
                    return response.json();
                } else {
                    // If no saved graph, try dummy
                    return fetch("http://localhost:8000/api/graph/dummy").then(
                        (r) => r.json()
                    );
                }
            })
            .then((data) => setGraphData(data))
            .catch((error) => {
                console.error("Error loading graph:", error);
                // Fallback to empty state
                setGraphData(null);
            });
    }, []);

    const showAgentArchitecture = async () => {
        try {
            const response = await fetch(
                "http://localhost:8000/api/agent/architecture"
            );
            const data = await response.json();
            if (data.mermaid) {
                setAgentArchitectureDiagram(data.mermaid);
                setShowMermaid(true);
            }
        } catch (error) {
            console.error("Error fetching architecture:", error);
        }
    };

    const handleSearch = async (query) => {
        if (!query.trim()) return;

        // Reset highlight path when asking a new question
        setHighlightPath(null);

        setIsSearching(true);
        setShowChatPanel(true);

        // Add question to chat history immediately with loading state
        const questionEntry = {
            question: query,
            answer: null, // null indicates loading
            timestamp: new Date().toLocaleTimeString(),
        };
        setChatHistory((prev) => [...prev, questionEntry]);

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
            const response = await fetch("http://localhost:8000/api/search", {
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
            if (data.mermaid && data.path && data.path.nodes) {
                setHighlightPath(data.path);
            } else {
                setHighlightPath(null);
            }

            // Update the last entry with the answer
            setChatHistory((prev) => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                    ...updated[updated.length - 1],
                    answer:
                        data.status === "search_results"
                            ? "SEARCH_RESULTS"
                            : data.answer || data.error || "No response",
                    search_results: data.search_results || null,
                    mermaid: data.mermaid || null, // Store mermaid diagram from response (null if not present)
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

                return updated;
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
                return updated;
            });
        } finally {
            setIsSearching(false);
        }
    };

    const handleFileUpload = async (event) => {
        const files = event.target.files;
        if (files.length > 0) {
            setIsUploading(true);
            setUploadError(null);

            try {
                // Clear existing graph data first
                setGraphData(null);

                // Create FormData to send files
                const formData = new FormData();
                for (let i = 0; i < files.length; i++) {
                    formData.append("files", files[i]);
                }

                // Upload files to backend
                const response = await fetch(
                    "http://localhost:8000/api/graph/upload",
                    {
                        method: "POST",
                        body: formData,
                    }
                );

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(
                        errorData.detail || "Failed to upload files"
                    );
                }

                const data = await response.json();
                // Replace the graph with new data
                setGraphData(data);
                setUploadError(null); // Clear any previous errors
                console.log(
                    "Graph replaced with",
                    data.papers.length,
                    "papers"
                );
            } catch (error) {
                console.error("Upload error:", error);
                setUploadError(
                    error.message || "Failed to upload and process files"
                );
                // Keep graph data as null on error
            } finally {
                setIsUploading(false);
                // Reset file input
                event.target.value = "";
            }
        }
    };

    return (
        <div className={`app ${isDarkMode ? "dark" : "light"}`}>
            <div className="theme-toggle">
                <button onClick={() => setIsDarkMode(!isDarkMode)}>
                    {isDarkMode ? "☀️" : "🌙"}
                </button>
            </div>
            <header className="app-header">
                <h1>Paper Graph Visualization</h1>
            </header>
            <main className="app-main">
                {isUploading ? (
                    <div className="loading">
                        Processing papers and extracting topics...
                    </div>
                ) : graphData ? (
                    <GraphVisualization
                        ref={graphRef}
                        data={graphData}
                        isDarkMode={isDarkMode}
                        onShowArchitecture={showAgentArchitecture}
                        highlightPath={highlightPath}
                    />
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
                    webkitdirectory=""
                />
                <button
                    onClick={() =>
                        document.getElementById("file-upload").click()
                    }
                    className="upload-button"
                    disabled={isUploading}
                >
                    {isUploading ? "⏳ Processing..." : "📁 Upload Papers"}
                </button>
                {uploadError && (
                    <div style={{ color: "red", marginTop: "10px" }}>
                        Error: {uploadError}
                    </div>
                )}
                {!showChatPanel && (
                    <input
                        type="text"
                        placeholder="Ask a Question..."
                        value={searchTerm}
                        onChange={(e) => {
                            setSearchTerm(e.target.value);
                        }}
                        onKeyDown={(e) => {
                            if (e.key === "Enter") {
                                handleSearch(searchTerm);
                                setSearchTerm("");
                            }
                        }}
                        onFocus={() => setIsSearchExpanded(true)}
                        onBlur={() => setIsSearchExpanded(false)}
                        onWheel={(e) => {
                            if (e.deltaY > 0 && chatHistory.length > 0) {
                                // Scrolling down
                                e.preventDefault();
                                setShowChatPanel(true);
                            }
                        }}
                        className={`search-bar unified-input ${
                            isSearchExpanded ? "expanded" : ""
                        }`}
                        ref={searchInputRef}
                    />
                )}
                {showChatPanel && (
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
                                                    {entry.mermaid && (
                                                        <div
                                                            id={`chat-mermaid-${index}`}
                                                            className="chat-mermaid-diagram"
                                                            style={{
                                                                width: "100%",
                                                                marginTop:
                                                                    "15px",
                                                                padding: "15px",
                                                                backgroundColor:
                                                                    isDarkMode
                                                                        ? "#2d2d2d"
                                                                        : "#f5f5f5",
                                                                borderRadius:
                                                                    "8px",
                                                                overflow:
                                                                    "auto",
                                                            }}
                                                        />
                                                    )}
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
                                                    {entry.mermaid && (
                                                        <div
                                                            id={`chat-mermaid-${index}`}
                                                            className="chat-mermaid-diagram"
                                                            style={{
                                                                width: "100%",
                                                                marginTop:
                                                                    "15px",
                                                                padding: "15px",
                                                                backgroundColor:
                                                                    isDarkMode
                                                                        ? "#2d2d2d"
                                                                        : "#f5f5f5",
                                                                borderRadius:
                                                                    "8px",
                                                                overflow:
                                                                    "auto",
                                                            }}
                                                        />
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
