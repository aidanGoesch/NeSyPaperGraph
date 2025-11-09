import React, { useState, useEffect } from "react";
import GraphVisualization from "./GraphVisualization";
import "./App.css";

function App() {
    const [graphData, setGraphData] = useState(null);
    const [isDarkMode, setIsDarkMode] = useState(false);
    const [searchTerm, setSearchTerm] = useState("");
    const [isSearchExpanded, setIsSearchExpanded] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadError, setUploadError] = useState(null);

    // Removed initial dummy data load - graph will be empty until papers are uploaded
    // useEffect(() => {
    //     fetch("http://localhost:8000/api/graph/dummy")
    //         .then((response) => response.json())
    //         .then((data) => setGraphData(data))
    //         .catch((error) => {
    //             console.error("Error:", error);
    //         });
    // }, []);

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
                        data={graphData}
                        isDarkMode={isDarkMode}
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
                <input
                    type="text"
                    placeholder="Ask a Question..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    onFocus={() => setIsSearchExpanded(true)}
                    onBlur={() => setIsSearchExpanded(false)}
                    className={`search-bar ${
                        isSearchExpanded ? "expanded" : ""
                    }`}
                />
            </main>
        </div>
    );
}

export default App;
