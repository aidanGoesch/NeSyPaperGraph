import React, { useState, useEffect } from 'react';
import GraphVisualization from './GraphVisualization';
import './App.css';

function App() {
  const [graphData, setGraphData] = useState(null);
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    fetch('http://localhost:8000/api/graph/dummy')
      .then(response => response.json())
      .then(data => setGraphData(data))
      .catch(error => {
        console.error('Error:', error);
        // Fallback to dummy data if backend is not running
        const fallbackData = {
          papers: [
            { title: "Paper A", topics: ["Topic 1", "Topic 2", "Topic 3"] },
            { title: "Paper B", topics: ["Topic 1", "Topic 2", "Topic 3"] },
            { title: "Paper C", topics: ["Topic 1", "Topic 2", "Topic 3"] },
            { title: "Paper D", topics: ["Topic 1", "Topic 2", "Topic 3"] },
            { title: "Paper E", topics: ["Topic 1", "Topic 2", "Topic 3"] }
          ],
          topics: ["Topic 1", "Topic 2", "Topic 3"]
        };
        setGraphData(fallbackData);
      });
  }, []);

  return (
    <div className={`app ${isDarkMode ? 'dark' : 'light'}`}>
      <div className="theme-toggle">
        <button onClick={() => setIsDarkMode(!isDarkMode)}>
          {isDarkMode ? '☀️' : '🌙'}
        </button>
      </div>
      <header className="app-header">
        <h1>Paper Graph Visualization</h1>
      </header>
      <main className="app-main">
        {graphData ? (
          <GraphVisualization data={graphData} />
        ) : (
          <div className="loading">Loading graph...</div>
        )}
      </main>
    </div>
  );
}

export default App;
