import React, { useState, useEffect } from 'react';
import GraphVisualization from './GraphVisualization';

function App() {
  const [graphData, setGraphData] = useState(null);

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
    <div style={{ padding: '20px' }}>
      <h1>Paper Graph Visualization</h1>
      {graphData ? (
        <GraphVisualization data={graphData} />
      ) : (
        <p>Loading graph...</p>
      )}
    </div>
  );
}

export default App;
