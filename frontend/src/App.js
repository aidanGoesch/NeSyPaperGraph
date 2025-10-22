import React, { useState, useEffect } from 'react';
import GraphVisualization from './GraphVisualization';

function App() {
  const [graphData, setGraphData] = useState(null);

  useEffect(() => {
    // Create fully connected graph - every paper connects to every topic
    const dummyData = {
      papers: [
        { title: "Paper A", topics: ["Topic 1", "Topic 2", "Topic 3"] },
        { title: "Paper B", topics: ["Topic 1", "Topic 2", "Topic 3"] },
        { title: "Paper C", topics: ["Topic 1", "Topic 2", "Topic 3"] },
        { title: "Paper D", topics: ["Topic 1", "Topic 2", "Topic 3"] },
        { title: "Paper E", topics: ["Topic 1", "Topic 2", "Topic 3"] }
      ],
      topics: ["Topic 1", "Topic 2", "Topic 3"]
    };
    setGraphData(dummyData);
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
