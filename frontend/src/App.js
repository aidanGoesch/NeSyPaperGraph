import React, { useState, useEffect, useRef } from 'react';
import GraphVisualization from './GraphVisualization';
import mermaid from 'mermaid';
import './App.css';

function App() {
  const [graphData, setGraphData] = useState(null);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [isSearchExpanded, setIsSearchExpanded] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [showMermaid, setShowMermaid] = useState(false);
  const [mermaidDiagram, setMermaidDiagram] = useState('');
  const mermaidRef = useRef();

  useEffect(() => {
    mermaid.initialize({ startOnLoad: true });
  }, []);

  useEffect(() => {
    if (showMermaid && mermaidDiagram && mermaidRef.current) {
      mermaidRef.current.innerHTML = mermaidDiagram;
      mermaid.init(undefined, mermaidRef.current);
    }
  }, [showMermaid, mermaidDiagram]);

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

  const showAgentArchitecture = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/agent/architecture');
      const data = await response.json();
      if (data.mermaid) {
        setMermaidDiagram(data.mermaid);
        setShowMermaid(true);
      }
    } catch (error) {
      console.error('Error fetching architecture:', error);
    }
  };

  const handleSearch = async (query) => {
    if (!query.trim()) return;
    
    try {
      const response = await fetch('http://localhost:8000/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });
      
      const data = await response.json();
      console.log('Search response:', data);
    } catch (error) {
      console.error('Search error:', error);
    }
  };

  const handleFileUpload = async (event) => {
    const files = event.target.files;
    if (files.length > 0) {
      setIsGenerating(true);
      console.log('Selected files:', Array.from(files).map(f => f.name));
      
      try {
        const formData = new FormData();
        Array.from(files).forEach(file => {
          formData.append('files', file);
        });

        const response = await fetch('http://localhost:8000/api/graph/build', {
          method: 'POST',
          body: formData,
        });
        
        const data = await response.json();
        if (data.error) {
          console.error('Error building graph:', data.error);
          alert('Error building graph: ' + data.error);
        } else {
          console.log('Received data:', data);
          setGraphData(data);
          console.log('Graph built successfully');
        }
      } catch (error) {
        console.error('Error:', error);
        alert('Error connecting to backend: ' + error.message);
      } finally {
        setIsGenerating(false);
      }
    }
  };

  return (
    <div className={`app ${isDarkMode ? 'dark' : 'light'}`}>
      {isGenerating && <div className="progress-bar"></div>}
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
          <GraphVisualization data={graphData} isDarkMode={isDarkMode} onShowArchitecture={showAgentArchitecture} />
        ) : (
          <div className="loading">Loading graph...</div>
        )}
        <input
          type="file"
          multiple
          accept=".pdf"
          onChange={handleFileUpload}
          style={{ display: 'none' }}
          id="file-upload"
          webkitdirectory=""
        />
        <button 
          onClick={() => document.getElementById('file-upload').click()}
          className="upload-button"
        >
          📁 Upload Papers
        </button>
        <input
          type="text"
          placeholder="Ask a Question..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              handleSearch(searchTerm);
              setSearchTerm('');
            }
          }}
          onFocus={() => setIsSearchExpanded(true)}
          onBlur={() => setIsSearchExpanded(false)}
          className={`search-bar ${isSearchExpanded ? 'expanded' : ''}`}
        />
        {showMermaid && (
          <div style={{
            position: 'fixed',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '8px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
            zIndex: 2000,
            maxWidth: '80vw',
            maxHeight: '80vh',
            overflow: 'auto'
          }}>
            <button
              onClick={() => setShowMermaid(false)}
              style={{
                position: 'absolute',
                top: '10px',
                right: '10px',
                background: 'none',
                border: 'none',
                fontSize: '24px',
                cursor: 'pointer'
              }}
            >
              ×
            </button>
            <h3>Agent Architecture</h3>
            <div ref={mermaidRef}></div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
