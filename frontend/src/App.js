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
  const [showChatPanel, setShowChatPanel] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [followUpQuestion, setFollowUpQuestion] = useState('');
  const mermaidRef = useRef();
  const chatContentRef = useRef();

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
    // Auto-scroll to bottom when chat history updates
    if (chatContentRef.current) {
      chatContentRef.current.scrollTop = chatContentRef.current.scrollHeight;
    }
  }, [chatHistory]);

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
    
    setIsSearching(true);
    setShowChatPanel(true);
    
    // Add question to chat history immediately with loading state
    const questionEntry = {
      question: query,
      answer: null, // null indicates loading
      timestamp: new Date().toLocaleTimeString()
    };
    setChatHistory(prev => [...prev, questionEntry]);
    
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
      
      // Update the last entry with the answer
      setChatHistory(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          answer: data.answer || data.error || 'No response'
        };
        return updated;
      });
    } catch (error) {
      console.error('Search error:', error);
      // Update the last entry with error
      setChatHistory(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          answer: 'Error: Could not connect to server'
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
        {!showChatPanel && (
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
        )}
        {showChatPanel && (
          <div className={`chat-panel-wrapper ${isDarkMode ? 'dark' : 'light'}`}>
            <div className={`chat-panel ${isDarkMode ? 'dark' : 'light'}`}>
              
              {/* X CLOSE BUTTON — now in top right corner */}
              <button 
                onClick={() => setShowChatPanel(false)}
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
                      <span className="timestamp">{entry.timestamp}</span>
                    </div>
                    <div className="answer">
                      <strong>A:</strong> 
                      {entry.answer === null ? (
                        <div className="loading-dots">
                          <span></span><span></span><span></span>
                        </div>
                      ) : (
                        <span> {entry.answer}</span>
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
                onChange={(e) => setFollowUpQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !isSearching) {
                    handleSearch(followUpQuestion);
                    setFollowUpQuestion('');
                  }
                }}
                disabled={isSearching}
                className={`chat-input ${isDarkMode ? 'dark' : 'light'}`}
              />
            </div>
          </div>
        )}
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