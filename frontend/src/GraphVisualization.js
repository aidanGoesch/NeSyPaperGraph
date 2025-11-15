import React, { useEffect, useRef, useState, forwardRef, useImperativeHandle } from 'react';
import * as d3 from 'd3';

const GraphVisualization = forwardRef(({ data, isDarkMode, onShowArchitecture, onTopicClick }, ref) => {
  const svgRef = useRef();
  const containerRef = useRef();
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [panelPosition, setPanelPosition] = useState({ x: 0, y: 0 });
  const [showSemanticEdges, setShowSemanticEdges] = useState(true);
  const [showMenu, setShowMenu] = useState(false);
  const [semanticThreshold, setSemanticThreshold] = useState(0.25);
  const selectedNodeRef = useRef(null);
  const simulationRef = useRef(null);
  const nodesRef = useRef([]);
  const zoomRef = useRef(null);

  useImperativeHandle(ref, () => ({
    focusOnTopic: (topicName) => {
      if (!nodesRef.current || !zoomRef.current) return;
      
      const topicNode = nodesRef.current.find(node => node.type === 'topic' && node.id === topicName);
      if (!topicNode) return;
      
      selectedNodeRef.current = topicNode;
      
      // Navigate to topic location
      const svg = d3.select(svgRef.current);
      const container = containerRef.current;
      const width = container.clientWidth;
      const height = container.clientHeight;
      const scale = 1.5;
      const newX = -topicNode.x * scale + width / 2;
      const newY = -topicNode.y * scale + height / 2;
      
      svg.transition()
        .duration(750)
        .call(zoomRef.current.transform, d3.zoomIdentity.translate(newX, newY).scale(scale));
      
      // Show topic panel
      const transform = d3.zoomTransform(svg.node());
      const [x, y] = transform.apply([topicNode.x, topicNode.y]);
      const svgRect = svgRef.current.getBoundingClientRect();
      
      setPanelPosition({
        x: svgRect.left + x + 20,
        y: svgRect.top + y - 10,
      });
      
      // Find connected papers
      const links = data.edges || [];
      if (links.length === 0) {
        data.papers.forEach(paper => {
          paper.topics.forEach(topic => {
            links.push({ source: paper.title, target: topic });
          });
        });
      }
      
      const connectedPapers = links
        .filter(link => link.target.id === topicName || link.source.id === topicName)
        .map(link => link.target.id === topicName ? link.source : link.target)
        .filter(node => node.type === 'paper');
      
      setSelectedPaper(null);
      setSelectedTopic({
        name: topicName,
        papers: connectedPapers
      });
    }
  }));

  useEffect(() => {
    if (!data) return;

    const container = containerRef.current;
    const svg = d3.select(svgRef.current);
    
    // Always clear and recreate when data changes
    svg.selectAll("*").remove();
    simulationRef.current = null;

    const width = container.clientWidth;
    const height = container.clientHeight;

    svg.attr("width", width).attr("height", height)
      .style("background-color", isDarkMode ? "#2a2a2a" : "white")
      .style("transition", "background-color 0.5s ease");

    // Update existing elements if simulation exists
    if (simulationRef.current) {
      svg.selectAll("line")
        .transition()
        .duration(500)
        .attr("stroke", d => d.type === 'semantic' ? (isDarkMode ? "#888" : "#999") : (isDarkMode ? "#ccc" : "#000"));
      
      svg.selectAll("text")
        .transition()
        .duration(500)
        .attr("fill", isDarkMode ? "#ccc" : "#333");
      
      return;
    }

    const nodes = [
      ...data.papers.map(paper => ({ id: paper.title, type: 'paper', ...paper })),
      ...data.topics.map(topic => ({ id: topic, type: 'topic' }))
    ];

    const links = data.edges || [];
    // Fallback: if no edges provided, create topic links
    if (links.length === 0) {
      data.papers.forEach(paper => {
        paper.topics.forEach(topic => {
          links.push({ source: paper.title, target: topic });
        });
      });
    }

    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(120))
      .force("charge", d3.forceManyBody().strength(-400))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(30));

    simulationRef.current = simulation;
    nodesRef.current = nodes;

    const g = svg.append("g");

    const zoom = d3.zoom()
      .scaleExtent([0.1, 5])
      .on("zoom", (event) => {
        g.attr("transform", event.transform);
        updatePanelPosition(); // move with zoom/pan
      })
      .filter(function(event) {
        return !(event.target && event.target.closest && event.target.closest('.info-panel'));
      });

    zoomRef.current = zoom;

    svg.call(zoom);

    const link = g.append("g")
      .selectAll("line")
      .data(links)
      .enter().append("line")
      .attr("stroke", d => d.type === 'semantic' ? (isDarkMode ? "#888" : "#999") : (isDarkMode ? "#ccc" : "#000"))
      .attr("stroke-width", d => d.type === 'semantic' ? ((d.weight - 0.25) * 5) : 5)
      .attr("stroke-opacity", d => {
        if (d.type === 'semantic') {
          return (d.weight >= semanticThreshold && showSemanticEdges) ? 0.6 : 0;
        }
        return 0.6;
      })
      .style("cursor", d => d.type === 'semantic' ? "pointer" : "default")
      .on("click", function(event, d) {
        if (d.type === 'semantic') {
          event.stopPropagation();
          alert(`Semantic connection between "${d.source.id || d.source}" and "${d.target.id || d.target}" (similarity: ${d.weight?.toFixed(3) || 'N/A'})`);
        }
      });

    const node = g.append("g")
      .selectAll("circle")
      .data(nodes)
      .enter().append("circle")
      .attr("r", d => d.type === 'paper' ? 10 : 15)
      .attr("fill", d => d.type === 'paper' ? "#4CAF50" : "#FF6B6B")
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .style("cursor", "pointer")
      .on("mouseover", function(event, d) {
        if (d.type === 'paper') {
          label.filter(labelData => labelData.id === d.id)
            .style("opacity", 1);
        }
      })
      .on("mouseout", function(event, d) {
        if (d.type === 'paper') {
          label.filter(labelData => labelData.id === d.id)
            .style("opacity", 0);
        }
      })
      .on("click", function(event, d) {
        event.stopPropagation();
        
        selectedNodeRef.current = d; // keep live reference

        const transform = d3.zoomTransform(svg.node());
        const [x, y] = transform.apply([d.x, d.y]);
        const svgRect = svgRef.current.getBoundingClientRect();

        const offsetX = 20;
        const offsetY = -10;

        setPanelPosition({
          x: svgRect.left + x + offsetX,
          y: svgRect.top + y + offsetY,
        });

        if (d.type === 'paper') {
          setSelectedTopic(null);
          setSelectedPaper({
            title: d.title,
            authors: d.authors || ['Dr. Jane Smith', 'Dr. John Doe', 'Dr. Alice Johnson'],
            year: d.year || 2024,
            citations: d.citations || Math.floor(Math.random() * 500),
            abstract: d.abstract || 'This is a dummy abstract for the paper...',
            topics: d.topics || []
          });
        } else if (d.type === 'topic') {
          setSelectedPaper(null);
          // Find connected papers
          const connectedPapers = links
            .filter(link => link.target.id === d.id || link.source.id === d.id)
            .map(link => link.target.id === d.id ? link.source : link.target)
            .filter(node => node.type === 'paper');
          
          setSelectedTopic({
            name: d.id,
            papers: connectedPapers
          });
        }

        // zoom smoothly toward node
        const scale = 1.5;
        const newX = -d.x * scale + width / 2;
        const newY = -d.y * scale + height / 2;

        svg.transition()
          .duration(750)
          .call(zoom.transform, d3.zoomIdentity.translate(newX, newY).scale(scale));
      })
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

    const label = g.append("g")
      .selectAll("text")
      .data(nodes)
      .enter().append("text")
      .text(d => d.type === 'topic' ? d.id : (d.id.length > 30 ? d.id.substring(0, 30) + '...' : d.id))
      .attr("font-size", d => d.type === 'topic' ? 14 : 10)
      .attr("font-family", "Arial, sans-serif")
      .attr("fill", isDarkMode ? "#ccc" : "#333")
      .attr("text-anchor", "middle")
      .attr("dy", d => d.type === 'topic' ? 5 : -20)
      .style("pointer-events", "none")
      .style("opacity", d => d.type === 'topic' ? 1 : 0)
      .style("transition", "opacity 0.3s ease, fill 0.5s ease")
      .style("paint-order", "stroke fill")
      .style("stroke", isDarkMode ? "#2a2a2a" : "white")
      .style("stroke-width", "3px")
      .style("stroke-linejoin", "round");

    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);

      label
        .attr("x", d => d.x)
        .attr("y", d => d.y);

      updatePanelPosition(); // follow node as simulation moves
    });

    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    function updatePanelPosition() {
      if (!selectedNodeRef.current) return;
      const transform = d3.zoomTransform(svg.node());
      const [x, y] = transform.apply([
        selectedNodeRef.current.x,
        selectedNodeRef.current.y,
      ]);
      const svgRect = svgRef.current.getBoundingClientRect();
      const offsetX = 20;
      const offsetY = -10;
      setPanelPosition({
        x: svgRect.left + x + offsetX,
        y: svgRect.top + y + offsetY,
      });
    }

    const handleResize = () => {
      const newWidth = container.clientWidth;
      const newHeight = container.clientHeight;
      svg.attr("width", newWidth).attr("height", newHeight);
      simulation.force("center", d3.forceCenter(newWidth / 2, newHeight / 2));
      simulation.alpha(0.3).restart();
    };

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      if (simulationRef.current) {
        simulationRef.current.stop();
      }
    };
  }, [data]);

  // Separate effect for theme changes
  useEffect(() => {
    if (!simulationRef.current) return;
    
    const svg = d3.select(svgRef.current);
    svg.style("background-color", isDarkMode ? "#2a2a2a" : "white");
    
    svg.selectAll("line")
      .attr("stroke", d => d.type === 'semantic' ? (isDarkMode ? "#888" : "#999") : (isDarkMode ? "#ccc" : "#000"));
    
    svg.selectAll("text")
      .attr("fill", isDarkMode ? "#ccc" : "#333")
      .style("stroke", isDarkMode ? "#2a2a2a" : "white");
  }, [isDarkMode]);

  // Effect for semantic edges toggle and threshold
  useEffect(() => {
    if (!simulationRef.current || !data) return;
    
    const svg = d3.select(svgRef.current);
    svg.select("g").selectAll("line")
      .attr("stroke-opacity", d => {
        if (d.type === 'semantic') {
          return (d.weight >= semanticThreshold && showSemanticEdges) ? 0.6 : 0;
        }
        return 0.6;
      });
    
    // Update simulation forces only for layout
    const links = data.edges || [];
    if (links.length === 0) {
      data.papers.forEach(paper => {
        paper.topics.forEach(topic => {
          links.push({ source: paper.title, target: topic });
        });
      });
    }
    
    const activeLinks = showSemanticEdges ? 
      links.filter(d => d.type !== 'semantic' || d.weight >= semanticThreshold) : 
      links.filter(link => link.type !== 'semantic');
    
    simulationRef.current.force("link").links(activeLinks);
    simulationRef.current.alpha(0.3).restart();
  }, [semanticThreshold, showSemanticEdges, data, isDarkMode]);



  return (
    <div ref={containerRef} style={{ width: '100%', height: '100vh', position: 'relative', backgroundColor: 'white' }}>
      <div style={{ position: 'absolute', top: '20px', right: '20px', zIndex: 1001 }}>
        <button
          onClick={() => setShowMenu(!showMenu)}
          style={{
            background: isDarkMode ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)',
            color: isDarkMode ? '#ccc' : '#333',
            border: isDarkMode ? '1px solid rgba(255, 255, 255, 0.1)' : '1px solid rgba(0, 0, 0, 0.1)',
            padding: '12px',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '16px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            width: '44px',
            height: '44px',
            transition: 'all 0.5s ease'
          }}
          onMouseEnter={(e) => {
            e.target.style.background = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
          }}
          onMouseLeave={(e) => {
            e.target.style.background = isDarkMode ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)';
          }}
        >
          <div style={{ width: '16px', height: '2px', backgroundColor: isDarkMode ? '#ccc' : '#333', margin: '2px 0' }}></div>
          <div style={{ width: '16px', height: '2px', backgroundColor: isDarkMode ? '#ccc' : '#333', margin: '2px 0' }}></div>
          <div style={{ width: '16px', height: '2px', backgroundColor: isDarkMode ? '#ccc' : '#333', margin: '2px 0' }}></div>
        </button>
        {showMenu && (
          <div style={{
            position: 'absolute',
            top: '50px',
            right: '0',
            backgroundColor: 'white',
            border: '1px solid #ccc',
            borderRadius: '4px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
            padding: '10px',
            minWidth: '200px'
          }}>
            <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', marginBottom: '15px' }}>
              <input
                type="checkbox"
                checked={showSemanticEdges}
                onChange={(e) => setShowSemanticEdges(e.target.checked)}
                style={{ marginRight: '8px' }}
              />
              Show Semantic Edges
            </label>
            <div style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '14px' }}>
                Semantic Threshold: {semanticThreshold.toFixed(2)}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={semanticThreshold}
                onChange={(e) => setSemanticThreshold(parseFloat(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
            <button
              onClick={onShowArchitecture}
              style={{
                width: '100%',
                padding: '8px 12px',
                backgroundColor: '#4CAF50',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '14px'
              }}
            >
              Show Agent Architecture
            </button>
          </div>
        )}
      </div>
      <svg ref={svgRef}></svg>
      {selectedPaper && (
        <div className="info-panel" style={{
          position: 'absolute',
          top: `${panelPosition.y}px`,
          left: `${panelPosition.x}px`,
          width: '350px',
          maxHeight: '400px',
          backgroundColor: 'white',
          borderRadius: '8px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
          padding: '20px',
          overflowY: 'auto',
          zIndex: 1000,
          pointerEvents: 'auto'
        }}>
          <button 
            onClick={() => {
              setSelectedPaper(null);
              setPanelPosition({ x: 0, y: 0 });
            }}
            style={{
              position: 'absolute',
              top: '10px',
              right: '10px',
              background: 'none',
              border: 'none',
              fontSize: '24px',
              cursor: 'pointer',
              color: '#666',
              lineHeight: '1'
            }}
          >
            ×
          </button>
          <h3 style={{ marginTop: 0, marginBottom: '15px', fontSize: '18px', color: '#333' }}>
            {selectedPaper.title}
          </h3>
          <div style={{ marginBottom: '12px' }}>
            <strong>Authors:</strong>
            <div style={{ marginTop: '4px', color: '#555' }}>
              {selectedPaper.authors.join(', ')}
            </div>
          </div>
          <div style={{ marginBottom: '12px' }}>
            <strong>Year:</strong> <span style={{ color: '#555' }}>{selectedPaper.year}</span>
          </div>
          <div style={{ marginBottom: '12px' }}>
            <strong>Citations:</strong> <span style={{ color: '#555' }}>{selectedPaper.citations}</span>
          </div>
          <div style={{ marginBottom: '12px' }}>
            <strong>Topics:</strong>
            <div style={{ marginTop: '4px', display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
              {selectedPaper.topics && selectedPaper.topics.map((topic, index) => (
                <span
                  key={index}
                  style={{
                    backgroundColor: '#FF6B6B',
                    color: 'white',
                    padding: '2px 8px',
                    borderRadius: '12px',
                    fontSize: '12px',
                    cursor: 'pointer',
                    transition: 'background-color 0.2s ease'
                  }}
                  onMouseEnter={(e) => e.target.style.backgroundColor = '#FF5252'}
                  onMouseLeave={(e) => e.target.style.backgroundColor = '#FF6B6B'}
                  onClick={() => {
                    // Find the topic node and show its panel
                    const topicNode = nodesRef.current.find(node => node.type === 'topic' && node.id === topic);
                    if (topicNode && zoomRef.current) {
                      selectedNodeRef.current = topicNode;
                      
                      // Navigate to topic location
                      const svg = d3.select(svgRef.current);
                      const container = containerRef.current;
                      const width = container.clientWidth;
                      const height = container.clientHeight;
                      const scale = 1.5;
                      const newX = -topicNode.x * scale + width / 2;
                      const newY = -topicNode.y * scale + height / 2;
                      
                      svg.transition()
                        .duration(750)
                        .call(zoomRef.current.transform, d3.zoomIdentity.translate(newX, newY).scale(scale));
                      
                      // Find connected papers for topic panel
                      const links = data.edges || [];
                      if (links.length === 0) {
                        data.papers.forEach(paper => {
                          paper.topics.forEach(paperTopic => {
                            links.push({ source: paper.title, target: paperTopic });
                          });
                        });
                      }
                      
                      const connectedPapers = links
                        .filter(link => link.target.id === topic || link.source.id === topic)
                        .map(link => link.target.id === topic ? link.source : link.target)
                        .filter(node => node.type === 'paper');
                      
                      setSelectedPaper(null);
                      setSelectedTopic({
                        name: topic,
                        papers: connectedPapers
                      });
                    }
                  }}
                >
                  {topic}
                </span>
              ))}
            </div>
          </div>
          <div style={{ marginBottom: '12px' }}>
            <strong>Abstract:</strong>
            <p style={{ marginTop: '4px', color: '#555', lineHeight: '1.5' }}>
              {selectedPaper.abstract}
            </p>
          </div>
        </div>
      )}
      {selectedTopic && (
        <div className="info-panel" style={{
          position: 'absolute',
          top: `${panelPosition.y}px`,
          left: `${panelPosition.x}px`,
          width: '350px',
          maxHeight: '400px',
          backgroundColor: 'white',
          borderRadius: '8px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
          padding: '20px',
          overflowY: 'auto',
          zIndex: 1000,
          pointerEvents: 'auto'
        }}>
          <button 
            onClick={() => {
              setSelectedTopic(null);
              setPanelPosition({ x: 0, y: 0 });
            }}
            style={{
              position: 'absolute',
              top: '10px',
              right: '10px',
              background: 'none',
              border: 'none',
              fontSize: '24px',
              cursor: 'pointer',
              color: '#666',
              lineHeight: '1'
            }}
          >
            ×
          </button>
          <h3 style={{ marginTop: 0, marginBottom: '15px', fontSize: '18px', color: '#333' }}>
            Topic: {selectedTopic.name}
          </h3>
          <div style={{ marginBottom: '12px' }}>
            <strong>Connected Papers ({selectedTopic.papers.length}):</strong>
          </div>
          <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
            {selectedTopic.papers.map((paper, index) => (
              <div key={index} style={{
                padding: '10px',
                marginBottom: '8px',
                backgroundColor: '#f5f5f5',
                borderRadius: '4px',
                cursor: 'pointer',
                transition: 'background-color 0.2s ease'
              }}
              onMouseEnter={(e) => e.target.style.backgroundColor = '#e0e0e0'}
              onMouseLeave={(e) => e.target.style.backgroundColor = '#f5f5f5'}
              onClick={() => {
                // Find the paper node in the graph
                const paperNode = nodesRef.current.find(node => node.id === paper.title);
                if (paperNode && zoomRef.current) {
                  selectedNodeRef.current = paperNode;
                  
                  // Navigate to paper location
                  const svg = d3.select(svgRef.current);
                  const container = containerRef.current;
                  const width = container.clientWidth;
                  const height = container.clientHeight;
                  const scale = 1.5;
                  const newX = -paperNode.x * scale + width / 2;
                  const newY = -paperNode.y * scale + height / 2;
                  
                  svg.transition()
                    .duration(750)
                    .call(zoomRef.current.transform, d3.zoomIdentity.translate(newX, newY).scale(scale));
                }
                
                setSelectedTopic(null);
                setSelectedPaper({
                  title: paper.title,
                  authors: paper.authors || ['Dr. Jane Smith', 'Dr. John Doe'],
                  year: paper.year || 2024,
                  citations: paper.citations || Math.floor(Math.random() * 500),
                  abstract: paper.abstract || 'This is a dummy abstract for the paper...',
                  topics: paper.topics || []
                });
              }}>
                <div style={{ fontWeight: 'bold', fontSize: '14px', color: '#333', marginBottom: '4px' }}>
                  {paper.title}
                </div>
                <div style={{ fontSize: '12px', color: '#666' }}>
                  {paper.authors ? paper.authors.join(', ') : 'Authors not available'}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});

GraphVisualization.displayName = 'GraphVisualization';

export default GraphVisualization;