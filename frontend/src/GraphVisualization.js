import React, { useEffect, useRef, useState, forwardRef, useImperativeHandle } from 'react';
import ReactMarkdown from 'react-markdown';
import * as d3 from 'd3';

const GraphVisualization = forwardRef(({ data, isDarkMode, onShowArchitecture, onTopicClick, highlightPath }, ref) => {
  const svgRef = useRef();
  const containerRef = useRef();
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [panelPosition, setPanelPosition] = useState({ x: 0, y: 0 });
  const [showSemanticEdges, setShowSemanticEdges] = useState(true);
  const [showMenu, setShowMenu] = useState(false);
  const [semanticThreshold, setSemanticThreshold] = useState(0.25);
  const selectedNodeRef = useRef(null);
  const pinnedNodeRef = useRef(null);
  const simulationRef = useRef(null);
  const nodesRef = useRef([]);
  const zoomRef = useRef(null);
  const panelMetrics = { width: 350, maxHeight: 400, margin: 12 };

  const normalizeLinkEndpoint = (endpoint) => {
    if (endpoint && typeof endpoint === "object") {
      return endpoint.id || endpoint.title || endpoint.name || "";
    }
    return endpoint;
  };

  const buildLinks = (papers, rawEdges) => {
    const normalizedLinks = Array.isArray(rawEdges)
      ? rawEdges.map((edge) => ({
          ...edge,
          source: normalizeLinkEndpoint(edge.source),
          target: normalizeLinkEndpoint(edge.target),
        }))
      : [];

    if (normalizedLinks.length > 0) {
      return normalizedLinks;
    }

    const fallbackLinks = [];
    (papers || []).forEach((paper) => {
      (paper.topics || []).forEach((topic) => {
        fallbackLinks.push({ source: paper.title, target: topic });
      });
    });
    return fallbackLinks;
  };

  const getConnectedPapersForTopic = (topicName, links, papers) => {
    const connectedTitles = new Set();
    (links || []).forEach((link) => {
      const sourceId = normalizeLinkEndpoint(link.source);
      const targetId = normalizeLinkEndpoint(link.target);
      if (sourceId === topicName && targetId) connectedTitles.add(targetId);
      if (targetId === topicName && sourceId) connectedTitles.add(sourceId);
    });
    return (papers || []).filter((paper) => connectedTitles.has(paper.title));
  };

  const clampPanelToContainer = (x, y) => {
    const container = containerRef.current;
    if (!container) return { x, y };
    const maxX = Math.max(
      panelMetrics.margin,
      container.clientWidth - panelMetrics.width - panelMetrics.margin
    );
    const maxY = Math.max(
      panelMetrics.margin,
      container.clientHeight - panelMetrics.maxHeight - panelMetrics.margin
    );
    return {
      x: Math.min(Math.max(x, panelMetrics.margin), maxX),
      y: Math.min(Math.max(y, panelMetrics.margin), maxY),
    };
  };

  const getPanelPositionForNode = (node, transformOverride) => {
    if (!node || !svgRef.current) return null;
    const transform =
      transformOverride || d3.zoomTransform(d3.select(svgRef.current).node());
    const [x, y] = transform.apply([node.x, node.y]);
    return clampPanelToContainer(x + 20, y - 10);
  };

  const pinNode = (node) => {
    if (!node) return;
    if (pinnedNodeRef.current && pinnedNodeRef.current !== node) {
      pinnedNodeRef.current.fx = null;
      pinnedNodeRef.current.fy = null;
    }
    node.fx = node.x;
    node.fy = node.y;
    pinnedNodeRef.current = node;
  };

  const unpinNode = () => {
    if (!pinnedNodeRef.current) return;
    pinnedNodeRef.current.fx = null;
    pinnedNodeRef.current.fy = null;
    pinnedNodeRef.current = null;
  };

  useImperativeHandle(ref, () => ({
    focusOnTopic: (topicName) => {
      if (!nodesRef.current || !zoomRef.current) return;
      
      const topicNode = nodesRef.current.find(node => node.type === 'topic' && node.id === topicName);
      if (!topicNode) return;
      
      selectedNodeRef.current = topicNode;
      pinNode(topicNode);
      
      // Navigate to topic location
      const svg = d3.select(svgRef.current);
      const container = containerRef.current;
      const width = container.clientWidth;
      const height = container.clientHeight;
      const scale = 1.5;
      const newX = -topicNode.x * scale + width / 2;
      const newY = -topicNode.y * scale + height / 2;
      const targetTransform = d3.zoomIdentity.translate(newX, newY).scale(scale);
      
      svg.transition()
        .duration(750)
        .call(zoomRef.current.transform, targetTransform);
      
      // Show topic panel near node, constrained inside container.
      const nextPanelPosition = getPanelPositionForNode(topicNode, targetTransform);
      if (nextPanelPosition) setPanelPosition(nextPanelPosition);
      
      // Find connected papers
      const links = buildLinks(data.papers, data.edges);
      const connectedPapers = getConnectedPapersForTopic(
        topicName,
        links,
        data.papers
      );
      
      setSelectedPaper(null);
      setSelectedTopic({
        name: topicName,
        papers: connectedPapers
      });
    },
    focusOnPaper: (paperTitle) => {
      if (!nodesRef.current || !zoomRef.current) return;
      
      const paperNode = nodesRef.current.find(node => node.type === 'paper' && node.id === paperTitle);
      if (!paperNode) return;
      
      selectedNodeRef.current = paperNode;
      pinNode(paperNode);
      
      // Navigate to paper location
      const svg = d3.select(svgRef.current);
      const container = containerRef.current;
      const width = container.clientWidth;
      const height = container.clientHeight;
      const scale = 1.5;
      const newX = -paperNode.x * scale + width / 2;
      const newY = -paperNode.y * scale + height / 2;
      const targetTransform = d3.zoomIdentity.translate(newX, newY).scale(scale);
      
      svg.transition()
        .duration(750)
        .call(zoomRef.current.transform, targetTransform);
      
      // Show paper panel near node, constrained inside container.
      const nextPanelPosition = getPanelPositionForNode(paperNode, targetTransform);
      if (nextPanelPosition) setPanelPosition(nextPanelPosition);
      
      // Find the paper data
      const paper = data.papers.find(p => p.title === paperTitle);
      if (paper) {
        setSelectedTopic(null);
        setSelectedPaper({
          title: paper.title,
          authors: paper.authors || ['Dr. Jane Smith', 'Dr. John Doe', 'Dr. Alice Johnson'],
          year: paper.year,
          publication_date: paper.publication_date,
          abstract: paper.abstract || 'This is a dummy abstract for the paper...',
          topics: paper.topics || []
        });
      }
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

    const links = buildLinks(data.papers, data.edges);

    const highlightedNodeSet = new Set(
      Array.isArray(highlightPath?.nodes) ? highlightPath.nodes : []
    );
    const pathNodes =
      highlightPath?.mode === "path" && Array.isArray(highlightPath?.nodes)
        ? highlightPath.nodes
        : [];
    const isEdgeInHighlightedPath = (edge) => {
      if (pathNodes.length < 2) return false;
      const sourceId = edge.source.id || edge.source;
      const targetId = edge.target.id || edge.target;
      for (let i = 0; i < pathNodes.length - 1; i++) {
        if (
          (pathNodes[i] === sourceId && pathNodes[i + 1] === targetId) ||
          (pathNodes[i] === targetId && pathNodes[i + 1] === sourceId)
        ) {
          return true;
        }
      }
      return false;
    };

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
      .attr("stroke", d => {
        if (isEdgeInHighlightedPath(d)) {
          return "#FFD600";
        }
        
        return d.type === 'semantic' ? (isDarkMode ? "#888" : "#999") : (isDarkMode ? "#ccc" : "#000");
      })
      .attr("stroke-width", d => {
        if (isEdgeInHighlightedPath(d)) {
          return 6;
        }
        
        return d.type === 'semantic' ? ((d.weight - 0.25) * 5) : 5;
      })
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
      .attr("fill", d => {
        if (highlightedNodeSet.has(d.id)) {
          return d.type === 'paper' ? "#FF6B35" : "#FF1744";
        }
        return d.type === 'paper' ? "#4CAF50" : "#FF6B6B";
      })
      .attr("stroke", d => {
        if (highlightedNodeSet.has(d.id)) {
          return "#FFD600";
        }
        return "#fff";
      })
      .attr("stroke-width", d => {
        if (highlightedNodeSet.has(d.id)) {
          return 4;
        }
        return 2;
      })
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

        const nextPanelPosition = getPanelPositionForNode(d);
        if (nextPanelPosition) setPanelPosition(nextPanelPosition);

        if (d.type === 'paper') {
          setSelectedTopic(null);
          setSelectedPaper({
            title: d.title,
            authors: d.authors || ['Dr. Jane Smith', 'Dr. John Doe', 'Dr. Alice Johnson'],
            year: d.year,
            publication_date: d.publication_date,
            abstract: d.abstract || 'This is a dummy abstract for the paper...',
            topics: d.topics || []
          });
        } else if (d.type === 'topic') {
          setSelectedPaper(null);
          // Find connected papers
          const connectedPapers = getConnectedPapersForTopic(
            d.id,
            links,
            data.papers
          );
          
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
      const nextPanelPosition = getPanelPositionForNode(selectedNodeRef.current);
      if (nextPanelPosition) setPanelPosition(nextPanelPosition);
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
      unpinNode();
      if (simulationRef.current) {
        simulationRef.current.stop();
      }
    };
  }, [data, highlightPath]);

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
    const links = buildLinks(data.papers, data.edges);
    
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
            <div style={{ marginBottom: '15px', padding: '8px', backgroundColor: '#f5f5f5', borderRadius: '4px' }}>
              <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '5px' }}>Graph Stats</div>
              <div style={{ fontSize: '13px' }}>📄 Papers: {data.papers?.length || 0}</div>
              <div style={{ fontSize: '13px' }}>🏷️ Topics: {data.topics?.length || 0}</div>
            </div>
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
              unpinNode();
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
            <strong>Year:</strong> <span style={{ color: '#555' }}>{selectedPaper.year || selectedPaper.publication_date || 'Unknown'}</span>
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
                      const links = buildLinks(data.papers, data.edges);
                      const connectedPapers = getConnectedPapersForTopic(
                        topic,
                        links,
                        data.papers
                      );
                      
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
            <strong>Summary:</strong>
            <div style={{ marginTop: '4px', color: '#555', lineHeight: '1.5' }}>
              <ReactMarkdown>{selectedPaper.abstract || 'No summary available.'}</ReactMarkdown>
            </div>
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
              unpinNode();
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
                  year: paper.year,
                  publication_date: paper.publication_date,
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