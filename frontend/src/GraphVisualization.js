import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

const GraphVisualization = ({ data, isDarkMode }) => {
  const svgRef = useRef();
  const containerRef = useRef();
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [panelPosition, setPanelPosition] = useState({ x: 0, y: 0 });
  const selectedNodeRef = useRef(null);
  const simulationRef = useRef(null);
  const nodesRef = useRef([]);

  useEffect(() => {
    if (!data) return;

    const container = containerRef.current;
    const svg = d3.select(svgRef.current);
    
    // Only clear if this is the first render or data changed
    if (!simulationRef.current) {
      svg.selectAll("*").remove();
    }

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
      .scaleExtent([0.5, 3])
      .on("zoom", (event) => {
        g.attr("transform", event.transform);
        updatePanelPosition(); // move with zoom/pan
      })
      .filter(function(event) {
        return !(event.target && event.target.closest && event.target.closest('.info-panel'));
      });

    svg.call(zoom);

    const link = g.append("g")
      .selectAll("line")
      .data(links)
      .enter().append("line")
      .attr("stroke", d => d.type === 'semantic' ? (isDarkMode ? "#888" : "#999") : (isDarkMode ? "#ccc" : "#000"))
      .attr("stroke-width", d => d.type === 'semantic' ? ((d.weight - 0.25) * 5) : 5)
      .attr("stroke-opacity", 0.6)
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
      .on("click", function(event, d) {
        event.stopPropagation();
        if (d.type !== 'paper') return;

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

        setSelectedPaper({
          title: d.title,
          authors: d.authors || ['Dr. Jane Smith', 'Dr. John Doe', 'Dr. Alice Johnson'],
          year: d.year || 2024,
          citations: d.citations || Math.floor(Math.random() * 500),
          abstract: d.abstract || 'This is a dummy abstract for the paper...'
        });

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
      .text(d => d.id)
      .attr("font-size", 12)
      .attr("font-family", "Arial, sans-serif")
      .attr("fill", isDarkMode ? "#ccc" : "#333")
      .attr("text-anchor", "middle")
      .attr("dy", -20)
      .style("pointer-events", "none")
      .style("transition", "fill 0.5s ease");

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
      .attr("fill", isDarkMode ? "#ccc" : "#333");
  }, [isDarkMode]);



  return (
    <div ref={containerRef} style={{ width: '100%', height: '100vh', position: 'relative', backgroundColor: 'white' }}>
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
            <strong>Abstract:</strong>
            <p style={{ marginTop: '4px', color: '#555', lineHeight: '1.5' }}>
              {selectedPaper.abstract}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default GraphVisualization;