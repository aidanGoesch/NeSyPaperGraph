import React, { useEffect, useMemo, useRef, useState, forwardRef, useImperativeHandle } from 'react';
import ReactMarkdown from 'react-markdown';
import * as d3 from 'd3';

const PAPER_NODE_RADIUS = 10;
const TOPIC_NODE_MIN_RADIUS = 18;
const TOPIC_NODE_MAX_RADIUS = 50;
const TOPIC_SIZE_EXPONENT = 1.35;
const COLLISION_PADDING = 12;
const OVERVIEW_SCALE = 0.18;
const FOCUS_SCALE = 0.72;
const ACTIVATION_SOURCE_FOCUS_SCALE = 0.56;
const ACTIVATION_CONTEXT_PADDING = 1200;
const ACTIVATION_CONTEXT_MIN_SPAN = 820;
const ACTIVATION_CONTEXT_MIN_SCALE = 0.1;
const ACTIVATION_CONTEXT_MAX_SCALE = 0.28;
const ACTIVATION_SOURCE_HOLD_MS = 230;
const LABEL_HOVER_DELAY_MS = 250;
const GRAPH_COLORS = {
  accentBlue: "#177cbc",
  gray1: "#44454d",
  gray2: "#31323a",
  gray3: "#292a2f",
  gray4: "#232427",
  gray5: "#1b1c1f",
  text: "#d8dce6",
  stroke: "#0f1013",
};
const ACTIVATION_COLOR = "#4da3db";
const HIGHLIGHT_COLOR = "#f97316";

const normalizeToken = (value) => String(value || '').trim().toLowerCase();

const scoreLocalSimilarity = (seedPaper, candidatePaper) => {
  if (!seedPaper || !candidatePaper) return 0;
  const seedTopics = new Set((seedPaper.topics || []).map(normalizeToken).filter(Boolean));
  const candidateTopics = new Set((candidatePaper.topics || []).map(normalizeToken).filter(Boolean));
  let topicOverlap = 0;
  seedTopics.forEach((topic) => {
    if (candidateTopics.has(topic)) topicOverlap += 1;
  });

  const seedAuthors = new Set((seedPaper.authors || []).map(normalizeToken).filter(Boolean));
  const candidateAuthors = new Set((candidatePaper.authors || []).map(normalizeToken).filter(Boolean));
  let authorOverlap = 0;
  seedAuthors.forEach((author) => {
    if (candidateAuthors.has(author)) authorOverlap += 1;
  });

  const seedYear = Number(seedPaper.year || seedPaper.publication_date || 0) || null;
  const candidateYear = Number(candidatePaper.year || candidatePaper.publication_date || 0) || null;
  const yearScore =
    seedYear && candidateYear ? Math.max(0, 1 - Math.min(Math.abs(seedYear - candidateYear), 8) / 8) : 0;

  return topicOverlap * 3 + authorOverlap * 2 + yearScore;
};

const buildLocalGraphRecommendations = (seedPaper, graphPapers, limit = 8) => {
  if (!seedPaper || !Array.isArray(graphPapers)) return [];
  const seedTitle = normalizeToken(seedPaper.title);
  return graphPapers
    .filter((paper) => normalizeToken(paper.title) && normalizeToken(paper.title) !== seedTitle)
    .map((paper) => ({
      ...paper,
      _localScore: scoreLocalSimilarity(seedPaper, paper),
    }))
    .filter((paper) => paper._localScore > 0)
    .sort((left, right) => right._localScore - left._localScore)
    .slice(0, Math.max(1, limit))
    .map((paper) => ({
      paperId: paper.paperId || null,
      title: paper.title,
      authors: paper.authors || [],
      year: paper.year || paper.publication_date || null,
      venue: paper.venue || null,
      abstract: paper.abstract || '',
      source: 'graph',
      reason: 'Similar to selected paper in your graph',
    }));
};

const interleaveRecommendations = (localItems, externalItems, limit = 8) => {
  const merged = [];
  const seenTitles = new Set();
  const locals = Array.isArray(localItems) ? localItems : [];
  const externals = Array.isArray(externalItems) ? externalItems : [];
  let localIdx = 0;
  let externalIdx = 0;
  while (merged.length < limit && (localIdx < locals.length || externalIdx < externals.length)) {
    if (localIdx < locals.length) {
      const nextLocal = locals[localIdx++];
      const key = normalizeToken(nextLocal?.title);
      if (key && !seenTitles.has(key)) {
        seenTitles.add(key);
        merged.push(nextLocal);
        if (merged.length >= limit) break;
      }
    }
    if (externalIdx < externals.length) {
      const nextExternal = externals[externalIdx++];
      const key = normalizeToken(nextExternal?.title);
      if (key && !seenTitles.has(key)) {
        seenTitles.add(key);
        merged.push({ ...nextExternal, source: nextExternal?.source || 'semanticScholar' });
      }
    }
  }
  return merged.slice(0, limit);
};

const recommendationSourceLabel = (source) =>
  source === 'graph' ? 'In graph' : 'Semantic Scholar';

const recommendationBrowserUrl = (paper) => {
  if (paper?.url) return paper.url;
  if (paper?.paperId) return `https://www.semanticscholar.org/paper/${paper.paperId}`;
  return '';
};

const GraphVisualization = forwardRef(({ data, isDarkMode, onShowArchitecture, onTopicClick, highlightPath, activationFrame, hasChatOverlay = false, apiBase, apiFetch, onAddRecommendationToReadingList }, ref) => {
  const svgRef = useRef();
  const containerRef = useRef();
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [panelPosition, setPanelPosition] = useState({ x: 0, y: 0 });
  const [showSemanticEdges, setShowSemanticEdges] = useState(true);
  const [showMenu, setShowMenu] = useState(false);
  const [semanticThreshold, setSemanticThreshold] = useState(0);
  const [similarPapers, setSimilarPapers] = useState([]);
  const [similarState, setSimilarState] = useState('idle');
  const [similarError, setSimilarError] = useState('');
  const [expandedSimilarKey, setExpandedSimilarKey] = useState(null);
  const selectedNodeRef = useRef(null);
  const pinnedNodeRef = useRef(null);
  const simulationRef = useRef(null);
  const nodesRef = useRef([]);
  const zoomRef = useRef(null);
  const hoveredLabelTimeoutRef = useRef(null);
  const activationCameraStateRef = useRef({
    key: "",
    didSourceFocus: false,
    didContextFit: false,
  });
  const activationCameraTimeoutRef = useRef(null);
  const [hoveredLabelNodeId, setHoveredLabelNodeId] = useState(null);
  const panelMetrics = { width: 350, maxHeight: 400, margin: 12 };
  const paperByTitle = useMemo(
    () => new Map((data?.papers || []).map((paper) => [paper.title, paper])),
    [data]
  );
  const palette = {
    background: GRAPH_COLORS.gray5,
    link: GRAPH_COLORS.gray1,
    semanticLink: GRAPH_COLORS.gray2,
    paper: GRAPH_COLORS.accentBlue,
    topic: GRAPH_COLORS.gray3,
    label: GRAPH_COLORS.text,
    stroke: GRAPH_COLORS.stroke,
  };
  const activationNodeMap = activationFrame?.nodes || {};
  const nodeColors = activationFrame?.nodeColors || {};
  const traversedEdges = activationFrame?.traversedEdges || {};
  const edgeColors = activationFrame?.edgeColors || {};
  const maxTraversalCount = Number(activationFrame?.maxTraversalCount || 0);
  const activePromptColor = activationFrame?.promptColor || ACTIVATION_COLOR;
  const activeSeedColor = d3.interpolateRgb(activePromptColor, "#ffffff")(0.35);
  const getNodeActivation = (nodeId) => activationNodeMap?.[nodeId] || null;

  const getNodeFillColor = (node, highlightedNodeSet = new Set()) => {
    if (highlightedNodeSet.has(node.id)) {
      return HIGHLIGHT_COLOR;
    }
    const activation = getNodeActivation(node.id);
    const baseColor = node.type === 'paper' ? palette.paper : palette.topic;
    if (!activation) return baseColor;
    const nodePromptColor = nodeColors[node.id] || activePromptColor;
    if (activation.stage === "seed") {
      const seedColor = d3.interpolateRgb(nodePromptColor, "#ffffff")(0.35);
      return d3.interpolateRgb(baseColor, seedColor)(Math.min(1, Math.max(0, activation.score || 0.35)));
    }
    return d3.interpolateRgb(baseColor, nodePromptColor)(Math.min(1, Math.max(0, activation.score || 0)));
  };

  const getNodeStrokeColor = (node, highlightedNodeSet = new Set()) => {
    if (highlightedNodeSet.has(node.id)) return "#fde68a";
    const activation = getNodeActivation(node.id);
    const nodePromptColor = nodeColors[node.id] || activePromptColor;
    if (activation?.stage === "seed") {
      return d3.interpolateRgb(nodePromptColor, "#ffffff")(0.25);
    }
    if (activation) return nodePromptColor;
    return palette.stroke;
  };

  const getNodeStrokeWidth = (node, highlightedNodeSet = new Set()) => {
    if (highlightedNodeSet.has(node.id)) return 4;
    const activation = getNodeActivation(node.id);
    if (!activation) return 1.6;
    return 2.3 + (Math.min(1, Math.max(0, activation.score || 0)) * 6.2);
  };

  const getNodeRadius = (node) => {
    const activation = getNodeActivation(node.id);
    const baseRadius = node.radius || PAPER_NODE_RADIUS;
    if (!activation) return baseRadius;
    return baseRadius + (Math.min(1, Math.max(0, activation.score || 0)) * 8.4);
  };

  const getLabelOpacity = (node, highlightedNodeSet = new Set()) => {
    if (hoveredLabelNodeId === node.id) return 1;
    if (node.type !== "topic") return 0;
    if (highlightedNodeSet.has(node.id)) return 1;
    const activation = getNodeActivation(node.id);
    if (!activation) return 0;
    return Math.min(1, 0.25 + (Math.max(0, Number(activation.score) || 0) * 0.75));
  };

  const getEdgeActivationScore = (edge) => {
    const sourceId = edge?.source?.id || edge?.source;
    const targetId = edge?.target?.id || edge?.target;
    const sourceScore = Number(getNodeActivation(sourceId)?.score || 0);
    const targetScore = Number(getNodeActivation(targetId)?.score || 0);
    return Math.max(sourceScore, targetScore);
  };

  const getEdgeTraversalRatio = (edge) => {
    const sourceId = edge?.source?.id || edge?.source;
    const targetId = edge?.target?.id || edge?.target;
    const edgeKey = [String(sourceId || ""), String(targetId || "")].sort().join("___");
    const traversalCount = Number(traversedEdges[edgeKey] || 0);
    if (maxTraversalCount <= 0) return 0;
    // Boost lower traversal counts so activation remains visible when zoomed out.
    return Math.min(1, Math.sqrt(traversalCount / maxTraversalCount));
  };

  const getEdgeVisualActivationRatio = (edge) => {
    const traversalRatio = getEdgeTraversalRatio(edge);
    const activationRatio = Math.min(1, Math.max(0, getEdgeActivationScore(edge) || 0));
    return Math.max(traversalRatio, activationRatio * 0.55);
  };

  const getEdgeStrokeColor = (edge, isHighlightedPathEdge) => {
    if (isHighlightedPathEdge) return "#fde68a";
    const sourceId = edge?.source?.id || edge?.source;
    const targetId = edge?.target?.id || edge?.target;
    const edgeKey = [String(sourceId || ""), String(targetId || "")].sort().join("___");
    const traversalRatio = getEdgeVisualActivationRatio(edge);
    const baseColor = edge.type === 'semantic' ? palette.semanticLink : palette.link;
    if (traversalRatio <= 0) return baseColor;
    const edgePromptColor = edgeColors[edgeKey] || activePromptColor;
    return d3.interpolateRgb(baseColor, edgePromptColor)(Math.min(1, traversalRatio));
  };

  const getEdgeStrokeWidth = (edge, isHighlightedPathEdge) => {
    if (isHighlightedPathEdge) return 8;
    const traversalRatio = getEdgeVisualActivationRatio(edge);
    const baseWidth = edge.type === 'semantic' ? 2.2 : 1.8;
    return baseWidth + (traversalRatio * 10.8);
  };

  const getEdgeOpacity = (edge, isHighlightedPathEdge) => {
    if (isHighlightedPathEdge) return 0.95;
    const traversalRatio = getEdgeVisualActivationRatio(edge);
    if (edge.type === 'semantic') {
      if (!showSemanticEdges || Number(edge.weight || 0) < semanticThreshold) return 0;
      return Math.min(0.97, 0.34 + (traversalRatio * 0.56));
    }
    return Math.min(0.96, 0.3 + (traversalRatio * 0.6));
  };

  const toSelectedPaper = (paper) => ({
    title: paper?.title || "",
    authors: paper?.authors || [],
    year: paper?.year,
    publication_date: paper?.publication_date,
    abstract: paper?.abstract || "No summary available.",
    topics: paper?.topics || [],
    semanticScholarPaperId: paper?.semanticScholarPaperId || null,
  });

  const parseResponseError = async (response) => {
    try {
      const payload = await response.json();
      return payload?.detail || payload?.error || `HTTP ${response.status}`;
    } catch {
      return `HTTP ${response.status}`;
    }
  };

  const requestSimilarPapers = async () => {
    if (!apiBase || !apiFetch || !selectedPaper) {
      setSimilarError("Recommendation API is unavailable.");
      setSimilarState("error");
      return;
    }
    setSimilarState("loading");
    setSimilarError("");
    setSimilarPapers([]);
    try {
      const requestBody = JSON.stringify({
        semanticScholarPaperId: selectedPaper.semanticScholarPaperId || null,
        title: selectedPaper.title || null,
        authors: selectedPaper.authors || [],
        year: selectedPaper.year || selectedPaper.publication_date || null,
        abstract: selectedPaper.abstract || null,
        limit: 8,
      });
      let response = await apiFetch(`${apiBase}/api/recommendations/paper`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: requestBody,
      });
      if (response.status === 404) {
        response = await apiFetch(`${apiBase}/api/workspace/recommendations/paper`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: requestBody,
        });
      }
      if (!response.ok) {
        throw new Error(await parseResponseError(response));
      }
      const payload = await response.json();
      const external = Array.isArray(payload?.results) ? payload.results : [];
      const local = buildLocalGraphRecommendations(selectedPaper, data?.papers || [], 8);
      setSimilarPapers(interleaveRecommendations(local, external, 8));
      setExpandedSimilarKey(null);
      setSimilarState("success");
    } catch (error) {
      setSimilarError(`Failed to load recommendations: ${error?.message || "Unknown error"}`);
      setSimilarState("error");
    }
  };

  useEffect(() => {
    setSimilarPapers([]);
    setSimilarState("idle");
    setSimilarError("");
    setExpandedSimilarKey(null);
  }, [selectedPaper?.title]);

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

  const buildTopicPaperCounts = (topics, links, paperTitles) => {
    const topicToConnectedPapers = new Map(
      (topics || []).map((topic) => [topic, new Set()])
    );

    (links || []).forEach((link) => {
      if (link?.type === 'semantic') return;

      const sourceId = normalizeLinkEndpoint(link.source);
      const targetId = normalizeLinkEndpoint(link.target);
      const sourceIsTopic = topicToConnectedPapers.has(sourceId);
      const targetIsTopic = topicToConnectedPapers.has(targetId);

      if (sourceIsTopic && paperTitles.has(targetId)) {
        topicToConnectedPapers.get(sourceId).add(targetId);
      }
      if (targetIsTopic && paperTitles.has(sourceId)) {
        topicToConnectedPapers.get(targetId).add(sourceId);
      }
    });

    return new Map(
      Array.from(topicToConnectedPapers.entries()).map(([topic, connectedPapers]) => [
        topic,
        connectedPapers.size,
      ])
    );
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

  function getViewportMetrics() {
    const container = containerRef.current;
    if (!container) {
      return {
        width: 0,
        height: 0,
        rightInset: 0,
        mapWidth: 0,
        centerX: 0,
        centerY: 0,
      };
    }
    const width = container.clientWidth;
    const height = container.clientHeight;
    const rightInset = hasChatOverlay ? Math.min(430, width * 0.36) : 0;
    const mapWidth = Math.max(120, width - rightInset);
    return {
      width,
      height,
      rightInset,
      mapWidth,
      centerX: mapWidth / 2,
      centerY: height / 2,
    };
  }

  const clampPanelToContainer = (x, y) => {
    const container = containerRef.current;
    if (!container) return { x, y };
    const viewport = getViewportMetrics();
    const maxX = Math.max(
      panelMetrics.margin,
      viewport.mapWidth - panelMetrics.width - panelMetrics.margin
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

  const getOverviewTransformForSize = (centerX, centerY) => {
    const cx = centerX;
    const cy = centerY;
    return d3.zoomIdentity
      .translate(cx, cy)
      .scale(OVERVIEW_SCALE)
      .translate(-cx, -cy);
  };

  const transitionToTransform = (transform, durationMs = 650) => {
    if (!zoomRef.current || !svgRef.current || !transform) return;
    const svg = d3.select(svgRef.current);
    svg.interrupt();
    svg.transition().duration(durationMs).call(zoomRef.current.transform, transform);
  };

  const resetOverviewZoom = (durationMs = 650) => {
    const viewport = getViewportMetrics();
    transitionToTransform(
      getOverviewTransformForSize(viewport.centerX, viewport.centerY),
      durationMs
    );
  };

  const zoomToNodes = (nodeIds, options = {}) => {
    if (!zoomRef.current || !containerRef.current || !Array.isArray(nodeIds) || nodeIds.length === 0) {
      return;
    }
    const {
      durationMs = 700,
      padding = 720,
      minSpan = 260,
      minScale = 0.2,
      maxScale = 0.72,
    } = options;
    const targetNodes = nodesRef.current.filter((node) => nodeIds.includes(node.id));
    if (targetNodes.length === 0) return;

    const viewport = getViewportMetrics();
    const width = viewport.mapWidth;
    const height = viewport.height;
    const minX = Math.min(...targetNodes.map((node) => node.x));
    const maxX = Math.max(...targetNodes.map((node) => node.x));
    const minY = Math.min(...targetNodes.map((node) => node.y));
    const maxY = Math.max(...targetNodes.map((node) => node.y));
    const spanX = Math.max(minSpan, maxX - minX + padding);
    const spanY = Math.max(minSpan, maxY - minY + padding);
    const scale = Math.max(
      minScale,
      Math.min(maxScale, Math.min(width / spanX, height / spanY))
    );
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const transform = d3.zoomIdentity
      .translate(viewport.centerX - centerX * scale, viewport.centerY - centerY * scale)
      .scale(scale);
    transitionToTransform(transform, durationMs);
  };

  const focusNodeWithZoom = (node, durationMs = 750, scale = FOCUS_SCALE) => {
    if (!node || !containerRef.current) return;
    const viewport = getViewportMetrics();
    const newX = -node.x * scale + viewport.centerX;
    const newY = -node.y * scale + viewport.centerY;
    const transform = d3.zoomIdentity.translate(newX, newY).scale(scale);
    transitionToTransform(transform, durationMs);
    return transform;
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
      const targetTransform = focusNodeWithZoom(topicNode);
      
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
      const targetTransform = focusNodeWithZoom(paperNode);
      
      // Show paper panel near node, constrained inside container.
      const nextPanelPosition = getPanelPositionForNode(paperNode, targetTransform);
      if (nextPanelPosition) setPanelPosition(nextPanelPosition);
      
      // Find the paper data
      const paper = paperByTitle.get(paperTitle);
      if (paper) {
        setSelectedTopic(null);
        setSelectedPaper(toSelectedPaper(paper));
      }
    },
    resetOverviewZoom: () => {
      resetOverviewZoom();
    },
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
    const viewport = getViewportMetrics();

    svg.attr("width", width).attr("height", height)
      .style("background-color", palette.background)
      .style("transition", "background-color 0.5s ease");

    // Update existing elements if simulation exists
    if (simulationRef.current) {
      svg.selectAll("line")
        .transition()
        .duration(500)
        .attr("stroke", d => d.type === 'semantic' ? palette.semanticLink : palette.link);
      
      svg.selectAll("text")
        .transition()
        .duration(500)
        .attr("fill", palette.label);
      
      return;
    }

    const links = buildLinks(data.papers, data.edges);
    const paperTitles = new Set((data.papers || []).map((paper) => paper.title));
    const topicPaperCounts = buildTopicPaperCounts(data.topics, links, paperTitles);
    const topicCounts = Array.from(topicPaperCounts.values());
    const minTopicCount = topicCounts.length > 0 ? Math.min(...topicCounts) : 0;
    const maxTopicCount = topicCounts.length > 0 ? Math.max(...topicCounts) : 0;

    const getTopicNodeRadius =
      minTopicCount === maxTopicCount
        ? () =>
            minTopicCount <= 1
              ? TOPIC_NODE_MIN_RADIUS
              : (TOPIC_NODE_MIN_RADIUS + TOPIC_NODE_MAX_RADIUS) / 2
        : d3
            .scalePow()
            .exponent(TOPIC_SIZE_EXPONENT)
            .domain([minTopicCount, maxTopicCount])
            .range([TOPIC_NODE_MIN_RADIUS, TOPIC_NODE_MAX_RADIUS]);

    const nodes = [
      ...data.papers.map((paper) => ({
        id: paper.title,
        type: 'paper',
        title: paper.title,
        radius: PAPER_NODE_RADIUS,
      })),
      ...data.topics.map((topic) => {
        const paperCount = topicPaperCounts.get(topic) || 0;
        return {
          id: topic,
          type: 'topic',
          paperCount,
          radius: Math.max(TOPIC_NODE_MIN_RADIUS, getTopicNodeRadius(paperCount)),
        };
      }),
    ];

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
      .force("link", d3.forceLink(links)
        .id(d => d.id)
        .distance((d) => {
          const targetId = d?.target?.id || d?.target;
          const degree = links.filter((l) => {
            const sourceId = l?.source?.id || l?.source;
            return sourceId === targetId;
          }).length;
          return Math.max(60, 150 - degree * 5);
        }))
      .force("charge", d3.forceManyBody().strength(-400))
      .force("center", d3.forceCenter(viewport.centerX, viewport.centerY))
      .force("collision", d3.forceCollide().radius((d) => (d.radius || PAPER_NODE_RADIUS) + COLLISION_PADDING))
      .alphaDecay(0.01);

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
    svg.call(
      zoom.transform,
      getOverviewTransformForSize(viewport.centerX, viewport.centerY)
    );

    const link = g.append("g")
      .selectAll("line")
      .data(links)
      .enter().append("line")
      .attr("stroke", d => getEdgeStrokeColor(d, isEdgeInHighlightedPath(d)))
      .attr("stroke-width", d => getEdgeStrokeWidth(d, isEdgeInHighlightedPath(d)))
      .attr("stroke-opacity", d => getEdgeOpacity(d, isEdgeInHighlightedPath(d)))
      .style("filter", d => {
        const traversalRatio = getEdgeVisualActivationRatio(d);
        return traversalRatio > 0.24 ? `drop-shadow(0 0 2px ${activePromptColor}66)` : "none";
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
      .attr("r", (d) => getNodeRadius(d))
      .attr("fill", d => getNodeFillColor(d, highlightedNodeSet))
      .attr("stroke", d => getNodeStrokeColor(d, highlightedNodeSet))
      .attr("stroke-width", d => getNodeStrokeWidth(d, highlightedNodeSet))
      .style("filter", d => {
        const activation = getNodeActivation(d.id);
        if (!activation) return "none";
        const nodePromptColor = nodeColors[d.id] || activePromptColor;
        const alpha = Math.min(0.7, 0.22 + (Number(activation.score || 0) * 0.5));
        const hexAlpha = Math.round(alpha * 255).toString(16).padStart(2, "0");
        return `drop-shadow(0 0 7px ${nodePromptColor}${hexAlpha})`;
      })
      .style("cursor", "pointer")
      .on("mouseover", function(event, d) {
        if (hoveredLabelTimeoutRef.current) {
          window.clearTimeout(hoveredLabelTimeoutRef.current);
        }
        hoveredLabelTimeoutRef.current = window.setTimeout(() => {
          setHoveredLabelNodeId(d.id);
        }, LABEL_HOVER_DELAY_MS);
      })
      .on("mouseout", function(event, d) {
        if (hoveredLabelTimeoutRef.current) {
          window.clearTimeout(hoveredLabelTimeoutRef.current);
          hoveredLabelTimeoutRef.current = null;
        }
        setHoveredLabelNodeId((previous) => (previous === d.id ? null : previous));
      })
      .on("click", function(event, d) {
        event.stopPropagation();
        
        selectedNodeRef.current = d; // keep live reference

        const nextPanelPosition = getPanelPositionForNode(d);
        if (nextPanelPosition) setPanelPosition(nextPanelPosition);

        if (d.type === 'paper') {
          setSelectedTopic(null);
          const selectedPaperData = paperByTitle.get(d.id);
          if (selectedPaperData) {
            setSelectedPaper(toSelectedPaper(selectedPaperData));
          }
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
        focusNodeWithZoom(d);
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
      .attr("fill", palette.label)
      .attr("text-anchor", "middle")
      .attr("dy", d => d.type === 'topic' ? 5 : -20)
      .style("pointer-events", "none")
      .style("opacity", d => getLabelOpacity(d, highlightedNodeSet))
      .style("transition", "opacity 0.3s ease, fill 0.5s ease")
      .style("paint-order", "stroke fill")
      .style("stroke", palette.background)
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

    let preDragVelocityDecay = simulation.velocityDecay();

    function dragstarted(event, d) {
      const syrupAnchors = new Map();
      nodes.forEach((node) => {
        if (node.id !== d.id) {
          syrupAnchors.set(node.id, { x: node.x, y: node.y });
        }
      });

      preDragVelocityDecay = simulation.velocityDecay();
      simulation
        .force(
          "syrupX",
          d3
            .forceX((node) => syrupAnchors.get(node.id)?.x ?? node.x)
            .strength((node) => (node.id === d.id ? 0 : 0.12))
        )
        .force(
          "syrupY",
          d3
            .forceY((node) => syrupAnchors.get(node.id)?.y ?? node.y)
            .strength((node) => (node.id === d.id ? 0 : 0.12))
        )
        .velocityDecay(0.9)
        .alphaTarget(0.1)
        .restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event, d) {
      simulation
        .force("syrupX", null)
        .force("syrupY", null)
        .velocityDecay(preDragVelocityDecay)
        .alphaTarget(0);
      d.fx = event.x;
      d.fy = event.y;
    }

    function updatePanelPosition() {
      if (!selectedNodeRef.current) return;
      const nextPanelPosition = getPanelPositionForNode(selectedNodeRef.current);
      if (nextPanelPosition) setPanelPosition(nextPanelPosition);
    }

    const handleResize = () => {
      const newWidth = container.clientWidth;
      const newHeight = container.clientHeight;
      const resizedViewport = getViewportMetrics();
      svg.attr("width", newWidth).attr("height", newHeight);
      simulation.force(
        "center",
        d3.forceCenter(resizedViewport.centerX, resizedViewport.centerY)
      );
      simulation.alpha(0.3).restart();
    };

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      if (hoveredLabelTimeoutRef.current) {
        window.clearTimeout(hoveredLabelTimeoutRef.current);
        hoveredLabelTimeoutRef.current = null;
      }
      unpinNode();
      if (simulationRef.current) {
        simulationRef.current.stop();
      }
    };
  }, [data]);

  useEffect(() => {
    if (!simulationRef.current || !containerRef.current || !svgRef.current) return;
    const viewport = getViewportMetrics();
    simulationRef.current.force(
      "center",
      d3.forceCenter(viewport.centerX, viewport.centerY)
    );
    // Keep this subtle to avoid a visible hitch when switching views.
    simulationRef.current.alpha(Math.max(0.08, simulationRef.current.alpha())).restart();
  }, [hasChatOverlay]);

  // Separate effect for theme changes
  useEffect(() => {
    if (!simulationRef.current) return;
    
    const svg = d3.select(svgRef.current);
    svg.style("background-color", palette.background);
    
    svg.selectAll("line")
      .attr("stroke", d => getEdgeStrokeColor(d, false))
      .attr("stroke-width", d => getEdgeStrokeWidth(d, false))
      .attr("stroke-opacity", d => getEdgeOpacity(d, false));
    
    svg.selectAll("text")
      .attr("fill", palette.label)
      .style("stroke", palette.background);
  }, [isDarkMode, palette.background, palette.label, palette.link, palette.semanticLink, semanticThreshold, showSemanticEdges]);

  // Update highlighted nodes/edges without recreating simulation.
  useEffect(() => {
    if (!simulationRef.current || !data || !svgRef.current) return;

    const highlightedNodeSet = new Set(
      Array.isArray(highlightPath?.nodes) ? highlightPath.nodes : []
    );
    const pathNodes =
      highlightPath?.mode === "path" && Array.isArray(highlightPath?.nodes)
        ? highlightPath.nodes
        : [];
    const isEdgeInHighlightedPath = (edge) => {
      if (pathNodes.length < 2) return false;
      const sourceId = edge?.source?.id || edge?.source;
      const targetId = edge?.target?.id || edge?.target;
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

    const svg = d3.select(svgRef.current);
    const g = svg.select("g");
    if (g.empty()) return;

    g.selectAll("line")
      .attr("stroke", (d) => {
        const inPath = isEdgeInHighlightedPath(d);
        return getEdgeStrokeColor(d, inPath);
      })
      .attr("stroke-width", (d) => {
        const inPath = isEdgeInHighlightedPath(d);
        return getEdgeStrokeWidth(d, inPath);
      })
      .attr("stroke-opacity", (d) => {
        const inPath = isEdgeInHighlightedPath(d);
        return getEdgeOpacity(d, inPath);
      })
      .style("filter", (d) => {
        const traversalRatio = getEdgeVisualActivationRatio(d);
        return traversalRatio > 0.24 ? `drop-shadow(0 0 2px ${activePromptColor}66)` : "none";
      });

    g.selectAll("circle")
      .attr("fill", (d) => getNodeFillColor(d, highlightedNodeSet))
      .attr("stroke", (d) => getNodeStrokeColor(d, highlightedNodeSet))
      .attr("stroke-width", (d) => getNodeStrokeWidth(d, highlightedNodeSet))
      .attr("r", (d) => getNodeRadius(d))
      .style("filter", (d) => {
        const activation = getNodeActivation(d.id);
        if (!activation) return "none";
        const nodePromptColor = nodeColors[d.id] || activePromptColor;
        const alpha = Math.min(0.7, 0.22 + (Number(activation.score || 0) * 0.5));
        const hexAlpha = Math.round(alpha * 255).toString(16).padStart(2, "0");
        return `drop-shadow(0 0 7px ${nodePromptColor}${hexAlpha})`;
      });

    g.selectAll("text")
      .style("opacity", (d) => getLabelOpacity(d, highlightedNodeSet));
  }, [highlightPath, isDarkMode, data, activationFrame, palette.link, palette.semanticLink, semanticThreshold, showSemanticEdges, hoveredLabelNodeId]);

  // Effect for semantic edges toggle and threshold
  useEffect(() => {
    if (!simulationRef.current || !data) return;
    
    const svg = d3.select(svgRef.current);
    svg.select("g").selectAll("line")
      .attr("stroke", d => getEdgeStrokeColor(d, false))
      .attr("stroke-width", d => getEdgeStrokeWidth(d, false))
      .attr("stroke-opacity", d => getEdgeOpacity(d, false));
    
    // Update simulation forces only for layout
    const links = buildLinks(data.papers, data.edges);
    
    const activeLinks = showSemanticEdges ? 
      links.filter(d => d.type !== 'semantic' || d.weight >= semanticThreshold) : 
      links.filter(link => link.type !== 'semantic');
    
    simulationRef.current.force("link").links(activeLinks);
    simulationRef.current.alpha(0.3).restart();
  }, [semanticThreshold, showSemanticEdges, data, isDarkMode]);

  useEffect(() => {
    if (!activationFrame || !zoomRef.current) return;
    if (selectedPaper || selectedTopic) return;

    const roundKey = `${activationFrame.promptIndex || 0}-${activationFrame.roundIndex || 0}`;
    const cameraState = activationCameraStateRef.current;
    if (cameraState.key !== roundKey) {
      cameraState.key = roundKey;
      cameraState.didSourceFocus = false;
      cameraState.didContextFit = false;
      if (activationCameraTimeoutRef.current) {
        window.clearTimeout(activationCameraTimeoutRef.current);
        activationCameraTimeoutRef.current = null;
      }
    }

    const traceNodeIds = Array.isArray(activationFrame.traceNodeIds) && activationFrame.traceNodeIds.length > 0
      ? activationFrame.traceNodeIds
      : (Array.isArray(activationFrame.seedNodeIds) ? activationFrame.seedNodeIds : []);
    if (traceNodeIds.length === 0) return;

    const cameraPhase = String(activationFrame.cameraPhase || "");
    if (cameraPhase === "source_focus" && !cameraState.didSourceFocus) {
      const focusNodeId = activationFrame.focusNodeId || traceNodeIds[0];
      const focusNode = nodesRef.current.find((candidate) => candidate.id === focusNodeId);
      if (!focusNode) return;
      focusNodeWithZoom(focusNode, 220, ACTIVATION_SOURCE_FOCUS_SCALE);
      cameraState.didSourceFocus = true;
      activationCameraTimeoutRef.current = window.setTimeout(() => {
        zoomToNodes(traceNodeIds, {
          durationMs: 650,
          padding: ACTIVATION_CONTEXT_PADDING,
          minSpan: ACTIVATION_CONTEXT_MIN_SPAN,
          minScale: ACTIVATION_CONTEXT_MIN_SCALE,
          maxScale: ACTIVATION_CONTEXT_MAX_SCALE,
        });
        const latestState = activationCameraStateRef.current;
        if (latestState.key === roundKey) {
          latestState.didContextFit = true;
        }
        activationCameraTimeoutRef.current = null;
      }, ACTIVATION_SOURCE_HOLD_MS);
      return;
    }

    if (
      (cameraPhase === "zoomed_out_context" || cameraPhase === "source_focus") &&
      !cameraState.didContextFit
    ) {
      zoomToNodes(traceNodeIds, {
        durationMs: 650,
        padding: ACTIVATION_CONTEXT_PADDING,
        minSpan: ACTIVATION_CONTEXT_MIN_SPAN,
        minScale: ACTIVATION_CONTEXT_MIN_SCALE,
        maxScale: ACTIVATION_CONTEXT_MAX_SCALE,
      });
      cameraState.didContextFit = true;
    }
  }, [activationFrame, selectedPaper, selectedTopic]);

  useEffect(() => {
    return () => {
      if (activationCameraTimeoutRef.current) {
        window.clearTimeout(activationCameraTimeoutRef.current);
        activationCameraTimeoutRef.current = null;
      }
    };
  }, []);



  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'relative', backgroundColor: palette.background }}>
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
          zIndex: 1900,
          pointerEvents: 'auto'
        }}>
          <button 
            onClick={() => {
              setSelectedPaper(null);
              unpinNode();
              setPanelPosition({ x: 0, y: 0 });
              resetOverviewZoom();
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
                      focusNodeWithZoom(topicNode);
                      
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
          <div style={{ marginBottom: '12px' }}>
            <button
              type="button"
              onClick={requestSimilarPapers}
              style={{
                padding: '8px 12px',
                borderRadius: '6px',
                border: '1px solid #d0d0d0',
                backgroundColor: '#f7f7f7',
                cursor: 'pointer',
              }}
            >
              See similar papers
            </button>
          </div>
          {similarState === 'loading' && (
            <p style={{ color: '#555' }}>Loading recommendations...</p>
          )}
          {similarState === 'error' && (
            <p style={{ color: '#b00020' }}>{similarError}</p>
          )}
          {similarState === 'success' && similarPapers.length === 0 && (
            <p style={{ color: '#555' }}>
              No similar papers found for this selection.
            </p>
          )}
          {similarState === 'success' && similarPapers.length > 0 && (
            <div style={{ marginTop: '8px' }}>
              <strong>Similar papers</strong>
              <div style={{ marginTop: '8px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {similarPapers.map((paper, index) => (
                  <div
                    key={paper.paperId || `${paper.title}-${index}`}
                    style={{
                      border: '1px solid #e0e0e0',
                      backgroundColor: '#fafafa',
                      borderRadius: '6px',
                      padding: '8px',
                    }}
                  >
                    <button
                      type="button"
                      onClick={() => {
                        const cardKey = paper.paperId || paper.title || String(index);
                        setExpandedSimilarKey((previous) =>
                          previous === cardKey ? null : cardKey
                        );
                      }}
                      style={{
                        width: '100%',
                        textAlign: 'left',
                        border: 'none',
                        background: 'transparent',
                        padding: 0,
                        cursor: 'pointer',
                      }}
                    >
                      <strong style={{ display: 'block' }}>{paper.title || 'Untitled paper'}</strong>
                      <span
                        style={{
                          fontSize: '11px',
                          fontWeight: 700,
                          padding: '2px 8px',
                          borderRadius: '999px',
                          display: 'inline-flex',
                          backgroundColor:
                            paper.source === 'graph'
                              ? 'rgba(76, 175, 80, 0.18)'
                              : 'rgba(33, 150, 243, 0.18)',
                          color: paper.source === 'graph' ? '#215f25' : '#0f4d82',
                        }}
                      >
                        {recommendationSourceLabel(paper.source)}
                      </span>
                    </button>
                    {expandedSimilarKey === (paper.paperId || paper.title || String(index)) && (
                      <div style={{ marginTop: '8px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        <p style={{ margin: 0, color: '#555', fontSize: '12px' }}>
                          {(paper.authors || []).length
                            ? paper.authors.join(', ')
                            : 'Unknown authors'}{' '}
                          | {paper.year || 'Unknown year'}
                        </p>
                        <p style={{ margin: 0, color: '#555', lineHeight: 1.4, fontSize: '12px' }}>
                          {paper.abstract || 'No summary available.'}
                        </p>
                        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                          {paper.source === 'graph' && paper.title && paperByTitle.get(paper.title) && (
                            <button
                              type="button"
                              onClick={() => {
                                const localPaper = paperByTitle.get(paper.title);
                                if (localPaper) {
                                  setSelectedPaper(toSelectedPaper(localPaper));
                                }
                              }}
                              style={{
                                padding: '6px 10px',
                                borderRadius: '6px',
                                border: '1px solid #d0d0d0',
                                backgroundColor: '#fff',
                                cursor: 'pointer',
                              }}
                            >
                              Focus in graph
                            </button>
                          )}
                          {paper.source !== 'graph' && onAddRecommendationToReadingList && (
                            <button
                              type="button"
                              onClick={() => onAddRecommendationToReadingList(paper)}
                              style={{
                                padding: '6px 10px',
                                borderRadius: '6px',
                                border: '1px solid #d0d0d0',
                                backgroundColor: '#fff',
                                cursor: 'pointer',
                              }}
                            >
                              Add to reading list
                            </button>
                          )}
                          {paper.source !== 'graph' && recommendationBrowserUrl(paper) && (
                            <button
                              type="button"
                              onClick={() =>
                                window.open(
                                  recommendationBrowserUrl(paper),
                                  '_blank',
                                  'noopener,noreferrer'
                                )
                              }
                              style={{
                                padding: '6px 10px',
                                borderRadius: '6px',
                                border: '1px solid #d0d0d0',
                                backgroundColor: '#fff',
                                color: '#333',
                                fontSize: '13px',
                                cursor: 'pointer',
                              }}
                            >
                              Open in browser
                            </button>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
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
          zIndex: 1900,
          pointerEvents: 'auto'
        }}>
          <button 
            onClick={() => {
              setSelectedTopic(null);
              unpinNode();
              setPanelPosition({ x: 0, y: 0 });
              resetOverviewZoom();
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
                  focusNodeWithZoom(paperNode);
                }
                
                setSelectedTopic(null);
                const selectedPaperData = paperByTitle.get(paper.title);
                if (selectedPaperData) {
                  setSelectedPaper(toSelectedPaper(selectedPaperData));
                }
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