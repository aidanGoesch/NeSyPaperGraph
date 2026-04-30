import React, { useMemo, useState } from "react";

function collectLeafTopics(node, limit = 6) {
    if (!node) return [];
    if (!node.children || node.children.length === 0) {
        return node.topics?.slice(0, 1) || [];
    }
    const collected = [];
    for (const child of node.children) {
        const topics = collectLeafTopics(child, limit - collected.length);
        for (const topic of topics) {
            if (!collected.includes(topic)) {
                collected.push(topic);
            }
            if (collected.length >= limit) return collected;
        }
    }
    return collected;
}

function buildNodeLabel(node) {
    const hasChildren = node.children && node.children.length > 0;
    const isTopicLeaf = !hasChildren && node.topics?.length === 1;
    if (isTopicLeaf) return node.topics[0];

    const previewTopics = collectLeafTopics(node, 3);
    const remaining = Math.max((node.topics?.length || 0) - previewTopics.length, 0);
    if (previewTopics.length === 0) {
        return `${node.children.length} topics`;
    }
    return `${previewTopics.join(" • ")}${remaining > 0 ? ` + ${remaining} more` : ""}`;
}

function TreeNode({ node, depth, onSelectNode, selectedNodeId }) {
    const [isOpen, setIsOpen] = useState(depth < 2);
    const hasChildren = node.children && node.children.length > 0;
    const isTopicLeaf = !hasChildren && node.topics?.length === 1;
    const nodeLabel = buildNodeLabel(node);

    return (
        <div className="cluster-tree-node" style={{ marginLeft: depth * 10 }}>
            <button
                type="button"
                className={`cluster-tree-row ${
                    selectedNodeId === node.id ? "selected" : ""
                }`}
                onClick={() => {
                    onSelectNode(node);
                    if (hasChildren) {
                        setIsOpen((prev) => !prev);
                    }
                }}
            >
                <span className="cluster-tree-caret">
                    {hasChildren ? (isOpen ? "▾" : "▸") : "•"}
                </span>
                <span className="cluster-tree-label">
                    {nodeLabel}
                </span>
                {hasChildren && (
                    <span className="cluster-tree-count">
                        {node.children.length} next
                    </span>
                )}
                {!hasChildren && (
                    <span className="cluster-tree-count">
                        topic
                    </span>
                )}
            </button>

            {hasChildren && isOpen && (
                <div className="cluster-tree-children">
                    {node.children.map((child) => (
                        <TreeNode
                            key={child.id}
                            node={child}
                            depth={depth + 1}
                            onSelectNode={onSelectNode}
                            selectedNodeId={selectedNodeId}
                        />
                    ))}
                </div>
            )}
        </div>
    );
}

export default function ClusterTree({
    clusters,
    selectedClusterId,
    selectedNodeId,
    onSelectCluster,
    onSelectNode,
}) {
    const sortedClusters = useMemo(
        () => [...(clusters || [])].sort((a, b) => b.paperCount - a.paperCount),
        [clusters]
    );

    return (
        <section className="workspace-panel workspace-panel-left">
            <div className="workspace-panel-header">
                <h3>Topic Clusters</h3>
                <span>{sortedClusters.length} groups</span>
            </div>
            <div className="cluster-list">
                {sortedClusters.map((cluster) => (
                    <div
                        key={cluster.id}
                        className={`cluster-card ${
                            selectedClusterId === cluster.id ? "active" : ""
                        }`}
                    >
                        <button
                            type="button"
                            className="cluster-card-header"
                            onClick={() => onSelectCluster(cluster.id)}
                        >
                            <strong>{cluster.label}</strong>
                            <span>{cluster.paperCount} papers</span>
                        </button>
                        {selectedClusterId === cluster.id && (
                            <div className="cluster-card-tree">
                                <TreeNode
                                    node={cluster.tree}
                                    depth={0}
                                    onSelectNode={onSelectNode}
                                    selectedNodeId={selectedNodeId}
                                />
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </section>
    );
}
