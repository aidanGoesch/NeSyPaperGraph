import React, { useEffect, useMemo, useState } from "react";
import ClusterTree from "./ClusterTree";
import PaperWorkbenchList from "./PaperWorkbenchList";
import ThemeNotebook from "./ThemeNotebook";
import ToReadInbox from "./ToReadInbox";
import ThemeAssignmentModal from "./ThemeAssignmentModal";
import { buildTopicClusters } from "../../utils/clustering";

function filterPapers(graphData, selectedCluster, selectedTreeNode) {
    const papers = graphData?.papers || [];
    if (selectedTreeNode?.topics?.length) {
        return papers.filter((paper) =>
            (paper.topics || []).some((topic) =>
                selectedTreeNode.topics.includes(topic)
            )
        );
    }
    if (!selectedCluster) return papers;
    return papers.filter((paper) =>
        (paper.topics || []).some((topic) => selectedCluster.topics.includes(topic))
    );
}

export default function TopicWorkspace({
    graphData,
    workspaceStore,
    onFocusPaper,
    onSetGraphHighlight,
}) {
    const { state, actions, selectors } = workspaceStore;
    const { clusters } = useMemo(() => buildTopicClusters(graphData), [graphData]);

    const [selectedClusterId, setSelectedClusterId] = useState(null);
    const [selectedTreeNode, setSelectedTreeNode] = useState(null);
    const [hasAutoSelectedCluster, setHasAutoSelectedCluster] = useState(false);
    const [selectedThemeId, setSelectedThemeId] = useState(null);
    const [requestedPaperTitle, setRequestedPaperTitle] = useState(null);
    const [isThemeModalOpen, setIsThemeModalOpen] = useState(false);
    const [themeModalPaperTitle, setThemeModalPaperTitle] = useState(null);

    useEffect(() => {
        if (!hasAutoSelectedCluster && !selectedClusterId && clusters.length > 0) {
            setSelectedClusterId(clusters[0].id);
            setHasAutoSelectedCluster(true);
        }
    }, [clusters, hasAutoSelectedCluster, selectedClusterId]);

    const selectedCluster =
        clusters.find((cluster) => cluster.id === selectedClusterId) || null;
    const filteredPapers = filterPapers(graphData, selectedCluster, selectedTreeNode);
    const themeQueueItems = selectedThemeId
        ? state.readingItems.filter((item) => item.linkedThemeId === selectedThemeId)
        : [];
    const hasActiveFilter = Boolean(selectedClusterId || selectedTreeNode);

    useEffect(() => {
        const highlightNodes = selectedTreeNode
            ? [...selectedTreeNode.topics, ...filteredPapers.map((paper) => paper.title)]
            : selectedCluster
              ? [...selectedCluster.topics, ...filteredPapers.map((paper) => paper.title)]
              : filteredPapers.map((paper) => paper.title);
        onSetGraphHighlight({
            nodes: Array.from(new Set(highlightNodes)),
        });
    }, [filteredPapers, onSetGraphHighlight, selectedCluster, selectedTreeNode]);

    return (
        <div className="topic-workspace">
            <div className="topic-workspace-grid">
                <ClusterTree
                    clusters={clusters}
                    selectedClusterId={selectedClusterId}
                    selectedNodeId={selectedTreeNode?.id || null}
                    onSelectCluster={(clusterId) => {
                        setSelectedClusterId(clusterId);
                        setSelectedTreeNode(null);
                    }}
                    onSelectNode={(node) => {
                        setSelectedTreeNode(node);
                    }}
                />
                <PaperWorkbenchList
                    papers={filteredPapers}
                    selectedTopic={
                        selectedTreeNode?.topics?.length === 1
                            ? selectedTreeNode.topics[0]
                            : null
                    }
                    selectedTopicLabel={
                        selectedTreeNode
                            ? selectedTreeNode.topics?.length === 1
                                ? selectedTreeNode.topics[0]
                                : `${selectedTreeNode.topics.length} selected topics`
                            : null
                    }
                    hasActiveFilter={hasActiveFilter}
                    onClearFilters={() => {
                        setSelectedClusterId(null);
                        setSelectedTreeNode(null);
                    }}
                    onFocusPaper={onFocusPaper}
                    requestedPaperTitle={requestedPaperTitle}
                    onOpenThemeAssignmentModal={(paperTitle) => {
                        setThemeModalPaperTitle(paperTitle);
                        setIsThemeModalOpen(true);
                    }}
                    getPaperAnnotation={selectors.getPaperAnnotation}
                    onUpdatePaperAnnotation={actions.upsertPaperAnnotation}
                />
                <ThemeNotebook
                    themeNotes={state.themeNotes}
                    selectedThemeId={selectedThemeId}
                    themeQueueItems={themeQueueItems}
                    onSelectTheme={setSelectedThemeId}
                    onUpsertTheme={actions.upsertThemeNote}
                    onSelectThemePaper={(paperTitle) => {
                        setSelectedClusterId(null);
                        setSelectedTreeNode(null);
                        setRequestedPaperTitle(paperTitle);
                    }}
                />
            </div>
            <ToReadInbox
                readingItems={state.readingItems}
                topics={graphData?.topics || []}
                themeNotes={state.themeNotes}
                onAddReadingItem={actions.addReadingItem}
                onUpdateReadingItem={actions.updateReadingItem}
                onRemoveReadingItem={actions.removeReadingItem}
                onFocusPaper={onFocusPaper}
            />
            {isThemeModalOpen && themeModalPaperTitle && (
                <ThemeAssignmentModal
                    paperTitle={themeModalPaperTitle}
                    themeNotes={state.themeNotes}
                    onClose={() => {
                        setIsThemeModalOpen(false);
                        setThemeModalPaperTitle(null);
                    }}
                    onSave={(themeIds) => {
                        actions.setPaperThemeMembership(themeModalPaperTitle, themeIds);
                        setIsThemeModalOpen(false);
                        setThemeModalPaperTitle(null);
                    }}
                />
            )}
        </div>
    );
}
