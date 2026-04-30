function createPairKey(a, b) {
    return a < b ? `${a}__${b}` : `${b}__${a}`;
}

const HIERARCHICAL_MAX_TOPICS = 90;

function averagePairwiseSimilarity(topicsA, topicsB, similarityMap) {
    let total = 0;
    let pairs = 0;
    for (const topicA of topicsA) {
        for (const topicB of topicsB) {
            if (topicA === topicB) continue;
            total += similarityMap.get(createPairKey(topicA, topicB)) || 0;
            pairs += 1;
        }
    }
    return pairs === 0 ? 0 : total / pairs;
}

export function computeTopicSimilarityMap(graphData) {
    const topicToPaperSet = new Map();
    const papers = graphData?.papers || [];
    const topics = graphData?.topics || [];

    topics.forEach((topic) => topicToPaperSet.set(topic, new Set()));

    papers.forEach((paper) => {
        (paper.topics || []).forEach((topic) => {
            if (!topicToPaperSet.has(topic)) {
                topicToPaperSet.set(topic, new Set());
            }
            topicToPaperSet.get(topic).add(paper.title);
        });
    });

    const similarityMap = new Map();
    const topicList = Array.from(topicToPaperSet.keys());
    for (let i = 0; i < topicList.length; i += 1) {
        for (let j = i + 1; j < topicList.length; j += 1) {
            const a = topicList[i];
            const b = topicList[j];
            const papersA = topicToPaperSet.get(a) || new Set();
            const papersB = topicToPaperSet.get(b) || new Set();
            const intersection = new Set(
                [...papersA].filter((paperTitle) => papersB.has(paperTitle))
            );
            const unionSize = new Set([...papersA, ...papersB]).size;
            const score = unionSize === 0 ? 0 : intersection.size / unionSize;
            similarityMap.set(createPairKey(a, b), score);
        }
    }

    return { similarityMap, topicToPaperSet };
}

export function buildTopicClusters(graphData) {
    const topics = graphData?.topics || [];
    if (topics.length === 0) {
        return {
            clusters: [],
            topicToCluster: {},
        };
    }

    const papers = graphData?.papers || [];
    const topicToPaperSet = new Map();
    topics.forEach((topic) => topicToPaperSet.set(topic, new Set()));
    papers.forEach((paper) => {
        (paper.topics || []).forEach((topic) => {
            if (!topicToPaperSet.has(topic)) {
                topicToPaperSet.set(topic, new Set());
            }
            topicToPaperSet.get(topic).add(paper.title);
        });
    });

    if (topics.length > HIERARCHICAL_MAX_TOPICS) {
        return buildFastClusters(topics, topicToPaperSet);
    }

    const { similarityMap } = computeTopicSimilarityMap(graphData);

    let currentNodes = topics.map((topic) => ({
        id: `topic-${topic}`,
        name: topic,
        topics: [topic],
        score: 1,
        children: [],
    }));

    let mergeCounter = 0;
    while (currentNodes.length > 1) {
        let bestI = 0;
        let bestJ = 1;
        let bestScore = -1;

        for (let i = 0; i < currentNodes.length; i += 1) {
            for (let j = i + 1; j < currentNodes.length; j += 1) {
                const score = averagePairwiseSimilarity(
                    currentNodes[i].topics,
                    currentNodes[j].topics,
                    similarityMap
                );
                if (score > bestScore) {
                    bestScore = score;
                    bestI = i;
                    bestJ = j;
                }
            }
        }

        const left = currentNodes[bestI];
        const right = currentNodes[bestJ];
        const merged = {
            id: `cluster-${mergeCounter}`,
            name: `Cluster ${mergeCounter + 1}`,
            topics: [...left.topics, ...right.topics],
            score: Math.max(bestScore, 0),
            children: [left, right],
        };
        mergeCounter += 1;

        currentNodes = currentNodes.filter(
            (_, idx) => idx !== bestI && idx !== bestJ
        );
        currentNodes.push(merged);
    }

    const root = currentNodes[0];
    const maxClusters = Math.max(2, Math.ceil(Math.sqrt(topics.length)));
    const splitThreshold = 0.2;
    const workList = [root];

    // Create top-level groups by splitting weak joins.
    while (workList.length < maxClusters) {
        let splitIndex = -1;
        let lowestScore = Number.POSITIVE_INFINITY;

        for (let i = 0; i < workList.length; i += 1) {
            const node = workList[i];
            if (!node.children || node.children.length < 2) continue;
            if (node.score < lowestScore) {
                lowestScore = node.score;
                splitIndex = i;
            }
        }

        if (splitIndex === -1) break;
        if (lowestScore > splitThreshold) break;

        const [splitNode] = workList.splice(splitIndex, 1);
        workList.push(...splitNode.children);
    }

    const clusters = workList
        .map((node, index) => ({
            id: `cluster-top-${index}`,
            label:
                node.topics.length === 1
                    ? node.topics[0]
                    : `${node.topics[0]} + ${node.topics.length - 1} more`,
            score: node.score,
            topics: node.topics,
            paperCount: new Set(
                node.topics.flatMap((topic) =>
                    Array.from(topicToPaperSet.get(topic) || [])
                )
            ).size,
            tree: node,
        }))
        .sort((a, b) => b.paperCount - a.paperCount);

    const topicToCluster = {};
    clusters.forEach((cluster) => {
        cluster.topics.forEach((topic) => {
            topicToCluster[topic] = cluster.id;
        });
    });

    return {
        clusters,
        topicToCluster,
    };
}

function jaccardSimilarity(setA, setB) {
    if (!setA || !setB || setA.size === 0 || setB.size === 0) return 0;
    let intersection = 0;
    for (const value of setA) {
        if (setB.has(value)) intersection += 1;
    }
    const union = setA.size + setB.size - intersection;
    return union === 0 ? 0 : intersection / union;
}

function buildFastClusters(topics, topicToPaperSet) {
    const sortedTopics = [...topics].sort((left, right) => {
        const leftCount = (topicToPaperSet.get(left) || new Set()).size;
        const rightCount = (topicToPaperSet.get(right) || new Set()).size;
        return rightCount - leftCount;
    });
    const seedCount = Math.max(3, Math.min(14, Math.ceil(Math.sqrt(sortedTopics.length))));
    const seedTopics = sortedTopics.slice(0, seedCount);
    const seedToMembers = new Map(seedTopics.map((seed) => [seed, [seed]]));

    for (const topic of sortedTopics.slice(seedCount)) {
        let bestSeed = seedTopics[0];
        let bestScore = -1;
        const topicSet = topicToPaperSet.get(topic) || new Set();
        for (const seed of seedTopics) {
            const seedSet = topicToPaperSet.get(seed) || new Set();
            const score = jaccardSimilarity(topicSet, seedSet);
            if (score > bestScore) {
                bestScore = score;
                bestSeed = seed;
            }
        }
        seedToMembers.get(bestSeed).push(topic);
    }

    const clusters = Array.from(seedToMembers.entries())
        .map(([seed, members], index) => {
            const leafChildren = members.map((topic) => ({
                id: `topic-${topic}`,
                name: topic,
                topics: [topic],
                score: 1,
                children: [],
            }));
            const paperCount = new Set(
                members.flatMap((topic) => Array.from(topicToPaperSet.get(topic) || []))
            ).size;
            return {
                id: `cluster-fast-${index}`,
                label: members.length === 1 ? members[0] : `${seed} + ${members.length - 1} more`,
                score: 0.5,
                topics: members,
                paperCount,
                tree: {
                    id: `cluster-fast-tree-${index}`,
                    name: seed,
                    topics: members,
                    score: 0.5,
                    children: leafChildren,
                },
            };
        })
        .sort((a, b) => b.paperCount - a.paperCount);

    const topicToCluster = {};
    clusters.forEach((cluster) => {
        cluster.topics.forEach((topic) => {
            topicToCluster[topic] = cluster.id;
        });
    });

    return { clusters, topicToCluster };
}
