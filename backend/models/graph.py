import math
import networkx as nx
from .paper import Paper

class PaperGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.topic_synonyms = {}  # Dictionary mapping topics to their synonyms
        self.topic_merge_groups = {}  # Dictionary mapping group_id -> list of topics to merge
        self.new_nodes = set()  # Track nodes added since last verification
        self.new_edges = set()  # Track edges added since last verification
    
    def add_paper(self, paper: Paper):
        self.graph.add_node(paper.title, type='paper', data=paper)
        self.new_nodes.add(paper.title)
        for topic_name in paper.topics:
            if topic_name not in self.graph:
                self.new_nodes.add(topic_name)
            self.graph.add_node(topic_name, type='topic')
            edge = (paper.title, topic_name)
            self.graph.add_edge(paper.title, topic_name)
            self.new_edges.add(edge)
    
    def clear_incremental_tracking(self):
        """Clear tracking after verification"""
        self.new_nodes.clear()
        self.new_edges.clear()
    
    def add_semantic_edges(self):
        """Add edges between semantically similar papers"""
        paper_nodes = [n for n, attr in self.graph.nodes(data=True) if attr['type'] == 'paper']
        for i, paper1 in enumerate(paper_nodes):
            for paper2 in paper_nodes[i+1:]:
                paper1_data = self.graph.nodes[paper1]['data']
                paper2_data = self.graph.nodes[paper2]['data']
                if paper1_data.embedding and paper2_data.embedding:
                    similarity = cosine_similarity(paper1_data.embedding, paper2_data.embedding)
                    if similarity > 0.5:  # threshold for semantic similarity
                        self.graph.add_edge(paper1, paper2, weight=similarity, type='semantic')
    
    def merge_topics(self, merge_groups: dict):
        """Merge topics based on merge groups. Each group is merged into a single representative topic."""
        for group_id, topics in merge_groups.items():
            if len(topics) <= 1:
                continue
            
            # Use first topic as representative
            representative = topics[0]
            
            # Store merged topic names in representative's data
            if representative in self.graph:
                node_data = self.graph.nodes[representative]
                if 'merged_topics' not in node_data:
                    node_data['merged_topics'] = []
            
            # Merge all other topics into representative
            for topic in topics[1:]:
                if topic not in self.graph:
                    continue
                
                # Store the merged topic name
                if representative in self.graph:
                    self.graph.nodes[representative]['merged_topics'].append(topic)
                
                # Transfer all edges from this topic to representative
                for neighbor in list(self.graph.neighbors(topic)):
                    if not self.graph.has_edge(representative, neighbor):
                        self.graph.add_edge(representative, neighbor)
                
                # Remove the merged topic
                self.graph.remove_node(topic)
    
    def find_path(self, source: str, target: str):
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return []
        
    def find_min_topic(self):
        """
        Finds the topic with the least connections
        """
        topic_nodes = [n for n, attr in self.graph.nodes(data=True) if attr['type'] == 'topic']
        if not topic_nodes:
            return None
        min_topic = min(topic_nodes, key=lambda t: self.graph.degree(t))
        return min_topic
    
    def find_max_topic(self):
        """
        Finds the topic with the most connections
        """
        topic_nodes = [n for n, attr in self.graph.nodes(data=True) if attr['type'] == 'topic']
        if not topic_nodes:
            return None
        max_topic = max(topic_nodes, key=lambda t: self.graph.degree(t))
        return max_topic


def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude_vec1 = math.sqrt(sum(a**2 for a in vec1))
    magnitude_vec2 = math.sqrt(sum(b**2 for b in vec2))
    
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0  # Handle division by zero if a vector is all zeros
    
    return dot_product / (magnitude_vec1 * magnitude_vec2)

