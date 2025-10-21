import networkx as nx
from .paper import Paper
from .topic import Topic

class PaperGraph:
    def __init__(self):
        self.graph = nx.Graph()
    
    def add_paper(self, paper: Paper):
        self.graph.add_node(paper.title, type='paper', data=paper)
        for topic_name in paper.topics:
            self.graph.add_node(topic_name, type='topic')
            self.graph.add_edge(paper.title, topic_name)
    
    def add_topic(self, topic: Topic):
        self.graph.add_node(topic.name, type='topic', data=topic)
    
    def find_path(self, source: str, target: str):
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return []
