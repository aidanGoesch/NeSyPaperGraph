from models.graph import PaperGraph
from models.paper import Paper
from models.topic import Topic
from services.llm_service import TopicExtractor, HuggingFaceLLMClient
from services.pdf_preprocessor import extract_text_from_pdf
import os
from pathlib import Path


class GraphBuilder:
    def __init__(self):
        self.papers = []
        self.topics = set()


    def build_graph(self, file_path: str) -> PaperGraph:
        """
        Builds a graph from the given file path of pdfs
        """
        self.get_papers(file_path)

        # get the topics from the papers
        client = HuggingFaceLLMClient()
        extractor = TopicExtractor(client)
        for paper in self.papers:
            topics = extractor.extract_topics(paper.text)
            paper.topics = topics
            self.topics.update(set(topics)) # add the topics to the set of topics member variable
        
        # build the graph
        graph = PaperGraph()
        for paper in self.papers:
            graph.add_paper(paper)
        
        return graph


    def get_papers(self, file_path: str) -> list[Paper]:
        """
        Gets all of the pdfs in the given file path and all of its sub folders
        """
        path = Path(file_path)
        
        for pdf_file in path.rglob("*.pdf"):
            paper = Paper(
                title=pdf_file.stem,
                file_path=str(pdf_file),
                text=extract_text_from_pdf(str(pdf_file))
            )
            self.papers.append(paper)
        
        return self.papers


def create_dummy_graph() -> PaperGraph:
    """Creates a dummy graph with semantically similar papers"""
    
    # Create similar embeddings for papers that should be semantically related
    similar_embedding_1 = [0.8, 0.6, 0.2, 0.1, 0.3]
    similar_embedding_2 = [0.85, 0.65, 0.25, 0.15, 0.35]  # Very similar to embedding_1
    
    papers = [
        Paper(title="Neural Networks in AI", file_path="path/a.pdf", text="Deep learning and neural networks", 
              topics=["Machine Learning", "AI"], embedding=similar_embedding_1),
        Paper(title="Deep Learning Applications", file_path="path/b.pdf", text="Applications of deep learning", 
              topics=["Machine Learning"], embedding=similar_embedding_2),
        Paper(title="Quantum Computing", file_path="path/c.pdf", text="Quantum algorithms and computing", 
              topics=["Quantum Physics", "Computing"], embedding=[0.1, 0.2, 0.9, 0.8, 0.1]),
        Paper(title="Blockchain Security", file_path="path/d.pdf", text="Distributed ledger security", 
              topics=["Cryptography", "Cyber Security"], embedding=[0.3, 0.1, 0.4, 0.7, 0.9]),
        Paper(title="Computer Vision", file_path="path/e.pdf", text="Image processing and recognition", 
              topics=["Machine Learning", "Computer Vision"], embedding=[0.7, 0.5, 0.3, 0.2, 0.4])
    ]
    
    graph = PaperGraph()
    for paper in papers:
        graph.add_paper(paper)
    
    graph.add_semantic_edges()
    
    return graph