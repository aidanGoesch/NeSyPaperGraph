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
    """Creates a dummy graph with 3 topics and 5 papers"""
    papers = [
        Paper(title="Paper A", file_path="path/a.pdf", text="text", topics=["Topic 1", "Topic 2"]),
        Paper(title="Paper B", file_path="path/b.pdf", text="text", topics=["Topic 1"]),
        Paper(title="Paper C", file_path="path/c.pdf", text="text", topics=["Topic 2", "Topic 3"]),
        Paper(title="Paper D", file_path="path/d.pdf", text="text", topics=["Topic 3"]),
        Paper(title="Paper E", file_path="path/e.pdf", text="text", topics=["Topic 1", "Topic 3"])
    ]
    
    graph = PaperGraph()
    for paper in papers:
        graph.add_paper(paper)
    
    return graph