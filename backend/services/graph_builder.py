from ..models.graph import PaperGraph
from ..models.paper import Paper
from ..models.topic import Topic
from .llm_service import TopicExtractor, HuggingFaceLLMClient
from .pdf_preprocessor import extract_text_from_pdf
import os
from pathlib import Path

class GraphBuilder:
    def __init__(self):
        self.papers = []


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