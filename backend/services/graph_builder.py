from models.graph import PaperGraph
from models.paper import Paper
from models.topic import Topic
from services.llm_service import TopicExtractor, OpenAILLMClient
from services.pdf_preprocessor import extract_text_from_pdf
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Global set to store all topics seen across all uploads (in-memory, not persistent)
_all_topics_seen = set()


class GraphBuilder:
    def __init__(self):
        self.papers = []
        self.topics = set()


    def build_graph(self, files_data=None, file_path: str = None) -> PaperGraph:
        """
        Builds a graph from PDF files.
        Processes papers sequentially, adding each to the graph immediately,
        and passing accumulated topics to the next paper's topic extraction.
        
        Args:
            files_data: List of tuples (filename, file_content_bytes), or None
            file_path: Directory path to scan for PDFs (legacy support), or None
        """
        # Reset state for new build
        self.papers = []
        self.topics = set()
        
        if files_data is not None:
            self.get_papers_from_data(files_data)
        elif file_path is not None:
            self.get_papers(file_path)
        else:
            raise ValueError("Either files_data or file_path must be provided")

        # Initialize graph
        graph = PaperGraph()
        
        # Get all topics seen across all previous uploads
        global _all_topics_seen
        accumulated_topics = _all_topics_seen.copy()  # Start with all previously seen topics
        
        # Use OpenAI assistant (reads assistant_id from environment)
        client = OpenAILLMClient()
        extractor = TopicExtractor(client)
        
        # Process papers one at a time, adding each to graph immediately
        for paper in self.papers:
            # Truncate very long texts to avoid excessive processing time
            # Keep first 50000 characters (enough for topic extraction)
            text_for_extraction = paper.text[:50000] if len(paper.text) > 50000 else paper.text
            
            # Extract topics, passing in all accumulated topics (global + from previous papers in this batch)
            topics = extractor.extract_topics(text_for_extraction, current_topics=accumulated_topics)
            paper.topics = topics
            
            # Update accumulated topics with new topics from this paper
            new_topics = set(topics) - accumulated_topics
            if new_topics:
                accumulated_topics.update(new_topics)
            
            self.topics.update(set(topics))  # Track topics for this batch
            
            # Add topics to global set of all topics seen (if not already there)
            global_new_topics = set(topics) - _all_topics_seen
            if global_new_topics:
                _all_topics_seen.update(global_new_topics)
            
            # Add paper to graph immediately after processing
            graph.add_paper(paper)
        
        logger.info(f"Processed {len(self.papers)} papers, {len(self.topics)} unique topics in this batch")
        
        return graph


    def get_papers_from_data(self, files_data: list[tuple]) -> list[Paper]:
        """
        Creates Paper objects from file data (filename, content bytes).
        
        Args:
            files_data: List of tuples (filename, file_content_bytes)
        """
        for filename, file_content in files_data:
            # Extract filename without extension for title
            title = Path(filename).stem
            paper = Paper(
                title=title,
                file_path=filename,  # Keep original filename for reference
                text=extract_text_from_pdf(file_content)
            )
            self.papers.append(paper)
        
        return self.papers

    def get_papers(self, file_path: str) -> list[Paper]:
        """
        Gets all of the pdfs in the given file path and all of its sub folders.
        Legacy method for backward compatibility.
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


def get_all_topics_seen():
    """Get the set of all topics seen across all uploads"""
    return _all_topics_seen.copy()


def clear_all_topics_seen():
    """Clear the set of all topics seen (useful for testing or reset)"""
    global _all_topics_seen
    _all_topics_seen.clear()


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