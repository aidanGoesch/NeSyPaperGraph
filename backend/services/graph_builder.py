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


    def build_graph(self, files_data=None, file_path: str = None, existing_graph=None) -> PaperGraph:
        """
        Builds a graph from PDF files.
        Processes papers sequentially, adding each to the graph immediately,
        and passing accumulated topics to the next paper's topic extraction.
        
        Args:
            files_data: List of tuples (filename, file_content_bytes), or None
            file_path: Directory path to scan for PDFs (legacy support), or None
            existing_graph: Existing PaperGraph to update, or None to create new
        """
        # Use existing graph or create new one
        if existing_graph:
            graph = existing_graph
            # Extract existing papers and topics
            self.papers = []
            self.topics = set()
            for node, data in graph.graph.nodes(data=True):
                if data.get('type') == 'paper':
                    self.papers.append(data['data'])
                elif data.get('type') == 'topic':
                    self.topics.add(node)
        else:
            # Reset state for new build
            self.papers = []
            self.topics = set()
            graph = PaperGraph()
        
        if files_data is not None:
            self.get_papers_from_data(files_data)
        elif file_path is not None:
            self.get_papers(file_path)
        else:
            raise ValueError("Either files_data or file_path must be provided")

        # Initialize graph (already done above if existing_graph provided)
        if not existing_graph:
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
            
            # Generate summary
            paper.summary = client.generate_summary(text_for_extraction)
            
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
        # Initialize LLM client for metadata extraction
        from .llm_service import OpenAILLMClient, TopicExtractor
        client = OpenAILLMClient()
        extractor = TopicExtractor(client)
        
        for filename, file_content in files_data:
            # Extract text from PDF
            text = extract_text_from_pdf(file_content)
            
            # Extract metadata using LLM
            metadata = extractor.extract_paper_metadata(text)
            
            # Use extracted title if available, otherwise use filename
            title = metadata.get('title') or Path(filename).stem
            
            paper = Paper(
                title=title,
                file_path=filename,  # Keep original filename for reference
                text=text,
                authors=metadata.get('authors', []),
                publication_date=metadata.get('publication_date')
            )
            self.papers.append(paper)
        
        return self.papers

    def get_papers(self, file_path: str) -> list[Paper]:
        """
        Gets all of the pdfs in the given file path and all of its sub folders.
        Legacy method for backward compatibility.
        """
        # Initialize LLM client for metadata extraction
        from .llm_service import OpenAILLMClient, TopicExtractor
        client = OpenAILLMClient()
        extractor = TopicExtractor(client)
        
        path = Path(file_path)
        
        for pdf_file in path.rglob("*.pdf"):
            # Extract text from PDF
            text = extract_text_from_pdf(str(pdf_file))
            
            # Extract metadata using LLM
            metadata = extractor.extract_paper_metadata(text)
            
            # Use extracted title if available, otherwise use filename
            title = metadata.get('title') or pdf_file.stem
            
            paper = Paper(
                title=title,
                file_path=str(pdf_file),
                text=text,
                authors=metadata.get('authors', []),
                publication_date=metadata.get('publication_date')
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
    """Creates a dummy graph with realistic papers for testing grounding responses"""
    
    papers = [
        Paper(
            title="Attention Is All You Need",
            file_path="path/transformer.pdf",
            text="We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs.",
            topics=["Natural Language Processing", "Deep Learning", "Attention Mechanisms"],
            authors=["Vaswani", "Shazeer", "Parmar", "Uszkoreit", "Jones", "Gomez", "Kaiser", "Polosukhin"],
            embedding=[0.9, 0.8, 0.7, 0.2, 0.1]
        ),
        Paper(
            title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            file_path="path/bert.pdf", 
            text="We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.",
            topics=["Natural Language Processing", "Deep Learning", "Transfer Learning"],
            authors=["Devlin", "Chang", "Lee", "Toutanova"],
            embedding=[0.85, 0.75, 0.65, 0.25, 0.15]
        ),
        Paper(
            title="Mastering the Game of Go with Deep Neural Networks and Tree Search",
            file_path="path/alphago.pdf",
            text="The game of Go has long been viewed as the most challenging of classic games for artificial intelligence owing to its enormous search space and the difficulty of evaluating board positions and moves. Here we introduce a new approach to computer Go that uses 'value networks' to evaluate board positions and 'policy networks' to select moves. These deep neural networks are trained by a novel combination of supervised learning from human expert games, and reinforcement learning from games of self-play. We also introduce a new search algorithm that combines Monte Carlo simulation with value and policy networks.",
            topics=["Reinforcement Learning", "Game Theory", "Deep Learning"],
            authors=["Silver", "Huang", "Maddison", "Guez", "Sifre", "van den Driessche", "Schrittwieser", "Antonoglou", "Panneershelvam", "Lanctot", "Dieleman", "Grewe", "Nham", "Kalchbrenner", "Sutskever", "Lillicrap", "Leach", "Kavukcuoglu", "Graepel", "Hassabis"],
            embedding=[0.6, 0.7, 0.8, 0.9, 0.3]
        ),
        Paper(
            title="Quantum Supremacy Using a Programmable Superconducting Processor",
            file_path="path/quantum_supremacy.pdf",
            text="The promise of quantum computers is that certain computational tasks might be executed exponentially faster on a quantum processor than on a classical processor. A fundamental challenge is to build a high-fidelity processor capable of running quantum algorithms in an exponentially large computational space. Here we report the use of a processor with programmable superconducting qubits to create quantum states on 53 qubits, corresponding to a computational state-space of dimension 2^53. Measurements from repeated experiments sample the resulting probability distribution, which we verify using classical simulations.",
            topics=["Quantum Computing", "Quantum Physics", "Computational Complexity"],
            authors=["Arute", "Arya", "Babbush", "Bacon", "Bardin", "Barends", "Biswas", "Boixo", "Brandao", "Buell"],
            embedding=[0.1, 0.2, 0.1, 0.9, 0.8]
        ),
        Paper(
            title="Bitcoin: A Peer-to-Peer Electronic Cash System",
            file_path="path/bitcoin.pdf",
            text="A purely peer-to-peer version of electronic cash would allow online payments to be sent directly from one party to another without going through a financial institution. Digital signatures provide part of the solution, but the main benefits are lost if a trusted third party is still required to prevent double-spending. We propose a solution to the double-spending problem using a peer-to-peer network. The network timestamps transactions by hashing them into an ongoing chain of hash-based proof-of-work, forming a record that cannot be changed without redoing the proof-of-work.",
            topics=["Cryptography", "Distributed Systems", "Digital Currency"],
            authors=["Nakamoto"],
            embedding=[0.2, 0.1, 0.3, 0.4, 0.9]
        ),
        Paper(
            title="ImageNet Classification with Deep Convolutional Neural Networks",
            file_path="path/alexnet.pdf",
            text="We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0% which is considerably better than the previous state-of-the-art. The neural network, which has 60 million parameters and 650,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and three fully-connected layers with a final 1000-way softmax.",
            topics=["Computer Vision", "Deep Learning", "Convolutional Neural Networks"],
            authors=["Krizhevsky", "Sutskever", "Hinton"],
            embedding=[0.8, 0.6, 0.7, 0.3, 0.2]
        ),
        Paper(
            title="Generative Adversarial Networks",
            file_path="path/gan.pdf",
            text="We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere.",
            topics=["Generative Models", "Deep Learning", "Game Theory"],
            authors=["Goodfellow", "Pouget-Abadie", "Mirza", "Xu", "Warde-Farley", "Ozair", "Courville", "Bengio"],
            embedding=[0.7, 0.8, 0.6, 0.4, 0.3]
        ),
        Paper(
            title="The PageRank Citation Ranking: Bringing Order to the Web",
            file_path="path/pagerank.pdf",
            text="The importance of a Web page is an inherently subjective matter, which depends on the readers interests, knowledge and attitudes. But there is still much that can be said objectively about the relative importance of Web pages. This paper describes PageRank, a method for rating Web pages objectively and mechanically, effectively measuring the human interest and attention devoted to them. We compare PageRank with an idealized random Web surfer. We show that PageRank can be efficiently computed by an iterative algorithm.",
            topics=["Web Search", "Graph Algorithms", "Information Retrieval"],
            authors=["Page", "Brin", "Motwani", "Winograd"],
            embedding=[0.3, 0.4, 0.5, 0.6, 0.7]
        ),
        Paper(
            title="A Few Useful Things to Know About Machine Learning",
            file_path="path/ml_guide.pdf",
            text="Machine learning algorithms can figure out how to perform important tasks by generalizing from examples. This is often feasible and cost-effective where manual programming is not. As more data becomes available, more ambitious problems can be tackled. As a result, machine learning is widely used in computer science and other fields. However, developing successful machine learning applications requires a substantial amount of 'black art' that is hard to find in textbooks. This article summarizes twelve key lessons that machine learning researchers and practitioners have learned.",
            topics=["Machine Learning", "Data Science", "Best Practices"],
            authors=["Domingos"],
            embedding=[0.6, 0.5, 0.4, 0.3, 0.4]
        ),
        Paper(
            title="Deep Residual Learning for Image Recognition",
            file_path="path/resnet.pdf",
            text="Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth.",
            topics=["Computer Vision", "Deep Learning", "Neural Architecture"],
            authors=["He", "Zhang", "Ren", "Sun"],
            embedding=[0.75, 0.65, 0.55, 0.35, 0.25]
        )
    ]
    
    graph = PaperGraph()
    for paper in papers:
        graph.add_paper(paper)
    
    graph.add_semantic_edges()
    
    return graph