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
    """Creates a dummy graph with realistic papers for testing grounding responses"""
    
    papers = [
        Paper(
            title="Attention Is All You Need",
            file_path="path/transformer.pdf",
            text="We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs.",
            topics=["Natural Language Processing", "Deep Learning", "Attention Mechanisms"],
            authors="Vaswani et al.",
            embedding=[0.9, 0.8, 0.7, 0.2, 0.1]
        ),
        Paper(
            title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            file_path="path/bert.pdf", 
            text="We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.",
            topics=["Natural Language Processing", "Deep Learning", "Transfer Learning"],
            authors="Devlin et al.",
            embedding=[0.85, 0.75, 0.65, 0.25, 0.15]
        ),
        Paper(
            title="Mastering the Game of Go with Deep Neural Networks and Tree Search",
            file_path="path/alphago.pdf",
            text="The game of Go has long been viewed as the most challenging of classic games for artificial intelligence owing to its enormous search space and the difficulty of evaluating board positions and moves. Here we introduce a new approach to computer Go that uses 'value networks' to evaluate board positions and 'policy networks' to select moves. These deep neural networks are trained by a novel combination of supervised learning from human expert games, and reinforcement learning from games of self-play. We also introduce a new search algorithm that combines Monte Carlo simulation with value and policy networks.",
            topics=["Reinforcement Learning", "Game Theory", "Deep Learning"],
            authors="Silver et al.",
            embedding=[0.6, 0.7, 0.8, 0.9, 0.3]
        ),
        Paper(
            title="Quantum Supremacy Using a Programmable Superconducting Processor",
            file_path="path/quantum_supremacy.pdf",
            text="The promise of quantum computers is that certain computational tasks might be executed exponentially faster on a quantum processor than on a classical processor. A fundamental challenge is to build a high-fidelity processor capable of running quantum algorithms in an exponentially large computational space. Here we report the use of a processor with programmable superconducting qubits to create quantum states on 53 qubits, corresponding to a computational state-space of dimension 2^53. Measurements from repeated experiments sample the resulting probability distribution, which we verify using classical simulations.",
            topics=["Quantum Computing", "Quantum Physics", "Computational Complexity"],
            authors="Arute et al.",
            embedding=[0.1, 0.2, 0.1, 0.9, 0.8]
        ),
        Paper(
            title="Bitcoin: A Peer-to-Peer Electronic Cash System",
            file_path="path/bitcoin.pdf",
            text="A purely peer-to-peer version of electronic cash would allow online payments to be sent directly from one party to another without going through a financial institution. Digital signatures provide part of the solution, but the main benefits are lost if a trusted third party is still required to prevent double-spending. We propose a solution to the double-spending problem using a peer-to-peer network. The network timestamps transactions by hashing them into an ongoing chain of hash-based proof-of-work, forming a record that cannot be changed without redoing the proof-of-work.",
            topics=["Cryptography", "Distributed Systems", "Digital Currency"],
            authors="Nakamoto, S.",
            embedding=[0.2, 0.1, 0.3, 0.4, 0.9]
        ),
        Paper(
            title="ImageNet Classification with Deep Convolutional Neural Networks",
            file_path="path/alexnet.pdf",
            text="We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0% which is considerably better than the previous state-of-the-art. The neural network, which has 60 million parameters and 650,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and three fully-connected layers with a final 1000-way softmax.",
            topics=["Computer Vision", "Deep Learning", "Convolutional Neural Networks"],
            authors="Krizhevsky et al.",
            embedding=[0.8, 0.6, 0.7, 0.3, 0.2]
        ),
        Paper(
            title="Generative Adversarial Networks",
            file_path="path/gan.pdf",
            text="We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere.",
            topics=["Generative Models", "Deep Learning", "Game Theory"],
            authors="Goodfellow et al.",
            embedding=[0.7, 0.8, 0.6, 0.4, 0.3]
        ),
        Paper(
            title="The PageRank Citation Ranking: Bringing Order to the Web",
            file_path="path/pagerank.pdf",
            text="The importance of a Web page is an inherently subjective matter, which depends on the readers interests, knowledge and attitudes. But there is still much that can be said objectively about the relative importance of Web pages. This paper describes PageRank, a method for rating Web pages objectively and mechanically, effectively measuring the human interest and attention devoted to them. We compare PageRank with an idealized random Web surfer. We show that PageRank can be efficiently computed by an iterative algorithm.",
            topics=["Web Search", "Graph Algorithms", "Information Retrieval"],
            authors="Page et al.",
            embedding=[0.3, 0.4, 0.5, 0.6, 0.7]
        ),
        Paper(
            title="A Few Useful Things to Know About Machine Learning",
            file_path="path/ml_guide.pdf",
            text="Machine learning algorithms can figure out how to perform important tasks by generalizing from examples. This is often feasible and cost-effective where manual programming is not. As more data becomes available, more ambitious problems can be tackled. As a result, machine learning is widely used in computer science and other fields. However, developing successful machine learning applications requires a substantial amount of 'black art' that is hard to find in textbooks. This article summarizes twelve key lessons that machine learning researchers and practitioners have learned.",
            topics=["Machine Learning", "Data Science", "Best Practices"],
            authors="Domingos, P.",
            embedding=[0.6, 0.5, 0.4, 0.3, 0.4]
        ),
        Paper(
            title="Deep Residual Learning for Image Recognition",
            file_path="path/resnet.pdf",
            text="Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth.",
            topics=["Computer Vision", "Deep Learning", "Neural Architecture"],
            authors="He et al.",
            embedding=[0.75, 0.65, 0.55, 0.35, 0.25]
        )
    ]
    
    graph = PaperGraph()
    for paper in papers:
        graph.add_paper(paper)
    
    graph.add_semantic_edges()
    
    return graph