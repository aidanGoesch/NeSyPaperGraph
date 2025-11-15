import os
from typing import TypedDict, List, Tuple
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    search_results: list

class QuestionAgent:
    def __init__(self, graph_data=None, graph_obj=None):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.graph_data = graph_data
        self.graph_obj = graph_obj
        self.graph = self._build_graph()
    
    def _get_graph_object(self):
        """Get the actual graph object, either from stored object or by loading from file"""
        if self.graph_obj:
            return self.graph_obj
        
        # Try to load the saved graph
        try:
            import pickle
            save_path = "storage/saved_graph.pkl"
            if os.path.exists(save_path):
                with open(save_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Could not load saved graph: {e}")
        
        # Fallback to dummy graph
        try:
            from services.graph_builder import create_dummy_graph
            return create_dummy_graph()
        except Exception as e:
            print(f"Could not create dummy graph: {e}")
            return None
    
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("route_question", self._route_question)
        workflow.add_node("bridge_question", self._bridge_question)
        workflow.add_node("explain_question", self._explain_question)
        workflow.add_node("keyword_search", self._keyword_search)
        workflow.add_node("graph_properties", self._graph_properties)
        workflow.add_node("generate_answer", self._generate_answer)
        
        workflow.set_entry_point("route_question")
        workflow.add_conditional_edges(
            "route_question",
            self._route_decision,
            {
                "bridge": "bridge_question",
                "explain": "explain_question", 
                "search": "keyword_search",
                "properties": "graph_properties"
            }
        )
        workflow.add_edge("bridge_question", "generate_answer")
        workflow.add_edge("explain_question", "generate_answer")
        workflow.add_edge("keyword_search", "generate_answer")
        workflow.add_edge("graph_properties", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()
    
    def _route_question(self, state: AgentState) -> AgentState:
        """Route the question based on its type"""
        question = state["question"]
        state["context"] = f"Routing question: {question}"
        return state
    
    def _route_decision(self, state: AgentState) -> str:
        """Decide which type of question this is"""
        question = state["question"].lower()
        
        # Graph properties questions
        property_patterns = [
            "how many", "number of", "count", "connections", "degree", 
            "neighbors", "edges", "nodes", "size", "statistics",
            "min", "max", "most", "least", "which topics", "what topics"
        ]
        
        # Bridge questions - looking for relationships/connections
        bridge_patterns = [
            "connect", "relationship", "between", "how are", 
            "related to", "connection", "compare", "versus", 
            "vs", "difference between", "how is", "relate"
        ]
        
        if any(pattern in question for pattern in property_patterns):
            question_type = "properties"
        elif any(pattern in question for pattern in bridge_patterns):
            question_type = "bridge"
        elif any(word in question for word in ["explain", "what is", "define", "describe"]):
            question_type = "explain"
        else:
            question_type = "search"
        
        print(f"Question type detected: {question_type} for question: '{state['question']}'")
        return question_type
    
    def _bridge_question(self, state: AgentState) -> AgentState:
        """Answers questions of how X and Y are connected based on the graph"""
        question = state["question"]
        print(f"Processing bridge question: {question}")
        
        start_node, end_node = self._extract_entities(question)
        
        if start_node and end_node:
            path_result = self._find_path_in_graph(start_node, end_node)
            state["context"] = path_result
            print(f"Bridge analysis: {start_node} -> {end_node}")
        else:
            state["context"] = f"Could not identify two entities to connect in: {question}"
            print(f"Could not parse entities from bridge question: {question}")
        
        return state
    
    def _extract_entities(self, question):
        """Extract start and end entities from bridge questions using LLM"""
        extraction_prompt = f"""
Extract the two main concepts/entities that need to be connected from this question.
Return ONLY the two entities separated by a pipe (|), nothing else.

Examples:
"How are generative models and transfer learning related?" -> generative models|transfer learning
"What's the connection between DNA and RNA?" -> DNA|RNA
"Compare neural networks and decision trees" -> neural networks|decision trees

Question: {question}
"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
            entities_text = response.content.strip()
            
            if '|' in entities_text:
                entities = [e.strip() for e in entities_text.split('|')]
                if len(entities) == 2:
                    return entities[0], entities[1]
        except Exception as e:
            print(f"Entity extraction failed: {e}")
        
        return None, None
    
    def _explain_segment(self, topic1: str, paper_content: str, paper_title: str, topic2: str) -> str:
        """Use LLM to explain how topic1 connects to topic2 through a paper"""
        prompt = f"""You are analyzing a research paper to understand how two topics are connected.

Topic 1: {topic1}
Topic 2: {topic2}

Paper Title: {paper_title}

Paper Content:
{paper_content}

Task: Explain how this paper connects "{topic1}" to "{topic2}". Focus on the key concepts, methods, or findings that bridge these two topics. Be concise but thorough.

Your explanation:"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            explanation = response.content.strip()
            print(f"Segment explanation generated: {topic1} -> {paper_title} -> {topic2}")
            return explanation
        except Exception as e:
            print(f"Error generating segment explanation: {e}")
            return f"Unable to explain connection through {paper_title}"
    
    def _synthesize_chain_explanation(self, segment_explanations: List[Tuple[str, str, str, str]], 
                                     start_entity: str, end_entity: str, path_str: str) -> str:
        """Synthesize individual segment explanations into a coherent overall explanation"""
        
        # Build the synthesis prompt
        segments_text = []
        for i, (topic1, paper_title, topic2, explanation) in enumerate(segment_explanations, 1):
            segments_text.append(f"Segment {i}: {topic1} → {paper_title} → {topic2}\n{explanation}")
        
        synthesis_prompt = f"""You have analyzed a chain of research papers connecting "{start_entity}" to "{end_entity}".

Here are the individual segment explanations:

{chr(10).join(segments_text)}

Connection Path: {path_str}

Task: Synthesize these segment explanations into a coherent, flowing explanation of how "{start_entity}" relates to "{end_entity}" through this chain of papers. Show how each connection builds on the previous one to form a complete picture.

Your synthesized explanation:"""

        try:
            response = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
            synthesis = response.content.strip()
            
            # Add the visual path at the end
            synthesis += f"\n\n**Connection Path:**\n{path_str}"
            
            print(f"Chain synthesis completed for: {start_entity} -> {end_entity}")
            return synthesis
        except Exception as e:
            print(f"Error synthesizing chain explanation: {e}")
            # Fallback: just concatenate the segments
            fallback = f"Connection from {start_entity} to {end_entity}:\n\n"
            for i, (_, _, _, explanation) in enumerate(segment_explanations, 1):
                fallback += f"{i}. {explanation}\n\n"
            fallback += f"\n**Connection Path:**\n{path_str}"
            return fallback
    
    def _find_path_in_graph(self, start_entity, end_entity):
        """Find path between two entities and use chain reasoning to explain the connection"""
        try:
            # Get the actual graph object
            graph_obj = self._get_graph_object()
            if not graph_obj:
                return "Graph not available"
            
            # Find nodes that match the entities
            start_node = None
            end_node = None
            
            for node, data in graph_obj.graph.nodes(data=True):
                node_str = str(node).lower()
                if start_entity.lower() in node_str:
                    start_node = node
                if end_entity.lower() in node_str:
                    end_node = node
            
            if not start_node or not end_node:
                return "Those are not related"
            
            # Use NetworkX to find shortest path
            import networkx as nx
            try:
                path = nx.shortest_path(graph_obj.graph, start_node, end_node)
                path_str = " → ".join([str(node) for node in path])
                
                print(f"Path found: {path_str}")
                print(f"Path length: {len(path)} nodes")
                
                # Extract segments: topic -> paper -> topic chains
                segment_explanations = []
                
                i = 0
                while i < len(path) - 1:
                    current_node = path[i]
                    current_data = graph_obj.graph.nodes[current_node]
                    
                    # Look for topic -> paper -> topic pattern
                    if current_data.get('type') == 'topic' and i + 2 < len(path):
                        paper_node = path[i + 1]
                        next_topic_node = path[i + 2]
                        
                        paper_data = graph_obj.graph.nodes[paper_node]
                        next_topic_data = graph_obj.graph.nodes[next_topic_node]
                        
                        if (paper_data.get('type') == 'paper' and 
                            next_topic_data.get('type') == 'topic' and 
                            'data' in paper_data):
                            
                            # Get full paper content
                            paper = paper_data['data']
                            paper_content = getattr(paper, 'text', '')
                            
                            # If paper text is too long, truncate but keep substantial content
                            if len(paper_content) > 8000:
                                paper_content = paper_content[:8000] + "... [truncated]"
                            
                            # Generate explanation for this segment
                            explanation = self._explain_segment(
                                topic1=str(current_node),
                                paper_content=paper_content,
                                paper_title=paper.title,
                                topic2=str(next_topic_node)
                            )
                            
                            segment_explanations.append((
                                str(current_node),
                                paper.title,
                                str(next_topic_node),
                                explanation
                            ))
                            
                            # Move to next topic
                            i += 2
                        else:
                            i += 1
                    else:
                        i += 1
                
                # If we have segment explanations, synthesize them
                if segment_explanations:
                    print(f"Generated {len(segment_explanations)} segment explanations, synthesizing...")
                    synthesized = self._synthesize_chain_explanation(
                        segment_explanations, 
                        start_entity, 
                        end_entity,
                        path_str
                    )
                    return f"CHAIN_REASONING_RESULT:\n\n{synthesized}"
                else:
                    # Fallback to simple path description
                    return f"Path found: {path_str}\n\nNo detailed paper content available for analysis."
                    
            except nx.NetworkXNoPath:
                return "Those are not related"
                
        except Exception as e:
            print(f"Error in pathfinding: {e}")
            import traceback
            traceback.print_exc()
            return "Those are not related"
    
    def _explain_question(self, state: AgentState) -> AgentState:
        """Answers questions of 'explain y', where y could be either a topic or a paper"""
        question = state["question"]
        print(f"Processing explain question: {question}")
        
        # Extract what needs to be explained
        explain_terms = []
        words = question.lower().split()
        for i, word in enumerate(words):
            if word in ["explain", "what", "is", "define", "describe"]:
                # Get remaining words as the thing to explain
                explain_terms.extend(words[i+1:])
                break
        
        if not explain_terms:
            explain_terms = words  # fallback to all words
        
        try:
            # Get the actual graph object
            graph_obj = self._get_graph_object()
            if not graph_obj:
                state["context"] = "Graph not available"
                return state
            
            relevant_papers = []
            relevant_nodes = []
            
            # Find directly matching nodes
            for node, data in graph_obj.graph.nodes(data=True):
                node_str = str(node).lower()
                
                # Check if any explain term matches this node
                for term in explain_terms:
                    if term in node_str:
                        relevant_nodes.append(str(node))
                        
                        # If it's a paper, collect its summary
                        if data.get('type') == 'paper' and 'data' in data:
                            paper = data['data']
                            summary = getattr(paper, 'summary', None) or paper.text[:1000] + "..."
                            relevant_papers.append(f"Paper: {paper.title}\nSummary: {summary}")
                        
                        # Also find semantically connected papers
                        for neighbor in graph_obj.graph.neighbors(node):
                            neighbor_data = graph_obj.graph.nodes[neighbor]
                            neighbor_str = str(neighbor)
                            if neighbor_str not in relevant_nodes:
                                relevant_nodes.append(neighbor_str)
                                
                                # If neighbor is a paper, collect its summary too
                                if neighbor_data.get('type') == 'paper' and 'data' in neighbor_data:
                                    paper = neighbor_data['data']
                                    summary = getattr(paper, 'summary', None) or paper.text[:1000] + "..."
                                    relevant_papers.append(f"Paper: {paper.title}\nSummary: {summary}")
                        break
            
            if relevant_papers:
                # Return paper summaries for LLM grounding
                state["context"] = f"Explaining with relevant papers: {', '.join(relevant_nodes[:10])}\n\nPaper summaries for grounding:\n\n" + "\n\n---\n\n".join(relevant_papers)
                print(f"Explain found {len(relevant_papers)} relevant papers with summaries")
            elif relevant_nodes:
                state["context"] = f"Explaining with relevant nodes: {', '.join(relevant_nodes[:10])}"
                print(f"Explain found {len(relevant_nodes)} relevant nodes: {relevant_nodes[:5]}...")
            else:
                state["context"] = "No relevant information found in graph"
                print("No relevant nodes found for explanation")
                
        except Exception as e:
            print(f"Error in explain search: {e}")
            state["context"] = "Error finding relevant information"
        
        return state
    
    def _keyword_search(self, state: AgentState) -> AgentState:
        """Finds papers that match keywords in title or have topics that match keywords"""
        question = state["question"]
        print(f"Processing keyword search: {question}")
        
        # Extract keywords from question (remove common words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'about', 'what', 'how', 'find', 'search', 'show', 'me', 'papers', 'documents'}
        keywords = [word.lower() for word in question.split() if word.lower() not in stop_words]
        
        try:
            # Get the actual graph object
            graph_obj = self._get_graph_object()
            if not graph_obj:
                state["context"] = "Graph not available"
                return state
            
            matching_papers = []
            
            # Search through all paper nodes in the graph
            for node, data in graph_obj.graph.nodes(data=True):
                if data.get('type') == 'paper' and 'data' in data:
                    paper = data['data']
                    node_str = str(node).lower()
                    
                    # Check if any keyword matches the paper title
                    title_match = any(keyword in node_str for keyword in keywords)
                    
                    # Check if any keyword matches the paper's topics
                    topic_match = False
                    if hasattr(paper, 'topics') and paper.topics:
                        for topic in paper.topics:
                            if any(keyword in topic.lower() for keyword in keywords):
                                topic_match = True
                                break
                    
                    if title_match or topic_match:
                        summary = getattr(paper, 'summary', None) or paper.text[:500] + "..."
                        matching_papers.append({
                            "title": paper.title,
                            "author": getattr(paper, 'authors', None) or 'Unknown Author',
                            "summary": summary,
                            "topics": getattr(paper, 'topics', []),
                            "node_id": node
                        })
            
            if matching_papers:
                # Return structured data for UI blocks instead of LLM context
                state["context"] = f"KEYWORD_RESULTS:{len(matching_papers)} papers found"
                state["search_results"] = matching_papers
                print(f"Keyword search found {len(matching_papers)} papers")
            else:
                state["context"] = f"No papers found matching keywords: {', '.join(keywords)}"
                state["search_results"] = []
                print("No matching papers found for keywords")
                
        except Exception as e:
            print(f"Error in keyword search: {e}")
            state["context"] = "Error searching for keywords"
            state["search_results"] = []
        
        return state
    
    def _graph_properties(self, state: AgentState) -> AgentState:
        """Answers questions about graph properties and statistics"""
        question = state["question"]
        print(f"Processing graph properties question: {question}")
        
        try:
            # Get the actual graph object
            graph_obj = self._get_graph_object()
            if not graph_obj:
                state["context"] = "Graph not available"
                return state
            
            # Extract what property is being asked about
            question_lower = question.lower()
            
            if "which topics" in question_lower or "what topics" in question_lower:
                if "most" in question_lower or "read" in question_lower:
                    # Find topics with most papers (most read about)
                    topic_nodes = [n for n, attr in graph_obj.graph.nodes(data=True) if attr.get('type') == 'topic']
                    if topic_nodes:
                        # Sort topics by degree (number of connected papers)
                        sorted_topics = sorted(topic_nodes, key=lambda t: graph_obj.graph.degree(t), reverse=True)
                        top_topics = sorted_topics[:5]  # Top 5
                        topic_info = [f"{topic} ({graph_obj.graph.degree(topic)} papers)" for topic in top_topics]
                        state["context"] = f"Topics you've read most about: {', '.join(topic_info)}"
                    else:
                        state["context"] = "No topics found in graph"
                else:
                    # General topic listing
                    topic_nodes = [n for n, attr in graph_obj.graph.nodes(data=True) if attr.get('type') == 'topic']
                    state["context"] = f"All topics in graph: {', '.join(topic_nodes[:10])}"
            
            elif "min" in question_lower and ("topic" in question_lower or "connected" in question_lower):
                min_topic = graph_obj.find_min_topic()
                if min_topic:
                    degree = graph_obj.graph.degree(min_topic)
                    state["context"] = f"Least connected topic: '{min_topic}' with {degree} connections"
                else:
                    state["context"] = "No topics found in graph"
            
            elif "max" in question_lower and ("topic" in question_lower or "connected" in question_lower):
                max_topic = graph_obj.find_max_topic()
                if max_topic:
                    degree = graph_obj.graph.degree(max_topic)
                    state["context"] = f"Most connected topic: '{max_topic}' with {degree} connections"
                else:
                    state["context"] = "No topics found in graph"
            
            elif "path" in question_lower and "between" in question_lower:
                # Extract entities for pathfinding
                start_node, end_node = self._extract_entities(question)
                if start_node and end_node:
                    path = graph_obj.find_path(start_node, end_node)
                    if path:
                        path_str = " -> ".join(path)
                        state["context"] = f"Path between '{start_node}' and '{end_node}': {path_str}"
                    else:
                        state["context"] = f"No path found between '{start_node}' and '{end_node}'"
                else:
                    state["context"] = "Could not identify entities for pathfinding"
            
            elif "how many" in question_lower or "number of" in question_lower:
                if "nodes" in question_lower:
                    count = graph_obj.graph.number_of_nodes()
                    state["context"] = f"Graph has {count} nodes total"
                elif "edges" in question_lower or "connections" in question_lower:
                    count = graph_obj.graph.number_of_edges()
                    state["context"] = f"Graph has {count} edges total"
                else:
                    # General statistics
                    num_nodes = graph_obj.graph.number_of_nodes()
                    num_edges = graph_obj.graph.number_of_edges()
                    state["context"] = f"Graph statistics: {num_nodes} nodes, {num_edges} edges"
            
            else:
                # Default to general graph statistics
                num_nodes = graph_obj.graph.number_of_nodes()
                num_edges = graph_obj.graph.number_of_edges()
                min_topic = graph_obj.find_min_topic()
                max_topic = graph_obj.find_max_topic()
                state["context"] = f"Graph: {num_nodes} nodes, {num_edges} edges. Min connected: {min_topic}, Max connected: {max_topic}"
            
            print(f"Graph properties result: {state['context']}")
                
        except Exception as e:
            print(f"Error in graph properties: {e}")
            state["context"] = "Error analyzing graph properties"
        
        return state
    
    def _generate_answer(self, state: AgentState) -> AgentState:
        """Generate answer using OpenAI, grounded only in provided context"""
        question = state["question"]
        context = state.get("context", "")
        
        # Handle keyword search results differently - don't send to LLM
        if context.startswith("KEYWORD_RESULTS:"):
            state["answer"] = "SEARCH_RESULTS"  # Special marker for frontend
            return state
        
        # Check if this is a chain reasoning result (already synthesized by LLM)
        if "CHAIN_REASONING_RESULT:" in context:
            # Extract the synthesized explanation
            state["answer"] = context.replace("CHAIN_REASONING_RESULT:\n\n", "")
            return state
        
        # Check if context contains paper summaries (for grounding)
        if "Paper summaries for grounding:" in context:
            system_message = """You are a helpful assistant that answers questions about research papers and academic topics. 
            
            IMPORTANT: You must base your response ONLY on the paper summaries provided in the context. Do not use any external knowledge or make assumptions beyond what is explicitly stated in the provided summaries. If the provided summaries don't contain enough information to answer the question, say so clearly."""
        elif "Path found:" in context:
            system_message = """You are a helpful assistant that answers questions about research papers and academic topics.

            IMPORTANT: You must base your response ONLY on the information provided in the context. Explain the connection concisely."""
        else:
            system_message = "You are a helpful assistant that answers questions about research papers and academic topics."
        
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=f"Context: {context}\n\nQuestion: {question}")
        ]
        
        response = self.llm.invoke(messages)
        state["answer"] = response.content
        
        return state
    
    def get_mermaid_diagram(self) -> str:
        """Generate mermaid diagram dynamically from the actual graph structure"""
        mermaid_lines = [
            "graph TD",
            "    START([User Question]) --> route_question",
            "    route_question -->|Connection Query| bridge_question",
            "    route_question -->|Explanation Query| explain_question",
            "    route_question -->|Search Query| keyword_search",
            "    route_question -->|Properties Query| graph_properties",
            "    bridge_question --> generate_answer",
            "    explain_question --> generate_answer", 
            "    keyword_search --> generate_answer",
            "    graph_properties --> generate_answer",
            "    generate_answer --> END([Final Answer])",
            "    generate_answer -.-> OpenAI[OpenAI GPT-3.5]",
            "    OpenAI -.-> generate_answer",
            "    bridge_question -.-> ChainReasoning[Chain of LLM Calls]",
            "    ChainReasoning -.-> bridge_question",
            "",
            "    style START fill:#e1f5fe",
            "    style END fill:#c8e6c9", 
            "    style OpenAI fill:#fff3e0",
            "    style ChainReasoning fill:#fff3e0",
            "    style route_question fill:#ffecb3",
            "    style bridge_question fill:#e8f5e8",
            "    style explain_question fill:#f3e5f5",
            "    style keyword_search fill:#fce4ec",
            "    style graph_properties fill:#e3f2fd",
            "    style generate_answer fill:#e1f5fe"
        ]
        
        return "\n".join(mermaid_lines)
    
    async def answer_question(self, question: str) -> str:
        """Main method to answer a question"""
        initial_state = {
            "question": question,
            "context": "",
            "answer": "",
            "search_results": []
        }
        
        result = self.graph.invoke(initial_state)
        self._last_state = result  # Store the last state
        return result["answer"]