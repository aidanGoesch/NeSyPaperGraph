import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    question: str
    context: str
    answer: str

class QuestionAgent:
    def __init__(self, graph_data=None):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.graph_data = graph_data
        self.graph = self._build_graph()
    
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
            state["context"] = f"Finding path between '{start_node}' and '{end_node}': {path_result}"
            print(f"Bridge analysis: {start_node} -> {end_node} | Result: {path_result}")
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
    
    def _find_path_in_graph(self, start_entity, end_entity):
        """Find path between two entities using the graph object's pathfinding"""
        try:
            # Get the actual graph object
            from services.graph_builder import create_dummy_graph
            graph_obj = create_dummy_graph()
            
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
                path_str = " -> ".join([str(node) for node in path])
                return f"Path found: {path_str}"
            except nx.NetworkXNoPath:
                return "Those are not related"
                
        except Exception as e:
            print(f"Error in pathfinding: {e}")
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
            from services.graph_builder import create_dummy_graph
            graph_obj = create_dummy_graph()
            
            relevant_nodes = []
            
            # Find directly matching nodes
            for node, data in graph_obj.graph.nodes(data=True):
                node_str = str(node).lower()
                
                # Check if any explain term matches this node
                for term in explain_terms:
                    if term in node_str:
                        relevant_nodes.append(str(node))
                        
                        # Also find semantically connected nodes
                        for neighbor in graph_obj.graph.neighbors(node):
                            neighbor_str = str(neighbor)
                            if neighbor_str not in relevant_nodes:
                                relevant_nodes.append(neighbor_str)
                        break
            
            if relevant_nodes:
                state["context"] = f"Explaining with relevant nodes: {', '.join(relevant_nodes[:10])}"  # Limit output
                print(f"Explain found {len(relevant_nodes)} relevant nodes: {relevant_nodes[:5]}...")
            else:
                state["context"] = "No relevant information found in graph"
                print("No relevant nodes found for explanation")
                
        except Exception as e:
            print(f"Error in explain search: {e}")
            state["context"] = "Error finding relevant information"
        
        return state
    
    def _keyword_search(self, state: AgentState) -> AgentState:
        """Finds papers according to a keyword search"""
        question = state["question"]
        print(f"Processing keyword search: {question}")
        
        # Extract keywords from question
        keywords = question.lower().split()
        
        try:
            # Get the actual graph object
            from services.graph_builder import create_dummy_graph
            graph_obj = create_dummy_graph()
            
            matching_nodes = []
            
            # Search through all nodes in the graph
            for node, data in graph_obj.graph.nodes(data=True):
                node_str = str(node).lower()
                
                # Check if any keyword matches this node
                for keyword in keywords:
                    if keyword in node_str:
                        matching_nodes.append(str(node))
                        break
            
            if matching_nodes:
                state["context"] = f"Found matching nodes: {', '.join(matching_nodes)}"
                print(f"Keyword search found: {matching_nodes}")
            else:
                state["context"] = "No matching papers or topics found"
                print("No matching nodes found for keywords")
                
        except Exception as e:
            print(f"Error in keyword search: {e}")
            state["context"] = "Error searching for keywords"
        
        return state
    
    def _graph_properties(self, state: AgentState) -> AgentState:
        """Answers questions about graph properties and statistics"""
        question = state["question"]
        print(f"Processing graph properties question: {question}")
        
        try:
            # Get the actual graph object
            from services.graph_builder import create_dummy_graph
            graph_obj = create_dummy_graph()
            
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
        """Generate answer using OpenAI"""
        question = state["question"]
        context = state.get("context", "")
        
        messages = [
            SystemMessage(content="You are a helpful assistant that answers questions about research papers and academic topics."),
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
            "",
            "    style START fill:#e1f5fe",
            "    style END fill:#c8e6c9", 
            "    style OpenAI fill:#fff3e0",
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
            "answer": ""
        }
        
        result = self.graph.invoke(initial_state)
        return result["answer"]
