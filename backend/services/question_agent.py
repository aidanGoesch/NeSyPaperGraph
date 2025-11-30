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
        self.conversation_history = []  # Store conversation context
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
        
        # Check conversation history for context
        previous_context = ""
        if len(self.conversation_history) > 0:
            last_entry = self.conversation_history[-1]
            last_question = last_entry.get("question", "").lower()
            previous_context = last_question
            
            # If previous question was about recommendations/gaps and current is a follow-up
            if any(word in last_question for word in ["gaps", "missing", "recommendations", "suggest", "what could i read", "what should i read"]):
                if any(phrase in question for phrase in ["what about", "tell me about", "about", "and"]):
                    print(f"Detected follow-up to recommendation question, routing to properties")
                    return "properties"
        
        # Graph properties questions
        property_patterns = [
            "how many", "number of", "count", "connections", "degree", 
            "neighbors", "edges", "nodes", "size", "statistics",
            "min", "max", "most", "least", "which topics", "what topics",
            "read most", "reading", "studied", "focus on", "interested in",
            "gaps", "missing", "weak spots", "what could i read", "what should i read",
            "recommendations", "suggest", "areas to explore", "underexplored"
        ]
        
        # Semantic edge patterns (check before bridge patterns)
        semantic_patterns = ["semantic", "strongest", "similar papers", "similarity", "most similar"]
        
        # Bridge questions - looking for relationships/connections
        bridge_patterns = [
            "connect", "relationship", "between", "how are", 
            "related to", "connection", "compare", "versus", 
            "vs", "difference between", "how is", "relate"
        ]
        
        # Check semantic patterns first (more specific)
        if any(pattern in question for pattern in semantic_patterns):
            question_type = "properties"
        # Check for gap/recommendation queries (very specific)
        elif any(word in question for word in ["gaps", "missing", "weak spots", "recommendations", "suggest", "underexplored", "what could i read", "what should i read"]):
            question_type = "properties"
        # Check explain patterns before other properties (more specific)
        elif any(phrase in question for phrase in ["what is", "what are", "explain", "define", "describe", "what about", "tell me about"]):
            question_type = "explain"
        elif any(pattern in question for pattern in property_patterns):
            question_type = "properties"
        elif any(pattern in question for pattern in bridge_patterns):
            question_type = "bridge"
        elif "about" in question:
            question_type = "explain"
        else:
            question_type = "search"
        
        print(f"Question type detected: {question_type} for question: '{state['question']}' (previous: '{previous_context}')")
        return question_type
    
    def _bridge_question(self, state: AgentState) -> AgentState:
        """Answers questions of how X and Y are connected based on the graph"""
        question = state["question"]
        print(f"Processing bridge question: {question}")
        
        # Clear any previous path at the start of bridge question processing
        # It will be set again if we successfully find a path
        if hasattr(self, '_last_path'):
            delattr(self, '_last_path')
        
        start_node, end_node = self._extract_entities(question)
        
        if start_node and end_node:
            path_result = self._find_path_in_graph(start_node, end_node)
            state["context"] = path_result
            
            # Store path information for mermaid diagram (only if path was found)
            if hasattr(self, '_last_path'):
                state["path_info"] = self._last_path
            
            print(f"Bridge analysis: {start_node} -> {end_node}")
        else:
            state["context"] = f"Could not identify two entities to connect in: {question}"
            print(f"Could not parse entities from bridge question: {question}")
            # Ensure no path is set if we couldn't extract entities
            if hasattr(self, '_last_path'):
                delattr(self, '_last_path')
        
        return state
    
    def _extract_entities(self, question):
        """Extract start and end entities from bridge questions using LLM"""
        
        # Get available topics from graph for context
        graph_obj = self._get_graph_object()
        topics_context = ""
        if graph_obj:
            topics = [str(node) for node, data in graph_obj.graph.nodes(data=True) if data.get('type') == 'topic']
            if topics:
                topics_context = f"\n\nAvailable topics in the research graph:\n{', '.join(topics[:50])}"  # Limit to 50 topics
        
        extraction_prompt = f"""
You are extracting entities from a question about a research paper knowledge graph.

CRITICAL RULES:
1. Extract EXACTLY what the user wrote, do NOT expand abbreviations
2. If user says "LLMs", extract "LLMs" (not "Large Language Models")
3. If user says "RL", extract "RL" (not "Reinforcement Learning")
4. Return ONLY the two entities separated by a pipe (|)
{topics_context}

Examples:
"How are generative models and transfer learning related?" -> generative models|transfer learning
"How are LLMs and RL related?" -> LLMs|RL
"Compare neural networks and decision trees" -> neural networks|decision trees

Question: {question}

Entities (exactly as written in question):"""
        
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
    
    def _extract_topics_from_question(self, question, graph_obj):
        """Extract topic names from a question using LLM"""
        topics_in_graph = [str(node) for node, data in graph_obj.graph.nodes(data=True) if data.get('type') == 'topic']
        
        extraction_prompt = f"""Extract the topic names from this question. Return ONLY the topic names, one per line.

Available topics in the graph:
{', '.join(topics_in_graph[:100])}

Question: {question}

Topics (one per line):"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
            topics_text = response.content.strip()
            
            # Parse topics from response
            extracted = [line.strip() for line in topics_text.split('\n') if line.strip()]
            
            # Resolve each extracted topic to actual graph nodes
            resolved_topics = []
            for topic in extracted:
                resolved = self._resolve_synonym(topic, graph_obj)
                if resolved:
                    resolved_topics.append(resolved)
            
            return resolved_topics
        except Exception as e:
            print(f"Topic extraction failed: {e}")
            return []
    
    def _resolve_synonym(self, entity, graph_obj):
        """Resolve entity to actual graph node, checking synonyms"""
        entity_lower = entity.lower().strip()
        
        # Common abbreviation mappings - map to key terms that MUST appear
        abbrev_map = {
            'llms': 'language model',
            'llm': 'language model',
            'rl': 'reinforcement',
            'nlp': 'natural language processing',
            'cv': 'computer vision',
            'ml': 'machine learning',
            'dl': 'deep learning',
            'nn': 'neural network',
            'cnn': 'convolutional',
            'rnn': 'recurrent'
        }
        
        # If it's an abbreviation, use the expanded form
        if entity_lower in abbrev_map:
            search_term = abbrev_map[entity_lower]
            print(f"Expanding abbreviation '{entity}' -> '{search_term}'")
        else:
            search_term = entity_lower
        
        # Direct match in node names
        for node, data in graph_obj.graph.nodes(data=True):
            node_lower = str(node).lower()
            if search_term in node_lower:
                print(f"Resolved '{entity}' -> '{node}' (direct match)")
                return node
        
        # Check merged topics
        for node, data in graph_obj.graph.nodes(data=True):
            if data.get('type') == 'topic' and 'merged_topics' in data:
                for merged in data['merged_topics']:
                    if search_term in merged.lower():
                        print(f"Resolved '{entity}' -> '{node}' (merged topic)")
                        return node
        
        # Check topic_synonyms
        if hasattr(graph_obj, 'topic_synonyms'):
            for topic, synonyms in graph_obj.topic_synonyms.items():
                for syn in synonyms:
                    if search_term in syn.lower():
                        for node in graph_obj.graph.nodes():
                            if topic.lower() in str(node).lower():
                                print(f"Resolved '{entity}' -> '{node}' (synonym)")
                                return node
        
        print(f"Could not resolve '{entity}' to any graph node")
        return None
    
    def _find_path_in_graph(self, start_entity, end_entity):
        """Find path between two entities and use chain reasoning to explain the connection"""
        try:
            # Get the actual graph object
            graph_obj = self._get_graph_object()
            if not graph_obj:
                return "Graph not available"
            
            print(f"Attempting to resolve: '{start_entity}' and '{end_entity}'")
            
            # Resolve entities to actual nodes (handles synonyms)
            start_node = self._resolve_synonym(start_entity, graph_obj)
            end_node = self._resolve_synonym(end_entity, graph_obj)
            
            print(f"Resolved to: start_node='{start_node}', end_node='{end_node}'")
            
            if not start_node:
                # List similar topics to help user
                topics = [str(n) for n, d in graph_obj.graph.nodes(data=True) if d.get('type') == 'topic']
                similar = [t for t in topics if any(word in t.lower() for word in start_entity.lower().split())]
                msg = f"Could not find '{start_entity}' in the graph."
                if similar:
                    msg += f" Did you mean: {', '.join(similar[:5])}?"
                return msg
            
            if not end_node:
                # List similar topics to help user
                topics = [str(n) for n, d in graph_obj.graph.nodes(data=True) if d.get('type') == 'topic']
                similar = [t for t in topics if any(word in t.lower() for word in end_entity.lower().split())]
                msg = f"Could not find '{end_entity}' in the graph."
                if similar:
                    msg += f" Did you mean: {', '.join(similar[:5])}?"
                return msg
            
            # Use NetworkX to find shortest path
            import networkx as nx
            try:
                path = nx.shortest_path(graph_obj.graph, start_node, end_node)
                path_str = " → ".join([str(node) for node in path])
                
                # Store path for mermaid diagram (only when path is successfully found)
                self._last_path = {
                    "nodes": path,
                    "start_entity": start_entity,
                    "end_entity": end_entity
                }
                
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
                # Clear path if no path exists
                if hasattr(self, '_last_path'):
                    delattr(self, '_last_path')
                return "Those are not related"
                
        except Exception as e:
            print(f"Error in pathfinding: {e}")
            import traceback
            traceback.print_exc()
            # Clear path on error
            if hasattr(self, '_last_path'):
                delattr(self, '_last_path')
            return "Those are not related"
    
    def _explain_question(self, state: AgentState) -> AgentState:
        """Answers questions of 'explain y', grounded in actual graph content"""
        question = state["question"]
        print(f"Processing explain question: {question}")
        
        # Extract what needs to be explained
        explain_terms = []
        words = question.lower().split()
        skip_words = {"explain", "what", "is", "are", "define", "describe", "the", "a", "an"}
        
        # Skip initial question words and articles
        start_idx = 0
        for i, word in enumerate(words):
            if word not in skip_words:
                start_idx = i
                break
        
        explain_terms = words[start_idx:]
        
        if not explain_terms:
            explain_terms = words  # fallback to all words
        
        # Use keyword search to find relevant papers
        search_query = " ".join(explain_terms)
        search_state = {"question": search_query, "context": "", "search_results": []}
        search_result = self._keyword_search(search_state)
        
        if search_result.get("search_results"):
            papers = search_result["search_results"][:3]  # Use top 3 papers for explanation
            
            # Collect paper content for grounding
            paper_content = []
            for paper in papers:
                content = f"Paper: {paper['title']}\n"
                if paper.get('summary'):
                    content += f"Summary: {paper['summary']}\n"
                if paper.get('topics'):
                    content += f"Topics: {', '.join(paper['topics'])}\n"
                paper_content.append(content)
            
            # Create grounded context
            grounding_text = "\n\n---\n\n".join(paper_content)
            state["context"] = f"Based on papers in your collection about '{search_query}':\n\n{grounding_text}\n\nPlease explain '{search_query}' based ONLY on the information from these papers. IMPORTANT: When citing papers, use square brackets around the exact paper title like [Paper Title]. Do not use quotes. For example: 'According to [Attention Is All You Need]...' or 'As described in [BERT: Pre-training of Deep Bidirectional Transformers]...'. Do not include information not found in these papers."
            
            print(f"Explain grounded in {len(papers)} papers from graph")
        else:
            # Fallback: no relevant papers found
            state["context"] = f"No papers found in your collection about '{search_query}'. Cannot provide explanation based on your graph content."
            print("No relevant papers found for explanation")
        
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
                # Check if user wants more than 5 results
                question_lower = question.lower()
                wants_all = any(phrase in question_lower for phrase in ['all', 'every', 'complete list', 'full list', 'everything'])
                wants_specific_number = any(word.isdigit() and int(word) > 5 for word in question.split())
                
                # Limit to 5 unless specifically requested otherwise
                if not wants_all and not wants_specific_number:
                    matching_papers = matching_papers[:5]
                
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
            
            # Check for set cover queries
            if ("fewest" in question_lower or "minimum" in question_lower or "minimal" in question_lower) and \
               ("papers" in question_lower or "read" in question_lower) and \
               ("learn" in question_lower or "cover" in question_lower or "understand" in question_lower):
                
                print(f"Detected set cover query: {question}")
                
                # Extract topics from the question
                topics = self._extract_topics_from_question(question, graph_obj)
                
                if not topics:
                    state["context"] = "Could not identify topics from your question."
                    return state
                
                print(f"Extracted topics: {topics}")
                
                # Create subgraph with these topics and their neighbors
                import networkx as nx
                subgraph_nodes = set(topics)
                for topic in topics:
                    if topic in graph_obj.graph:
                        subgraph_nodes.update(graph_obj.graph.neighbors(topic))
                
                subgraph = graph_obj.graph.subgraph(subgraph_nodes).copy()
                
                # Create temporary PaperGraph object for set_cover
                from models.graph import PaperGraph
                temp_graph = PaperGraph()
                temp_graph.graph = subgraph
                
                # Call set_cover
                from services.verification import set_cover
                chosen_papers = set_cover(temp_graph)
                
                if chosen_papers:
                    # Format as search results
                    search_results = []
                    for paper_title in chosen_papers:
                        paper_data = graph_obj.graph.nodes[paper_title].get('data')
                        if paper_data:
                            search_results.append({
                                "title": paper_title,
                                "author": getattr(paper_data, 'authors', None) or 'Unknown',
                                "summary": getattr(paper_data, 'summary', None) or paper_data.text[:500] + "...",
                                "topics": getattr(paper_data, 'topics', []),
                                "node_id": paper_title
                            })
                    
                    state["context"] = f"SET_COVER_RESULTS:{len(chosen_papers)} papers needed to cover {len(topics)} topics"
                    state["search_results"] = search_results
                    print(f"Set cover found {len(chosen_papers)} papers for {len(topics)} topics")
                else:
                    state["context"] = "Could not find a minimal set of papers to cover those topics."
                    state["search_results"] = []
                
                return state
            
            # Check for semantic edge queries
            if ("semantic" in question_lower) or \
               ("most similar" in question_lower) or \
               ("strongest" in question_lower and "similar" in question_lower) or \
               ("most related" in question_lower and "paper" in question_lower):
                
                print(f"Detected semantic edge query: {question}")
                
                # Get all semantic edges (edges between papers)
                semantic_edges = []
                for n1, n2, data in graph_obj.graph.edges(data=True):
                    if data.get('type') == 'semantic':
                        semantic_edges.append((n1, n2, data.get('weight', 0)))
                
                print(f"Found {len(semantic_edges)} semantic edges")
                
                if semantic_edges:
                    # Sort by weight (similarity)
                    semantic_edges.sort(key=lambda x: x[2], reverse=True)
                    
                    # Get top 10 strongest connections
                    top_edges = semantic_edges[:10]
                    
                    # Format as search results (pairs with nested papers)
                    search_results = []
                    for paper1, paper2, weight in top_edges:
                        # Get paper data for both papers
                        paper1_data = graph_obj.graph.nodes[paper1].get('data')
                        paper2_data = graph_obj.graph.nodes[paper2].get('data')
                        
                        search_results.append({
                            "type": "semantic_pair",
                            "similarity": weight,
                            "papers": [
                                {
                                    "title": paper1,
                                    "authors": paper1_data.authors if paper1_data and paper1_data.authors else [],
                                    "publication_date": paper1_data.publication_date if paper1_data else None,
                                    "topics": paper1_data.topics if paper1_data else [],
                                    "node_id": paper1
                                },
                                {
                                    "title": paper2,
                                    "authors": paper2_data.authors if paper2_data and paper2_data.authors else [],
                                    "publication_date": paper2_data.publication_date if paper2_data else None,
                                    "topics": paper2_data.topics if paper2_data else [],
                                    "node_id": paper2
                                }
                            ]
                        })
                    
                    state["context"] = f"SEMANTIC_RESULTS:{len(semantic_edges)} semantic connections found"
                    state["search_results"] = search_results
                    print(f"Returning {len(search_results)} semantic pair results")
                else:
                    state["context"] = "No semantic edges found in the graph. Papers may not have embeddings yet."
                    state["search_results"] = []
                    print("No semantic edges found")
                
                return state
            
            if ("which topics" in question_lower or "what topics" in question_lower or 
                "read most" in question_lower or "reading" in question_lower or
                "studied" in question_lower or "focus on" in question_lower):
                
                if ("most" in question_lower or "read" in question_lower or 
                    "studied" in question_lower or "focus" in question_lower):
                    # Find topics with most papers (most read about)
                    topic_nodes = [n for n, attr in graph_obj.graph.nodes(data=True) if attr.get('type') == 'topic']
                    if topic_nodes:
                        # Sort topics by degree (number of connected papers)
                        sorted_topics = sorted(topic_nodes, key=lambda t: graph_obj.graph.degree(t), reverse=True)
                        
                        if len(sorted_topics) == 1:
                            top_topic = sorted_topics[0]
                            degree = graph_obj.graph.degree(top_topic)
                            state["context"] = f"You've read most about '{top_topic}' with {degree} papers"
                        else:
                            top_topics = sorted_topics[:5]  # Top 5
                            topic_info = [f"'{topic}' ({graph_obj.graph.degree(topic)} papers)" for topic in top_topics]
                            state["context"] = f"Topics you've read most about: {', '.join(topic_info)}"
                    else:
                        state["context"] = "No topics found in graph"
                else:
                    # General topic listing
                    topic_nodes = [n for n, attr in graph_obj.graph.nodes(data=True) if attr.get('type') == 'topic']
                    if len(topic_nodes) <= 10:
                        state["context"] = f"All topics in your collection: {', '.join(topic_nodes)}"
                    else:
                        state["context"] = f"You have {len(topic_nodes)} topics. Top topics: {', '.join(topic_nodes[:10])}"
            
            elif ("gaps" in question_lower or "missing" in question_lower or 
                  "weak spots" in question_lower or "what could i read" in question_lower or
                  "what should i read" in question_lower or "recommendations" in question_lower or
                  "suggest" in question_lower or "areas to explore" in question_lower or
                  "underexplored" in question_lower):
                
                # Use Z3-based research gap identification
                from services.verification import identify_research_gap
                
                try:
                    gaps = identify_research_gap(graph_obj, k=5, weight=1)
                    
                    if gaps:
                        # Build context for LLM to explain gaps
                        gap_descriptions = []
                        for topic_a, topic_b in gaps:
                            # Get paper counts for each topic
                            papers_a = len(list(graph_obj.graph.neighbors(topic_a)))
                            papers_b = len(list(graph_obj.graph.neighbors(topic_b)))
                            
                            # Get path length
                            path = graph_obj.find_path(topic_a, topic_b)
                            path_length = len(path) if path else 0
                            
                            gap_descriptions.append(
                                f"- **{topic_a}** ({papers_a} papers) ↔ **{topic_b}** ({papers_b} papers): "
                                f"{'No direct connection' if path_length == 0 else f'Distant connection ({path_length} hops)'}"
                            )
                        
                        state["context"] = (
                            "**Research Gaps Identified (using Z3 optimization):**\n\n"
                            "These topic pairs are semantically related but poorly connected in your collection, "
                            "representing potential novel research directions:\n\n" +
                            "\n".join(gap_descriptions) +
                            "\n\n*Gaps are ranked by interestingness: path length × semantic similarity between topics.*"
                        )
                    else:
                        state["context"] = "No significant research gaps found. Your collection has good coverage across related topics."
                    
                except Exception as e:
                    print(f"Error in gap identification: {e}")
                    import traceback
                    traceback.print_exc()
                    state["context"] = "Error analyzing research gaps. Make sure papers have embeddings for semantic analysis."
            
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
        if context.startswith("KEYWORD_RESULTS:") or context.startswith("SEMANTIC_RESULTS:") or context.startswith("SET_COVER_RESULTS:"):
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
    
    @staticmethod
    def get_agent_architecture_diagram() -> str:
        """Get the static agent architecture diagram (always the same) - static method that doesn't depend on instance state"""
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
    
    def get_mermaid_diagram(self) -> str:
        """Generate mermaid diagram dynamically from the actual graph structure (for chat responses)"""
        
        # If we have path information from a bridge question, show the path
        if hasattr(self, '_last_path') and self._last_path:
            path_info = self._last_path
            mermaid_lines = ["graph LR"]
            
            # Add nodes and connections for the path
            for i, node in enumerate(path_info["nodes"]):
                node_id = f"node{i}"
                node_label = str(node)
                
                # Truncate long labels
                if len(node_label) > 30:
                    node_label = node_label[:27] + "..."
                
                # Style nodes differently based on type
                if i == 0:  # Start node
                    mermaid_lines.append(f'    {node_id}["{node_label}"]')
                    mermaid_lines.append(f"    style {node_id} fill:#e1f5fe,stroke:#01579b")
                elif i == len(path_info["nodes"]) - 1:  # End node
                    mermaid_lines.append(f'    {node_id}["{node_label}"]')
                    mermaid_lines.append(f"    style {node_id} fill:#c8e6c9,stroke:#2e7d32")
                else:  # Intermediate nodes
                    mermaid_lines.append(f'    {node_id}["{node_label}"]')
                    mermaid_lines.append(f"    style {node_id} fill:#fff3e0,stroke:#ef6c00")
                
                # Add connection to next node
                if i < len(path_info["nodes"]) - 1:
                    next_node_id = f"node{i+1}"
                    mermaid_lines.append(f"    {node_id} --> {next_node_id}")
            
            return "\n".join(mermaid_lines)
        
        # If no path, return None (no diagram for chat)
        return None
    
    async def answer_question(self, question: str) -> str:
        """Main method to answer a question"""
        # Clear previous path at the start of each new question
        # This ensures mermaid/path are only returned if the current query uses them
        if hasattr(self, '_last_path'):
            delattr(self, '_last_path')
        
        initial_state = {
            "question": question,
            "context": "",
            "answer": "",
            "search_results": []
        }
        
        result = self.graph.invoke(initial_state)
        self._last_state = result  # Store the last state
        
        # Store in conversation history for context
        self.conversation_history.append({
            "question": question,
            "answer": result["answer"],
            "type": getattr(self, '_last_question_type', 'unknown')
        })
        
        # Keep only last 5 conversations for context
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]
        
        return result["answer"]