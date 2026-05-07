import json
import os
import re
import uuid
from typing import TypedDict, List, Tuple, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from services.graph_activation_service import GraphActivationService, ActivationConfig
from services.llm_service import OpenAILLMClient

load_dotenv()

class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    search_results: list
    sources_used: list
    answer_structured: dict

class QuestionAgent:
    def __init__(self, graph_data=None, graph_obj=None):
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini-2025-08-07"),
            temperature=1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.graph_data = graph_data
        self.graph_obj = graph_obj
        self.conversation_history = []  # Store conversation context
        self.graph = self._build_graph()
        self._embedding_client = None

    def _default_structured_answer(self, answer_text: str, warning: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "segments": [{"text": str(answer_text or ""), "claim_id": None}],
            "claims": [],
            "warnings": [],
        }
        if warning:
            payload["warnings"].append(str(warning))
        return payload

    @staticmethod
    def _coerce_claim(claim: Any) -> dict[str, Any] | None:
        if not isinstance(claim, dict):
            return None
        claim_id = str(claim.get("id") or f"claim_{uuid.uuid4().hex[:8]}")
        claim_text = str(claim.get("text") or "").strip()
        purpose = str(claim.get("purpose") or "").strip()
        relation_to_question = str(claim.get("relation_to_question") or "").strip()
        citations = claim.get("citations")
        if (
            not claim_text
            or not purpose
            or not relation_to_question
            or not isinstance(citations, list)
        ):
            return None
        normalized_citations = []
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            paper_title = str(citation.get("paper_title") or "").strip()
            excerpt = str(citation.get("excerpt") or "").strip()
            if not paper_title or not excerpt:
                continue
            normalized_citations.append(
                {
                    "paper_title": paper_title,
                    "excerpt": excerpt,
                }
            )
        if not normalized_citations:
            return None
        return {
            "id": claim_id,
            "text": claim_text,
            "purpose": purpose,
            "relation_to_question": relation_to_question,
            "citations": normalized_citations,
        }

    def _validate_structured_answer(self, payload: Any) -> tuple[bool, str]:
        if not isinstance(payload, dict):
            return False, "response is not a JSON object"
        segments = payload.get("segments")
        claims = payload.get("claims")
        if not isinstance(segments, list) or len(segments) == 0:
            return False, "segments must be a non-empty array"
        if not isinstance(claims, list):
            return False, "claims must be an array"

        claim_ids = {str(c.get("id")) for c in claims if isinstance(c, dict) and c.get("id")}
        for segment in segments:
            if not isinstance(segment, dict):
                return False, "segment entry is not an object"
            text = str(segment.get("text") or "")
            if not text.strip():
                return False, "segment text cannot be empty"
            claim_id = segment.get("claim_id")
            if claim_id is not None and str(claim_id) not in claim_ids:
                return False, "segment references unknown claim_id"

        for claim in claims:
            normalized = self._coerce_claim(claim)
            if normalized is None:
                return False, "each claim must contain text, purpose, relation_to_question, and at least one citation with paper_title + excerpt"
        return True, ""

    def _normalize_structured_answer(
        self, payload: dict[str, Any], fallback_answer: str = ""
    ) -> dict[str, Any]:
        claims_raw = payload.get("claims") if isinstance(payload.get("claims"), list) else []
        normalized_claims: list[dict[str, Any]] = []
        for claim in claims_raw:
            normalized = self._coerce_claim(claim)
            if normalized:
                normalized_claims.append(normalized)
        claim_id_set = {claim["id"] for claim in normalized_claims}

        segments_raw = payload.get("segments") if isinstance(payload.get("segments"), list) else []
        normalized_segments: list[dict[str, Any]] = []
        for segment in segments_raw:
            if not isinstance(segment, dict):
                continue
            text = str(segment.get("text") or "")
            if not text.strip():
                continue
            claim_id = segment.get("claim_id")
            claim_id_value = str(claim_id) if claim_id is not None else None
            if claim_id_value not in claim_id_set:
                claim_id_value = None
            normalized_segments.append({"text": text, "claim_id": claim_id_value})

        if not normalized_segments:
            normalized_segments = [{"text": fallback_answer or "", "claim_id": None}]

        warnings = payload.get("warnings")
        normalized_warnings = (
            [str(item) for item in warnings if str(item).strip()]
            if isinstance(warnings, list)
            else []
        )
        return {
            "segments": normalized_segments,
            "claims": normalized_claims,
            "warnings": normalized_warnings,
        }

    @staticmethod
    def _structured_answer_to_text(answer_structured: dict[str, Any] | None) -> str:
        if not isinstance(answer_structured, dict):
            return ""
        segments = answer_structured.get("segments")
        if not isinstance(segments, list):
            return ""
        return "".join(str(segment.get("text") or "") for segment in segments if isinstance(segment, dict)).strip()

    def _build_structured_answer(
        self, question: str, context: str, base_system_message: str
    ) -> tuple[str, dict[str, Any]]:
        structured_instructions = (
            f"{base_system_message.strip()}\n\n"
            "Output requirements:\n"
            "1) Respond with valid JSON only. No markdown code fences.\n"
            "2) Keep the answer concise and directly answer the user question first.\n"
            "3) JSON schema:\n"
            "{\n"
            '  "segments": [{"text": "string", "claim_id": "string|null"}],\n'
            '  "claims": [{"id": "string", "text": "string", "purpose": "string", "relation_to_question": "string", "citations": [{"paper_title": "string", "excerpt": "string"}]}],\n'
            '  "warnings": ["string"]\n'
            "}\n"
            "4) Every substantive claim must map to a claim_id segment and include at least one citation with a direct excerpt.\n"
            "5) Every claim must include a clear purpose and an explicit explanation of how it helps answer the user's question.\n"
            "6) In prose, follow each claim quickly with relevance to the user's question.\n"
            "7) If context is insufficient, keep claims empty and place a short note in warnings."
        )
        user_prompt = f"Context: {context}\n\nQuestion: {question}"
        initial_response = self.llm.invoke(
            [
                SystemMessage(content=structured_instructions),
                HumanMessage(content=user_prompt),
            ]
        )
        initial_content = str(initial_response.content or "").strip()

        parsed_payload = None
        parse_error = ""
        try:
            parsed_payload = json.loads(initial_content)
        except Exception as exc:
            parse_error = f"invalid JSON: {exc}"

        is_valid = False
        validation_error = parse_error
        if parsed_payload is not None:
            is_valid, validation_error = self._validate_structured_answer(parsed_payload)

        if is_valid:
            normalized = self._normalize_structured_answer(parsed_payload, fallback_answer="")
            normalized = self._ensure_clickable_claims(normalized, context=context)
            plain_answer = self._structured_answer_to_text(normalized)
            return plain_answer, normalized

        repair_prompt = (
            "Repair the previous response into valid JSON with the required schema.\n"
            f"Validation error: {validation_error or 'unknown'}\n"
            "Do not add new facts not present in Context.\n"
            "Return JSON only."
        )
        repaired_response = self.llm.invoke(
            [
                SystemMessage(content=structured_instructions),
                HumanMessage(content=user_prompt),
                HumanMessage(
                    content=f"Previous invalid response:\n{initial_content}\n\n{repair_prompt}"
                ),
            ]
        )
        repaired_content = str(repaired_response.content or "").strip()

        try:
            repaired_payload = json.loads(repaired_content)
            repaired_valid, repaired_error = self._validate_structured_answer(repaired_payload)
            if repaired_valid:
                normalized = self._normalize_structured_answer(repaired_payload, fallback_answer="")
                normalized = self._ensure_clickable_claims(normalized, context=context)
                plain_answer = self._structured_answer_to_text(normalized)
                if validation_error:
                    normalized["warnings"].append("Output required one repair pass before validation.")
                return plain_answer, normalized
            failure_reason = repaired_error
        except Exception as exc:
            failure_reason = f"repair JSON parse failed: {exc}"

        fallback_text = initial_content or repaired_content or "I could not produce a structured answer."
        warning = (
            "Structured claim-citation validation failed after one repair pass; "
            f"showing best-effort answer ({failure_reason})."
        )
        fallback_structured = self._default_structured_answer(fallback_text, warning=warning)
        fallback_structured = self._ensure_clickable_claims(fallback_structured, context=context)
        return fallback_text, fallback_structured

    @staticmethod
    def _extract_context_citations(context: str) -> list[dict[str, str]]:
        if not context or "Paper summaries for grounding:" not in context:
            return []
        pattern = re.compile(
            r"-\s*(?P<title>.+?)\s*\(activation=.*?\)\s*\n\s*Topics:.*?\n\s*Summary:\s*(?P<summary>.*?)(?=\n-\s|\Z)",
            re.DOTALL,
        )
        citations: list[dict[str, str]] = []
        for match in pattern.finditer(context):
            title = str(match.group("title") or "").strip()
            summary = " ".join(str(match.group("summary") or "").split()).strip()
            if not title or not summary:
                continue
            citations.append(
                {
                    "paper_title": title,
                    "excerpt": summary[:360],
                }
            )
        return citations

    def _ensure_clickable_claims(
        self, answer_structured: dict[str, Any], context: str
    ) -> dict[str, Any]:
        if not isinstance(answer_structured, dict):
            return answer_structured
        existing_claims = answer_structured.get("claims")
        if isinstance(existing_claims, list) and len(existing_claims) > 0:
            return answer_structured

        segments = answer_structured.get("segments")
        if not isinstance(segments, list) or len(segments) == 0:
            return answer_structured
        first_text = ""
        first_segment_index = 0
        for idx, segment in enumerate(segments):
            if not isinstance(segment, dict):
                continue
            text = str(segment.get("text") or "").strip()
            if text:
                first_text = text
                first_segment_index = idx
                break
        if not first_text:
            return answer_structured

        citations = self._extract_context_citations(context)[:2]
        if not citations:
            return answer_structured

        synthesized_claim_id = f"claim_{uuid.uuid4().hex[:8]}"
        answer_structured["claims"] = [
            {
                "id": synthesized_claim_id,
                "text": first_text[:240],
                "purpose": "Provide the most grounded available statement from retrieved summaries.",
                "relation_to_question": "This indicates what the current corpus can support for answering the user question.",
                "citations": citations,
            }
        ]
        segment_entry = segments[first_segment_index]
        if isinstance(segment_entry, dict):
            segment_entry["claim_id"] = synthesized_claim_id

        warnings = answer_structured.get("warnings")
        if not isinstance(warnings, list):
            warnings = []
            answer_structured["warnings"] = warnings
        warnings.append(
            "Auto-generated one claim citation mapping from grounding summaries."
        )
        return answer_structured

    def _plan_activation_query(
        self, current_question: str, conversation_history: list[dict[str, str]] | None = None
    ) -> str:
        history = conversation_history or []
        recent = history[-4:]
        history_lines: list[str] = []
        for turn in recent:
            q = str(turn.get("question", "") or "").strip()
            a = str(turn.get("answer", "") or "").strip()
            if q:
                history_lines.append(f"Question: {q[:220]}")
            if a:
                history_lines.append(f"Answer gist: {a[:240]}")
        history_block = "\n".join(history_lines) if history_lines else "(none)"
        planner_prompt = (
            "You are planning a graph retrieval query for paper-memory activation.\n"
            "Return ONLY JSON: {\"retrieval_query\": \"...\"}\n"
            "Prioritize the current question. Use conversation context only when needed to resolve references.\n"
            "Focus query terms on entities, concepts, and relation intent likely to retrieve relevant papers."
        )
        user_prompt = (
            f"Current question:\n{current_question}\n\n"
            f"Recent conversation context:\n{history_block}\n"
        )
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=planner_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            content = str(response.content or "").strip()
            parsed = json.loads(content)
            candidate = str(parsed.get("retrieval_query", "") or "").strip()
            if candidate:
                return candidate[:420]
        except Exception as exc:
            print(f"Activation query planner fallback: {exc}")
        return str(current_question or "").strip()
    
    def _get_graph_object(self):
        """Get the actual graph object, either from memory or by loading from S3."""
        if self.graph_obj:
            return self.graph_obj
        
        # Try to load the saved graph
        try:
            from services.storage_service import load_graph
            graph = load_graph()
            if graph:
                self.graph_obj = graph
                return graph
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
                # Track papers in path as sources
                path = self._last_path.get("nodes", [])
                graph_obj = self._get_graph_object()
                if graph_obj:
                    for node in path:
                        node_data = graph_obj.graph.nodes.get(node, {})
                        if node_data.get('type') == 'paper':
                            state["sources_used"].append(str(node))
            
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
        """Resolve entity to actual graph node, checking synonyms. Prioritize topics over papers."""
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
        
        # PRIORITY 1: Direct match in TOPIC nodes only
        for node, data in graph_obj.graph.nodes(data=True):
            if data.get('type') == 'topic':
                node_lower = str(node).lower()
                if search_term in node_lower:
                    print(f"Resolved '{entity}' -> '{node}' (topic direct match)")
                    return node
        
        # PRIORITY 2: Check merged topics
        for node, data in graph_obj.graph.nodes(data=True):
            if data.get('type') == 'topic' and 'merged_topics' in data:
                for merged in data['merged_topics']:
                    if search_term in merged.lower():
                        print(f"Resolved '{entity}' -> '{node}' (merged topic)")
                        return node
        
        # PRIORITY 3: Check topic_synonyms
        if hasattr(graph_obj, 'topic_synonyms'):
            for topic, synonyms in graph_obj.topic_synonyms.items():
                for syn in synonyms:
                    if search_term in syn.lower():
                        for node in graph_obj.graph.nodes():
                            if topic.lower() in str(node).lower():
                                print(f"Resolved '{entity}' -> '{node}' (synonym)")
                                return node
        
        # PRIORITY 4: Fallback to paper nodes (only if no topic match)
        for node, data in graph_obj.graph.nodes(data=True):
            if data.get('type') == 'paper':
                node_lower = str(node).lower()
                if search_term in node_lower:
                    print(f"Resolved '{entity}' -> '{node}' (paper fallback match)")
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
            
            # Use NetworkX to find shortest path (excluding semantic edges)
            import networkx as nx
            try:
                # Create a view of the graph without semantic edges
                def edge_filter(n1, n2):
                    edge_data = graph_obj.graph.get_edge_data(n1, n2)
                    return edge_data.get('type') != 'semantic'
                
                filtered_graph = nx.subgraph_view(graph_obj.graph, filter_edge=edge_filter)
                
                path = nx.shortest_path(filtered_graph, start_node, end_node)
                
                # Debug: print node types and edges in path
                print(f"Path nodes with types and connections:")
                for i, node in enumerate(path):
                    node_type = graph_obj.graph.nodes[node].get('type', 'unknown')
                    print(f"  [{i}] {node} ({node_type})")
                    if i < len(path) - 1:
                        next_node = path[i + 1]
                        has_edge = graph_obj.graph.has_edge(node, next_node)
                        print(f"      -> edge to next: {has_edge}")
                
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
                            paper_content = getattr(paper, 'text', '') or ''
                            
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
        search_state = {"question": search_query, "context": "", "search_results": [], "sources_used": []}
        search_result = self._keyword_search(search_state)
        
        if search_result.get("search_results"):
            papers = search_result["search_results"][:3]  # Use top 3 papers for explanation
            
            # Track papers as sources
            for paper in papers:
                state["sources_used"].append(paper['title'])
            
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
                        paper_text = getattr(paper, "text", "") or ""
                        summary = getattr(paper, 'summary', None) or paper_text[:500] + "..."
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
                                "summary": getattr(paper_data, 'summary', None) or ((getattr(paper_data, "text", "") or "")[:500] + "..."),
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
        state["answer_structured"] = self._default_structured_answer("")
        
        # Handle keyword search results differently - don't send to LLM
        if context.startswith("KEYWORD_RESULTS:") or context.startswith("SEMANTIC_RESULTS:") or context.startswith("SET_COVER_RESULTS:"):
            state["answer"] = "SEARCH_RESULTS"  # Special marker for frontend
            state["answer_structured"] = self._default_structured_answer("SEARCH_RESULTS")
            return state
        
        # Check if this is a chain reasoning result (already synthesized by LLM)
        if "CHAIN_REASONING_RESULT:" in context:
            # Extract the synthesized explanation
            chain_answer = context.replace("CHAIN_REASONING_RESULT:\n\n", "")
            state["answer"] = chain_answer
            state["answer_structured"] = self._default_structured_answer(
                chain_answer,
                warning="Chain reasoning answer was not emitted as structured claim-citation JSON.",
            )
            return state
        
        # Check if context contains paper summaries (for grounding)
        if "Paper summaries for grounding:" in context:
            system_message = """You are a helpful assistant that answers questions about research papers and academic topics.

IMPORTANT: You must base your response ONLY on the paper summaries provided in the context. Do not use external knowledge or assumptions beyond what is explicitly stated. If the summaries do not contain enough information, state that clearly."""
        elif "Path found:" in context:
            system_message = """You are a helpful assistant that answers questions about research papers and academic topics.

IMPORTANT: You must base your response ONLY on the information provided in the context."""
        else:
            system_message = "You are a helpful assistant that answers questions about research papers and academic topics."

        answer_text, answer_structured = self._build_structured_answer(
            question=question,
            context=context,
            base_system_message=system_message,
        )
        state["answer"] = answer_text
        state["answer_structured"] = answer_structured
        
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
            "    generate_answer -.-> OpenAI[OpenAI GPT-5-nano]",
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
            nodes = path_info["nodes"]
            
            # Get graph object to check node types
            graph_obj = self._get_graph_object()
            
            mermaid_lines = ["graph TD"]
            
            # Add nodes and connections
            for i, node in enumerate(nodes):
                node_id = f"node{i}"
                node_label = str(node)
                
                # Truncate long labels
                if len(node_label) > 30:
                    node_label = node_label[:27] + "..."
                
                # Get node type for styling
                node_type = 'unknown'
                if graph_obj:
                    node_data = graph_obj.graph.nodes.get(node, {})
                    node_type = node_data.get('type', 'unknown')
                
                # All nodes are rectangles
                mermaid_lines.append(f'    {node_id}["{node_label}"]')
                
                # Color based on type
                if node_type == 'paper':
                    mermaid_lines.append(f"    style {node_id} fill:#fff3e0,stroke:#ef6c00")
                else:  # topic
                    mermaid_lines.append(f"    style {node_id} fill:#f3e5f5,stroke:#7b1fa2")
                
                # Add connection to next node
                if i < len(nodes) - 1:
                    next_node_id = f"node{i+1}"
                    mermaid_lines.append(f"    {node_id} --> {next_node_id}")
            
            return "\n".join(mermaid_lines)
        
        # If no path, return None (no diagram for chat)
        return None

    def _get_embedding_client(self):
        if self._embedding_client is not None:
            return self._embedding_client
        try:
            self._embedding_client = OpenAILLMClient()
        except Exception as exc:
            print(f"Embedding client unavailable: {exc}")
            self._embedding_client = False
        return self._embedding_client

    def _embed_text(self, text: str) -> list[float]:
        client = self._get_embedding_client()
        if not client:
            return []
        try:
            return client.generate_embedding(text)
        except Exception as exc:
            print(f"Embedding generation failed: {exc}")
            return []

    def _build_grounding_context(
        self, activated_nodes: list[dict[str, Any]], max_papers: int = 6
    ) -> tuple[str, list[str]]:
        paper_nodes = [
            item for item in activated_nodes if item.get("node_type") == "paper"
        ][: max(1, max_papers)]
        if not paper_nodes or not self.graph_obj:
            return "Insufficient paper context from current activation.", []

        context_lines = ["Paper summaries for grounding:"]
        sources: list[str] = []
        for paper_activation in paper_nodes:
            node_id = paper_activation.get("node_id")
            node_data = self.graph_obj.graph.nodes.get(node_id, {})
            paper = node_data.get("data")
            if not paper:
                continue
            title = str(getattr(paper, "title", node_id) or node_id)
            summary = str(getattr(paper, "summary", "") or "No summary available.")
            topics = ", ".join(getattr(paper, "topics", []) or [])
            score = paper_activation.get("score", 0.0)
            context_lines.append(
                f"- {title} (activation={score:.2f})\n  Topics: {topics}\n  Summary: {summary}"
            )
            sources.append(title)
        if len(context_lines) == 1:
            return "Insufficient paper context from current activation.", []
        return "\n".join(context_lines), sources

    @staticmethod
    def _estimate_confidence(answer: str, activated_nodes: list[dict[str, Any]]) -> float:
        paper_nodes = [
            item for item in activated_nodes if item.get("node_type") == "paper"
        ]
        if not paper_nodes:
            return 0.2
        top_nodes = paper_nodes[:4]
        avg_score = sum(float(item.get("score", 0.0)) for item in top_nodes) / max(
            1, len(top_nodes)
        )
        coverage = min(1.0, len(paper_nodes) / 4.0)
        confidence = 0.3 + (0.45 * avg_score) + (0.25 * coverage)

        lower_answer = (answer or "").lower()
        if any(
            phrase in lower_answer
            for phrase in [
                "don't have enough",
                "do not have enough",
                "insufficient",
                "not enough information",
            ]
        ):
            confidence *= 0.6
        return max(0.0, min(0.99, round(confidence, 4)))

    @staticmethod
    def _formulate_followup_query(
        question: str, activated_nodes: list[dict[str, Any]]
    ) -> str:
        topic_focus = [
            item.get("node_id")
            for item in activated_nodes
            if item.get("node_type") == "topic"
        ][:3]
        if not topic_focus:
            return f"Expand retrieval context for: {question}"
        return f"{question}\nFocus follow-up retrieval on: {', '.join(topic_focus)}"

    async def answer_question_with_activation(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        confidence_threshold: float = 0.62,
        max_rounds: int = 2,
        seed_count: int = 6,
    ) -> dict[str, Any]:
        if hasattr(self, "_last_path"):
            delattr(self, "_last_path")

        graph_obj = self._get_graph_object()
        activation_service = GraphActivationService(graph_obj, self._embed_text)
        running_history = list(conversation_history or self.conversation_history or [])
        rounds: list[dict[str, Any]] = []
        final_answer = ""
        final_sources: list[str] = []
        confidence = 0.0
        query_used = question

        for round_index in range(max(1, max_rounds)):
            activation_query = self._plan_activation_query(
                current_question=query_used,
                conversation_history=running_history,
            )
            activation_payload = activation_service.activate(
                question=query_used,
                conversation_history=running_history,
                retrieval_query_text=activation_query,
                config=ActivationConfig(
                    seed_count=max(1, seed_count),
                    surfer_steps=90 + (round_index * 20),
                    restart_probability=0.22,
                    rng_seed=7 + round_index,
                    max_activated_nodes=80,
                ),
            )
            context, sources_used = self._build_grounding_context(
                activation_payload.get("activated_nodes", [])
            )
            workflow_state = {
                "question": question,
                "context": context,
                "answer": "",
                "search_results": [],
                "sources_used": list(sources_used),
                "answer_structured": self._default_structured_answer(""),
            }
            workflow_state = self._generate_answer(workflow_state)
            round_answer = str(workflow_state.get("answer", "") or "")
            round_answer_structured = workflow_state.get("answer_structured") or self._default_structured_answer(round_answer)
            round_confidence = self._estimate_confidence(
                round_answer, activation_payload.get("activated_nodes", [])
            )
            rounds.append(
                {
                    "round_index": round_index + 1,
                    "query_used": activation_query,
                    "seed_nodes": activation_payload.get("seed_nodes", []),
                    "activated_nodes": activation_payload.get("activated_nodes", []),
                    "step_trace": activation_payload.get("step_trace", []),
                    "sources_used": sources_used,
                    "answer": round_answer,
                    "answer_structured": round_answer_structured,
                    "confidence": round_confidence,
                }
            )

            final_answer = round_answer
            final_sources = sources_used
            confidence = round_confidence

            if confidence >= confidence_threshold:
                break

            query_used = self._formulate_followup_query(
                question, activation_payload.get("activated_nodes", [])
            )
            running_history.append({"question": query_used})

        needs_more_context = confidence < confidence_threshold

        self._last_state = {
            "answer": final_answer,
            "answer_structured": rounds[-1].get("answer_structured")
            if rounds
            else self._default_structured_answer(final_answer),
            "search_results": [],
            "sources_used": final_sources,
            "activation_rounds": rounds,
            "confidence": confidence,
            "needs_more_context": needs_more_context,
        }
        self.conversation_history.append(
            {
                "question": question,
                "answer": final_answer,
                "type": "activation_chat",
            }
        )
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]

        return {
            "final_answer": final_answer,
            "answer_structured": self._last_state["answer_structured"],
            "confidence": confidence,
            "needs_more_context": needs_more_context,
            "rounds": rounds,
            "sources_used": final_sources,
        }
    
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
            "search_results": [],
            "sources_used": [],
            "answer_structured": self._default_structured_answer(""),
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