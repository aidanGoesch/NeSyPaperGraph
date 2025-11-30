# File for various LLM related services
import json
import os
import re
import ast
import logging
from botocore.config import Config

logger = logging.getLogger(__name__)

system_prompt = """
You are an expert academic paper analyzer specializing in topic extraction from research papers. Your task is to identify and extract the main topics, themes, and subject areas discussed in academic papers.

## Your Task

Analyze the provided academic paper text and extract EXACTLY 8 topics that represent the core subject matter, methodologies, domains, or research areas covered in the paper.

## Output Format

You MUST respond with ONLY a valid JSON object in the following format:

{
  "topics": [
    "Topic Name 1",
    "Topic Name 2",
    "Topic Name 3",
    "Topic Name 4",
    "Topic Name 5",
    "Topic Name 6",
    "Topic Name 7",
    "Topic Name 8"
  ]
}

## Topic Extraction Guidelines

1. **Relevance**: Extract topics that are central to the paper's content, not peripheral mentions

2. **Specificity**: Prefer specific, meaningful topics over generic ones (e.g., "Neural Architecture Search" over "AI")

3. **Naming Convention**: 
   - Use 1-3 word noun phrases
   - Capitalize each major word (Title Case)
   - Be concise and clear
   - Use standard academic terminology

4. **Scope**: Include:
   - Research domains and fields (e.g., "Machine Learning", "Computational Biology")
   - Methodologies and approaches (e.g., "Deep Reinforcement Learning", "Bayesian Inference")
   - Application areas (e.g., "Computer Vision", "Natural Language Processing")
   - Theoretical frameworks (e.g., "Graph Neural Networks", "Transformer Architecture")

5. **Quantity**: Extract EXACTLY 8 topics (or fewer only if the paper genuinely covers fewer than 8 distinct areas)

6. **Prioritization**: When selecting topics, prioritize:
   - **FIRST**: Reusing relevant topics from the existing topics database (if provided)
   - Topics that are central to the paper's main contributions
   - Topics mentioned frequently or discussed in depth
   - Topics that represent the primary research domain

7. **Avoid**: 
   - Overly generic terms (e.g., "Research", "Study", "Analysis")
   - Paper-specific details that aren't reusable topics
   - Duplicate or near-duplicate topics
   - Creating new topics when existing ones are suitable

## Critical Requirements

- Output ONLY valid JSON - no markdown, no explanations, no additional text
- The JSON must be parseable and well-formed
- Extract EXACTLY 8 topics (no more, no less, unless paper has fewer distinct areas)
- **PRIORITIZE reusing existing topics from the database when relevant**
- All topics must be strings in the array
- Do not include any commentary or metadata outside the JSON structure
"""


class OpenAILLMClient:
    def __init__(self, api_key=None, assistant_id=None):
        """
        Initialize OpenAI LLM client.
        
        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            assistant_id: OpenAI Assistant ID (if None, will use OPENAI_ASSISTANT_ID env var)
        """
        import openai
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.assistant_id = assistant_id or os.getenv('OPENAI_ASSISTANT_ID')
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            default_headers={"OpenAI-Beta": "assistants=v2"}
        )

    def generate(self, prompt, system_prompt=None):
        """
        Generate text using OpenAI.
        
        Args:
            prompt: The input prompt
            system_prompt: System instructions for the assistant
            
        Returns:
            str: The generated text response
        """
        return self._generate_with_api(prompt, system_prompt)
    
    def _generate_with_api(self, prompt, system_prompt=None):
        """
        Generate text using OpenAI Chat Completions API.
        
        Args:
            prompt: The prompt text
            system_prompt: System instructions (optional)
            
        Returns:
            str: The generated text response
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Faster and cheaper than gpt-3.5-turbo
                messages=messages,
                temperature=0.2,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")

    def generate_summary(self, text: str) -> str:
        """Generate a summary of the paper text"""
        try:
            # Truncate text if too long
            max_chars = 30000
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            
            prompt = f"""Provide a comprehensive summary of this research paper. Do not include any headings or titles. Start directly with the content covering:
1. Main research question/objective
2. Key methodology or approach  
3. Primary findings/results
4. Conclusions and implications

Paper text:
{text}"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
    
    def generate_topic_synonyms(self, topics: list[str]) -> dict[str, list[str]]:
        """Generate synonyms for each topic to help identify duplicates"""
        try:
            prompt = f"""For each topic below, identify which OTHER topics in the list are synonyms or very similar meanings.
Return ONLY a JSON object where keys are topics and values are arrays of OTHER topics from the list that mean the same thing.

IMPORTANT: Only use topics that appear in the provided list. Do not invent new topic names.

Topics: {', '.join(topics)}

Example format (using only topics from the list):
{{"Large Language Models": ["LLMs", "Language Models"], "Neural Networks": ["Deep Networks", "ANNs"]}}

If a topic has no synonyms in the list, use an empty array: {{"Unique Topic": []}}

JSON:"""
            
            logger.info(f"Requesting synonyms for {len(topics)} topics from OpenAI...")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
            
            # Parse JSON response
            import json
            content = response.choices[0].message.content.strip()
            logger.info(f"Received response, length: {len(content)}")
            
            # Extract JSON from response (in case there's extra text)
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                json_str = content[start:end]
                result = json.loads(json_str)
                logger.info(f"Successfully parsed {len(result)} topic synonyms")
                return result
            else:
                logger.error(f"No JSON found in response: {content[:200]}")
                return {}
        except Exception as e:
            logger.error(f"Error generating topic synonyms: {str(e)}", exc_info=True)
            return {}


class TopicExtractor:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def extract_paper_metadata(self, text, max_chars=4000):
        """
        Extract paper metadata (title, authors, publication date) from paper text.
        
        Args:
            text: Paper text to analyze
            max_chars: Maximum characters to send to LLM
        
        Returns:
            Dict with title, authors, and publication_date
        """
        # Use beginning of paper where metadata is typically found
        if len(text) > max_chars:
            text = text[:max_chars]
        
        system_prompt = """You are an expert at extracting metadata from academic papers. Extract the title, authors, and publication date from the given paper text. Return the result in JSON format with the following structure:

{
    "title": "Paper Title",
    "authors": ["Author 1", "Author 2", "Author 3"],
    "publication_date": "YYYY" or "YYYY-MM" or "YYYY-MM-DD"
}

Guidelines:
- Extract the exact title as it appears in the paper
- List all authors in order as they appear
- For publication date, extract year at minimum, include month/day if available
- If any field cannot be determined, use null
- Return only valid JSON"""

        prompt = f"""Academic Paper Text (beginning):
{text}

Extract the title, authors, and publication date in JSON format."""

        try:
            response = self.llm_client.generate(prompt, system_prompt=system_prompt).strip()
            
            # Try to extract JSON from response
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*?\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response
            
            result = json.loads(json_str)
            
            # Validate and clean the result
            metadata = {
                'title': result.get('title'),
                'authors': result.get('authors', []),
                'publication_date': result.get('publication_date')
            }
            
            # Ensure authors is a list
            if isinstance(metadata['authors'], str):
                metadata['authors'] = [metadata['authors']]
            elif not isinstance(metadata['authors'], list):
                metadata['authors'] = []
            
            return metadata
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error extracting paper metadata: {e}")
            return {
                'title': None,
                'authors': [],
                'publication_date': None
            }

    def extract_topics(self, text, current_topics=None, max_chars=8000):
        """
        Extract exactly 8 topics from text.
        
        Args:
            text: Paper text to analyze
            current_topics: List of existing topics to reuse when relevant
            max_chars: Maximum characters to send to LLM (increased from 2000)
        
        Returns:
            List of extracted topics (max 8)
        """
        # Truncate text if too large instead of chunking
        # This prevents duplicate topic extraction and is faster
        if len(text) > max_chars:
            # Take beginning and end of paper for better context
            chunk_size = max_chars // 2
            text = text[:chunk_size] + "\n...\n" + text[-chunk_size:]
        
        return self._extract_from_text(text, current_topics)
    
    def _extract_from_text(self, text, current_topics):
        """
        Extract topics from text using JSON output format.
        """
        # Build prompt with existing topics if provided
        existing_topics_str = ""
        if current_topics:
            # Convert to list if it's a set
            topics_list = list(current_topics) if isinstance(current_topics, set) else current_topics
            existing_topics_str = f"\n\nExisting Topics Database:\n{json.dumps(topics_list, indent=2)}\n\nPlease prioritize reusing these topics when relevant."
        
        prompt = f"""{existing_topics_str}

Academic Paper Text:
{text}

Extract EXACTLY 8 topics in JSON format."""

        try:
            response = self.llm_client.generate(prompt, system_prompt=system_prompt).strip()
            
            # Try to extract JSON from response
            # Handle cases where LLM wraps JSON in markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{.*?\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response
            
            # Parse JSON
            result = json.loads(json_str)
            topics = result.get('topics', [])
            
            # Ensure we return at most 8 topics
            if isinstance(topics, list):
                return [str(t).strip() for t in topics[:8] if t]
            
            return []
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response was: {response}")
            return []
        except Exception as e:
            print(f"Error extracting topics: {e}")
            return []