# File for various LLM related services
import json
import os
import re
import ast
import logging
import time
import threading
from botocore.config import Config

logger = logging.getLogger(__name__)

# Global configuration for LLM behaviour
DEBUG_LLM = os.getenv("DEBUG_LLM", "0").lower() in {"1", "true", "yes", "y"}
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3") or "3")
LLM_BACKOFF_BASE_SECONDS = float(os.getenv("LLM_BACKOFF_BASE_SECONDS", "1.0") or "1.0")
LLM_SLEEP_BETWEEN_CALLS_MS = int(os.getenv("LLM_SLEEP_BETWEEN_CALLS_MS", "0") or "0")

_LLM_MAX_CONCURRENT = int(os.getenv("LLM_MAX_CONCURRENT", "0") or "0")
_llm_semaphore = threading.Semaphore(_LLM_MAX_CONCURRENT) if _LLM_MAX_CONCURRENT > 0 else None

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
        # Allow overriding the chat model via environment variable
        self.model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini-2025-08-07")
        self.embedding_model_name = os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        )
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            default_headers={"OpenAI-Beta": "assistants=v2"}
        )

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding vector for text using OpenAI embeddings API."""
        embedding_input = (text or "").strip()
        if not embedding_input:
            return []

        response = self.client.embeddings.create(
            model=self.embedding_model_name,
            input=embedding_input,
        )
        return response.data[0].embedding

    def generate(self, prompt, system_prompt=None, context: str | None = None):
        """
        Generate text using OpenAI.
        
        Args:
            prompt: The input prompt
            system_prompt: System instructions for the assistant
            
        Returns:
            str: The generated text response
        """
        return self._generate_with_api(prompt, system_prompt, context=context)
    
    def _generate_with_api(self, prompt, system_prompt=None, context: str | None = None, max_tokens: int = 2000):
        """
        Generate text using OpenAI Chat Completions API.
        
        Args:
            prompt: The prompt text
            system_prompt: System instructions (optional)
            context: Context label for logging (optional)
            max_tokens: Maximum completion tokens (default 2000, increased from 1000)
            
        Returns:
            str: The generated text response
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        prompt_len = len(prompt) if isinstance(prompt, str) else 0
        context_label = context or "generic"

        for attempt in range(LLM_MAX_RETRIES):
            try:
                # Increase max_tokens on retry if we hit length limit
                current_max_tokens = max_tokens
                if attempt > 0:
                    # Increase by 50% on each retry if previous attempt hit length limit
                    current_max_tokens = int(max_tokens * (1.5 ** attempt))
                    # Cap at 4000 tokens
                    current_max_tokens = min(current_max_tokens, 4000)

                if DEBUG_LLM:
                    logger.info(
                        f"[LLM] Request start | context={context_label} | "
                        f"model={self.model_name} | prompt_chars={prompt_len} | "
                        f"has_system={bool(system_prompt)} | max_tokens={current_max_tokens} | "
                        f"attempt={attempt + 1}/{LLM_MAX_RETRIES}"
                    )

                def _do_request():
                    return self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_completion_tokens=current_max_tokens
                    )

                if _llm_semaphore:
                    with _llm_semaphore:
                        response = _do_request()
                else:
                    response = _do_request()

                content = response.choices[0].message.content
                finish_reason = getattr(response.choices[0], "finish_reason", None)
                model_used = getattr(response, "model", self.model_name)

                if DEBUG_LLM:
                    logger.info(
                        f"[LLM] Response received | context={context_label} | "
                        f"model={model_used} | finish_reason={finish_reason} | "
                        f"content_len={len(content) if content else 0}"
                    )

                if not content:
                    logger.warning(
                        f"[LLM] Empty content from OpenAI | context={context_label} | "
                        f"model={model_used} | finish_reason={finish_reason} | "
                        f"attempt={attempt + 1}/{LLM_MAX_RETRIES} | max_tokens_used={current_max_tokens}"
                    )
                    # If finish_reason is 'length', the model hit the token limit - increase on retry
                    if finish_reason == "length" and attempt < LLM_MAX_RETRIES - 1:
                        logger.info(
                            f"[LLM] Hit token limit (finish_reason=length), will increase max_tokens on next attempt"
                        )
                    # Treat empty content as transient and retry if attempts remain
                    if attempt < LLM_MAX_RETRIES - 1:
                        backoff = LLM_BACKOFF_BASE_SECONDS * (2 ** attempt)
                        logger.info(f"[LLM] Retrying after empty content in {backoff:.2f}s")
                        time.sleep(backoff)
                        continue

                if LLM_SLEEP_BETWEEN_CALLS_MS > 0:
                    time.sleep(LLM_SLEEP_BETWEEN_CALLS_MS / 1000.0)

                return content or ""

            except Exception as e:
                message = str(e)
                lower_msg = message.lower()
                is_transient = any(
                    kw in lower_msg
                    for kw in ["rate limit", "429", "timeout", "timed out", "server error", "503"]
                )

                if is_transient and attempt < LLM_MAX_RETRIES - 1:
                    backoff = LLM_BACKOFF_BASE_SECONDS * (2 ** attempt)
                    logger.warning(
                        f"[LLM] Transient OpenAI error (will retry) | context={context_label} | "
                        f"attempt={attempt + 1}/{LLM_MAX_RETRIES} | backoff={backoff:.2f}s | error={message}"
                    )
                    time.sleep(backoff)
                    continue

                logger.error(
                    f"[LLM] OpenAI API error (giving up) | context={context_label} | "
                    f"attempt={attempt + 1}/{LLM_MAX_RETRIES} | error={message}"
                )
                raise RuntimeError(f"OpenAI API error: {message}")

    def generate_summary(self, text: str) -> str:
        """Generate a summary of the paper text"""
        # Truncate text if too long
        max_chars = 30000
        original_len = len(text)
        truncated = False
        if original_len > max_chars:
            text = text[:max_chars] + "..."
            truncated = True
        
        prompt = f"""Provide a comprehensive summary of this research paper. Do not include any headings or titles. Start directly with the content covering:
1. Main research question/objective
2. Key methodology or approach  
3. Primary findings/results
4. Conclusions and implications

Paper text:
{text}"""

        # Retry up to 3 times on failure
        for attempt in range(LLM_MAX_RETRIES):
            try:
                if DEBUG_LLM:
                    logger.info(
                        f"[LLM] Summary request start | model={self.model_name} | "
                        f"prompt_chars={len(prompt)} | text_truncated={truncated} | "
                        f"attempt={attempt + 1}/{LLM_MAX_RETRIES}"
                    )

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=500
                )
                
                summary = response.choices[0].message.content
                if summary:
                    if DEBUG_LLM:
                        finish_reason = getattr(response.choices[0], "finish_reason", None)
                        logger.info(
                            f"[LLM] Summary response received | model={getattr(response, 'model', self.model_name)} | "
                            f"finish_reason={finish_reason} | summary_len={len(summary)}"
                        )
                    return summary.strip()
                
                finish_reason = getattr(response.choices[0], "finish_reason", None)
                logger.warning(
                    f"Empty summary response (attempt {attempt + 1}/{LLM_MAX_RETRIES}) | "
                    f"model={getattr(response, 'model', self.model_name)} | finish_reason={finish_reason}"
                )
                if attempt < LLM_MAX_RETRIES - 1:
                    backoff = LLM_BACKOFF_BASE_SECONDS * (2 ** attempt)
                    time.sleep(backoff)  # Wait before retry
                    
            except Exception as e:
                message = str(e)
                lower_msg = message.lower()
                is_transient = any(
                    kw in lower_msg
                    for kw in ["rate limit", "429", "timeout", "timed out", "server error", "503"]
                )
                logger.warning(
                    f"Summary generation error (attempt {attempt + 1}/{LLM_MAX_RETRIES}): {message} | "
                    f"transient={is_transient}"
                )
                if is_transient and attempt < LLM_MAX_RETRIES - 1:
                    backoff = LLM_BACKOFF_BASE_SECONDS * (2 ** attempt)
                    time.sleep(backoff)
                else:
                    # For non-transient errors, break early
                    break

        # All retries failed - attempt alternate simple prompt
        logger.error("Failed to generate summary after primary attempts, trying alternate prompt")
        alt_prompt = f"""Summarize this research paper in 3-5 sentences as an abstract suitable for a reader familiar with the field.

Paper text:
{text}"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": alt_prompt}],
                max_completion_tokens=300
            )
            summary = response.choices[0].message.content or ""
            if summary:
                logger.warning(
                    "Using summary generated from alternate prompt after failures with primary prompt"
                )
                return summary.strip()
        except Exception as e:
            logger.warning(f"Alternate summary prompt also failed: {e}")

        # Final local heuristic fallback: use first few sentences of the text
        logger.error("Falling back to heuristic summary based on source text")
        return self._heuristic_summary(text)

    def _heuristic_summary(self, text: str, max_chars: int = 2000, max_sentences: int = 5) -> str:
        """
        Create a simple extractive summary from the start of the text.
        This is a last-resort fallback when LLM-based summaries fail.
        """
        if not text:
            return ""

        snippet = text[:max_chars]
        # Naive sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', snippet)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return snippet.strip()
        return " ".join(sentences[:max_sentences])
    
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
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=2000
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
        original_len = len(text)
        if original_len > max_chars:
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

        # Retry up to 3 times
        for attempt in range(LLM_MAX_RETRIES):
            try:
                if DEBUG_LLM:
                    logger.info(
                        f"[Metadata] LLM metadata extraction attempt {attempt + 1}/{LLM_MAX_RETRIES} | "
                        f"text_chars={len(text)} | original_chars={original_len} | "
                        f"text_truncated={original_len > max_chars}"
                    )

                response = self.llm_client.generate(
                    prompt, system_prompt=system_prompt, context="metadata"
                ).strip()
                
                # Check for empty response
                if not response:
                    logger.warning(f"Empty metadata response (attempt {attempt + 1}/{LLM_MAX_RETRIES})")
                    if attempt < LLM_MAX_RETRIES - 1:
                        time.sleep(LLM_BACKOFF_BASE_SECONDS * (2 ** attempt))  # Wait before retry
                    continue
                
                # Try to extract JSON from response
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
                json_path = None
                if json_match:
                    json_str = json_match.group(1)
                    json_path = "markdown_block"
                else:
                    json_match = re.search(r'\{.*?\}', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        json_path = "bare_object"
                    else:
                        json_str = response
                        json_path = "raw_response"
                
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

                # Basic validation: require non-empty title and authors list
                title_ok = isinstance(metadata['title'], str) and metadata['title'].strip() != ""
                authors_ok = isinstance(metadata['authors'], list) and len(metadata['authors']) > 0

                if not title_ok or not authors_ok:
                    logger.warning(
                        f"[Metadata] Incomplete or invalid metadata from LLM "
                        f"(attempt {attempt + 1}/{LLM_MAX_RETRIES}) | "
                        f"title_ok={title_ok} | authors_ok={authors_ok}"
                    )
                    if attempt < LLM_MAX_RETRIES - 1:
                        continue
                    # Fall through to heuristic fallback after loop
                else:
                    if DEBUG_LLM:
                        logger.info(
                            f"[Metadata] Successfully extracted metadata via LLM | "
                            f"title={metadata['title']!r} | authors_count={len(metadata['authors'])}"
                        )
                    return metadata

            except json.JSONDecodeError as e:
                snippet = response[:500] if isinstance(response, str) else ""
                logger.warning(
                    f"Metadata JSON parsing error (attempt {attempt + 1}/{LLM_MAX_RETRIES}) via {json_path}: {e}. "
                    f"Response snippet: {snippet!r}"
                )
            except Exception as e:
                logger.warning(
                    f"Error extracting paper metadata (attempt {attempt + 1}/{LLM_MAX_RETRIES}): {e}"
                )
        
        # All retries failed or produced invalid metadata - use heuristic fallback
        logger.error("Failed to extract valid metadata via LLM after all attempts; using heuristic parser")
        return self._heuristic_metadata(text)

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

        if DEBUG_LLM:
            logger.info(
                f"[Topics] Starting topic extraction | text_chars={len(text)} | "
                f"max_chars={max_chars} | has_current_topics={bool(current_topics)} | "
                f"current_topics_count={len(current_topics) if current_topics else 0}"
            )

        topics = self._extract_from_text(text, current_topics)
        if topics:
            return topics

        # Fallback: use KeyBERT-based topic extraction if LLM could not provide topics
        logger.warning("[Topics] LLM topic extraction failed; falling back to KeyBERT-based topics")
        try:
            from .pdf_preprocessor import extract_topics as keybert_extract_topics

            kb_topics = keybert_extract_topics(text, current_topics=current_topics)
            if not kb_topics:
                logger.error("[Topics] KeyBERT fallback also returned no topics")
                return []

            # Normalize and limit to 8 topics
            cleaned = []
            for t in kb_topics:
                if not t:
                    continue
                # Title-case while preserving common acronyms reasonably well
                normalized = str(t).strip()
                if not normalized:
                    continue
                cleaned.append(normalized.title())
                if len(cleaned) >= 8:
                    break

            logger.warning(
                f"[Topics] Using {len(cleaned)} fallback topics from KeyBERT after LLM failure"
            )
            return cleaned
        except Exception as e:
            logger.error(f"[Topics] Error during KeyBERT fallback topic extraction: {e}", exc_info=True)
            return []
    
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

        # Retry up to 3 times on failure
        for attempt in range(LLM_MAX_RETRIES):
            try:
                if DEBUG_LLM:
                    logger.info(
                        f"[Topics] LLM topic extraction attempt {attempt + 1}/{LLM_MAX_RETRIES} | "
                        f"prompt_chars={len(prompt)} | has_existing_topics={bool(existing_topics_str)}"
                    )

                response = self.llm_client.generate(
                    prompt, system_prompt=system_prompt, context="topics"
                ).strip()
                
                # Check for empty response
                if not response:
                    logger.warning(
                        f"Empty response from LLM for topics (attempt {attempt + 1}/{LLM_MAX_RETRIES})"
                    )
                    if attempt < LLM_MAX_RETRIES - 1:
                        time.sleep(LLM_BACKOFF_BASE_SECONDS * (2 ** attempt))  # Wait before retry
                    continue
                
                # Try to extract JSON from response
                # Handle cases where LLM wraps JSON in markdown code blocks
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
                json_path = None
                if json_match:
                    json_str = json_match.group(1)
                    json_path = "markdown_block"
                else:
                    # Try to find JSON object directly
                    json_match = re.search(r'\{.*?\}', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        json_path = "bare_object"
                    else:
                        json_str = response
                        json_path = "raw_response"
                
                # Parse JSON
                result = json.loads(json_str)
                topics = result.get('topics', [])
                
                # Ensure we return at most 8 topics
                if isinstance(topics, list) and topics:
                    cleaned = [str(t).strip() for t in topics[:8] if t]
                    if DEBUG_LLM:
                        logger.info(
                            f"[Topics] Extracted {len(cleaned)} topics via LLM on attempt "
                            f"{attempt + 1}/{LLM_MAX_RETRIES}"
                        )
                    return cleaned
                
                logger.warning(
                    f"No topics extracted from LLM response (attempt {attempt + 1}/{LLM_MAX_RETRIES})"
                )
                
            except json.JSONDecodeError as e:
                snippet = response[:500] if isinstance(response, str) else ""
                logger.warning(
                    f"JSON parsing error while extracting topics (attempt {attempt + 1}/{LLM_MAX_RETRIES}) "
                    f"via {json_path}: {e}. Response snippet: {snippet!r}"
                )
            except Exception as e:
                logger.warning(
                    f"Error extracting topics (attempt {attempt + 1}/{LLM_MAX_RETRIES}): {e}"
                )
        
        # All retries failed
        logger.error("Failed to extract topics via LLM after all attempts")
        return []

    def _heuristic_metadata(self, text: str, max_lines: int = 40) -> dict:
        """
        Heuristic/local metadata extraction when LLM-based extraction fails.
        Attempts to infer title, authors, and publication date from the early
        part of the paper text.
        """
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        lines = lines[:max_lines]

        title = None
        authors: list[str] = []
        pub_date = None

        # Heuristic title: first non-obviously-non-title line
        title_exclude_patterns = [
            r"arxiv",
            r"preprint",
            r"doi",
            r"copyright",
            r"all rights reserved",
        ]

        def looks_like_title(line: str) -> bool:
            if len(line) < 5 or len(line) > 300:
                return False
            lower = line.lower()
            if any(re.search(pat, lower) for pat in title_exclude_patterns):
                return False
            # Avoid pure page numbers
            if re.fullmatch(r"\d+(\s+of\s+\d+)?", line):
                return False
            return True

        for ln in lines:
            if looks_like_title(ln):
                title = ln.strip()
                break

        # Heuristic authors: look at lines after title
        start_idx = 0
        if title and title in lines:
            start_idx = lines.index(title) + 1

        author_candidates = lines[start_idx:start_idx + 5]
        for ln in author_candidates:
            # Look for commas and 'and' as separators, with multiple capitalized words
            if "," in ln or " and " in ln:
                parts = re.split(r",| and ", ln)
                candidate_authors = []
                for p in parts:
                    p = p.strip()
                    if not p:
                        continue
                    # Require at least one space and some alphabetic characters
                    if " " in p and re.search(r"[A-Za-z]", p):
                        candidate_authors.append(p)
                if len(candidate_authors) >= 1:
                    authors = candidate_authors
                    break

        # Heuristic publication date: look for YYYY or YYYY-MM(-DD)
        date_pattern = re.compile(r"(19|20)\d{2}(?:-\d{2}){0,2}")
        for ln in lines:
            m = date_pattern.search(ln)
            if m:
                pub_date = m.group(0)
                break

        logger.warning(
            f"[Metadata] Using heuristic metadata | "
            f"title={title!r} | authors_count={len(authors)} | publication_date={pub_date!r}"
        )
        return {
            "title": title,
            "authors": authors,
            "publication_date": pub_date,
        }