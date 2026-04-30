from models.graph import PaperGraph
from models.paper import Paper
from services.llm_service import TopicExtractor, OpenAILLMClient
from services.docling_service import get_docling_service
from services.pdf_preprocessor import extract_text_from_pdf
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterator, Optional
from services.observability import timed_block, log_memory

logger = logging.getLogger(__name__)

# Global set to store all topics seen across all uploads (in-memory, not persistent)
_all_topics_seen = set()

# Global dict to store topic synonyms (persistent across uploads)
_topic_synonyms_cache = {}
MAX_TOPICS_TRACKED = int(os.getenv("MAX_TOPICS_TRACKED", "5000") or "5000")
MAX_TOPIC_SYNONYMS_CACHE = int(os.getenv("MAX_TOPIC_SYNONYMS_CACHE", "5000") or "5000")
MAX_PERSISTED_TEXT_CHARS = int(os.getenv("MAX_PERSISTED_TEXT_CHARS", "0") or "0")
INGEST_EMBED_BATCH_SIZE = int(os.getenv("INGEST_EMBED_BATCH_SIZE", "8") or "8")
SUMMARY_SOURCE_MAX_CHARS = int(os.getenv("SUMMARY_SOURCE_MAX_CHARS", "7000") or "7000")


def _cap_set_size(items: set, max_size: int) -> None:
    if max_size <= 0 or len(items) <= max_size:
        return
    # Keep arbitrary subset when we exceed the cap.
    while len(items) > max_size:
        items.pop()


def _cap_dict_size(items: dict, max_size: int) -> None:
    if max_size <= 0 or len(items) <= max_size:
        return
    keys = list(items.keys())
    for key in keys[: len(items) - max_size]:
        items.pop(key, None)

class GraphBuilder:
    def __init__(self):
        self.papers = []
        self.topics = set()

    @staticmethod
    def _normalize_title(title: str) -> str:
        """Normalize title for duplicate detection: lowercase, remove punctuation and extra whitespace"""
        import re
        normalized = title.lower()
        normalized = re.sub(r'[^\w\s]', '', normalized)  # Remove punctuation
        normalized = re.sub(r'\s+', ' ', normalized).strip()  # Normalize whitespace
        return normalized

    @staticmethod
    def _prefer_heuristic_title(
        docling_title: str | None, heuristic_title: str | None
    ) -> str | None:
        """Prefer heuristic title when Docling returns likely venue/container labels."""
        if not heuristic_title:
            return docling_title
        if not docling_title:
            return heuristic_title

        normalized_docling = docling_title.strip()
        normalized_heuristic = heuristic_title.strip()
        if not normalized_docling:
            return normalized_heuristic
        if not normalized_heuristic:
            return normalized_docling

        container_markers = [
            r"\bjournal\b",
            r"\bproceedings\b",
            r"\btransactions\b",
            r"\bconference\b",
            r"\bvolume\b",
            r"\bvol\.\b",
            r"\bissue\b",
            r"\bissn\b",
            r"\bpublisher\b",
        ]
        lower_docling = normalized_docling.lower()
        looks_like_container = any(
            re.search(pattern, lower_docling) for pattern in container_markers
        )
        very_short_docling = len(normalized_docling) < 15

        if (
            looks_like_container or very_short_docling
        ) and len(normalized_heuristic) >= len(normalized_docling):
            return normalized_heuristic
        return normalized_docling

    @staticmethod
    def _is_likely_valid_title(title: str | None) -> bool:
        if not title:
            return False
        candidate = title.strip()
        if len(candidate) < 8 or len(candidate) > 300:
            return False
        # Titles should contain meaningful alphabetic content.
        alpha_count = sum(1 for ch in candidate if ch.isalpha())
        if alpha_count < 6:
            return False
        non_space_count = sum(1 for ch in candidate if not ch.isspace())
        if non_space_count == 0:
            return False
        symbol_ratio = (
            sum(1 for ch in candidate if not ch.isalnum() and not ch.isspace())
            / non_space_count
        )
        # Reject OCR noise like "1234567890();;"
        if symbol_ratio > 0.35:
            return False
        digit_ratio = sum(1 for ch in candidate if ch.isdigit()) / non_space_count
        if digit_ratio > 0.4:
            return False
        if re.fullmatch(r"[\d\W_]+", candidate):
            return False
        lower_candidate = candidate.lower()
        if lower_candidate.startswith("tmp") and len(lower_candidate) <= 24:
            return False
        if lower_candidate.endswith(".pdf") and (
            lower_candidate.startswith("tmp")
            or lower_candidate.startswith("nesy_upload_")
        ):
            return False
        if candidate.startswith("#") or candidate.startswith("<!--"):
            return False
        if re.search(r"[(){};,:]{3,}", candidate):
            return False
        return True

    @staticmethod
    def _heuristic_title_from_text(text: str, max_lines: int = 30) -> str | None:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        lines = lines[:max_lines]
        title_exclude_patterns = [
            r"arxiv",
            r"preprint",
            r"doi",
            r"copyright",
            r"all rights reserved",
            r"\bjournal\b",
            r"\bproceedings\b",
            r"\bvolume\b",
            r"\bissue\b",
        ]
        for line in lines:
            if len(line) < 8 or len(line) > 300:
                continue
            lower = line.lower()
            if any(re.search(pattern, lower) for pattern in title_exclude_patterns):
                continue
            if re.fullmatch(r"\d+(\s+of\s+\d+)?", line):
                continue
            if not GraphBuilder._is_likely_valid_title(line):
                continue
            return line
        return None

    @staticmethod
    def _resolve_title(
        preferred_title: str | None, heuristic_title: str | None, fallback_stem: str
    ) -> str:
        if GraphBuilder._is_likely_valid_title(preferred_title):
            return preferred_title.strip()
        if GraphBuilder._is_likely_valid_title(heuristic_title):
            return heuristic_title.strip()
        cleaned_stem = fallback_stem.strip() if fallback_stem else "Untitled Paper"
        return cleaned_stem or "Untitled Paper"

    @staticmethod
    def _build_metadata_extractor() -> TopicExtractor:
        try:
            return TopicExtractor(OpenAILLMClient())
        except Exception as exc:
            logger.warning(
                "[Metadata] LLM metadata extractor unavailable, using non-LLM fallbacks: %s",
                exc,
            )
            return TopicExtractor(None)

    @staticmethod
    def _merge_metadata(base: dict, llm_metadata: dict | None) -> dict:
        if not llm_metadata:
            return base
        merged = {
            "title": base.get("title"),
            "authors": base.get("authors", []),
            "publication_date": base.get("publication_date"),
        }
        llm_title = llm_metadata.get("title")
        if GraphBuilder._is_likely_valid_title(llm_title):
            merged["title"] = llm_title.strip()
        llm_authors = llm_metadata.get("authors")
        if isinstance(llm_authors, list) and llm_authors:
            cleaned_authors = [str(author).strip() for author in llm_authors if str(author).strip()]
            if cleaned_authors:
                merged["authors"] = cleaned_authors
        llm_pub_date = llm_metadata.get("publication_date")
        if isinstance(llm_pub_date, (str, int)) and str(llm_pub_date).strip():
            merged["publication_date"] = str(llm_pub_date).strip()
        return merged
    
    def _is_duplicate_paper(self, paper: Paper, graph: PaperGraph) -> bool:
        """Check if paper already exists in graph by normalized title"""
        normalized_new = self._normalize_title(paper.title)
        for node, data in graph.graph.nodes(data=True):
            if data.get('type') == 'paper':
                existing_title = data['data'].title
                if self._normalize_title(existing_title) == normalized_new:
                    return True
        return False

    @staticmethod
    def _ensure_graph_metadata(graph: PaperGraph) -> None:
        """Backfill graph metadata fields for graphs loaded from old snapshots."""
        if not hasattr(graph, "paper_content_hashes") or graph.paper_content_hashes is None:
            graph.paper_content_hashes = set()

    def _is_duplicate_hash(self, paper: Paper, graph: PaperGraph) -> bool:
        if not paper.content_hash:
            return False
        return paper.content_hash in graph.paper_content_hashes


    def build_graph(
        self,
        files_data=None,
        file_path: str = None,
        existing_graph=None,
        on_paper_processed: Optional[Callable[[dict], None]] = None,
        total_papers: Optional[int] = None,
    ) -> PaperGraph:
        """
        Builds a graph from PDF files.
        Processes papers sequentially, adding each to the graph immediately,
        and passing accumulated topics to the next paper's topic extraction.
        
        Args:
            files_data: List of tuples (filename, file_content_bytes), or None
            file_path: Directory path to scan for PDFs (legacy support), or None
            existing_graph: Existing PaperGraph to update, or None to create new
        """
        global _topic_synonyms_cache
        log_memory("graph_builder_build_graph_start")
        
        # Use existing graph or create new one
        if existing_graph:
            graph = existing_graph
            self._ensure_graph_metadata(graph)
            # Extract existing papers and topics
            self.papers = []
            self.topics = set()
            for node, data in graph.graph.nodes(data=True):
                if data.get('type') == 'paper':
                    existing_paper = data['data']
                    self.papers.append(existing_paper)
                    if getattr(existing_paper, "content_hash", None):
                        graph.paper_content_hashes.add(existing_paper.content_hash)
                elif data.get('type') == 'topic':
                    self.topics.add(node)
            
            # Load existing synonyms into cache
            if hasattr(graph, 'topic_synonyms') and graph.topic_synonyms:
                _topic_synonyms_cache.update(graph.topic_synonyms)
        else:
            # Reset state for new build
            self.papers = []
            self.topics = set()
            graph = PaperGraph()
            self._ensure_graph_metadata(graph)
        
        if files_data is not None:
            papers_to_process = self.get_papers_from_data(files_data)
            if total_papers is None:
                total_papers = len(files_data)
        elif file_path is not None:
            papers_to_process = self.get_papers(file_path)
            if total_papers is None:
                total_papers = len(papers_to_process)
        else:
            raise ValueError("Either files_data or file_path must be provided")
        
        # Get all topics seen across all previous uploads
        global _all_topics_seen
        accumulated_topics = _all_topics_seen.copy()  # Start with all previously seen topics
        
        # Use OpenAI assistant (reads assistant_id from environment)
        client = OpenAILLMClient()
        log_memory("graph_builder_after_openai_client_init")
        extractor = TopicExtractor(client)
        from services.verification import verify_bipartite, find_optimal_topic_merge
        
        total_papers = total_papers or 0
        embed_batch_size = max(1, INGEST_EMBED_BATCH_SIZE)
        pending_embedding_batch: list[dict[str, Any]] = []
        input_paper_count = 0

        def finalize_paper_into_graph(
            paper: Paper, paper_index: int, paper_total: int
        ) -> None:
            # Drop or truncate retained full text to reduce graph memory footprint.
            if MAX_PERSISTED_TEXT_CHARS <= 0:
                paper.text = None
            elif paper.text:
                paper.text = paper.text[:MAX_PERSISTED_TEXT_CHARS]

            graph.add_paper(paper)
            if paper.content_hash:
                graph.paper_content_hashes.add(paper.content_hash)

            with timed_block("verify_bipartite_incremental"):
                valid_increment = verify_bipartite(
                    graph, graph.new_nodes, graph.new_edges
                )
            if not valid_increment:
                logger.error(
                    "Bipartite verification failed after adding paper: %s. Rolling back.",
                    paper.title,
                )
                if paper.title in graph.graph:
                    paper_topics = list(graph.graph.neighbors(paper.title))
                    graph.graph.remove_node(paper.title)
                    for topic in paper_topics:
                        if topic in graph.graph and graph.graph.degree(topic) == 0:
                            graph.graph.remove_node(topic)
                            logger.info("Removed orphaned topic: %s", topic)
                graph.clear_incremental_tracking()
                if paper.content_hash:
                    graph.paper_content_hashes.discard(paper.content_hash)
                if on_paper_processed:
                    on_paper_processed(
                        {
                            "status": "skipped",
                            "reason": "verification_failed",
                            "paper_title": paper.title,
                            "paper_index": paper_index,
                            "paper_total": paper_total,
                            "graph": graph,
                        }
                    )
                return

            graph.clear_incremental_tracking()
            if on_paper_processed:
                on_paper_processed(
                    {
                        "status": "processed",
                        "paper_title": paper.title,
                        "paper_index": paper_index,
                        "paper_total": paper_total,
                        "graph": graph,
                    }
                )

        def flush_embedding_batch() -> None:
            if not pending_embedding_batch:
                return
            embedding_texts = [
                item["embedding_text"] for item in pending_embedding_batch
            ]
            try:
                with timed_block("embed_batch_request"):
                    vectors = client.generate_embeddings(embedding_texts)
            except Exception as exc:
                logger.warning(
                    "Failed to generate batched embeddings for %s papers: %s",
                    len(pending_embedding_batch),
                    exc,
                )
                vectors = [[] for _ in pending_embedding_batch]

            for batch_index, item in enumerate(pending_embedding_batch):
                paper = item["paper"]
                paper.embedding = vectors[batch_index] if batch_index < len(vectors) else []
                finalize_paper_into_graph(
                    paper=paper,
                    paper_index=item["paper_index"],
                    paper_total=item["paper_total"],
                )
            pending_embedding_batch.clear()

        for paper_index, paper in enumerate(papers_to_process, start=1):
            input_paper_count = paper_index
            if self._is_duplicate_paper(paper, graph):
                logger.info(f"Skipping duplicate paper: {paper.title}")
                if on_paper_processed:
                    on_paper_processed(
                        {
                            "status": "skipped",
                            "reason": "duplicate_title",
                            "paper_title": paper.title,
                            "paper_index": paper_index,
                            "paper_total": total_papers,
                            "graph": graph,
                        }
                    )
                continue
            if self._is_duplicate_hash(paper, graph):
                logger.info(f"Skipping duplicate paper hash: {paper.title}")
                if on_paper_processed:
                    on_paper_processed(
                        {
                            "status": "skipped",
                            "reason": "duplicate_hash",
                            "paper_title": paper.title,
                            "paper_index": paper_index,
                            "paper_total": total_papers,
                            "graph": graph,
                        }
                    )
                continue

            text_for_extraction = (
                paper.text[:50000] if len(paper.text) > 50000 else paper.text
            )

            with timed_block("extract_topics_per_paper"):
                topics = extractor.extract_topics(
                    text_for_extraction, current_topics=accumulated_topics
                )
            if not topics:
                logger.warning(
                    "Skipping paper due to failed topic extraction: %s", paper.title
                )
                if on_paper_processed:
                    on_paper_processed(
                        {
                            "status": "skipped",
                            "reason": "topic_extraction_failed",
                            "paper_title": paper.title,
                            "paper_index": paper_index,
                            "paper_total": total_papers,
                            "graph": graph,
                        }
                    )
                continue

            paper.topics = topics

            summary_source_text = text_for_extraction[:SUMMARY_SOURCE_MAX_CHARS]
            with timed_block("generate_summary_per_paper"):
                summary = client.generate_summary(summary_source_text)
            if not summary and summary_source_text:
                logger.warning(
                    "[Summary] Empty summary returned for paper '%s'. Using heuristic summary based on source text.",
                    paper.title,
                )
                summary = summary_source_text[:1000]
            paper.summary = summary

            new_topics = set(topics) - accumulated_topics
            if new_topics:
                accumulated_topics.update(new_topics)
            self.topics.update(set(topics))
            global_new_topics = set(topics) - _all_topics_seen
            if global_new_topics:
                _all_topics_seen.update(global_new_topics)
                _cap_set_size(_all_topics_seen, MAX_TOPICS_TRACKED)

            embedding_text = (
                paper.summary if paper.summary else text_for_extraction[:1000]
            )
            pending_embedding_batch.append(
                {
                    "paper": paper,
                    "paper_index": paper_index,
                    "paper_total": total_papers,
                    "embedding_text": embedding_text,
                }
            )
            if len(pending_embedding_batch) >= embed_batch_size:
                flush_embedding_batch()

        flush_embedding_batch()
        log_memory("graph_builder_after_embedding_batches")
        
        logger.info(
            "Processed %s papers, %s unique topics in this batch",
            input_paper_count,
            len(self.topics),
        )
        
        # Generate synonyms for all topics in the graph
        all_topics = [node for node, data in graph.graph.nodes(data=True) if data.get('type') == 'topic']
        if all_topics:
            # Find topics that don't have synonyms yet
            new_topics = [t for t in all_topics if t not in _topic_synonyms_cache]
            
            if new_topics:
                logger.info(f"Generating synonyms for {len(new_topics)} new topics...")
                new_synonyms = client.generate_topic_synonyms(new_topics)
                _topic_synonyms_cache.update(new_synonyms)
                _cap_dict_size(_topic_synonyms_cache, MAX_TOPIC_SYNONYMS_CACHE)
                logger.info(f"Generated synonyms for {len(new_synonyms)} topics")
            else:
                logger.info("All topics already have cached synonyms")
            
            # Use all cached synonyms for this graph
            graph.topic_synonyms = {t: _topic_synonyms_cache[t] for t in all_topics if t in _topic_synonyms_cache}
            
            # Find optimal topic merges
            logger.info("Finding optimal topic merges...")
            merge_groups = find_optimal_topic_merge(all_topics, graph.topic_synonyms)
            graph.topic_merge_groups = merge_groups
            logger.info(f"Found {len(merge_groups)} topic groups for merging")
            
            # Apply the merges to the graph
            logger.info("Applying topic merges to graph...")
            graph.merge_topics(merge_groups)
            logger.info("Topic merges applied")
        
        # Add semantic edges between similar papers
        logger.info("Computing semantic similarities between papers...")
        with timed_block("add_semantic_edges"):
            graph.add_semantic_edges()
        logger.info("Semantic edges added")
        log_memory("graph_builder_build_graph_end")
        
        return graph

    def ingest_semantic_paper(
        self,
        semantic_paper: dict[str, Any],
        existing_graph: PaperGraph | None = None,
        llm_client: Any | None = None,
        topic_extractor: Any | None = None,
    ) -> PaperGraph:
        """
        Ingest one Semantic Scholar-derived paper into the graph.
        Topic extraction is intentionally driven from abstract text.
        """
        from services.verification import verify_bipartite, find_optimal_topic_merge

        graph = existing_graph or PaperGraph()
        self._ensure_graph_metadata(graph)

        if existing_graph:
            self.papers = []
            self.topics = set()
            for node, data in graph.graph.nodes(data=True):
                if data.get("type") == "paper":
                    existing_paper = data["data"]
                    self.papers.append(existing_paper)
                    if getattr(existing_paper, "content_hash", None):
                        graph.paper_content_hashes.add(existing_paper.content_hash)
                elif data.get("type") == "topic":
                    self.topics.add(node)
        else:
            self.papers = []
            self.topics = set()

        client = llm_client or OpenAILLMClient()
        extractor = topic_extractor or TopicExtractor(client)

        title = (semantic_paper.get("title") or "").strip() or "Untitled Paper"
        source_url = (semantic_paper.get("url") or "").strip() or title
        paper_id = (semantic_paper.get("semanticScholarPaperId") or "").strip()
        abstract = (semantic_paper.get("abstract") or "").strip()
        abstract = abstract[:50000] if len(abstract) > 50000 else abstract
        publication_date = semantic_paper.get("year")
        if publication_date is not None and str(publication_date).strip():
            publication_date = str(publication_date).strip()
        else:
            publication_date = None

        paper = Paper(
            title=title,
            file_path=source_url,
            content_hash=f"semantic_scholar:{paper_id}" if paper_id else None,
            text=abstract,
            summary=abstract,
            topics=[],
            authors=semantic_paper.get("authors") or [],
            publication_date=publication_date,
        )

        if self._is_duplicate_paper(paper, graph):
            return graph
        if self._is_duplicate_hash(paper, graph):
            return graph

        extraction_source = abstract or title
        topics = extractor.extract_topics(extraction_source, current_topics=self.topics)
        if not topics:
            return graph
        paper.topics = topics

        embedding_text = paper.summary if paper.summary else extraction_source
        vectors = client.generate_embeddings([embedding_text])
        paper.embedding = vectors[0] if vectors else []

        if MAX_PERSISTED_TEXT_CHARS <= 0:
            paper.text = None
        elif paper.text:
            paper.text = paper.text[:MAX_PERSISTED_TEXT_CHARS]

        graph.add_paper(paper)
        if paper.content_hash:
            graph.paper_content_hashes.add(paper.content_hash)
        if not verify_bipartite(graph, graph.new_nodes, graph.new_edges):
            if paper.title in graph.graph:
                graph.graph.remove_node(paper.title)
            graph.clear_incremental_tracking()
            if paper.content_hash:
                graph.paper_content_hashes.discard(paper.content_hash)
            return graph
        graph.clear_incremental_tracking()

        self.topics.update(set(topics))
        global _all_topics_seen
        _all_topics_seen.update(topics)
        _cap_set_size(_all_topics_seen, MAX_TOPICS_TRACKED)

        all_topics = [
            node for node, data in graph.graph.nodes(data=True) if data.get("type") == "topic"
        ]
        if all_topics:
            new_topics = [topic for topic in all_topics if topic not in _topic_synonyms_cache]
            if new_topics:
                _topic_synonyms_cache.update(client.generate_topic_synonyms(new_topics))
                _cap_dict_size(_topic_synonyms_cache, MAX_TOPIC_SYNONYMS_CACHE)
            graph.topic_synonyms = {
                topic: _topic_synonyms_cache[topic]
                for topic in all_topics
                if topic in _topic_synonyms_cache
            }
            merge_groups = find_optimal_topic_merge(all_topics, graph.topic_synonyms)
            graph.topic_merge_groups = merge_groups
            graph.merge_topics(merge_groups)

        graph.add_semantic_edges()
        return graph


    def get_papers_from_data(self, files_data: list[tuple]) -> Iterator[Paper]:
        """
        Creates Paper objects from file data (filename, content bytes).
        
        Args:
            files_data: List of tuples (filename, file_content_bytes)
        """
        metadata_extractor = self._build_metadata_extractor()
        docling = get_docling_service()
        log_memory("graph_builder_get_papers_from_data_start")
        docling_success_count = 0
        fallback_count = 0
        metadata_complete_count = 0
        
        for file_tuple in files_data:
            if len(file_tuple) == 4:
                filename, file_content, content_hash, _ = file_tuple
            elif len(file_tuple) == 3:
                filename, file_content, content_hash = file_tuple
            else:
                filename, file_content = file_tuple
                content_hash = None
            if isinstance(file_content, (str, Path)):
                with Path(file_content).open("rb") as handle:
                    file_bytes = handle.read()
            else:
                file_bytes = file_content
            parsed_doc = docling.parse_pdf(file_bytes)
            metadata = {
                "title": parsed_doc.get("title"),
                "authors": parsed_doc.get("authors", []),
                "publication_date": parsed_doc.get("publication_date"),
            }
            text = parsed_doc.get("text") or ""
            if parsed_doc.get("ok"):
                docling_success_count += 1
            else:
                fallback_count += 1
                with timed_block("docling_fallback_extract_text_per_paper"):
                    text = extract_text_from_pdf(
                        file_bytes, max_pages=getattr(docling, "max_pages", None)
                    )
                with timed_block("metadata_fallback_heuristic_per_paper"):
                    metadata = metadata_extractor.heuristic_metadata(text)
            heuristic_title = self._heuristic_title_from_text(text)
            metadata["title"] = self._prefer_heuristic_title(
                metadata.get("title"),
                heuristic_title,
            )
            if metadata_extractor.llm_client and text:
                with timed_block("metadata_llm_extract_per_paper"):
                    llm_metadata = metadata_extractor.extract_paper_metadata(text)
                metadata = self._merge_metadata(metadata, llm_metadata)
            if metadata.get("title") and metadata.get("authors") and metadata.get("publication_date"):
                metadata_complete_count += 1
            
            title = self._resolve_title(
                metadata.get("title"),
                heuristic_title,
                Path(filename).stem,
            )
            if title == Path(filename).stem:
                logger.warning(
                    f"[Metadata] Using filename fallback title for {filename}; extracted title looked invalid."
                )
            
            paper = Paper(
                title=title,
                file_path=filename,  # Keep original filename for reference
                content_hash=content_hash,
                text=text,
                authors=metadata.get('authors', []),
                publication_date=metadata.get('publication_date')
            )
            file_bytes = b""
            yield paper

        logger.info(
            "Metadata extraction summary | papers=%s | docling_success=%s | heuristic_fallback=%s | metadata_complete=%s",
            len(files_data),
            docling_success_count,
            fallback_count,
            metadata_complete_count,
        )
        log_memory("graph_builder_get_papers_from_data_end")

    def get_papers(self, file_path: str) -> list[Paper]:
        """
        Gets all of the pdfs in the given file path and all of its sub folders.
        Legacy method for backward compatibility.
        """
        metadata_extractor = self._build_metadata_extractor()
        docling = get_docling_service()
        docling_success_count = 0
        fallback_count = 0
        metadata_complete_count = 0
        
        path = Path(file_path)
        
        parsed_papers = []
        for pdf_file in path.rglob("*.pdf"):
            # Extract text from PDF
            with pdf_file.open("rb") as handle:
                file_content = handle.read()
            parsed_doc = docling.parse_pdf(file_content)
            metadata = {
                "title": parsed_doc.get("title"),
                "authors": parsed_doc.get("authors", []),
                "publication_date": parsed_doc.get("publication_date"),
            }
            text = parsed_doc.get("text") or ""
            if parsed_doc.get("ok"):
                docling_success_count += 1
            else:
                fallback_count += 1
                with timed_block("docling_fallback_extract_text_per_paper"):
                    text = extract_text_from_pdf(
                        str(pdf_file), max_pages=getattr(docling, "max_pages", None)
                    )
                with timed_block("metadata_fallback_heuristic_per_paper"):
                    metadata = metadata_extractor.heuristic_metadata(text)
            heuristic_title = self._heuristic_title_from_text(text)
            metadata["title"] = self._prefer_heuristic_title(
                metadata.get("title"),
                heuristic_title,
            )
            if metadata_extractor.llm_client and text:
                with timed_block("metadata_llm_extract_per_paper"):
                    llm_metadata = metadata_extractor.extract_paper_metadata(text)
                metadata = self._merge_metadata(metadata, llm_metadata)
            if metadata.get("title") and metadata.get("authors") and metadata.get("publication_date"):
                metadata_complete_count += 1
            
            title = self._resolve_title(
                metadata.get("title"),
                heuristic_title,
                pdf_file.stem,
            )
            if title == pdf_file.stem:
                logger.warning(
                    f"[Metadata] Using filename fallback title for {pdf_file}; extracted title looked invalid."
                )
            
            paper = Paper(
                title=title,
                file_path=str(pdf_file),
                text=text,
                authors=metadata.get('authors', []),
                publication_date=metadata.get('publication_date')
            )
            parsed_papers.append(paper)
            self.papers.append(paper)

        logger.info(
            "Metadata extraction summary | papers=%s | docling_success=%s | heuristic_fallback=%s | metadata_complete=%s",
            len(parsed_papers),
            docling_success_count,
            fallback_count,
            metadata_complete_count,
        )
        
        return parsed_papers


def get_all_topics_seen():
    """Get the set of all topics seen across all uploads"""
    return _all_topics_seen.copy()


def clear_all_topics_seen():
    """Clear the set of all topics seen (useful for testing or reset)"""
    global _all_topics_seen
    _all_topics_seen.clear()


def get_topic_synonyms_cache():
    """Get the cached topic synonyms"""
    return _topic_synonyms_cache.copy()


def clear_topic_synonyms_cache():
    """Clear the cached topic synonyms"""
    global _topic_synonyms_cache
    _topic_synonyms_cache.clear()


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