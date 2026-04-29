import os
from typing import Any
from tempfile import NamedTemporaryFile
import resource
import logging

from services.observability import timed_block

logger = logging.getLogger(__name__)


class DoclingService:
    def __init__(self) -> None:
        self.enabled = os.getenv("DOCLING_ENABLED", "true").lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        self.max_pages = int(os.getenv("DOCLING_MAX_PAGES", "2") or "2")
        self.max_text_chars = int(os.getenv("DOCLING_MAX_TEXT_CHARS", "8000") or "8000")
        self._converter = None

    def parse_pdf(self, pdf_bytes: bytes) -> dict[str, Any]:
        if not self.enabled:
            return self._empty_result("disabled")
        if not pdf_bytes:
            return self._empty_result("empty_input")

        try:
            with timed_block("docling_parse_per_paper"):
                parsed = self._run_docling(pdf_bytes)
            parsed["source"] = "docling"
            parsed["ok"] = bool(parsed.get("text"))
            return parsed
        except Exception:
            return self._empty_result("docling_error")

    def _run_docling(self, pdf_bytes: bytes) -> dict[str, Any]:
        from docling.document_converter import DocumentConverter

        def mem_mb() -> float:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

        if self._converter is None:
            logger.info("Before Docling init: %.0f MB", mem_mb())
            self._converter = DocumentConverter()
            logger.info("After Docling init: %.0f MB", mem_mb())

        with NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            conversion_result = self._converter.convert(tmp.name)
            document = getattr(conversion_result, "document", None)
        logger.info("After parse: %.0f MB", mem_mb())

        text = self._extract_text(document)
        if self.max_text_chars > 0:
            text = text[: self.max_text_chars]

        metadata = self._extract_metadata(document)
        return {
            "text": text,
            "title": metadata.get("title"),
            "authors": metadata.get("authors", []),
            "publication_date": metadata.get("publication_date"),
        }

    def _extract_text(self, document: Any) -> str:
        if document is None:
            return ""

        # Prefer markdown export if available (structured text).
        export_markdown = getattr(document, "export_to_markdown", None)
        if callable(export_markdown):
            rendered = export_markdown()
            if isinstance(rendered, str) and rendered.strip():
                return rendered.strip()

        # Fallback to plain text export.
        export_text = getattr(document, "export_to_text", None)
        if callable(export_text):
            rendered = export_text()
            if isinstance(rendered, str) and rendered.strip():
                return rendered.strip()

        # Final fallback: best-effort string conversion.
        rendered = str(document)
        return rendered.strip() if rendered else ""

    def _extract_metadata(self, document: Any) -> dict[str, Any]:
        title = None
        authors: list[str] = []
        publication_date = None

        if document is None:
            return {
                "title": title,
                "authors": authors,
                "publication_date": publication_date,
            }

        for attr in ("title", "doc_title", "name"):
            value = getattr(document, attr, None)
            if isinstance(value, str) and value.strip():
                title = value.strip()
                break

        for attr in ("authors", "author_list"):
            value = getattr(document, attr, None)
            if isinstance(value, list):
                normalized = []
                for item in value:
                    if isinstance(item, str) and item.strip():
                        normalized.append(item.strip())
                    elif hasattr(item, "name") and isinstance(item.name, str):
                        if item.name.strip():
                            normalized.append(item.name.strip())
                if normalized:
                    authors = normalized
                    break

        for attr in ("publication_date", "date", "year"):
            value = getattr(document, attr, None)
            if value is None:
                continue
            if isinstance(value, str) and value.strip():
                publication_date = value.strip()
                break
            if isinstance(value, int):
                publication_date = str(value)
                break

        return {
            "title": title,
            "authors": authors,
            "publication_date": publication_date,
        }

    @staticmethod
    def _empty_result(source: str) -> dict[str, Any]:
        return {
            "text": "",
            "title": None,
            "authors": [],
            "publication_date": None,
            "source": source,
            "ok": False,
        }
