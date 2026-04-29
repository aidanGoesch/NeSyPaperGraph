import os
from typing import Any
from urllib import error, request
import xml.etree.ElementTree as ET
from uuid import uuid4

from services.observability import timed_block


class GrobidService:
    def __init__(self) -> None:
        self.enabled = os.getenv("GROBID_ENABLED", "true").lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        self.base_url = os.getenv("GROBID_URL", "http://localhost:8070").rstrip("/")
        self.timeout_seconds = float(
            os.getenv("GROBID_TIMEOUT_SECONDS", "8.0") or "8.0"
        )

    def extract_metadata(self, pdf_bytes: bytes) -> dict[str, Any]:
        if not self.enabled:
            return {
                "title": None,
                "authors": [],
                "publication_date": None,
                "source": "disabled",
                "ok": False,
            }

        if not pdf_bytes:
            return {
                "title": None,
                "authors": [],
                "publication_date": None,
                "source": "empty_input",
                "ok": False,
            }

        try:
            with timed_block("grobid_extract_metadata_per_paper"):
                xml_text = self._process_header_document(pdf_bytes)
            metadata = self._parse_tei_header(xml_text)
            metadata["source"] = "grobid"
            metadata["ok"] = bool(metadata["title"] or metadata["authors"])
            return metadata
        except Exception:
            return {
                "title": None,
                "authors": [],
                "publication_date": None,
                "source": "grobid_error",
                "ok": False,
            }

    def _process_header_document(self, pdf_bytes: bytes) -> str:
        endpoint = f"{self.base_url}/api/processHeaderDocument"
        boundary = f"----NeSyBoundary{uuid4().hex}"
        multipart_body = (
            f"--{boundary}\r\n"
            "Content-Disposition: form-data; name=\"input\"; filename=\"paper.pdf\"\r\n"
            "Content-Type: application/pdf\r\n\r\n"
        ).encode("utf-8") + pdf_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        req = request.Request(
            endpoint,
            data=multipart_body,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Accept": "application/xml",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except error.URLError as exc:
            raise RuntimeError(f"GROBID request failed: {exc}") from exc

    def _parse_tei_header(self, xml_text: str) -> dict[str, Any]:
        root = ET.fromstring(xml_text)
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}

        title = self._first_nonempty(
            root.findall(".//tei:titleStmt/tei:title", ns),
            transform=lambda node: (node.text or "").strip(),
        )

        authors = []
        for author in root.findall(".//tei:sourceDesc//tei:author", ns):
            forename = self._first_nonempty(
                author.findall(".//tei:forename", ns),
                transform=lambda node: (node.text or "").strip(),
            )
            surname = self._first_nonempty(
                author.findall(".//tei:surname", ns),
                transform=lambda node: (node.text or "").strip(),
            )
            full = " ".join(part for part in [forename, surname] if part).strip()
            if full:
                authors.append(full)

        publication_date = self._extract_publication_date(root, ns)
        return {
            "title": title or None,
            "authors": authors,
            "publication_date": publication_date,
        }

    @staticmethod
    def _first_nonempty(nodes, transform):
        for node in nodes:
            value = transform(node)
            if value:
                return value
        return ""

    def _extract_publication_date(self, root, ns) -> str | None:
        date_nodes = root.findall(".//tei:sourceDesc//tei:date", ns)
        for date_node in date_nodes:
            when = (date_node.attrib.get("when") or "").strip()
            if when:
                return when
            text_value = (date_node.text or "").strip()
            if text_value:
                return text_value
        return None
