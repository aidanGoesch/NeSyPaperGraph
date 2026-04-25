import argparse
import logging
import os

from services.pdf_preprocessor import extract_text_from_pdf
from services.llm_service import OpenAILLMClient, TopicExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Debug the PDF → LLM pipeline (metadata, topics, summary) for a single PDF."
    )
    parser.add_argument("pdf_path", help="Path to a PDF file")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if os.getenv("DEBUG_LLM", "0").lower() in {"1", "true", "yes", "y"} else logging.INFO
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)

    logger.info(f"Reading PDF from {args.pdf_path}")
    text = extract_text_from_pdf(args.pdf_path)
    logger.info(f"Extracted text length: {len(text)} characters")

    client = OpenAILLMClient()
    extractor = TopicExtractor(client)

    logger.info("Running metadata extraction...")
    metadata = extractor.extract_paper_metadata(text)
    logger.info(f"Metadata result: {metadata}")

    logger.info("Running topic extraction...")
    topics = extractor.extract_topics(text)
    logger.info(f"Extracted {len(topics)} topics: {topics}")

    logger.info("Running summary generation...")
    summary = client.generate_summary(text)
    logger.info(f"Summary length: {len(summary)} characters")
    logger.debug(f"Summary preview:\n{summary[:800]}")


if __name__ == "__main__":
    main()

