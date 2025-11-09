from pypdf import PdfReader
import re
from keybert import KeyBERT
from rapidfuzz import fuzz
from .llm_service import LLMClient, TopicExtractor

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    extracted_text = ""
    for page in reader.pages:
        extracted_text += page.extract_text() + "\n"
    return extracted_text

def extract_topics(paper_text, current_topics=None):
    """Extract topics from paper text using AWS Bedrock."""
    kw_model = KeyBERT(model="all-MiniLM-L6-v2")

    keywords = kw_model.extract_keywords(paper_text, keyphrase_ngram_range=(1, 3), stop_words="english", top_n=10)

    return clean_keybert_topics(keywords)

def clean_keybert_topics(topics, similarity_threshold=80):
    phrases = [p for p, _ in topics]
    
    # Normalize
    clean_phrases = []
    for p in phrases:
        p = p.lower().strip()
        p = re.sub(r'\brewards?\b', 'reward', p)
        p = re.sub(r'\s+', ' ', p)
        clean_phrases.append(p)
    
    # Deduplicate
    unique_phrases = []
    for phrase in clean_phrases:
        if not any(fuzz.ratio(phrase, up) > similarity_threshold for up in unique_phrases):
            unique_phrases.append(phrase)
    
    return unique_phrases

