from pypdf import PdfReader
from io import BytesIO

def extract_text_from_pdf(pdf_source):
    """
    Extracts all text from a given PDF file.

    Args:
        pdf_source: Either a file path (str) or file content (bytes/BytesIO)

    Returns:
        str: A string containing all extracted text,
             with page content separated by newlines.
    """
    # If it's bytes, wrap in BytesIO; if it's already BytesIO or a path, use directly
    if isinstance(pdf_source, bytes):
        pdf_source = BytesIO(pdf_source)
    
    reader = PdfReader(pdf_source)
    extracted_text = ""
    for page in reader.pages:
        extracted_text += page.extract_text() + "\n"
    return extracted_text
