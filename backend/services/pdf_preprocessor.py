from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a given PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: A string containing all extracted text,
             with page content separated by newlines.
    """
    reader = PdfReader(pdf_path)
    extracted_text = ""
    for page in reader.pages:
        extracted_text += page.extract_text() + "\n"
    return extracted_text

if __name__ == "__main__":
    # Example usage:
    pdf_file = "Aidan Goeschel - CV.pdf"  # Replace with the path to your PDF file
    text_content = extract_text_from_pdf(pdf_file)
    print(text_content)
