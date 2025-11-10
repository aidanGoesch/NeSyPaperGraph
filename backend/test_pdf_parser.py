#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from services.pdf_preprocessor import extract_text_from_pdf

def test_pdf_parsing(pdf_path):
    """Test PDF parsing on a single file"""
    
    print(f"Testing PDF: {pdf_path}")
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return False
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    
    # Basic validation
    if not text or len(text.strip()) == 0:
        print("❌ No text extracted")
        return False
    
    print(f"✅ Successfully extracted {len(text)} characters")
    print(f"📄 First 500 characters:")
    print("-" * 50)
    print(text[:500])
    print("-" * 50)
        
    return True
    
    

def test_folder(folder_path):
    """Test PDF parsing on all PDFs in a folder"""
    folder = Path(folder_path)
    pdf_files = list(folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    
    success_count = 0
    for pdf_file in pdf_files:
        print(f"\n{'='*60}")
        if test_pdf_parsing(str(pdf_file)):
            success_count += 1
    
    print(f"\n📊 Results: {success_count}/{len(pdf_files)} PDFs parsed successfully")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_pdf_parser.py <pdf_file_or_folder>")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if os.path.isfile(path):
        test_pdf_parsing(path)

    elif os.path.isdir(path):
        test_folder(path)
    else:
        print(f"Path not found: {path}")
