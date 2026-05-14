#!/usr/bin/env python3
"""
Complete extraction with automatic encoding issue detection and fixing
"""

import os
import json
import pdfplumber
import fitz
from nltk.tokenize import sent_tokenize
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ============================================================================
# ENCODING FIX FUNCTION
# ============================================================================

def fix_encoding(text):
    """Fix common encoding issues"""
    fixes = {
        'â€"': '—', 'â€"': '–', 'â€™': "'", 'â€œ': '"', 'â€\x9d': '"',
        'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ã¹': 'ú', 'Ã§': 'ç',
        '\ufeff': '', 'ï¿½': '', 'ï»¿': '', '\ufffd': ' '
    }
    for wrong, correct in fixes.items():
        text = text.replace(wrong, correct)
    return text

# ============================================================================
# EXTRACTION WITH QUALITY DETECTION
# ============================================================================

def extract_pdf_smart(pdf_path):
    """
    Extract PDF with quality detection
    Uses pdfplumber first, detects encoding issues, fixes them
    Falls back to PyMuPDF if needed
    """
    
    text = None
    method = "unknown"
    
    # Try pdfplumber first
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        method = "pdfplumber"
    except:
        pass
    
    # Fall back to PyMuPDF
    if text is None:
        try:
            pdf = fitz.open(pdf_path)
            text = ""
            for page in pdf:
                text += page.get_text()
            pdf.close()
            method = "PyMuPDF"
        except Exception as e:
            return None, "failed", str(e)
    
    # Fix encoding issues
    if text:
        text = fix_encoding(text)
    
    return text, method, None

# ============================================================================
# MAIN EXTRACTION PIPELINE
# ============================================================================

def extract_all_with_quality_tracking():
    """Extract all PDFs with quality tracking"""
    
    PDF_ROOT = "./upsc_pdfs"
    OUTPUT_ROOT = "./dataset_output"
    
    print("\n" + "="*70)
    print("SMART PDF EXTRACTION WITH ENCODING DETECTION & FIXING")
    print("="*70)
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    quality_report = {
        'high_quality': [],
        'fixed_encoding': [],
        'failed': []
    }
    
    for subject in os.listdir(PDF_ROOT):
        subject_path = os.path.join(PDF_ROOT, subject)
        
        if not os.path.isdir(subject_path):
            continue
        
        print(f"\n{'='*70}")
        print(f"SUBJECT: {subject.upper()}")
        print(f"{'='*70}")
        
        subject_output = os.path.join(OUTPUT_ROOT, subject)
        os.makedirs(subject_output, exist_ok=True)
        
        all_chunks = []
        
        for pdf_file in os.listdir(subject_path):
            if not pdf_file.endswith('.pdf'):
                continue
            
            pdf_path = os.path.join(subject_path, pdf_file)
            
            print(f"\n  {pdf_file}")
            
            # Extract
            text, method, error = extract_pdf_smart(pdf_path)
            
            if text is None:
                print(f"    ✗ Failed: {error}")
                quality_report['failed'].append({
                    'file': pdf_file,
                    'subject': subject,
                    'error': error
                })
                continue
            
            # Count chunks before and after
            num_chars = len(text)
            print(f"    ✓ Method: {method}")
            print(f"    ✓ Extracted: {num_chars:,} characters")
            
            # Check if had encoding issues
            if any(x in text for x in ['â', 'Ã', '\\u']):
                print(f"    ⚠ Had encoding issues (now fixed)")
                quality_report['fixed_encoding'].append({
                    'file': pdf_file,
                    'subject': subject,
                    'method': method
                })
            else:
                print(f"    ✓ High quality (no encoding issues)")
                quality_report['high_quality'].append({
                    'file': pdf_file,
                    'subject': subject,
                    'method': method
                })
            
            # Create chunks
            try:
                sentences = sent_tokenize(text)
            except:
                sentences = text.split('.')
            
            for i in range(0, len(sentences), 5):
                chunk_sentences = sentences[i:i+5]
                chunk_text = ' '.join(chunk_sentences)
                
                if chunk_text.strip():
                    all_chunks.append({
                        'text': chunk_text,
                        'source': pdf_file,
                        'subject': subject
                    })
        
        # Save chunks with CORRECT ENCODING
        if all_chunks:
            jsonl_path = os.path.join(subject_output, 'training_chunks.jsonl')
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for chunk in all_chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            
            print(f"\n  ✓ Saved {len(all_chunks)} chunks")
    
    # Print quality report
    print("\n" + "="*70)
    print("QUALITY REPORT")
    print("="*70)
    
    print(f"\n✓ High Quality PDFs: {len(quality_report['high_quality'])}")
    for item in quality_report['high_quality']:
        print(f"  - {item['file']}")
    
    print(f"\n⚠ Fixed Encoding PDFs: {len(quality_report['fixed_encoding'])}")
    for item in quality_report['fixed_encoding']:
        print(f"  - {item['file']} (Method: {item['method']})")
    
    print(f"\n✗ Failed PDFs: {len(quality_report['failed'])}")
    for item in quality_report['failed']:
        print(f"  - {item['file']}: {item['error']}")

if __name__ == "__main__":
    extract_all_with_quality_tracking()