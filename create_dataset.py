#!/usr/bin/env python3
"""
Complete script to extract PDFs and create training dataset for UPSC SLM
Run: python create_dataset.py
"""

import os
import json
import pdfplumber
import re
from pathlib import Path
from nltk.tokenize import sent_tokenize
import nltk

# Download required data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ============================================================================
# CONFIGURATION
# ============================================================================

PDF_FOLDER = "./upsc_pdfs"  # Change this to your PDF folder
OUTPUT_FOLDER = "./dataset_output"
CHUNK_SIZE = 5  # Number of sentences per chunk

# ============================================================================
# STEP 1: EXTRACT TEXT FROM PDFs
# ============================================================================

def extract_pdfs(pdf_folder):
    """Extract text from all PDFs in folder"""
    print("=" * 60)
    print("STEP 1: EXTRACTING TEXT FROM PDFs")
    print("=" * 60)
    
    extracted_data = {}
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_folder}")
        return extracted_data
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"\nProcessing: {pdf_file}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                num_pages = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    
                    # Extract tables if any
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            table_text = "\n".join([
                                " | ".join(str(cell) for cell in row) 
                                for row in table
                            ])
                            text += f"\nTABLE:\n{table_text}\n"
                
                extracted_data[pdf_file] = text
                print(f"  ✓ Extracted {num_pages} pages, {len(text):,} characters")
        
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    return extracted_data

# ============================================================================
# STEP 2: CLEAN TEXT
# ============================================================================

def clean_text(text):
    """Clean extracted text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page markers
    text = re.sub(r'Page \d+', '', text)
    text = re.sub(r'---.*?---', '', text)
    # Fix encoding issues
    text = text.replace('ï¿½', '')
    text = text.replace('â€™', "'")
    text = text.replace('â€œ', '"')
    text = text.replace('â€\x9d', '"')
    text = text.replace('â€"', '—')
    
    return text.strip()

def clean_pdfs(extracted_data):
    """Clean all extracted texts"""
    print("\n" + "=" * 60)
    print("STEP 2: CLEANING TEXT")
    print("=" * 60)
    
    cleaned_data = {}
    for filename, text in extracted_data.items():
        cleaned = clean_text(text)
        cleaned_data[filename] = cleaned
        print(f"\n  ✓ {filename}")
        print(f"    Original: {len(text):,} chars → Cleaned: {len(cleaned):,} chars")
    
    return cleaned_data

# ============================================================================
# STEP 3: CREATE CHUNKS
# ============================================================================

def chunk_text(text, chunk_size=5):
    """Split text into chunks by sentences"""
    try:
        sentences = sent_tokenize(text)
    except:
        # Fallback if tokenizer fails
        sentences = text.split('.')
    
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = ' '.join(sentences[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def create_chunks(cleaned_data, chunk_size=5):
    """Create chunks from all texts"""
    print("\n" + "=" * 60)
    print("STEP 3: CREATING CHUNKS")
    print("=" * 60)
    
    all_chunks = []
    
    for filename, text in cleaned_data.items():
        chunks = chunk_text(text, chunk_size)
        
        # Add metadata
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                'text': chunk,
                'source': filename,
                'chunk_id': i
            })
        
        print(f"\n  ✓ {filename}")
        print(f"    Created {len(chunks)} chunks")
    
    print(f"\n  TOTAL CHUNKS: {len(all_chunks)}")
    return all_chunks

# ============================================================================
# STEP 4: FORMAT FOR TRAINING
# ============================================================================

def format_training_data(chunks):
    """Format chunks for training"""
    training_data = []
    
    for chunk in chunks:
        training_data.append({
            'text': chunk['text'],
            'source': chunk['source']
        })
    
    return training_data

# ============================================================================
# STEP 5: SAVE DATASET
# ============================================================================

def save_dataset(training_data, output_folder):
    """Save dataset to JSONL format"""
    print("\n" + "=" * 60)
    print("STEP 4: SAVING DATASET")
    print("=" * 60)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Save as JSONL
    jsonl_path = os.path.join(output_folder, 'domain_adaptation_dataset.jsonl')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n  ✓ Saved to {jsonl_path}")
    print(f"    Total examples: {len(training_data)}")
    
    return jsonl_path

# ============================================================================
# STEP 6: CREATE HUGGINGFACE DATASET
# ============================================================================

def create_huggingface_dataset(jsonl_path, output_folder):
    """Convert to HuggingFace dataset format"""
    print("\n" + "=" * 60)
    print("STEP 5: CREATING HUGGINGFACE DATASET")
    print("=" * 60)
    
    try:
        from datasets import Dataset, load_dataset
        import pandas as pd
        
        # Load from JSONL
        dataset = load_dataset('json', data_files=jsonl_path)['train']
        
        # Split
        split = dataset.train_test_split(test_size=0.1)
        
        # Save
        output_path = os.path.join(output_folder, 'hf_dataset')
        split.save_to_disk(output_path)
        
        print(f"\n  ✓ Train size: {len(split['train'])}")
        print(f"  ✓ Val size: {len(split['test'])}")
        print(f"  ✓ Saved to {output_path}")
        
        return split
    
    except ImportError:
        print("\n  ⚠ HuggingFace datasets not installed")
        print("  Run: pip install datasets")
        return None

# ============================================================================
# STEP 7: STATISTICS
# ============================================================================

def print_statistics(all_chunks, training_data):
    """Print dataset statistics"""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    text_lengths = [len(item['text']) for item in training_data]
    
    print(f"\nTotal examples: {len(training_data)}")
    print(f"Average text length: {sum(text_lengths)/len(text_lengths):.0f} characters")
    print(f"Min length: {min(text_lengths)}")
    print(f"Max length: {max(text_lengths)}")
    print(f"Median length: {sorted(text_lengths)[len(text_lengths)//2]}")
    
    # By source
    print(f"\nExamples by source:")
    source_counts = {}
    for item in training_data:
        source = item['source']
        source_counts[source] = source_counts.get(source, 0) + 1
    
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution"""
    with open('dataset.txt', 'r') as f:
        data = f.read().splitlines()
    for PDF_FOLDER in data:
        print("\n")
        print("╔" + "=" * 58 + "╗")
        print("║" + " " * 58 + "║")
        print("║  DATASET CREATION FOR UPSC SLM FINE-TUNING" + " " * 14 + "║")
        print("║" + " " * 58 + "║")
        print("╚" + "=" * 58 + "╝")
        
        # Check if PDF folder exists
        if not os.path.exists(PDF_FOLDER):
            print(f"\n✗ PDF folder not found: {PDF_FOLDER}")
            print(f"  Create folder and add PDF files:")
            print(f"    mkdir {PDF_FOLDER}")
            print(f"    # Copy NCERT, PIB PDFs to this folder")
            return
        
        # Run pipeline
        try:
            # Step 1: Extract
            extracted_data = extract_pdfs(PDF_FOLDER)
            if not extracted_data:
                print("\n✗ No PDFs found or extraction failed")
                return
            
            # Step 2: Clean
            cleaned_data = clean_pdfs(extracted_data)
            
            # Step 3: Chunk
            all_chunks = create_chunks(cleaned_data, CHUNK_SIZE)
            
            # Step 4: Format
            training_data = format_training_data(all_chunks)
            
            # Step 5: Save
            jsonl_path = save_dataset(training_data, OUTPUT_FOLDER)
            
            # Step 6: HuggingFace
            hf_dataset = create_huggingface_dataset(jsonl_path, OUTPUT_FOLDER)
            
            # Step 7: Statistics
            print_statistics(all_chunks, training_data)
            
            # Success
            print("\n" + "=" * 60)
            print("✓ DATASET CREATION COMPLETE!")
            print("=" * 60)
            print(f"\nOutput files:")
            print(f"  - {jsonl_path}")
            print(f"  - {OUTPUT_FOLDER}/hf_dataset/")
            print(f"\nYou can now use this dataset for Phase 1 training!")
            print("=" * 60 + "\n")
        
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()