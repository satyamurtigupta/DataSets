#!/usr/bin/env python3
"""
Subject-Wise PDF Extraction for UPSC SLM
Extracts PDFs from separate subject folders and creates separate datasets
"""

import os
import json
import re
from pathlib import Path
from collections import defaultdict
import pdfplumber
from nltk.tokenize import sent_tokenize
import pandas as pd
import nltk

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ============================================================================
# CONFIGURATION
# ============================================================================

PDF_ROOT_FOLDER = "./upsc_pdfs"  # Root folder containing subject subfolders
OUTPUT_ROOT_FOLDER = "./dataset_output"
CHUNK_SIZE = 5  # Sentences per chunk

# Automatically detect subjects from folder structure
SUBJECTS = [folder for folder in os.listdir(PDF_ROOT_FOLDER) 
            if os.path.isdir(os.path.join(PDF_ROOT_FOLDER, folder))]

print(f"Detected subjects: {SUBJECTS}")

# ============================================================================
# STEP 1: TEXT EXTRACTION
# ============================================================================

def extract_pdfs_from_subject(subject_folder_path, subject_name):
    """
    Extract text from all PDFs in a subject folder
    """
    print(f"\n{'='*70}")
    print(f"EXTRACTING: {subject_name.upper()}")
    print(f"{'='*70}")
    
    extracted_data = {}
    pdf_files = [f for f in os.listdir(subject_folder_path) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {subject_folder_path}")
        return extracted_data
    
    print(f"Found {len(pdf_files)} PDFs in {subject_name}/")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(subject_folder_path, pdf_file)
        print(f"\n  Processing: {pdf_file}")
        
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
                
                extracted_data[pdf_file] = {
                    'text': text,
                    'num_pages': num_pages,
                    'num_chars': len(text),
                    'subject': subject_name
                }
                
                print(f"    ✓ {num_pages} pages, {len(text):,} characters")
        
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    return extracted_data

# ============================================================================
# STEP 2: TEXT CLEANING
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

def clean_pdfs(extracted_data, subject_name):
    """Clean all extracted texts"""
    print(f"\nCleaning {subject_name.upper()} texts...")
    
    cleaned_data = {}
    for filename, data in extracted_data.items():
        cleaned = clean_text(data['text'])
        cleaned_data[filename] = {
            'text': cleaned,
            'original_length': len(data['text']),
            'cleaned_length': len(cleaned),
            'num_pages': data['num_pages'],
            'subject': subject_name
        }
        print(f"  ✓ {filename}: {len(data['text']):,} → {len(cleaned):,} chars")
    
    return cleaned_data

# ============================================================================
# STEP 3: TEXT CHUNKING
# ============================================================================

def chunk_text(text, chunk_size=5):
    """Split text into chunks by sentences"""
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = text.split('.')
    
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = ' '.join(sentences[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def create_chunks(cleaned_data, subject_name, chunk_size=5):
    """Create chunks from all texts"""
    print(f"\nCreating chunks for {subject_name.upper()}...")
    
    all_chunks = []
    
    for filename, data in cleaned_data.items():
        chunks = chunk_text(data['text'], chunk_size)
        
        # Add metadata
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                'text': chunk,
                'source': filename,
                'subject': subject_name,
                'chunk_id': i,
                'length': len(chunk)
            })
        
        print(f"  ✓ {filename}: {len(chunks)} chunks")
    
    print(f"  TOTAL CHUNKS ({subject_name}): {len(all_chunks)}")
    return all_chunks

# ============================================================================
# STEP 4: SAVE SUBJECT DATA
# ============================================================================

def save_subject_data(extracted_data, cleaned_data, all_chunks, subject_name):
    """Save all subject data to separate folders"""
    
    subject_output_dir = os.path.join(OUTPUT_ROOT_FOLDER, subject_name)
    os.makedirs(subject_output_dir, exist_ok=True)
    
    print(f"\nSaving {subject_name.upper()} data...")
    
    # Save extracted texts
    with open(os.path.join(subject_output_dir, 'extracted_texts.json'), 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved extracted texts")
    
    # Save cleaned texts
    with open(os.path.join(subject_output_dir, 'cleaned_texts.json'), 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved cleaned texts")
    
    # Save chunks as JSONL
    jsonl_path = os.path.join(subject_output_dir, 'training_chunks.jsonl')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    print(f"  ✓ Saved {len(all_chunks)} chunks to JSONL")
    
    return subject_output_dir

# ============================================================================
# STEP 5: CREATE HUGGINGFACE DATASET
# ============================================================================

def create_huggingface_dataset(jsonl_path, subject_output_dir):
    """Convert to HuggingFace dataset format"""
    from datasets import Dataset, load_dataset
    
    print(f"\nConverting to HuggingFace format...")
    
    try:
        # Load from JSONL
        dataset = load_dataset('json', data_files=jsonl_path)['train']
        
        # Split
        split = dataset.train_test_split(test_size=0.1)
        
        # Save
        output_path = os.path.join(subject_output_dir, 'hf_dataset')
        split.save_to_disk(output_path)
        
        print(f"  ✓ Train size: {len(split['train'])}")
        print(f"  ✓ Val size: {len(split['test'])}")
        print(f"  ✓ Saved to {output_path}")
        
        return split
    
    except ImportError:
        print("  ⚠ HuggingFace datasets not installed")
        return None

# ============================================================================
# STEP 6: STATISTICS
# ============================================================================

def compute_statistics(all_chunks, subject_name, subject_output_dir):
    """Compute and save dataset statistics"""
    
    text_lengths = [chunk['length'] for chunk in all_chunks]
    
    statistics = {
        'subject': subject_name,
        'total_chunks': len(all_chunks),
        'average_chunk_length': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
        'min_chunk_length': min(text_lengths) if text_lengths else 0,
        'max_chunk_length': max(text_lengths) if text_lengths else 0,
        'median_chunk_length': sorted(text_lengths)[len(text_lengths)//2] if text_lengths else 0,
    }
    
    # Count by source
    source_counts = defaultdict(int)
    for chunk in all_chunks:
        source_counts[chunk['source']] += 1
    
    statistics['chunks_by_source'] = dict(source_counts)
    
    # Save statistics
    stats_file = os.path.join(subject_output_dir, 'statistics.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2)
    
    return statistics

# ============================================================================
# STEP 7: PROCESS ALL SUBJECTS
# ============================================================================

def process_all_subjects():
    """Process all subjects"""
    
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║  SUBJECT-WISE PDF EXTRACTION FOR UPSC SLM" + " "*24 + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    os.makedirs(OUTPUT_ROOT_FOLDER, exist_ok=True)
    
    subject_stats = {}
    all_subjects_chunks = []
    
    # Process each subject
    for subject_name in SUBJECTS:
        subject_folder = os.path.join(PDF_ROOT_FOLDER, subject_name)
        
        # Extract
        extracted_data = extract_pdfs_from_subject(subject_folder, subject_name)
        if not extracted_data:
            print(f"\n⚠ Skipping {subject_name} (no PDFs found)")
            continue
        
        # Clean
        cleaned_data = clean_pdfs(extracted_data, subject_name)
        
        # Chunk
        all_chunks = create_chunks(cleaned_data, subject_name, CHUNK_SIZE)
        
        # Save
        subject_output_dir = save_subject_data(extracted_data, cleaned_data, all_chunks, subject_name)
        
        # HuggingFace
        hf_dataset = create_huggingface_dataset(
            os.path.join(subject_output_dir, 'training_chunks.jsonl'),
            subject_output_dir
        )
        
        # Statistics
        stats = compute_statistics(all_chunks, subject_name, subject_output_dir)
        subject_stats[subject_name] = stats
        
        # Add to combined
        all_subjects_chunks.extend(all_chunks)
        
        print(f"\n✓ {subject_name.upper()} COMPLETE")
        print(f"  Chunks: {len(all_chunks)}")
        print(f"  Avg length: {stats['average_chunk_length']:.0f} chars")

    # ========================================================================
    # STEP 8: CREATE COMBINED DATASET
    # ========================================================================
    
    print("\n" + "="*70)
    print("CREATING COMBINED DATASET")
    print("="*70)
    
    combined_output_dir = os.path.join(OUTPUT_ROOT_FOLDER, 'combined')
    os.makedirs(combined_output_dir, exist_ok=True)
    
    # Save combined chunks
    combined_jsonl = os.path.join(combined_output_dir, 'training_chunks.jsonl')
    with open(combined_jsonl, 'w', encoding='utf-8') as f:
        for chunk in all_subjects_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Saved {len(all_subjects_chunks)} combined chunks")
    
    # Create combined HuggingFace dataset
    create_huggingface_dataset(combined_jsonl, combined_output_dir)
    
    # Combined statistics
    combined_stats = {
        'total_chunks': len(all_subjects_chunks),
        'subjects': subject_stats,
        'total_by_subject': {subject: stats['total_chunks'] for subject, stats in subject_stats.items()}
    }
    
    with open(os.path.join(combined_output_dir, 'statistics.json'), 'w') as f:
        json.dump(combined_stats, f, indent=2)
    
    # ========================================================================
    # STEP 9: FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("✓ EXTRACTION COMPLETE!")
    print("="*70)
    
    print("\nSUMMARY BY SUBJECT:")
    print("-" * 70)
    
    summary_data = []
    for subject in SUBJECTS:
        if subject in subject_stats:
            stats = subject_stats[subject]
            summary_data.append({
                'Subject': subject,
                'Chunks': stats['total_chunks'],
                'Avg Length': f"{stats['average_chunk_length']:.0f}",
                'Min': stats['min_chunk_length'],
                'Max': stats['max_chunk_length']
            })
    
    # Create summary table
    if summary_data:
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
    
    print("\n" + "-" * 70)
    print(f"TOTAL CHUNKS (All subjects): {len(all_subjects_chunks)}")
    print(f"Average chunk size: {sum(c['length'] for c in all_subjects_chunks) / len(all_subjects_chunks):.0f} chars")
    print(f"\nOutput directories:")
    
    for subject in SUBJECTS:
        if subject in subject_stats:
            print(f"  {OUTPUT_ROOT_FOLDER}/{subject}/")
    print(f"  {OUTPUT_ROOT_FOLDER}/combined/")
    
    print("\n" + "="*70)
    print("READY FOR TRAINING!")
    print("="*70)
    print("\nYou can now train separate models for each subject:")
    print("  - Phase 1: domain_adaptation_by_subject.py")
    print("  - Phase 2: instruction_tuning_by_subject.py")
    print("  - Phase 3: quantization_by_subject.py")
    
    return all_subjects_chunks, subject_stats

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    all_chunks, stats = process_all_subjects()