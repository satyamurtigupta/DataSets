#!/usr/bin/env python3
"""
Diagnostic script to identify which PDFs have encoding issues
Analyzes each PDF separately to pinpoint problems
"""

import os
import json
import pdfplumber
from collections import defaultdict
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

PDF_ROOT_FOLDER = "./upsc_pdfs"
SUBJECTS = [folder for folder in os.listdir(PDF_ROOT_FOLDER) 
            if os.path.isdir(os.path.join(PDF_ROOT_FOLDER, folder))]

# ============================================================================
# ENCODING DETECTION
# ============================================================================

def detect_encoding_issues(text):
    """
    Analyze text to detect encoding problems
    Returns: list of encoding issues found
    """
    issues = []
    
    # Common encoding error patterns
    encoding_error_patterns = {
        'UTF-8 BOM (corrupted)': ['\ufeff'],
        'Windows-1252 mapped as UTF-8': [
            'â€"', 'â€"', 'â€™', 'â€œ', 'â€\x9d',
            'â€\x93', 'â€\x94', 'Â', 'Ã', 'Ä'
        ],
        'Double encoded': [
            'Ã¡', 'Ã©', 'Ã­', 'Ã³', 'Ã¹',  # Accents
            'Ã', 'Ã§'  # Special chars
        ],
        'Mojibake (wrong encoding)': [
            '?', '?', '?', '?',  # Replacement chars
            '\ufffd'  # Unicode replacement
        ],
        'Mixed encoding': ['ï¿½', 'ï»¿', 'â\x80'],
        'Special Unicode symbols': [
            '✏', '✖', '✎', '☞', '✑', '✔', '✗'
        ]
    }
    
    issue_count = defaultdict(int)
    
    for issue_type, patterns in encoding_error_patterns.items():
        for pattern in patterns:
            count = text.count(pattern)
            if count > 0:
                issue_count[issue_type] += count
    
    return dict(issue_count)

def assess_pdf_quality(pdf_path):
    """
    Assess encoding quality of a single PDF
    Returns detailed analysis
    """
    
    issues = {}
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            num_pages = len(pdf.pages)
            
            # Extract text
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
            
            # Detect issues
            encoding_issues = detect_encoding_issues(full_text)
            
            issues = {
                'num_pages': num_pages,
                'extracted_chars': len(full_text),
                'encoding_issues': encoding_issues,
                'has_issues': len(encoding_issues) > 0,
                'issue_severity': 'HIGH' if sum(encoding_issues.values()) > 100 else 'MEDIUM' if sum(encoding_issues.values()) > 10 else 'LOW' if sum(encoding_issues.values()) > 0 else 'NONE'
            }
    
    except Exception as e:
        issues = {
            'num_pages': 0,
            'extracted_chars': 0,
            'encoding_issues': {'Error': str(e)},
            'has_issues': True,
            'issue_severity': 'ERROR'
        }
    
    return issues

# ============================================================================
# ANALYZE ALL PDFS
# ============================================================================

def analyze_all_pdfs():
    """Analyze all PDFs and generate report"""
    
    print("\n" + "╔" + "="*68 + "╗")
    print("║  PDF ENCODING DIAGNOSTIC TOOL" + " "*37 + "║")
    print("║  Identifies which PDFs have encoding issues" + " "*22 + "║")
    print("╚" + "="*68 + "╝")
    
    all_results = []
    
    for subject in SUBJECTS:
        subject_folder = os.path.join(PDF_ROOT_FOLDER, subject)
        pdf_files = [f for f in os.listdir(subject_folder) if f.endswith('.pdf')]
        
        print(f"\n{'='*70}")
        print(f"ANALYZING: {subject.upper()}")
        print(f"{'='*70}")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(subject_folder, pdf_file)
            
            print(f"\n  {pdf_file}")
            
            # Assess
            result = assess_pdf_quality(pdf_path)
            result['filename'] = pdf_file
            result['subject'] = subject
            
            # Display results
            print(f"    Pages: {result['num_pages']}")
            print(f"    Extracted chars: {result['extracted_chars']:,}")
            print(f"    Severity: {result['issue_severity']}")
            
            if result['encoding_issues']:
                print(f"    Issues found:")
                for issue_type, count in result['encoding_issues'].items():
                    print(f"      - {issue_type}: {count}")
            else:
                print(f"    ✓ No encoding issues detected")
            
            all_results.append(result)
    
    # ========================================================================
    # GENERATE REPORT
    # ========================================================================
    
    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)
    
    # Create DataFrame
    df_results = []
    for result in all_results:
        df_results.append({
            'PDF': result['filename'],
            'Subject': result['subject'],
            'Pages': result['num_pages'],
            'Chars': result['extracted_chars'],
            'Severity': result['issue_severity'],
            'Issues': len(result['encoding_issues'])
        })
    
    df = pd.DataFrame(df_results)
    
    # Sort by severity
    severity_order = {'NONE': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'ERROR': 4}
    df['severity_rank'] = df['Severity'].map(severity_order)
    df = df.sort_values('severity_rank', ascending=False)
    
    print("\nPDFs with Encoding Issues:")
    print("-" * 70)
    
    problem_pdfs = df[df['Severity'] != 'NONE']
    if len(problem_pdfs) > 0:
        print(problem_pdfs[['PDF', 'Subject', 'Severity', 'Issues', 'Chars']].to_string(index=False))
    else:
        print("✓ No PDFs with encoding issues found!")
    
    print("\n\nPDFs without Encoding Issues:")
    print("-" * 70)
    
    good_pdfs = df[df['Severity'] == 'NONE']
    if len(good_pdfs) > 0:
        print(good_pdfs[['PDF', 'Subject', 'Pages', 'Chars']].to_string(index=False))
    else:
        print("⚠ All PDFs have some encoding issues")
    
    # ========================================================================
    # DETAILED REPORT
    # ========================================================================
    
    print("\n" + "="*70)
    print("DETAILED ANALYSIS")
    print("="*70)
    
    # Group by severity
    print("\nHIGH SEVERITY (Many encoding issues - need fixing):")
    print("-" * 70)
    high_severity = [r for r in all_results if r['issue_severity'] == 'HIGH']
    if high_severity:
        for result in high_severity:
            print(f"\n{result['filename']} ({result['subject']})")
            print(f"  Total issues: {sum(result['encoding_issues'].values())}")
            for issue, count in result['encoding_issues'].items():
                print(f"    - {issue}: {count}")
    else:
        print("✓ None found")
    
    print("\n\nMEDIUM SEVERITY (Some encoding issues - consider fixing):")
    print("-" * 70)
    medium_severity = [r for r in all_results if r['issue_severity'] == 'MEDIUM']
    if medium_severity:
        for result in medium_severity:
            print(f"\n{result['filename']} ({result['subject']})")
            print(f"  Total issues: {sum(result['encoding_issues'].values())}")
            for issue, count in result['encoding_issues'].items():
                print(f"    - {issue}: {count}")
    else:
        print("✓ None found")
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    
    total_pdfs = len(all_results)
    good_pdfs_count = len([r for r in all_results if r['issue_severity'] == 'NONE'])
    problem_pdfs_count = total_pdfs - good_pdfs_count
    
    print(f"\nTotal PDFs: {total_pdfs}")
    print(f"Good PDFs (no issues): {good_pdfs_count} ({100*good_pdfs_count/total_pdfs:.1f}%)")
    print(f"Problem PDFs: {problem_pdfs_count} ({100*problem_pdfs_count/total_pdfs:.1f}%)")
    
    # ========================================================================
    # SAVE REPORT
    # ========================================================================
    
    report_file = './pdf_encoding_diagnostic_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Detailed report saved to: {report_file}")
    
    return all_results

if __name__ == "__main__":
    results = analyze_all_pdfs()