#!/usr/bin/env python3
"""
PDF Metadata Extractor for Scientific Articles

Extracts title and other metadata from PDF files in the sources directory.
Detects duplicate articles based on title similarity.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import pandas as pd
import re
from difflib import SequenceMatcher

# Try different PDF libraries
try:
    import pypdf
    PDF_LIBRARY = 'pypdf'
except ImportError:
    try:
        from PyPDF2 import PdfReader
        PDF_LIBRARY = 'PyPDF2'
    except ImportError:
        PDF_LIBRARY = None


def extract_metadata_pypdf(pdf_path: str) -> Dict[str, Optional[str]]:
    """Extract metadata using pypdf library."""
    try:
        reader = pypdf.PdfReader(pdf_path)
        metadata = reader.metadata
        
        if metadata is None:
            return {'title': None, 'author': None, 'subject': None, 'creator': None}
        
        return {
            'title': metadata.get('/Title', None),
            'author': metadata.get('/Author', None),
            'subject': metadata.get('/Subject', None),
            'creator': metadata.get('/Creator', None),
        }
    except Exception as e:
        return {'title': None, 'author': None, 'subject': None, 'creator': None, 'error': str(e)}


def extract_metadata_pypdf2(pdf_path: str) -> Dict[str, Optional[str]]:
    """Extract metadata using PyPDF2 library."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        metadata = reader.metadata
        
        if metadata is None:
            return {'title': None, 'author': None, 'subject': None, 'creator': None}
        
        return {
            'title': metadata.get('/Title', None),
            'author': metadata.get('/Author', None),
            'subject': metadata.get('/Subject', None),
            'creator': metadata.get('/Creator', None),
        }
    except Exception as e:
        return {'title': None, 'author': None, 'subject': None, 'creator': None, 'error': str(e)}


def extract_metadata(pdf_path: str) -> Dict[str, Optional[str]]:
    """Extract metadata from PDF file using available library."""
    if PDF_LIBRARY == 'pypdf':
        return extract_metadata_pypdf(pdf_path)
    elif PDF_LIBRARY == 'PyPDF2':
        return extract_metadata_pypdf2(pdf_path)
    else:
        return {'title': None, 'error': 'No PDF library available'}


def find_pdf_files(base_dir: str) -> List[Path]:
    """Recursively find all PDF files in directory."""
    base_path = Path(base_dir)
    pdf_files = list(base_path.rglob('*.pdf'))
    return sorted(pdf_files)


def clean_title(title: Optional[str]) -> Optional[str]:
    """Clean up extracted title."""
    if title is None:
        return None
    
    # Remove leading/trailing whitespace
    title = title.strip()
    
    # Skip if empty or just whitespace
    if not title:
        return None
    
    # Skip if it's just a filename or path
    if title.endswith('.pdf') or '/' in title or '\\' in title:
        return None
    
    # Skip if it's too short (likely not a real title)
    if len(title) < 10:
        return None
    
    # Skip common non-title patterns
    skip_patterns = [
        'Microsoft Word',
        'untitled',
        'Untitled',
        'Document',
    ]
    for pattern in skip_patterns:
        if pattern in title:
            return None
    
    return title


def normalize_title(title: str) -> str:
    """Normalize title for comparison (lowercase, remove special chars, etc.)."""
    if not title:
        return ""
    
    # Convert to lowercase
    normalized = title.lower()
    
    # Remove HTML entities
    normalized = re.sub(r'&#x[0-9a-fA-F]+;', '', normalized)
    normalized = re.sub(r'&[a-z]+;', '', normalized)
    
    # Remove special characters but keep letters, numbers, spaces
    normalized = re.sub(r'[^a-z0-9\s]', ' ', normalized)
    
    # Normalize whitespace
    normalized = ' '.join(normalized.split())
    
    return normalized


def calculate_similarity(title1: str, title2: str) -> float:
    """Calculate similarity between two titles using SequenceMatcher."""
    norm1 = normalize_title(title1)
    norm2 = normalize_title(title2)
    
    if not norm1 or not norm2:
        return 0.0
    
    return SequenceMatcher(None, norm1, norm2).ratio()


def find_duplicates(df: pd.DataFrame, similarity_threshold: float = 0.85) -> pd.DataFrame:
    """
    Find duplicate articles based on title similarity.
    
    Returns DataFrame with additional columns:
    - is_duplicate: True if this is a duplicate of another file
    - duplicate_of: filename of the original (first occurrence)
    - duplicate_group: group ID for duplicates
    """
    # Initialize new columns
    df['is_duplicate'] = False
    df['duplicate_of'] = ''
    df['duplicate_group'] = 0
    
    # Get only rows with valid titles
    with_titles = df[df['has_title']].copy()
    
    if len(with_titles) == 0:
        return df
    
    # Create normalized titles for comparison
    with_titles['norm_title'] = with_titles['title'].apply(normalize_title)
    
    # Group by exact normalized title first (fast)
    exact_groups = with_titles.groupby('norm_title').groups
    
    group_id = 1
    processed_indices = set()
    duplicates_found = []
    
    # Process exact matches first
    for norm_title, indices in exact_groups.items():
        if len(indices) > 1:
            indices_list = list(indices)
            original_idx = indices_list[0]
            original_file = df.loc[original_idx, 'filename']
            
            df.loc[original_idx, 'duplicate_group'] = group_id
            
            for dup_idx in indices_list[1:]:
                df.loc[dup_idx, 'is_duplicate'] = True
                df.loc[dup_idx, 'duplicate_of'] = original_file
                df.loc[dup_idx, 'duplicate_group'] = group_id
                duplicates_found.append((df.loc[dup_idx, 'filename'], original_file, 1.0))
            
            processed_indices.update(indices_list)
            group_id += 1
    
    # Now check for fuzzy matches (slower, but catches near-duplicates)
    remaining = with_titles[~with_titles.index.isin(processed_indices)]
    remaining_list = list(remaining.iterrows())
    
    for i, (idx1, row1) in enumerate(remaining_list):
        if idx1 in processed_indices:
            continue
            
        current_group = []
        
        for idx2, row2 in remaining_list[i+1:]:
            if idx2 in processed_indices:
                continue
            
            similarity = calculate_similarity(row1['title'], row2['title'])
            
            if similarity >= similarity_threshold:
                if not current_group:
                    current_group = [idx1]
                    df.loc[idx1, 'duplicate_group'] = group_id
                
                current_group.append(idx2)
                df.loc[idx2, 'is_duplicate'] = True
                df.loc[idx2, 'duplicate_of'] = row1['filename']
                df.loc[idx2, 'duplicate_group'] = group_id
                duplicates_found.append((row2['filename'], row1['filename'], similarity))
                processed_indices.add(idx2)
        
        if current_group:
            processed_indices.add(idx1)
            group_id += 1
    
    return df, duplicates_found


def main():
    """Main function to extract and display PDF metadata."""
    # Find sources directory
    script_dir = Path(__file__).parent
    sources_dir = script_dir / 'sources'
    
    if not sources_dir.exists():
        print(f"Error: sources directory not found at {sources_dir}")
        sys.exit(1)
    
    print(f"PDF Library: {PDF_LIBRARY or 'None (install pypdf or PyPDF2)'}")
    print(f"Scanning: {sources_dir}")
    print("=" * 80)
    
    if PDF_LIBRARY is None:
        print("\nError: No PDF library found!")
        print("Please install one of the following:")
        print("  pip install pypdf")
        print("  pip install PyPDF2")
        sys.exit(1)
    
    # Find all PDF files
    pdf_files = find_pdf_files(sources_dir)
    print(f"Found {len(pdf_files)} PDF files\n")
    
    # Extract metadata from each file
    results = []
    
    for pdf_path in pdf_files:
        relative_path = pdf_path.relative_to(sources_dir)
        folder = str(relative_path.parent)
        filename = pdf_path.name
        
        metadata = extract_metadata(str(pdf_path))
        title = clean_title(metadata.get('title'))
        author = metadata.get('author')
        error = metadata.get('error')
        
        results.append({
            'folder': folder,
            'filename': filename,
            'title': title,
            'author': author,
            'has_title': title is not None,
            'error': error
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Find duplicates
    print("Analyzing duplicates...")
    df, duplicates_found = find_duplicates(df, similarity_threshold=0.85)
    
    # Summary statistics
    total = len(df)
    with_title = df['has_title'].sum()
    with_error = df['error'].notna().sum()
    n_duplicates = df['is_duplicate'].sum()
    n_groups = df[df['duplicate_group'] > 0]['duplicate_group'].nunique()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total PDF files:     {total}")
    print(f"With valid title:    {with_title} ({100*with_title/total:.1f}%)")
    print(f"Without title:       {total - with_title} ({100*(total-with_title)/total:.1f}%)")
    print(f"With errors:         {with_error}")
    print(f"Duplicate files:     {n_duplicates} (in {n_groups} groups)")
    print()
    
    # Display duplicates
    if n_duplicates > 0:
        print("=" * 80)
        print("DUPLICATE ARTICLES DETECTED")
        print("=" * 80)
        
        for group_id in sorted(df[df['duplicate_group'] > 0]['duplicate_group'].unique()):
            group_df = df[df['duplicate_group'] == group_id]
            original = group_df[~group_df['is_duplicate']].iloc[0] if len(group_df[~group_df['is_duplicate']]) > 0 else group_df.iloc[0]
            duplicates = group_df[group_df['is_duplicate']]
            
            print(f"\n📄 Group {group_id}:")
            title = original['title'] if pd.notna(original['title']) else "Unknown"
            title_short = title[:80] + "..." if len(title) > 80 else title
            print(f"   Title: {title_short}")
            print(f"   Original: {original['folder']}/{original['filename']}")
            
            for _, dup in duplicates.iterrows():
                # Find similarity
                sim = 1.0
                for d_file, o_file, s in duplicates_found:
                    if d_file == dup['filename']:
                        sim = s
                        break
                print(f"   ⚠️  Duplicate: {dup['folder']}/{dup['filename']} (similarity: {sim:.1%})")
        print()
    
    # Display table
    print("=" * 80)
    print("PDF FILES WITH METADATA")
    print("=" * 80)
    
    # Group by folder
    for folder in df['folder'].unique():
        folder_df = df[df['folder'] == folder]
        print(f"\n📁 {folder}/")
        print("-" * 78)
        
        for _, row in folder_df.iterrows():
            status = "✓" if row['has_title'] else "✗"
            title = row['title'] if pd.notna(row['title']) else None
            title_display = title[:60] + "..." if title and len(title) > 60 else title
            
            # Add duplicate marker
            dup_marker = ""
            if row['is_duplicate']:
                dup_marker = f" ⚠️ DUPLICATE of {row['duplicate_of']}"
            elif row['duplicate_group'] > 0:
                dup_marker = f" 📋 Has duplicates (group {row['duplicate_group']})"
            
            print(f"  {status} {row['filename']}{dup_marker}")
            if title:
                print(f"      Title: {title_display}")
            if pd.notna(row['error']):
                print(f"      Error: {row['error']}")
    
    # Save to CSV with duplicate info
    output_path = script_dir / 'sources_metadata.csv'
    df_export = df[['folder', 'filename', 'title', 'author', 'has_title', 'is_duplicate', 'duplicate_of', 'duplicate_group']]
    df_export.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n\nResults saved to: {output_path}")
    
    # Summary of duplicates at the end
    if n_duplicates > 0:
        print("\n" + "=" * 80)
        print(f"⚠️  DUPLICATE SUMMARY: {n_duplicates} duplicate files found in {n_groups} groups")
        print("=" * 80)
        print("\nDuplicate pairs:")
        for dup_file, orig_file, sim in duplicates_found:
            print(f"  • {dup_file} → {orig_file} ({sim:.1%} similar)")


if __name__ == '__main__':
    main()

