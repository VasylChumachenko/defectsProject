#!/usr/bin/env python3
"""
Experimental Section Extractor for Scientific Articles

Extracts "Experimental", "Materials and Methods", "Synthesis" sections from PDFs.
Outputs in JSON format ready for LLM processing.
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
import pandas as pd

# PDF libraries
try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


# ============================================================================
# SECTION DETECTION PATTERNS
# ============================================================================

# Patterns for START of experimental section (case-insensitive)
# IMPORTANT: Order matters! More specific patterns first.
# Two-pass approach: first try strict patterns, then relaxed ones
EXPERIMENTAL_START_PATTERNS_STRICT = [
    # Numbered main sections (most reliable)
    r'^\s*\d+\.?\s*Experimental\s+Section\s*$',
    r'^\s*\d+\.?\s*Experimental\s+Procedures?\s*$',
    r'^\s*\d+\.?\s*Experimental\s+Details?\s*$',
    r'^\s*\d+\.?\s*Experimental\s+Methods?\s*$',
    r'^\s*\d+\.?\s*Experimental\s*$',
    r'^\s*\d+\.?\s*Materials?\s+and\s+Methods?\s*$',
    r'^\s*\d+\.?\s*Methods?\s+and\s+Materials?\s*$',
    r'^\s*\d+\.?\s*Methods\s*$',
    # Non-numbered but clear section headers
    r'^\s*Experimental\s+Section\s*$',
    r'^\s*Experimental\s+Procedures?\s*$',
    r'^\s*Experimental\s+Details?\s*$',
    r'^\s*Experimental\s*$',
    r'^\s*EXPERIMENTAL\s*$',
    r'^\s*Materials?\s+and\s+Methods?\s*$',
    r'^\s*MATERIALS\s+AND\s+METHODS\s*$',
]

EXPERIMENTAL_START_PATTERNS_RELAXED = [
    # Numbered subsections (2.1, 2.2, etc.)
    r'^\s*\d+\.\d+\.?\s*Synthesis\s+of\s+.{0,50}$',
    r'^\s*\d+\.\d+\.?\s*Preparation\s+of\s+.{0,50}$',
    r'^\s*\d+\.\d+\.?\s*Synthesis\s*$',
    r'^\s*\d+\.\d+\.?\s*Sample\s+Preparation\s*$',
    r'^\s*\d+\.\d+\.?\s*Materials?\s*$',
    r'^\s*\d+\.\d+\.?\s*Chemicals?\s*$',
    # "2. Experimental" merged with next word (PDF artifact)
    r'^\s*\d+\.?\s*Experimental\s+\w',
    # Experimental without space after number
    r'^\s*\d+\.Experimental',
    # Supporting Information patterns
    r'^\s*S?\d+\.?\s*Experimental\s+Details?\s*$',
    r'^\s*S?\d+\.?\s*Experimental\s*$',
    # Synthesis subsection without number
    r'^\s*Synthesis\s+of\s+.{0,40}g-?C\s*3\s*N\s*4',
    r'^\s*Preparation\s+of\s+.{0,40}g-?C\s*3\s*N\s*4',
    # Words merged together (no spaces) - PDF artifact
    r'^\s*\d+\.\d+\.?\s*Synthesisof',
    r'^\s*\d+\.\d+\.?\s*Preparationof',
    r'^\s*Preparationof.{0,30}g-?C\s*3\s*N\s*4',
    # Characterizations section (often contains synthesis details)
    r'^\s*\d+\.?\s*Characterizations?\s*$',
]

# Combined for backward compatibility
EXPERIMENTAL_START_PATTERNS = EXPERIMENTAL_START_PATTERNS_STRICT + EXPERIMENTAL_START_PATTERNS_RELAXED

# Patterns for END of experimental section
EXPERIMENTAL_END_PATTERNS = [
    # Numbered sections (most reliable)
    r'^\s*\d+\.?\s*Results?\s+and\s+Discussions?\s*$',
    r'^\s*\d+\.?\s*Results?\s*$',
    r'^\s*\d+\.?\s*Discussions?\s*$',
    r'^\s*\d+\.?\s*Conclusions?\s*$',
    r'^\s*\d+\.?\s*Summary\s*$',
    # Non-numbered headers
    r'^\s*Results?\s+and\s+Discussions?\s*$',
    r'^\s*RESULTS?\s+AND\s+DISCUSSIONS?\s*$',
    r'^\s*Results?\s*$',
    r'^\s*RESULTS\s*$',
    r'^\s*Discussions?\s*$',
    r'^\s*DISCUSSION\s*$',
    r'^\s*Conclusions?\s*$',
    r'^\s*CONCLUSIONS?\s*$',
    r'^\s*Characterizations?\s*$',
    r'^\s*Results?\s+and\s+Characterizations?\s*$',
    # Paper end sections
    r'^\s*\d*\.?\s*Acknowledg[e]?ments?\s*$',
    r'^\s*ACKNOWLEDG',
    r'^\s*\d*\.?\s*References?\s*$',
    r'^\s*REFERENCES\s*$',
    r'^\s*Bibliography\s*$',
    r'^\s*Author\s+Contributions?\s*$',
    r'^\s*Conflicts?\s+of\s+Interest\s*$',
    r'^\s*Declaration\s+of\s+',
    r'^\s*Data\s+Availability\s*$',
    r'^\s*CRediT\s+',
    # Supplementary markers
    r'^\s*\d*\.?\s*Supporting\s+Information\s*$',
    r'^\s*\d*\.?\s*Supplementary\s+',
    r'^\s*ASSOCIATED\s+CONTENT',
    # Figure/Table captions (often indicate Results section)
    r'^Figure\s+\d+\.',
    r'^Fig\.\s*\d+\.',
    r'^Table\s+\d+\.',
]

# Patterns indicating content is in Supporting Information
SI_REDIRECT_PATTERNS = [
    r'detailed?\s+in\s+(?:the\s+)?support',
    r'provided\s+in\s+(?:the\s+)?support',
    r'see\s+(?:the\s+)?support',
    r'(?:in|see)\s+(?:the\s+)?(?:electronic\s+)?supplementary',
    r'described\s+in\s+(?:the\s+)?SI',
    r'(?:in|see)\s+ESI',
]

# Maximum reasonable section length (in characters)
MAX_SECTION_LENGTH = 20000  # ~3500 words

# Subsection patterns within Experimental (for context)
EXPERIMENTAL_SUBSECTIONS = [
    r'^\s*\d*\.?\d*\.?\s*Materials?\s*$',
    r'^\s*\d*\.?\d*\.?\s*Chemicals?\s*$',
    r'^\s*\d*\.?\d*\.?\s*Reagents?\s*$',
    r'^\s*\d*\.?\d*\.?\s*Synthesis\s+of\s+',
    r'^\s*\d*\.?\d*\.?\s*Preparation\s+of\s+',
    r'^\s*\d*\.?\d*\.?\s*Characterization\s*$',
    r'^\s*\d*\.?\d*\.?\s*Instruments?\s*$',
    r'^\s*\d*\.?\d*\.?\s*Measurements?\s*$',
    r'^\s*\d*\.?\d*\.?\s*Photocatalytic\s+',
    r'^\s*\d*\.?\d*\.?\s*Electrochemical\s+',
]


@dataclass
class ExtractionResult:
    """Result of experimental section extraction."""
    filename: str
    folder: str
    title: Optional[str]
    experimental_text: Optional[str]
    section_header: Optional[str]
    start_page: Optional[int]
    end_page: Optional[int]
    char_count: int
    word_count: int
    extraction_method: str
    confidence: str  # high, medium, low, none
    notes: str


def extract_text_pypdf(pdf_path: str) -> Tuple[str, List[str]]:
    """Extract text from PDF using pypdf."""
    try:
        reader = pypdf.PdfReader(pdf_path)
        pages = []
        full_text = ""
        
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
            full_text += page_text + "\n\n"
        
        return full_text, pages
    except Exception as e:
        return "", []


def extract_text_pdfplumber(pdf_path: str) -> Tuple[str, List[str]]:
    """Extract text from PDF using pdfplumber (better for complex layouts)."""
    try:
        pages = []
        full_text = ""
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                pages.append(page_text)
                full_text += page_text + "\n\n"
        
        return full_text, pages
    except Exception as e:
        return "", []


def extract_text_pymupdf(pdf_path: str) -> Tuple[str, List[str]]:
    """
    Extract text from PDF using PyMuPDF with column-aware extraction.
    Handles two-column layouts by sorting blocks by column then row.
    """
    try:
        doc = fitz.open(pdf_path)
        pages = []
        full_text = ""
        
        for page in doc:
            # Get text blocks with coordinates
            blocks = page.get_text("blocks")
            
            if not blocks:
                pages.append("")
                continue
            
            # Determine if this is a two-column layout
            # by checking x-coordinates of blocks
            x_coords = [b[0] for b in blocks if b[4].strip()]
            
            if len(x_coords) > 5:
                # Find the middle of the page
                page_width = page.rect.width
                mid_x = page_width / 2
                
                # Check if there are blocks on both sides of the middle
                left_blocks = [b for b in blocks if b[0] < mid_x - 20]
                right_blocks = [b for b in blocks if b[0] >= mid_x - 20]
                
                is_two_column = len(left_blocks) > 3 and len(right_blocks) > 3
            else:
                is_two_column = False
            
            if is_two_column:
                # Sort: first all left column blocks by y, then all right column blocks by y
                left_blocks = sorted(left_blocks, key=lambda b: b[1])
                right_blocks = sorted(right_blocks, key=lambda b: b[1])
                sorted_blocks = left_blocks + right_blocks
            else:
                # Single column: sort by y-coordinate
                sorted_blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
            
            # Extract text in order
            page_text = ""
            for block in sorted_blocks:
                text = block[4].strip()
                if text:
                    page_text += text + "\n"
            
            pages.append(page_text)
            full_text += page_text + "\n\n"
        
        doc.close()
        return full_text, pages
    except Exception as e:
        return "", []


def extract_text(pdf_path: str) -> Tuple[str, List[str], str]:
    """Extract text using best available method."""
    # Try PyMuPDF first (best for two-column layouts)
    if HAS_PYMUPDF:
        text, pages = extract_text_pymupdf(pdf_path)
        if text.strip():
            return text, pages, "pymupdf"
    
    # Try pdfplumber second
    if HAS_PDFPLUMBER:
        text, pages = extract_text_pdfplumber(pdf_path)
        if text.strip():
            return text, pages, "pdfplumber"
    
    # Fallback to pypdf
    if HAS_PYPDF:
        text, pages = extract_text_pypdf(pdf_path)
        if text.strip():
            return text, pages, "pypdf"
    
    return "", [], "none"


def check_si_redirect(text: str) -> bool:
    """Check if text indicates content is in Supporting Information."""
    text_lower = text.lower()
    for pattern in SI_REDIRECT_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def find_section_start(lines: List[str], text: str, patterns: List[str]) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Try to find section start using given patterns."""
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue
        
        # Skip very long lines (likely not headers)
        if len(line_clean) > 100:
            continue
        
        for pattern in patterns:
            if re.match(pattern, line_clean, re.IGNORECASE):
                start_pos = text.find(line)
                return start_pos, i, line_clean
    
    return None, None, None


def find_section_boundaries(text: str, pages: List[str]) -> Tuple[Optional[int], Optional[int], Optional[str], str, str]:
    """
    Find start and end positions of experimental section.
    Uses two-pass approach: strict patterns first, then relaxed.
    Returns: (start_pos, end_pos, section_header, confidence, notes)
    """
    lines = text.split('\n')
    
    # First pass: try strict patterns
    start_pos, start_line_idx, section_header = find_section_start(
        lines, text, EXPERIMENTAL_START_PATTERNS_STRICT
    )
    used_strict = start_pos is not None
    
    # Second pass: try relaxed patterns if strict didn't work
    if start_pos is None:
        start_pos, start_line_idx, section_header = find_section_start(
            lines, text, EXPERIMENTAL_START_PATTERNS_RELAXED
        )
    
    if start_pos is None:
        return None, None, None, "none", "Experimental section not found"
    
    # Find end of experimental section
    end_pos = None
    chars_searched = 0
    
    for i, line in enumerate(lines[start_line_idx + 1:], start=start_line_idx + 1):
        line_clean = line.strip()
        chars_searched += len(line) + 1
        
        # Hard limit on section length
        if chars_searched > MAX_SECTION_LENGTH:
            end_pos = start_pos + MAX_SECTION_LENGTH
            break
        
        if not line_clean:
            continue
        
        # Skip very long lines (not headers)
        if len(line_clean) > 100:
            continue
        
        for pattern in EXPERIMENTAL_END_PATTERNS:
            if re.match(pattern, line_clean, re.IGNORECASE):
                end_pos = text.find(line, start_pos + len(section_header))
                break
        
        if end_pos is not None:
            break
    
    # Apply max length limit if no end found
    if end_pos is None:
        end_pos = min(start_pos + MAX_SECTION_LENGTH, len(text))
        end_estimated = True
    else:
        end_estimated = False
    
    # Extract section and check for SI redirect
    section_text = text[start_pos:end_pos]
    is_si_redirect = check_si_redirect(section_text[:500])  # Check first 500 chars
    
    # Determine confidence and notes
    section_length = end_pos - start_pos
    word_count = len(section_text.split())
    notes = []
    
    if is_si_redirect and word_count < 100:
        confidence = "low"
        notes.append("Content likely in Supporting Information")
    elif section_length > 500 and section_length <= MAX_SECTION_LENGTH and not end_estimated:
        if word_count >= 100:
            # High confidence only if we used strict patterns
            confidence = "high" if used_strict else "medium"
            if not used_strict:
                notes.append("Matched subsection header")
        else:
            confidence = "medium"
            notes.append("Short section")
    elif end_estimated:
        confidence = "medium"
        notes.append("End boundary estimated")
    elif section_length > 200:
        confidence = "medium"
    else:
        confidence = "low"
        notes.append("Very short section")
    
    if word_count < 50:
        notes.append("May be incomplete")
    
    if not used_strict and confidence != "low":
        if "Matched subsection header" not in notes:
            notes.append("Matched subsection header")
    
    return start_pos, end_pos, section_header, confidence, "; ".join(notes) if notes else "OK"


def find_page_numbers(text: str, start_pos: int, end_pos: int, pages: List[str]) -> Tuple[Optional[int], Optional[int]]:
    """Find which pages contain the experimental section."""
    if not pages:
        return None, None
    
    cumulative_pos = 0
    start_page = None
    end_page = None
    
    for i, page_text in enumerate(pages):
        page_end = cumulative_pos + len(page_text) + 2  # +2 for \n\n
        
        if start_page is None and cumulative_pos <= start_pos < page_end:
            start_page = i + 1
        
        if end_pos is not None and cumulative_pos <= end_pos < page_end:
            end_page = i + 1
            break
        
        cumulative_pos = page_end
    
    if end_page is None and start_page is not None:
        end_page = len(pages)
    
    return start_page, end_page


def clean_extracted_text(text: str) -> str:
    """Clean extracted text for LLM processing."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove page numbers (common patterns)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'\n\s*Page\s+\d+\s*\n', '\n', text, flags=re.IGNORECASE)
    
    # Remove common artifacts
    text = re.sub(r'\(cid:\d+\)', '', text)  # PDF encoding artifacts
    
    # Remove URLs and DOIs (not needed for synthesis)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'doi:\s*\S+', '', text, flags=re.IGNORECASE)
    
    return text.strip()


def extract_experimental_section(pdf_path: str, metadata: Dict) -> ExtractionResult:
    """Extract experimental section from a single PDF."""
    filename = os.path.basename(pdf_path)
    folder = os.path.basename(os.path.dirname(pdf_path))
    title = metadata.get('title')
    
    # Extract text
    full_text, pages, method = extract_text(pdf_path)
    
    if not full_text.strip():
        return ExtractionResult(
            filename=filename,
            folder=folder,
            title=title,
            experimental_text=None,
            section_header=None,
            start_page=None,
            end_page=None,
            char_count=0,
            word_count=0,
            extraction_method=method,
            confidence="none",
            notes="Failed to extract text from PDF"
        )
    
    # Find experimental section
    start_pos, end_pos, section_header, confidence, section_notes = find_section_boundaries(full_text, pages)
    
    if start_pos is None:
        return ExtractionResult(
            filename=filename,
            folder=folder,
            title=title,
            experimental_text=None,
            section_header=None,
            start_page=None,
            end_page=None,
            char_count=0,
            word_count=0,
            extraction_method=method,
            confidence="none",
            notes=section_notes
        )
    
    # Extract the section text
    experimental_text = full_text[start_pos:end_pos]
    experimental_text = clean_extracted_text(experimental_text)
    
    # Find page numbers
    start_page, end_page = find_page_numbers(full_text, start_pos, end_pos or len(full_text), pages)
    
    # Calculate stats
    char_count = len(experimental_text)
    word_count = len(experimental_text.split())
    
    return ExtractionResult(
        filename=filename,
        folder=folder,
        title=title,
        experimental_text=experimental_text,
        section_header=section_header,
        start_page=start_page,
        end_page=end_page,
        char_count=char_count,
        word_count=word_count,
        extraction_method=method,
        confidence=confidence,
        notes=section_notes
    )


def load_metadata(metadata_path: str) -> Dict[str, Dict]:
    """Load metadata from CSV file."""
    if not os.path.exists(metadata_path):
        return {}
    
    df = pd.read_csv(metadata_path)
    metadata = {}
    
    for _, row in df.iterrows():
        key = f"{row['folder']}/{row['filename']}"
        metadata[key] = {
            'title': row.get('title'),
            'author': row.get('author'),
            'is_duplicate': row.get('is_duplicate', False),
            'duplicate_of': row.get('duplicate_of', '')
        }
    
    return metadata


def find_pdf_files(base_dir: str) -> List[Path]:
    """Recursively find all PDF files."""
    base_path = Path(base_dir)
    # Exclude supplementary/supporting info files
    pdf_files = []
    for pdf in base_path.rglob('*.pdf'):
        name_lower = pdf.name.lower()
        # Skip supplementary files
        if '_sup' in name_lower or '_si' in name_lower or 'supporting' in name_lower:
            continue
        pdf_files.append(pdf)
    return sorted(pdf_files)


def main():
    """Main function."""
    script_dir = Path(__file__).parent
    sources_dir = script_dir / 'sources'
    metadata_path = script_dir / 'sources_metadata.csv'
    output_json = script_dir / 'experimental_sections.json'
    output_csv = script_dir / 'experimental_sections_summary.csv'
    
    print("=" * 80)
    print("EXPERIMENTAL SECTION EXTRACTOR")
    print("=" * 80)
    
    # Check dependencies
    if not HAS_PYPDF and not HAS_PDFPLUMBER:
        print("Error: No PDF library found!")
        print("Install: pip install pypdf pdfplumber")
        sys.exit(1)
    
    print(f"PDF libraries: PyMuPDF={HAS_PYMUPDF}, pdfplumber={HAS_PDFPLUMBER}, pypdf={HAS_PYPDF}")
    
    # Load metadata
    metadata = load_metadata(metadata_path)
    print(f"Loaded metadata for {len(metadata)} files")
    
    # Find PDF files
    pdf_files = find_pdf_files(sources_dir)
    print(f"Found {len(pdf_files)} PDF files (excluding supplements)")
    print()
    
    # Process each PDF
    results = []
    stats = {'high': 0, 'medium': 0, 'low': 0, 'none': 0}
    
    for i, pdf_path in enumerate(pdf_files):
        relative_path = pdf_path.relative_to(sources_dir)
        key = str(relative_path)
        
        print(f"[{i+1}/{len(pdf_files)}] {relative_path}...", end=" ", flush=True)
        
        # Get metadata
        file_metadata = metadata.get(key, {})
        
        # Skip duplicates
        if file_metadata.get('is_duplicate', False):
            print("SKIPPED (duplicate)")
            continue
        
        # Extract experimental section
        result = extract_experimental_section(str(pdf_path), file_metadata)
        results.append(result)
        stats[result.confidence] += 1
        
        # Print status
        if result.confidence == "high":
            print(f"✓ {result.word_count} words")
        elif result.confidence == "medium":
            print(f"○ {result.word_count} words ({result.notes})")
        elif result.confidence == "low":
            print(f"△ {result.word_count} words ({result.notes})")
        else:
            print(f"✗ {result.notes}")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total processed:  {len(results)}")
    print(f"High confidence:  {stats['high']}")
    print(f"Medium confidence: {stats['medium']}")
    print(f"Low confidence:   {stats['low']}")
    print(f"Not found:        {stats['none']}")
    
    # Save results
    # JSON format for LLM processing
    json_data = {
        'metadata': {
            'total_files': len(results),
            'extraction_stats': stats,
            'ready_for_llm': stats['high'] + stats['medium'],
            'manual_entry_needed': stats['none'] + stats['low']
        },
        'articles': []
    }
    
    for result in results:
        article_entry = {
            'id': f"{result.folder}/{result.filename}",
            'filename': result.filename,
            'folder': result.folder,
            'title': result.title,
            'experimental_text': result.experimental_text if result.experimental_text else "",
            'section_header': result.section_header if result.section_header else "",
            'word_count': result.word_count,
            'confidence': result.confidence,
            'notes': result.notes,
            'manual_entry': result.confidence in ['none', 'low']
        }
        json_data['articles'].append(article_entry)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nJSON saved to: {output_json}")
    
    # CSV summary
    df = pd.DataFrame([asdict(r) for r in results])
    df = df.drop(columns=['experimental_text'])  # Don't include full text in CSV
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"CSV summary saved to: {output_csv}")
    
    # Show sample
    print()
    print("=" * 80)
    print("SAMPLE EXTRACTION (first high-confidence result)")
    print("=" * 80)
    
    for result in results:
        if result.confidence == "high" and result.experimental_text:
            print(f"\nFile: {result.folder}/{result.filename}")
            print(f"Title: {result.title}")
            print(f"Section: {result.section_header}")
            print(f"Pages: {result.start_page}-{result.end_page}")
            print("-" * 40)
            # Show first 1000 chars
            preview = result.experimental_text[:1000]
            if len(result.experimental_text) > 1000:
                preview += "\n... [truncated]"
            print(preview)
            break


if __name__ == '__main__':
    main()

