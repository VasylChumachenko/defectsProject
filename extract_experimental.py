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
    r'^\s*\d+\.?\s*Experiments?\s*$',
    r'^\s*\d+\.?\s*Experiments?\s+Section\s*$',
    r'^\s*\d+\.?\s*Materials?\s+and\s+Methods?\s*$',
    r'^\s*\d+\.?\s*Methods?\s+and\s+Materials?\s*$',
    r'^\s*\d+\.?\s*Methods\s*$',
    # Non-numbered but clear section headers
    r'^\s*Experimental\s+Section\s*$',
    r'^\s*Experimental\s+Procedures?\s*$',
    r'^\s*Experimental\s+Details?\s*$',
    r'^\s*Experimental\s*$',
    r'^\s*Experiments?\s*$',
    r'^\s*EXPERIMENTAL\s*$',
    r'^\s*EXPERIMENTAL\s+SECTION\s*$',
    r'^\s*Materials?\s+and\s+Methods?\s*$',
    r'^\s*MATERIALS\s+AND\s+METHODS\s*$',
    # Cell/Joule/Nature-style
    r'^\s*STAR\s*[\+\★]\s*METHODS\s*$',
    r'^\s*STAR\s*METHODS\s*$',
    r'^\s*Method\s+Details?\s*$',
    r'^\s*METHOD\s+DETAILS?\s*$',
]

EXPERIMENTAL_START_PATTERNS_RELAXED = [
    # Numbered subsections (2.1, 2.2, etc.)
    r'^\s*\d+\.\d+\.?\s*Synthesis\s+of\s+.{0,50}$',
    r'^\s*\d+\.\d+\.?\s*Preparation\s+of\s+.{0,50}$',
    r'^\s*\d+\.\d+\.?\s*Synthesis\s*$',
    r'^\s*\d+\.\d+\.?\s*Sample\s+Preparation\s*\.?$',
    r'^\s*\d+\.\d+\.?\s*Materials?\s*$',
    r'^\s*\d+\.\d+\.?\s*Chemicals?\s*$',
    # Standalone "Sample Preparation" / "Catalyst synthesis" (no number)
    r'^\s*Sample\s+Preparation\s*\.?\s*$',
    r'^\s*Catalyst\s+(?:synthesis|preparation)\s',
    # "Experiment section" (some papers use singular)
    r'^\s*\d+\.?\d*\.?\s*Experiment\s+[Ss]ection\s*$',
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
    # ESI-specific headers
    r'^\s*Experimental\s+procedure\s+for\s+',
    # "Detailed methods are provided" (Cell/STAR format)
    r'^\s*Detailed\s+methods\s+are\s+provided\s+',
]

# Combined for backward compatibility
EXPERIMENTAL_START_PATTERNS = EXPERIMENTAL_START_PATTERNS_STRICT + EXPERIMENTAL_START_PATTERNS_RELAXED

# Strong end patterns — always terminate, even with little content
EXPERIMENTAL_END_PATTERNS_STRONG = [
    # Numbered sections (most reliable)
    r'^\s*\d+\.?\s*Results?\s+and\s+Discussions?\s*$',
    r'^\s*\d+\.?\s*Results?\s*$',
    r'^\s*\d+\.?\s*Discussions?\s*$',
    r'^\s*\d+\.?\s*Conclusions?\s*$',
    r'^\s*\d+\.?\s*Summary\s*$',
    # Non-numbered headers
    r'^\s*Results?\s+and\s+Discussions?\s*$',
    r'^\s*RESULTS?\s+AND\s+DISCUSSIONS?\s*$',
    r'^\s*Results?\s+and\s+Characterizations?\s*$',
    r'^\s*Conclusions?\s*$',
    r'^\s*CONCLUSIONS?\s*$',
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
    # STAR METHODS sub-sections that mark end of synthesis details
    r'^\s*QUANTIFICATION\s+AND\s+STATISTICAL',
    r'^\s*DATA\s+AND\s+CODE\s+AVAILABILITY',
]

# Weak end patterns — only terminate after MIN_WORDS_BEFORE_WEAK_END words
# (these can appear as subsection headers within the experimental section)
MIN_WORDS_BEFORE_WEAK_END = 150

EXPERIMENTAL_END_PATTERNS_WEAK = [
    r'^\s*Results?\s*$',
    r'^\s*RESULTS\s*$',
    r'^\s*Discussions?\s*$',
    r'^\s*DISCUSSION\s*$',
    r'^\s*Characterizations?\s*$',
]

# Combined for backward compatibility
EXPERIMENTAL_END_PATTERNS = EXPERIMENTAL_END_PATTERNS_STRONG + EXPERIMENTAL_END_PATTERNS_WEAK

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

# Folders that are expected to have no experimental section
SKIP_FOLDERS = {'theoretical', 'reviews'}

# Keywords for paragraph-level fallback (at least N per paragraph to qualify)
_SYNTH_KEYWORDS = [
    r'was\s+(?:synthesized|prepared|calcined|heated|annealed)',
    r'were\s+(?:synthesized|prepared|calcined|heated|annealed)',
    r'(?:precursor|melamine|urea|dicyandiamide|DCDA)\s+(?:was|were|is)',
    r'(?:tube|muffle)\s+furnace',
    r'crucible',
    r'(?:heated|calcined|annealed)\s+(?:at|to)\s+\d+\s*[°℃]',
    r'\d+\s*[°℃]C?\s+for\s+\d+',
    r'(?:N2|Ar|air|NH3)\s+atmosphere',
    r'under\s+(?:N2|Ar|air|NH3|nitrogen|argon)',
    r'heating\s+rate\s+of\s+\d+',
    r'polycondensation',
    r'thermal\s+(?:polymerization|condensation)',
    r'hydrothermal|solvothermal|autoclave',
]
_SYNTH_RE = [re.compile(p, re.IGNORECASE) for p in _SYNTH_KEYWORDS]
_MIN_KEYWORD_HITS = 2  # paragraph must match at least this many patterns

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
    experimental_text: Optional[str]       # cleaned + trimmed (for LLM)
    experimental_text_raw: Optional[str]   # cleaned but NOT trimmed (for comparison)
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


# Patterns indicating a download-error / paywall page instead of a real article
_BROKEN_PDF_MARKERS = [
    'javascript is disabled',
    'please turn javascript on',
    'request id:',
    'access denied',
    'this content is not available',
    'cookie policy',
    'subscribe to read',
]


def is_broken_pdf(text: str, n_pages: int) -> bool:
    """Detect download error / paywall pages that are not real articles."""
    if n_pages <= 2 and len(text) < 3000:
        text_lower = text.lower()
        hits = sum(1 for m in _BROKEN_PDF_MARKERS if m in text_lower)
        if hits >= 2:
            return True
    return False


def _strip_leading_symbols(s: str) -> str:
    """Strip leading non-alphanumeric decorators (■, •, ►, ★, etc.) from header lines."""
    return re.sub(r'^[\W_]+', '', s).strip()


def find_section_start(lines: List[str], text: str, patterns: List[str]) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Try to find section start using given patterns."""
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue
        
        # Skip very long lines (likely not headers)
        if len(line_clean) > 100:
            continue
        
        # Try both the original line and a version with leading symbols stripped
        candidates = [line_clean]
        stripped = _strip_leading_symbols(line_clean)
        if stripped != line_clean:
            candidates.append(stripped)
        
        for candidate in candidates:
            for pattern in patterns:
                if re.match(pattern, candidate, re.IGNORECASE):
                    start_pos = text.find(line)
                    return start_pos, i, line_clean
    
    return None, None, None


def _extract_synthesis_paragraphs(text: str) -> Optional[str]:
    """Fallback: extract paragraphs that describe synthesis procedures.

    Used when no section header is found.  Returns concatenated paragraphs
    or None if nothing qualifies.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    hits = []
    for para in paragraphs:
        para_clean = para.strip()
        if len(para_clean) < 80:
            continue
        n_matches = sum(1 for rx in _SYNTH_RE if rx.search(para_clean))
        if n_matches >= _MIN_KEYWORD_HITS:
            hits.append(para_clean)
    if not hits:
        return None
    return '\n\n'.join(hits)


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
    words_so_far = 0
    chars_searched = 0
    
    for i, line in enumerate(lines[start_line_idx + 1:], start=start_line_idx + 1):
        line_clean = line.strip()
        chars_searched += len(line) + 1
        words_so_far += len(line_clean.split()) if line_clean else 0
        
        # Hard limit on section length
        if chars_searched > MAX_SECTION_LENGTH:
            end_pos = start_pos + MAX_SECTION_LENGTH
            break
        
        if not line_clean:
            continue
        
        # Skip very long lines (not headers)
        if len(line_clean) > 100:
            continue
        
        # Also try with leading symbols stripped for end patterns
        candidates = [line_clean]
        stripped = _strip_leading_symbols(line_clean)
        if stripped != line_clean:
            candidates.append(stripped)
        
        matched = False
        for candidate in candidates:
            # Strong patterns: always terminate
            for pattern in EXPERIMENTAL_END_PATTERNS_STRONG:
                if re.match(pattern, candidate, re.IGNORECASE):
                    end_pos = text.find(line, start_pos + len(section_header))
                    matched = True
                    break
            if matched:
                break
            # Weak patterns: only terminate if we have enough content
            if words_so_far >= MIN_WORDS_BEFORE_WEAK_END:
                for pattern in EXPERIMENTAL_END_PATTERNS_WEAK:
                    if re.match(pattern, candidate, re.IGNORECASE):
                        end_pos = text.find(line, start_pos + len(section_header))
                        matched = True
                        break
            if matched:
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


_LIGATURE_MAP = str.maketrans({
    '\ufb00': 'ff', '\ufb01': 'fi', '\ufb02': 'fl',
    '\ufb03': 'ffi', '\ufb04': 'ffl',
})

# Subsection headers that mark the END of synthesis-relevant content.
# Matched against each line; may have trailing text (e.g. "2.2 Characterization. XRD patterns...")
_TRIM_HEADERS = re.compile(
    r'^\s*\d*\.?\d*\.?\s*'
    r'(?:'
    r'(?:Physicochemical\s+|Sample\s+|Structural\s+|Material\s+)?Characterizations?\b'
    r'|Instruments?\b'
    r'|Measurements?\b'
    r'|Photocatalytic\b'
    r'|Photocatalysis\b'
    r'|Photoelectrochemical\b'
    r'|Electrochemical\b'
    r'|Hydrogen\s+(?:evolution|production)'
    r'|Photo-?degradation\b'
    r'|(?:Density.functional|DFT|Computational|Theoretical)\s+(?:theory|calculation|detail|method|studie)'
    r'|Photocurrent\b'
    r'|Impedance\b'
    r'|Catalytic\s+(?:activity|performance|test)'
    r'|(?:Quantum\s+)?(?:Efficiency|yield)\s+(?:measurement|test)'
    r'|(?:Water\s+)?(?:Contact\s+angle|Wettability)\b'
    r'|Nitrogen\s+(?:fixation|reduction)\s+(?:measurement|test|experiment)'
    r')',
    re.IGNORECASE | re.MULTILINE,
)


def clean_extracted_text(text: str) -> str:
    """Fix PDF artifacts; produce human-readable text."""
    if not text:
        return ""

    # ── Ligatures ──
    text = text.translate(_LIGATURE_MAP)

    # ── Soft hyphens ──
    text = text.replace('\xad', '')

    # ── Degree symbol variants ──
    text = text.replace('℃', '°C')

    # ── Join mid-word line breaks (e.g. "synthe-\nsized" → "synthesized") ──
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

    # ── (cid:XX) PDF encoding artifacts ──
    text = re.sub(r'\(cid:\d+\)', '', text)

    # ── Inline citation brackets [1], [2,3], [12-15] ──
    text = re.sub(r'\s*\[\d+(?:[,\-–]\s*\d+)*\]', '', text)

    # ── Inline figure/scheme refs "(Fig. 1a)", "(Scheme 1)" ──
    text = re.sub(r'\s*\((?:Fig\.|Figure|Scheme|Table)\s*\d+[a-z]?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\((?:Fig\.|Figure|Scheme|Table)\s*S?\d+[a-z]?\)', '', text, flags=re.IGNORECASE)

    # ── URLs and DOIs ──
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'doi:\s*\S+', '', text, flags=re.IGNORECASE)

    # ── Page numbers ──
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'\n\s*Page\s+\d+\s*\n', '\n', text, flags=re.IGNORECASE)

    # ── Whitespace normalisation ──
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def trim_to_synthesis(text: str) -> str:
    """Remove characterisation / photocatalytic / DFT subsections.

    Keeps everything from the start up to the first non-synthesis
    subsection header.  The match is done line-by-line to avoid
    false positives on mid-sentence occurrences like
    "characterization revealed that...".
    """
    if not text:
        return ""

    lines = text.split('\n')
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > 150:
            continue
        # Skip headers that combine synthesis + characterization
        # e.g. "2.1. Preparation and characterization of SCNNSs"
        if re.search(r'(?:preparation|synthesis)\s+and\s+characterization',
                      stripped, re.IGNORECASE):
            continue
        if _TRIM_HEADERS.match(stripped):
            trimmed = '\n'.join(lines[:i]).rstrip()
            if len(trimmed.split()) >= 30:
                return trimmed
    return text


def _try_si_extraction(pdf_path: str, main_text: str, main_pages: List[str]) -> Optional[str]:
    """Try to extract experimental section from a supplementary PDF in the same folder."""
    pdf_dir = Path(pdf_path).parent
    stem = Path(pdf_path).stem
    si_candidates = list(pdf_dir.glob(f'{stem}_sup*')) + list(pdf_dir.glob(f'{stem}_si*'))
    if not si_candidates:
        # Try broader patterns
        si_candidates = [p for p in pdf_dir.glob('*_sup*.pdf')] + \
                        [p for p in pdf_dir.glob('*_si*.pdf')] + \
                        [p for p in pdf_dir.glob('*supporting*.pdf')]
    for si_path in si_candidates:
        si_text, si_pages, _ = extract_text(str(si_path))
        if not si_text.strip():
            continue
        start, end, header, conf, notes = find_section_boundaries(si_text, si_pages)
        if start is not None:
            return si_text[start:end]
        # If no header found in SI, try paragraph fallback on SI text
        fallback = _extract_synthesis_paragraphs(si_text)
        if fallback:
            return fallback
    return None


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
            experimental_text_raw=None,
            section_header=None,
            start_page=None,
            end_page=None,
            char_count=0,
            word_count=0,
            extraction_method=method,
            confidence="none",
            notes="Failed to extract text from PDF"
        )
    
    if is_broken_pdf(full_text, len(pages)):
        return ExtractionResult(
            filename=filename,
            folder=folder,
            title=title,
            experimental_text=None,
            experimental_text_raw=None,
            section_header=None,
            start_page=None,
            end_page=None,
            char_count=0,
            word_count=0,
            extraction_method=method,
            confidence="none",
            notes="Broken PDF (download error / paywall page)"
        )
    
    # Find experimental section
    start_pos, end_pos, section_header, confidence, section_notes = find_section_boundaries(full_text, pages)
    
    if start_pos is None:
        # Paragraph-level fallback: extract synthesis-describing paragraphs
        fallback_text = _extract_synthesis_paragraphs(full_text)
        if fallback_text:
            fallback_text = clean_extracted_text(fallback_text)
            wc = len(fallback_text.split())
            return ExtractionResult(
                filename=filename,
                folder=folder,
                title=title,
                experimental_text=fallback_text,
                experimental_text_raw=fallback_text,
                section_header="(paragraph fallback)",
                start_page=None,
                end_page=None,
                char_count=len(fallback_text),
                word_count=wc,
                extraction_method=method + "+paragraph_fallback",
                confidence="low" if wc < 100 else "medium",
                notes="No section header; synthesis paragraphs extracted by keyword"
            )
        return ExtractionResult(
            filename=filename,
            folder=folder,
            title=title,
            experimental_text=None,
            experimental_text_raw=None,
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
    
    # SI fallback: if main text is very short and redirects to SI,
    # try to find a supplementary PDF in the same folder
    word_count = len(experimental_text.split())
    if word_count < 80 and check_si_redirect(experimental_text):
        si_text = _try_si_extraction(pdf_path, full_text, pages)
        if si_text:
            si_text = clean_extracted_text(si_text)
            si_wc = len(si_text.split())
            if si_wc > word_count:
                experimental_text = experimental_text + "\n\n--- From Supporting Information ---\n\n" + si_text
                word_count = len(experimental_text.split())
                confidence = "medium"
                section_notes = "Main text + SI extraction"
    
    # Store raw (cleaned but not trimmed) for comparison
    raw_text = experimental_text
    
    # Trim non-synthesis subsections (characterisation, photocatalytic, DFT)
    experimental_text = trim_to_synthesis(experimental_text)
    
    # Find page numbers
    start_page, end_page = find_page_numbers(full_text, start_pos, end_pos or len(full_text), pages)
    
    # Stats reflect the trimmed version (what the LLM sees)
    char_count = len(experimental_text)
    word_count = len(experimental_text.split())
    
    return ExtractionResult(
        filename=filename,
        folder=folder,
        title=title,
        experimental_text=experimental_text,
        experimental_text_raw=raw_text,
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
        
        # Skip folders that are expected to have no experimental section
        folder_name = pdf_path.parent.name.lower()
        if folder_name in SKIP_FOLDERS:
            print(f"SKIPPED ({folder_name} folder)")
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
    
    total_raw_words = 0
    total_trimmed_words = 0
    for result in results:
        raw = result.experimental_text_raw or ""
        trimmed = result.experimental_text or ""
        raw_wc = len(raw.split()) if raw else 0
        trimmed_wc = len(trimmed.split()) if trimmed else 0
        total_raw_words += raw_wc
        total_trimmed_words += trimmed_wc
        article_entry = {
            'id': f"{result.folder}/{result.filename}",
            'filename': result.filename,
            'folder': result.folder,
            'title': result.title,
            'experimental_text': trimmed,
            'experimental_text_raw': raw,
            'section_header': result.section_header if result.section_header else "",
            'word_count': trimmed_wc,
            'word_count_raw': raw_wc,
            'confidence': result.confidence,
            'notes': result.notes,
            'manual_entry': result.confidence in ['none', 'low']
        }
        json_data['articles'].append(article_entry)
    
    saved_words = total_raw_words - total_trimmed_words
    json_data['metadata']['total_words_raw'] = total_raw_words
    json_data['metadata']['total_words_trimmed'] = total_trimmed_words
    json_data['metadata']['words_saved_by_trimming'] = saved_words
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nSynthesis trimming: {total_raw_words} → {total_trimmed_words} words "
          f"(saved {saved_words} words, {100*saved_words/max(total_raw_words,1):.0f}%)")
    print(f"JSON saved to: {output_json}")
    
    # CSV summary
    df = pd.DataFrame([asdict(r) for r in results])
    df = df.drop(columns=['experimental_text', 'experimental_text_raw'])
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

