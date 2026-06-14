"""
Text processing utilities for medical free text analysis.

Functions for cleaning, normalizing, and analyzing German/English medical texts.
"""

import re
import unicodedata
from typing import Optional


# =============================================================================
# ENCODING FIXES
# =============================================================================

# Common UTF-8 mojibake patterns (UTF-8 bytes interpreted as Latin-1)
ENCODING_FIXES = {
    # German umlauts (most common in GSR data)
    "\xc3\xa4": "\xe4",      # Ã¤ -> ä
    "\xc3\xb6": "\xf6",      # Ã¶ -> ö
    "\xc3\xbc": "\xfc",      # Ã¼ -> ü
    "\xc3\x9f": "\xdf",      # ÃŸ -> ß
    "\xc3\x84": "\xc4",      # Ã„ -> Ä
    "\xc3\x96": "\xd6",      # Ã– -> Ö
    "\xc3\x9c": "\xdc",      # Ãœ -> Ü
    # Simpler string-based fixes (these work reliably)
    "Ã¤": "ä",
    "Ã¶": "ö",
    "Ã¼": "ü",
    "ÃŸ": "ß",
    "Ã„": "Ä",
    "Ã–": "Ö",
    "Ãœ": "Ü",
    # Artifact character
    "Â ": " ",  # Non-breaking space artifact
}


def fix_encoding(text: str) -> str:
    """
    Fix common UTF-8 encoding issues (mojibake).

    Handles cases where UTF-8 was interpreted as Latin-1.

    Args:
        text: Text with potential encoding issues

    Returns:
        Text with fixed encoding
    """
    if not text or not isinstance(text, str):
        return text

    result = text
    for wrong, correct in ENCODING_FIXES.items():
        result = result.replace(wrong, correct)

    return result


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode to NFC form.

    This ensures consistent representation of characters that can be
    composed in multiple ways (e.g., ä as single char vs a + combining umlaut).

    Args:
        text: Input text

    Returns:
        Unicode-normalized text
    """
    if not text:
        return text
    return unicodedata.normalize("NFC", text)


# =============================================================================
# WHITESPACE HANDLING
# =============================================================================

def clean_whitespace(text: str) -> str:
    """
    Clean and normalize whitespace.

    - Strips leading/trailing whitespace
    - Collapses multiple spaces to single space
    - Normalizes different types of spaces to regular space
    - Preserves newlines but collapses multiple newlines

    Args:
        text: Input text

    Returns:
        Whitespace-cleaned text
    """
    if not text:
        return text

    # Replace various space characters with regular space
    text = re.sub(r'[\u00A0\u2000-\u200B\u202F\u205F\u3000]', ' ', text)

    # Collapse multiple spaces (but preserve newlines)
    text = re.sub(r'[^\S\n]+', ' ', text)

    # Collapse multiple newlines to double newline
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# =============================================================================
# TEXT ANALYSIS
# =============================================================================

def get_text_stats(text: str) -> dict:
    """
    Get basic statistics about a text.

    Args:
        text: Input text

    Returns:
        Dict with char_count, word_count, sentence_count, has_numbers
    """
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "has_numbers": False,
        }

    return {
        "char_count": len(text),
        "word_count": len(text.split()),
        "sentence_count": len(re.findall(r'[.!?]+', text)),
        "has_numbers": bool(re.search(r'\d', text)),
    }


def is_likely_placeholder(text: str, max_length: int = 3) -> bool:
    """
    Check if text is likely a placeholder (very short, punctuation only, etc.).

    Args:
        text: Input text
        max_length: Maximum length to consider as placeholder

    Returns:
        True if text appears to be a placeholder
    """
    if not text:
        return True

    text = text.strip()

    # Very short text
    if len(text) <= max_length:
        return True

    # Only punctuation or special characters
    if re.match(r'^[\-–—_.,:;/\\()\[\]{}]+$', text):
        return True

    return False


# =============================================================================
# FULL CLEANING PIPELINE
# =============================================================================

def clean_text(text: str) -> str:
    """
    Apply full cleaning pipeline to text.

    Steps:
    1. Fix encoding issues
    2. Normalize Unicode
    3. Clean whitespace

    Does NOT:
    - Change case (preserves medical acronyms, drug names)
    - Remove special characters (may be medically relevant)

    Args:
        text: Raw input text

    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""

    text = fix_encoding(text)
    text = normalize_unicode(text)
    text = clean_whitespace(text)

    return text


def clean_for_display(text: str, max_length: int = 100) -> str:
    """
    Clean and truncate text for display purposes.

    Args:
        text: Input text
        max_length: Maximum display length

    Returns:
        Cleaned, possibly truncated text
    """
    cleaned = clean_text(text)
    if len(cleaned) > max_length:
        return cleaned[:max_length-3] + "..."
    return cleaned
