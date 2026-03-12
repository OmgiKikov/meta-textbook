"""Utility functions for text processing, chunking, etc."""

import re
from typing import List
import logging

# Configure basic logging (if not already configured elsewhere, e.g., in app.py)
# If it might be configured elsewhere, use logging.getLogger(__name__)
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def chunk_book(
    book_content: str,
    method: str = "paragraph",
    chunk_size: int = 1000, # Default size for fixed_size method
    chunk_overlap: int = 100, # Default overlap for fixed_size method
    min_chunk_size: int = 50 # Minimum size for paragraph method
) -> List[str]:
    """
    Splits book content into text chunks.

    Args:
        book_content: The full text content of the book.
        method: The chunking method ("paragraph" or "fixed_size").
        chunk_size: Target character size for "fixed_size" method.
        chunk_overlap: Character overlap for "fixed_size" method.
        min_chunk_size: Minimum character length for a chunk in "paragraph" method.

    Returns:
        A list of text chunks.
    """
    logger.info(f"Starting book chunking using method: '{method}'")
    chunks = []

    if not book_content or not isinstance(book_content, str):
        logger.warning("chunk_book received empty or invalid book_content.")
        return []

    if method == "paragraph":
        logger.info(f"Using paragraph splitting (min size: {min_chunk_size})...")
        potential_chunks = re.split(r'\n\s*\n+', book_content)
        chunks = [p.strip() for p in potential_chunks if p and len(p.strip()) >= min_chunk_size]
        logger.info(f"Split into {len(potential_chunks)} potential paragraphs, kept {len(chunks)} chunks >= {min_chunk_size} chars.")

    elif method == "fixed_size":
        if chunk_overlap < 0:
             logger.warning(f"Chunk overlap ({chunk_overlap}) cannot be negative. Setting to 0.")
             chunk_overlap = 0
        if chunk_size <= 0:
             logger.error(f"Chunk size ({chunk_size}) must be positive.")
             return [] # Cannot chunk with non-positive size
        if chunk_overlap >= chunk_size:
            logger.warning(f"Chunk overlap ({chunk_overlap}) >= chunk size ({chunk_size}). Setting overlap to chunk_size // 4 or 0 if size is small.")
            chunk_overlap = max(0, chunk_size // 4) # Ensure overlap isn't negative
            
        step = chunk_size - chunk_overlap
        if step <= 0:
             logger.error(f"Chunk size ({chunk_size}) minus overlap ({chunk_overlap}) results in a non-positive step ({step}). Cannot proceed.")
             return []

        logger.info(f"Using fixed size splitting (size: {chunk_size}, overlap: {chunk_overlap}, step: {step})...")
        start_index = 0
        while start_index < len(book_content):
            end_index = min(start_index + chunk_size, len(book_content))
            chunk = book_content[start_index:end_index].strip()
            if chunk: # Add non-empty chunks
                chunks.append(chunk)
            # Move start index for the next chunk using the calculated step
            start_index += step
            # The check 'start_index < len(book_content)' handles the loop exit
            # Removed the incorrect break condition
            
        logger.info(f"Split into {len(chunks)} fixed-size chunks.")

    else:
        logger.warning(f"Unknown chunking method '{method}'. Returning the entire content as a single chunk (if long enough).")
        if len(book_content.strip()) >= min_chunk_size: # Use min_chunk_size for fallback too
            chunks = [book_content.strip()]
        else:
            chunks = []
            logger.warning(f"Single chunk did not meet min_chunk_size ({min_chunk_size}), returning empty list.")

    if not chunks:
        logger.warning("Chunking resulted in an empty list of chunks.")

    logger.info(f"Generated {len(chunks)} final chunks.")
    return chunks 