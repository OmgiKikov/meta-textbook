"""Functions for parsing input files and extracting basic structures."""

from typing import List, Dict, Any, Optional, Tuple
import re
import logging
# Removed Node, Metadata, NodeType imports as they are not used here anymore
# from graph_structures import Node, Metadata, NodeType

logger = logging.getLogger(__name__)
# Optional: Configure basic logging here if it's the primary entry point for it
# if not logger.handlers:
#    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants (Defaults can likely be removed if not used here) ---
# DEFAULT_SUBJECT = "UnknownSubject"
# DEFAULT_GRADE = "UnknownGrade"
# --- End Constants ---

# Placeholder for topic info structure (should match definition in matching.py or be shared)
# Consider moving this to graph_structures.py if it becomes complex
class TopicInfo:
    def __init__(self, id: str, title: str, subtopics: List[str], persons: List[str], subject: str, grade: str):
        self.id = id
        self.title = title.strip()
        self.subtopics = [st.strip() for st in subtopics if st.strip()]
        self.persons = [p.strip() for p in persons if p.strip()]
        self.subject = subject
        self.grade = grade

    def __repr__(self):
        return f"TopicInfo(id='{self.id}', title='{self.title}', subtopics={len(self.subtopics)}, persons={len(self.persons)})"

# --- Candidate Node Extraction Function --- #
# Restore this function
def extract_candidate_nodes_from_text(text: str) -> List[str]:
    """Extracts potential node titles (capitalized words/phrases) from text."""
    candidates = re.findall(r'\b[А-ЯЁ][а-яё]+(?:[- ]+[А-ЯЁ][а-яё]+)*\b', text)
    # Basic filtering by length
    filtered_candidates = [cand.strip() for cand in candidates if len(cand.strip()) > 2]
    return list(set(filtered_candidates))

# TODO: Add parse_book function if needed (e.g., to extract chapters/sections)
# def parse_book(book_content: str) -> Any:
#     pass 