"""Main pipeline logic for building the knowledge graph from files."""

import logging
from typing import Dict, List, Any, Optional
import time # To measure execution time
import json # Add json import for caching
import os.path # Add os.path import for cache file check

# --- Project Imports ---
from parsers import TopicInfo # Only TopicInfo needed now
from utils import chunk_book
# Import BOTH LLM functions
from llm_interface import call_openrouter, extract_topics_with_llm
from graph_structures import Node, Edge, Metadata, NodeType # Import NodeType
import uuid # Import uuid for topic IDs
import re # Import re for subject/grade extraction
# --- End Project Imports ---

# --- Dependency Imports ---
try:
    import chromadb
    from chromadb.api.models.Collection import Collection # For type hinting
except ImportError:
    raise ImportError("ChromaDB is not installed. Please install it using: pip install chromadb")

import openai # Import openai
from openai import OpenAI, AuthenticationError # Import client and specific error
# --- End Dependency Imports ---

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
DEFAULT_SUBJECT = "UnknownSubject"
DEFAULT_GRADE = "UnknownGrade"
DEFAULT_TOPIC = "UnknownTopic"
DEFAULT_CHROMA_COLLECTION_NAME = "rp_topics_collection"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RP_CACHE_FILE = "rp_parsed_cache.json" # Define cache file name
# --- End Constants ---

def build_graph_from_files(
    rp_file_content: str,
    book_file_content: str,
    openrouter_api_key: str,
    openai_api_key: str
) -> Dict[str, List[Any]]:
    """Orchestrates the graph building process from RP and Book files."""
    start_time = time.time()
    logger.info("Starting knowledge graph building pipeline...")

    # --- Step 0: Initialize Clients Early --- #
    # Initialize clients first, as they are needed for RP parsing now
    logger.info("Pipeline Step: Initializing API Clients...")
    openrouter_client: Optional[OpenAI] = None
    # embedding_client: Optional[OpenAI] = None # Removed
    try:
        if not openrouter_api_key: raise ValueError("OpenRouter API Key is required.")
        openrouter_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=openrouter_api_key)
        logger.info("OpenRouter client initialized successfully.")
        # Removed embedding client initialization
        # if not openai_api_key: raise ValueError("OpenAI API Key for embeddings is required.")
        # embedding_client = OpenAI(api_key=openai_api_key)
        # logger.info("Direct OpenAI client (for embeddings) initialized successfully.")
    except (AuthenticationError, ValueError) as e:
        logger.error(f"API Client Initialization Error: {e}.")
        return {"nodes": [], "edges": []}
    except Exception as e:
        logger.error(f"Failed to initialize API clients: {e}", exc_info=True)
        return {"nodes": [], "edges": []}

    # --- Step 1: Parse RP (Load from Cache or Use LLM) --- #
    logger.info("Pipeline Step: Parsing RP (checking cache)...")
    parsed_rp_topics: List[TopicInfo] = []
    topics_map: Dict[str, TopicInfo] = {}
    rp_candidate_node_titles: List[str] = []
    subject = DEFAULT_SUBJECT
    grade = DEFAULT_GRADE
    llm_rp_result_data: Optional[Dict] = None # To store loaded/parsed data

    # --- Check for Cache --- 
    if os.path.exists(RP_CACHE_FILE):
        try:
            with open(RP_CACHE_FILE, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                # Validate essential keys exist in cache
                if all(k in cached_data for k in ('subject', 'grade', 'rp_topics', 'rp_candidate_nodes')):
                    subject = cached_data['subject']
                    grade = cached_data['grade']
                    # Store the relevant parts for later processing
                    llm_rp_result_data = {
                        'rp_topics': cached_data['rp_topics'],
                        'rp_candidate_nodes': cached_data['rp_candidate_nodes']
                    }
                    logger.info(f"Loaded RP data from cache file '{RP_CACHE_FILE}'. Subject: {subject}, Grade: {grade}")
                else:
                     logger.warning(f"Cache file '{RP_CACHE_FILE}' is missing required keys. Re-parsing RP.")
                     llm_rp_result_data = None # Force re-parsing
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error reading or parsing RP cache file '{RP_CACHE_FILE}': {e}. Re-parsing RP.")
            llm_rp_result_data = None # Force re-parsing
            
    # --- If Cache Miss or Invalid: Parse RP using Regex and LLM --- 
    if llm_rp_result_data is None:
        logger.info("RP cache not found or invalid. Proceeding with LLM parsing...")
        try:
            # 1a: Extract Subject/Grade using Regex (simple approach) - DO THIS EVEN IF CACHING FAILS
            lines_to_search = rp_file_content.splitlines()[:20]
            subject_regex = re.compile(r"^Предмет\s*:\s*(.+)", re.IGNORECASE)
            grade_regex = re.compile(r"^(Класс|Grade)\s*:\s*(.+)", re.IGNORECASE)
            temp_subject, temp_grade = DEFAULT_SUBJECT, DEFAULT_GRADE # Use temp vars first
            for line in lines_to_search:
                subj_match = subject_regex.match(line.strip())
                if subj_match: temp_subject = subj_match.group(1).strip(); logger.info(f"Extracted Subject: '{temp_subject}'")
                grade_match = grade_regex.match(line.strip())
                if grade_match: temp_grade = grade_match.group(2).strip(); logger.info(f"Extracted Grade: '{temp_grade}'")
                if temp_subject != DEFAULT_SUBJECT and temp_grade != DEFAULT_GRADE: break
            subject = temp_subject # Assign to main variables
            grade = temp_grade
            if subject == DEFAULT_SUBJECT: logger.warning("Could not find Subject via regex.")
            if grade == DEFAULT_GRADE: logger.warning("Could not find Grade via regex.")

            # --- Extract relevant section from RP (CASE-SENSITIVE) --- 
            rp_start_marker = "СОДЕРЖАНИЕ ОБУЧЕНИЯ"
            rp_end_marker = "ПЛАНИРУЕМЫЕ РЕЗУЛЬТАТЫ" # Using shorter, reliable part
            
            rp_start_index = rp_file_content.find(rp_start_marker)
            rp_end_index = rp_file_content.find(rp_end_marker)
            
            rp_text_to_llm = rp_file_content # Default to full content
            if rp_start_index != -1 and rp_end_index != -1 and rp_start_index < rp_end_index:
                rp_text_to_llm = rp_file_content[rp_start_index:rp_end_index]
                logger.info(f"Found '{rp_start_marker}' and '{rp_end_marker}' (case-sensitive). Sending slice of RP text to LLM (length: {len(rp_text_to_llm)}).")
            elif rp_start_index != -1:
                rp_text_to_llm = rp_file_content[rp_start_index:]
                logger.warning(f"Found '{rp_start_marker}' but not '{rp_end_marker}' (case-sensitive). Sending RP text from start marker to end (length: {len(rp_text_to_llm)}). This might be too long.")
            else:
                logger.warning(f"Could not find markers '{rp_start_marker}' or '{rp_end_marker}' (case-sensitive) in RP content. Attempting to process full text (length: {len(rp_text_to_llm)}). This might exceed token limits.")
             # --- End Updated Section --- 

            # 1b: Call LLM to get topics AND candidate node names
            llm_rp_result = extract_topics_with_llm(openrouter_client, rp_text_to_llm)

            if not llm_rp_result:
                logger.error("LLM failed to parse RP structure and candidates. Aborting.")
                return {"nodes": [], "edges": []}
            
            llm_rp_result_data = llm_rp_result # Store the freshly parsed data

            # --- Save Successful Parse to Cache --- 
            try:
                data_to_cache = {
                    'subject': subject,
                    'grade': grade,
                    'rp_topics': llm_rp_result_data['rp_topics'],
                    'rp_candidate_nodes': llm_rp_result_data['rp_candidate_nodes']
                }
                with open(RP_CACHE_FILE, 'w', encoding='utf-8') as f:
                    json.dump(data_to_cache, f, ensure_ascii=False, indent=4)
                logger.info(f"Successfully saved parsed RP data to cache file '{RP_CACHE_FILE}'.")
            except (IOError, TypeError, KeyError) as e: # Added KeyError
                logger.warning(f"Failed to save RP data to cache file '{RP_CACHE_FILE}': {e}")
                # Continue execution even if caching fails
        
        except Exception as e:
            logger.error(f"Error during RP parsing (LLM or conversion): {e}", exc_info=True)
            return {"nodes": [], "edges": []}

    # --- Post-Processing (using llm_rp_result_data from cache or fresh parse) --- 
    if not llm_rp_result_data: # Should not happen if logic above is correct, but safeguard
        logger.error("RP data is missing after cache check and parsing attempt. Aborting.")
        return {"nodes": [], "edges": []}

    # 1c: Process Topics from llm_rp_result_data
    llm_topics = llm_rp_result_data.get('rp_topics', [])
    for i, item in enumerate(llm_topics):
        topic_title = item.get('topic', f"Unnamed Topic {i+1}")
        subtopics = item.get('subtopics', [])
        if not isinstance(subtopics, list): subtopics = [] # Ensure list
        topic_id = f"rp_{subject}_{grade}_topic_{i+1}_{uuid.uuid4().hex[:8]}"
        topic_info = TopicInfo(
            id=topic_id, title=topic_title, subtopics=subtopics,
            persons=[], subject=subject, grade=grade
        )
        parsed_rp_topics.append(topic_info)
        topics_map[topic_id] = topic_info
    logger.info(f"Processed {len(parsed_rp_topics)} TopicInfo objects from RP data.")
    if not parsed_rp_topics:
         logger.warning("LLM RP result contained no valid topics.") # Warn instead of abort if candidates exist?
         # Decide if abort is necessary if no topics found but candidates might exist
         # return {"nodes": [], "edges": []}

    # 1d: Get Candidate Node Titles from llm_rp_result_data
    rp_candidate_node_titles = llm_rp_result_data.get('rp_candidate_nodes', [])
    if not isinstance(rp_candidate_node_titles, list):
        logger.warning("RP data had invalid format for candidate nodes. Proceeding without RP candidates.")
        rp_candidate_node_titles = []
    logger.info(f"Using {len(rp_candidate_node_titles)} candidate node titles from RP data.")

    # --- Step 1.5: Initialize Nodes from Candidates --- #
    all_nodes_dict: Dict[str, Node] = {}
    try:
        logger.info(f"Initializing graph with {len(rp_candidate_node_titles)} candidates from LLM...")
        # Use Subject/Grade extracted earlier
        base_metadata = Metadata(subject=subject, grade=grade, topic="RP Source", subtopic=None) # Generic RP metadata
        for i, title in enumerate(rp_candidate_node_titles):
            if not isinstance(title, str) or not title.strip(): continue # Skip invalid titles
            title = title.strip()
            lookup_key = title.lower()
            if lookup_key not in all_nodes_dict:
                 node_id = f"rp_node_{lookup_key.replace(' ', '_')}_{i}"
                 all_nodes_dict[lookup_key] = Node(
                     id=node_id,
                     title=title,
                     type=NodeType.CONCEPT, # Default type
                     importance=3, # Lower default importance for RP candidates
                     order=0,
                     summary=f"Candidate node from RP.", # Minimal initial summary
                     meta=[base_metadata], # Start with generic RP context
                     chunks=[], date=None, geo=None, media=[]
                 )
        logger.info(f"Created {len(all_nodes_dict)} initial candidate nodes.")
    except Exception as e:
         logger.error(f"Error creating initial nodes from RP candidates: {e}", exc_info=True)
         all_nodes_dict = {} # Start empty if node creation fails

    # --- Step 4: Chunk Book --- #
    logger.info("Pipeline Step: Chunking Book Content...")
    try:
        chunks: List[str] = chunk_book(
            book_file_content,
            method="fixed_size",
            chunk_size=4000,      # Increased chunk size
            chunk_overlap=400       # Increased overlap
            # min_chunk_size is not used for fixed_size
        )
        if not chunks:
            logger.error("Book chunking resulted in no chunks. Aborting pipeline.")
            return {"nodes": [], "edges": []}
        logger.info(f"Book Chunked successfully using fixed_size. Found {len(chunks)} chunks.")
    except Exception as e:
        logger.error(f"Error during book chunking: {e}", exc_info=True)
        return {"nodes": [], "edges": []}

    # --- Step 5: Process Chunks --- #
    # all_nodes_dict initialized with candidates, topics_map from LLM
    all_edges_dict: Dict[tuple, Edge] = {}
    processed_chunks = 0

    logger.info(f"Pipeline Step: Processing {len(chunks)} Book Chunks...")
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i}"
        chunk_start_time = time.time()
        logger.info(f"Processing chunk {i+1}/{len(chunks)} (ID: {chunk_id})...")

        # --- Step 5c (now simplified): Call LLM to extract entities/relationships --- #
        logger.info(f"  Calling LLM for chunk {chunk_id} analysis and context determination...")
        try:
            # Call the updated call_openrouter, passing the full RP structure
            extracted_data = call_openrouter(
                client=openrouter_client,
                chapter_text=chunk,
                all_rp_topics=parsed_rp_topics, # Pass the list of TopicInfo objects
                subject=subject, # Pass overall subject/grade
                grade=grade
            )

            if extracted_data is None: continue # Skip chunk on LLM failure
            if not extracted_data.get('nodes') and not extracted_data.get('edges'): continue # Skip if nothing found
            
            num_llm_nodes = len(extracted_data.get('nodes', [])); num_llm_edges = len(extracted_data.get('edges', []))
            logger.info(f"  LLM call for chunk {chunk_id} successful. Got {num_llm_nodes} nodes, {num_llm_edges} edges.")
            
            # --- Step 5d (Revised): Aggregate/Enrich Results with ID Mapping --- #
            nodes_processed_this_chunk = 0; nodes_enriched_this_chunk = 0; nodes_added_this_chunk = 0
            edges_added_this_chunk = 0
            # Map LLM-generated node IDs to the actual graph node IDs for this chunk
            llm_id_to_graph_id: Dict[str, str] = {}
            
            # --- Process Nodes and Build ID Map ---
            for llm_node in extracted_data.get('nodes', []):
                 nodes_processed_this_chunk += 1
                 if not (hasattr(llm_node, 'id') and hasattr(llm_node, 'title') and hasattr(llm_node, 'meta') and llm_node.meta):
                     logger.warning(f"  Skipping invalid LLM node object: {llm_node}")
                     continue # Moved continue inside the if block

                 lookup_title = llm_node.title.lower()
                 llm_node_id = llm_node.id # ID assigned by LLM
                 llm_node_metadata = llm_node.meta[0]
                 graph_node_id: Optional[str] = None

                 if lookup_title in all_nodes_dict:
                      # Enrich existing node (likely from RP)
                      existing_node = all_nodes_dict[lookup_title]
                      graph_node_id = existing_node.id # Get the actual ID stored in the graph
                      # Enrich fields...
                      if hasattr(llm_node, 'summary') and llm_node.summary: existing_node.summary = llm_node.summary if existing_node.summary.startswith("Candidate node from RP") else existing_node.summary + f"\n---\n{llm_node.summary}"
                      if not any(m == llm_node_metadata for m in existing_node.meta): existing_node.meta.append(llm_node_metadata)
                      if hasattr(llm_node, 'importance'): existing_node.importance = max(existing_node.importance, llm_node.importance)
                      if hasattr(llm_node, 'order'): existing_node.order = llm_node.order # Use order from LLM if provided
                      if chunk_id not in existing_node.chunks: existing_node.chunks.append(chunk_id)
                      if hasattr(llm_node, 'date') and llm_node.date and not existing_node.date: existing_node.date = llm_node.date
                      if hasattr(llm_node, 'geo') and llm_node.geo and not existing_node.geo: existing_node.geo = llm_node.geo
                      if hasattr(llm_node, 'media') and llm_node.media: existing_node.media = list(set(existing_node.media + llm_node.media)) # Combine unique media
                      nodes_enriched_this_chunk += 1
                 else:
                      # Add new node from LLM
                      llm_node.chunks = [chunk_id]
                      graph_node_id = llm_node.id # Use the LLM ID as the graph ID for new nodes
                      all_nodes_dict[lookup_title] = llm_node
                      nodes_added_this_chunk += 1

                 # Store the mapping from the LLM's ID to the actual graph ID
                 if graph_node_id:
                     llm_id_to_graph_id[llm_node_id] = graph_node_id
                 else:
                     logger.error(f"  Failed to determine graph_node_id for LLM node {llm_node_id}. This shouldn't happen.")

            # --- Process Edges using the ID Map and Global Dictionary ---
            for edge in extracted_data.get('edges', []):
                 # Get IDs assigned by LLM
                 llm_source_id = getattr(edge, 'source_id', None)
                 llm_target_id = getattr(edge, 'target_id', None)

                 if not llm_source_id or not llm_target_id:
                      logger.warning(f"  Edge missing source/target ID from LLM. Skipping edge: {edge}")
                      continue

                 graph_source_id = None
                 graph_target_id = None

                 # --- Find actual graph_source_id ---
                 # 1. Check current chunk's map
                 graph_source_id = llm_id_to_graph_id.get(llm_source_id)
                 # 2. If not found, check global dictionary by deriving title from LLM ID
                 if graph_source_id is None and llm_source_id.startswith("node_"):
                     source_title_key = llm_source_id[5:].lower() # Assumes "node_<title>" format
                     source_node = all_nodes_dict.get(source_title_key)
                     if source_node:
                         graph_source_id = source_node.id

                 # --- Find actual graph_target_id ---
                 # 1. Check current chunk's map
                 graph_target_id = llm_id_to_graph_id.get(llm_target_id)
                 # 2. If not found, check global dictionary by deriving title from LLM ID
                 if graph_target_id is None and llm_target_id.startswith("node_"):
                     target_title_key = llm_target_id[5:].lower() # Assumes "node_<title>" format
                     target_node = all_nodes_dict.get(target_title_key)
                     if target_node:
                         graph_target_id = target_node.id

                 # Check if both corresponding graph nodes were successfully resolved
                 if graph_source_id and graph_target_id:
                    # Update edge object to use the actual graph IDs
                    edge.source_id = graph_source_id
                    edge.target_id = graph_target_id

                    # Create edge key using the actual graph IDs
                    edge_key = (graph_source_id, graph_target_id, edge.relation_type, edge.description)

                    if edge_key not in all_edges_dict:
                        edge.chunks = [chunk_id]
                        all_edges_dict[edge_key] = edge
                        edges_added_this_chunk += 1
                    else:
                        # Enrich existing edge
                        existing_edge = all_edges_dict[edge_key]
                        if chunk_id not in existing_edge.chunks: existing_edge.chunks.append(chunk_id)
                        if hasattr(edge, 'meta') and edge.meta and existing_edge.meta and edge.meta[0] not in existing_edge.meta: existing_edge.meta.append(edge.meta[0]) # Check if existing_edge.meta exists
                        if hasattr(edge, 'strength'): existing_edge.strength = max(existing_edge.strength, edge.strength)
                 else:
                    # Log warning if source/target node wasn't resolved globally
                    missing_nodes_msg = []
                    if not graph_source_id: missing_nodes_msg.append(f"source '{llm_source_id}'")
                    if not graph_target_id: missing_nodes_msg.append(f"target '{llm_target_id}'")
                    logger.warning(f"  Edge references globally unresolved node ID(s) ({', '.join(missing_nodes_msg)}). Skipping edge: {llm_source_id} -> {llm_target_id}")

            logger.info(f"  Chunk {chunk_id}: Processed {nodes_processed_this_chunk} LLM nodes -> Enriched: {nodes_enriched_this_chunk}, Added: {nodes_added_this_chunk}. Added {edges_added_this_chunk} new edges.")
            processed_chunks += 1

        except AuthenticationError as e:
             logger.error(f"  OpenRouter Authentication Error for chunk {chunk_id}: {e}. Aborting.")
             return {"nodes": [], "edges": []}
        except Exception as e:
            logger.error(f"  Unexpected error processing chunk {chunk_id}: {e}", exc_info=True)
            continue # Skip chunk on error

        # ... (logging chunk time) ...
        chunk_time = time.time() - chunk_start_time
        logger.info(f"Finished processing chunk {i+1}/{len(chunks)} in {chunk_time:.2f} seconds.")

    # --- Step 6: Final Aggregation & Return --- #
    logger.info(f"Pipeline Step: Aggregation Complete. Processed {processed_chunks}/{len(chunks)} chunks.")

    final_nodes = list(all_nodes_dict.values()) # Nodes are already enriched/deduplicated
    final_edges = list(all_edges_dict.values()) # Edges are deduplicated

    total_time = time.time() - start_time
    logger.info(f"Knowledge graph building pipeline finished in {total_time:.2f} seconds.")
    logger.info(f"Final graph: {len(final_nodes)} nodes, {len(final_edges)} edges.")

    return {"nodes": final_nodes, "edges": final_edges}