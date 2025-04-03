"""Interface for interacting with the LLM (e.g., via OpenRouter)."""

import json
import openai
from openai import OpenAI
import streamlit as st
import uuid
from typing import Dict, List, Any, Optional, Union
import logging
from dataclasses import asdict
from enum import Enum

# Import TopicInfo from parsers
from parsers import TopicInfo

# Import base structures needed for type hinting and object creation
try:
    from graph_structures import Node, Edge, Metadata, NodeType, RelationType
except ImportError:
    logging.warning("Could not import graph structures. LLM interface might return basic dicts.")
    # Fallback to basic types if structures are not available
    Metadata = dict
    Node = dict
    Edge = dict
    class NodeType(Enum): CONCEPT = 'concept'; PERSON = 'person'; EVENT = 'event' # Minimal fallback
    class RelationType(Enum): LOGICAL = 'logical'; CAUSAL = 'causal'; STRUCTURAL = 'structural'; COMPARATIVE = 'comparative'; REFERENTIAL = 'referential' # Minimal fallback

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_topics_with_llm(client: OpenAI, rp_content: str) -> Optional[Dict[str, Union[List[Dict], List[str]]]]:
    """
    Uses LLM to extract topic/subtopic structure AND candidate node names from RP content.

    Args:
        client: Initialized OpenAI client.
        rp_content: The full text content of the Work Program (RP).

    Returns:
        A dictionary with keys 'rp_topics' (List[Dict]) and 'rp_candidate_nodes' (List[str]),
        or None if extraction fails.
    """
    if not client:
        logger.error("LLM client not provided for RP parsing.")
        return None
    if not rp_content:
        logger.warning("Empty RP content provided for parsing.")
        return {"rp_topics": [], "rp_candidate_nodes": []}

    prompt = f"""
    Analyze the provided educational Work Program (RP) text. Your goals are:
    1. Identify the main educational topics and their corresponding subtopics.
    2. Identify potential candidate entities (specific educational terms, concepts, persons, places, events) mentioned within the RP text (in topic titles or subtopic descriptions).

    Instructions:
    - Extract main topics (often starting with "Тема...") and the list of subtopics/points under each.
    - Separately, list all potential candidate entities found in the text. These should be specific names or terms, not generic words like "Введение", "Значение", "Структура" unless they are clearly defined terms within the RP context.
    - Structure the output STRICTLY as a single JSON object with exactly two top-level keys:
        - "rp_topics": A list of objects, where each object has keys "topic" (string) and "subtopics" (list of strings).
        - "rp_candidate_nodes": A list of strings, where each string is a potential candidate entity name identified in the RP.
    - Ensure the output is only the JSON object, with no other text.

    Example Input Text Snippet:
    ```
    Предмет: Биология
    Класс: 10

    Тема 1. Введение в Биологию
    Биология как наука. Уровни организации живой природы.

    Тема 2. Клетка - единица живого
    Основные положения клеточной теории. Роберт Гук и открытие клетки.
    Прокариотическая клетка. Эукариотическая клетка.
    ```

    Example JSON Output:
    ```json
    {{
      "rp_topics": [
        {{
          "topic": "Тема 1. Введение в Биологию",
          "subtopics": [
            "Биология как наука",
            "Уровни организации живой природы"
          ]
        }},
        {{
          "topic": "Тема 2. Клетка - единица живого",
          "subtopics": [
            "Основные положения клеточной теории",
            "Роберт Гук и открытие клетки",
            "Прокариотическая клетка",
            "Эукариотическая клетка"
          ]
        }}
      ],
      "rp_candidate_nodes": [
        "Биология",
        "Уровни организации живой природы",
        "Клетка",
        "Клеточная теория",
        "Роберт Гук",
        "Прокариотическая клетка",
        "Эукариотическая клетка"
      ]
    }}
    ```

    Now, analyze the following Work Program text:
    ---
    {rp_content}
    ---

    Return ONLY the JSON object with the two specified keys.
    """

    try:
        logger.info("Sending request to LLM for RP topic/subtopic and candidate node extraction...")
        completion = client.chat.completions.create(
            model="anthropic/claude-3.7-sonnet",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant specialized in parsing educational documents. Extract topics, subtopics, and candidate entity names strictly in the requested JSON object format with keys 'rp_topics' and 'rp_candidate_nodes'.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            # max_tokens=4000,
        )
        # --- ADDED: Validate API response before accessing content --- 
        if not completion or not completion.choices or not completion.choices[0].message or not completion.choices[0].message.content:
            logger.error(f"LLM response for RP parsing is missing expected structure or content. Completion object: {completion}")
            st.error("Received an invalid or empty response from the LLM for RP parsing.")
            return None
        # --- END Validation ---

        response_content = completion.choices[0].message.content
        logger.info("Received response from LLM for RP parsing.")

        # Use the improved JSON parsing logic
        raw_data = None
        cleaned_response = response_content.strip()
        try:
            raw_data = json.loads(cleaned_response)
            logger.info("Successfully parsed LLM response JSON directly.")
        except json.JSONDecodeError as e:
            # Log the full response when direct parsing fails
            logger.error(f"Direct JSON parsing failed: {e}. Raw response was: {cleaned_response}")
            logger.warning("Attempting cleaning...") # Change previous warning to debug/info if too verbose
            
            # Apply cleaning heuristics if direct parsing fails
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()
            
            # Sometimes it might be wrapped in a key like {"response": { ... json ... }}
            # Or just have leading/trailing text. Find the outermost {} or []
            json_start = -1
            json_end = -1
            first_brace = cleaned_response.find('{')
            first_bracket = cleaned_response.find('[')
            
            if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
                json_start = first_brace
                json_end = cleaned_response.rfind('}')
            elif first_bracket != -1:
                json_start = first_bracket
                json_end = cleaned_response.rfind(']')

            if json_start != -1 and json_end != -1:
                json_str = cleaned_response[json_start : json_end + 1]
                try:
                    raw_data = json.loads(json_str)
                    logger.info("Successfully parsed LLM response JSON for RP data after cleaning.")
                except json.JSONDecodeError as e_clean:
                    # Log the cleaned string that failed parsing
                    logger.error(f"JSON parsing failed even after cleaning: {e_clean}. Cleaned string was: {json_str}")
                    st.error(f"LLM returned invalid data format even after cleaning. See logs.")
                    return None
            else:
                # Log the cleaned response if no JSON structure found
                logger.error(f"Could not find valid JSON object structure in LLM response for RP data: {cleaned_response}")
                st.error(f"LLM returned invalid data format. See logs.")
                return None
        
        # Validate the structure of the parsed data
        if not isinstance(raw_data, dict) or \
           'rp_topics' not in raw_data or \
           'rp_candidate_nodes' not in raw_data or \
           not isinstance(raw_data['rp_topics'], list) or \
           not isinstance(raw_data['rp_candidate_nodes'], list):
            logger.error(f"LLM response JSON for RP data has incorrect structure. Keys: {raw_data.keys() if isinstance(raw_data, dict) else 'N/A'}")
            return None
        
        # Further validation of rp_topics structure can be added here if needed
        
        logger.info(f"Successfully extracted RP structure and {len(raw_data['rp_candidate_nodes'])} candidates using LLM.")
        return raw_data # Return the dictionary with the two keys

    except openai.APIError as e:
        logger.error(f"OpenRouter API Error during RP parsing: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred calling LLM for RP parsing: {e}", exc_info=True)
        return None

# --- ADDED: Function for Second LLM Pass (Refinement) ---
def refine_llm_extraction(
    client: OpenAI,
    initial_extraction: Dict[str, List[Dict]],
    subject: str,
    grade: str
) -> Optional[Dict[str, List[Dict]]]:
    """
    Performs a second LLM pass to refine the initial extraction results.
    Focuses on applying normalization rules and adding logical links.
    
    Args:
        client: Initialized OpenAI client.
        initial_extraction: The dictionary parsed from the first LLM call's JSON response
                             (containing 'entities' and 'relationships' lists of dictionaries).
        subject: Overall subject context.
        grade: Overall grade context.

    Returns:
        A refined dictionary in the same format, or None on failure.
    """
    if not client:
        logger.error("OpenAI client not provided for refinement LLM call.")
        return None
    if not initial_extraction or not isinstance(initial_extraction, dict):
        logger.warning("Invalid initial extraction data provided for refinement.")
        return initial_extraction # Return original if input invalid

    # Convert input dict to JSON string for the prompt
    try:
        input_json_str = json.dumps(initial_extraction, ensure_ascii=False, indent=2)
    except TypeError as e:
        logger.error(f"Could not serialize initial extraction data to JSON for refinement prompt: {e}")
        return initial_extraction # Return original if serialization fails
        
    # Get allowed relation types dynamically from the Enum
    allowed_relation_types_str = ', '.join([f'\"{e.value}\"' for e in RelationType])

    # --- Refinement Prompt --- #
    refinement_prompt = f"""
    You are an AI assistant tasked with refining structured data extracted from an educational text ({subject}, Grade {grade}).
    Your input is a JSON object containing lists of 'entities' and 'relationships' extracted in a previous step.
    Your goal is to improve this data by STRICTLY applying normalization rules and adding missing key logical links.

    INPUT JSON:
    ```json
    {input_json_str}
    ```

    TASKS:
    1.  **Analyze Input Entities:** Review the 'entities' list.
    2.  **Apply STRICT Normalization (Modify 'entities'):**
        - **Synonyms:** If multiple entities represent the same concept (e.g., 'шапероны' and 'белки теплового шока'), consolidate them into a SINGLE entity using the most appropriate canonical name. Remove the redundant entities.
        - **Hierarchy/Generalization:** If specific variants exist alongside a broader concept (e.g., 'Гемоглобин А', 'Гемоглобин F' together with 'Гемоглобин'), merge the specific variants into the broader concept entity, unless the distinction was clearly the main focus of the original text passage (which is rare). Remove the redundant specific entities.
        - **Update IDs/Names:** Ensure the `id` and `name` of the consolidated entity reflect the single canonical name.
    3.  **Analyze Input Relationships:** Review the 'relationships' list.
    4.  **Update Relationship IDs:** If any entities were merged in step 2, update the `source` and `target` IDs in the 'relationships' list to point to the consolidated entity's ID.
    5.  **Add Missing Logical Links (Modify 'relationships'):** Identify pairs of entities in the *refined* list where a key logical or functional relationship is clearly implied but missing (based on general biological knowledge relevant to the subject/grade). Add these relationships. Examples:
        - If 'Митохондрия' and 'АТФ' exist, add: `{{"source": "node_митохондрия", "target": "node_атф", "relation_type": "logical", "strength": 5, "description": "производит АТФ", ...}}`
        - Add other obvious links like organelle-function, process-result.
        - **DO NOT invent new entities.** Only add relationships between existing entities in the refined list.
    6.  **Format Output:** Return the *complete, corrected* data STRICTLY as a single JSON object with the original keys ('entities', 'relationships'). The lists should contain the modified/consolidated entities and the updated/added relationships.

    **CRITICAL RULES:**
    - Output ONLY the final, valid JSON object.
    - Ensure all `id` fields start with "node_".
    - Ensure `importance` and `strength` are integers 1-5.
    - Ensure `relation_type` is one of: {allowed_relation_types_str}.
    - Perform normalization aggressively to reduce redundancy.

    Return the refined JSON object now.
    """

    try:
        logger.info("Sending request to LLM for extraction refinement...")
        completion = client.chat.completions.create(
            model="anthropic/claude-3.5-sonnet", # Consider using the same model or a different one if needed
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant refining structured JSON data. Apply normalization rules and add missing logical links strictly as instructed. Output only the corrected JSON object."
                },
                {"role": "user", "content": refinement_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0, # Use low temperature for deterministic refinement
            max_tokens=8000, # Ensure enough tokens for potentially large JSON I/O
        )
        
        # --- Validate API response --- 
        if not completion or not completion.choices or not completion.choices[0].message or not completion.choices[0].message.content:
            logger.error(f"LLM response for refinement is missing expected structure or content. Completion object: {completion}")
            st.error("Received an invalid or empty response from the refinement LLM.")
            return None # Indicate failure
        # --- END Validation ---

        response_content = completion.choices[0].message.content
        logger.info("Received refined data response from LLM.")

        # --- Parse the refined JSON --- 
        try:
            refined_data = json.loads(response_content)
            # Basic validation of the refined structure
            if not isinstance(refined_data, dict) or \
               not isinstance(refined_data.get('entities'), list) or \
               not isinstance(refined_data.get('relationships'), list):
                logger.error(f"Refined JSON from LLM has incorrect structure: {refined_data}")
                st.error("Refinement LLM returned data with incorrect structure.")
                return None # Indicate failure
            logger.info("Successfully parsed refined JSON data.")
            return refined_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse refined JSON response from LLM: {e}. Response was: {response_content}")
            st.error("Could not parse the refined data JSON returned by the LLM.")
            return None # Indicate failure

    except openai.APIError as e:
        logger.error(f"OpenRouter API Error during refinement: {e}")
        st.error(f"LLM API Error during refinement: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred calling refinement LLM: {e}", exc_info=True)
        st.error(f"Unexpected error during refinement LLM call. See logs.")
        return None
# --- END Refinement Function ---

def call_openrouter(
    client: OpenAI, 
    chapter_text: str, 
    all_rp_topics: List[TopicInfo], # Pass the full RP structure
    subject: str, # Pass overall subject/grade if available
    grade: str
) -> Optional[Dict[str, List[Any]]]:
    """
    Uses LLM to analyze a chapter text chunk, determine the relevant RP topic/subtopic context,
    extract entities/relationships, and associate them with the determined context.
    
    Args:
        client: Initialized OpenAI client.
        chapter_text: The text chunk from the textbook.
        all_rp_topics: A list of TopicInfo objects representing the full RP structure.
        subject: The overall subject context.
        grade: The overall grade context.

    Returns:
        A dictionary containing lists of Node and Edge objects (with context in their meta), 
        or None on failure.
    """
    # Check if we have actual structure classes or fallbacks
    has_node_class = hasattr(Node, 'from_dict')
    has_edge_class = hasattr(Edge, 'from_dict')

    # Client initialization is now handled outside this function
    if not client:
        logger.error("OpenAI client object was not provided to call_openrouter.")
        st.error("LLM client not initialized.")
        return None # Return None instead of empty dict on fundamental errors
    if not chapter_text:
        logger.warning("call_openrouter received empty chapter_text.")
        return {"nodes": [], "edges": []} # Return empty dict if just text is missing

    # --- Format RP Structure for Prompt --- #
    rp_structure_str = ""
    if all_rp_topics:
        rp_structure_str += "Available RP Topics and Subtopics:\n"
        for i, topic_info in enumerate(all_rp_topics):
            rp_structure_str += f"{i+1}. Topic: {topic_info.title}\n"
            if topic_info.subtopics:
                for sub in topic_info.subtopics:
                     rp_structure_str += f"   - Subtopic: {sub}\n"
            else:
                 rp_structure_str += "   (No specific subtopics listed)\n"
    else:
        rp_structure_str = "No RP topic structure was provided."
    
    # --- Updated Prompt with Stricter Instructions (Corrected f-string escaping) --- #
    allowed_relation_types_str = ', '.join([f'\"{e.value}\"' for e in RelationType]) # Correctly escape quotes for JSON example

    prompt = f"""
Analyze the following textbook passage for Subject: '{subject}', Grade: '{grade}'.

First, understand the available structure of the corresponding Work Program (RP):
{rp_structure_str}

Now, perform the following tasks based *only* on the textbook passage provided below. FOLLOW ALL INSTRUCTIONS STRICTLY.

1.  **Determine Overall Context:** Identify the *single main RP Topic* from the list above that is most relevant to the content of this specific textbook passage.
2.  **Identify and NORMALIZE Entities:** 
    a. Read the passage and identify all phrases matching the **Definition of an Entity**.
    b. **CRITICAL NORMALIZATION STEP:** Before creating the entity list, apply the **Normalization Requirement** below. Ensure synonyms and specific variants are represented by a SINGLE canonical term.
3.  **Assign Entity Attributes:** For each *normalized* entity:
    a.  `id`: Create a unique identifier (e.g., "node_" + lowercase normalized name).
    b.  `name`: The **SINGLE normalized, canonical name** resulting from the normalization step.
    c.  `type`: Assign type: 'concept', 'term', 'person', 'event', 'location', 'organization'.
    d.  `importance`: Assign an **INTEGER score from 1 (lowest) to 5 (highest)** based on significance within the passage.
    e.  `context`: Sentence or short phrase from the passage providing context FOR THE NORMALIZED TERM.
    f.  `topic`: The *main RP Topic title* you identified in Step 1.
    g.  `subtopic`: The *most relevant subtopic* listed under the chosen main topic (from Step 1) that this entity relates to in the passage. Use null if no specific subtopic fits well.
4.  **Identify Relationships (Including Logical):** 
    a. Find direct relationships between the *normalized* entities within the passage.
    b. **CRITICALLY IMPORTANT:** Identify key logical and functional relationships between core concepts (e.g., 'Митохондрия производит АТФ'), even if not stated in a single sentence but strongly implied by the passage context. Use 'logical' or 'causal' types for these.
5.  **Assign Relationship Attributes:** For each relationship:
    a.  `source`, `target`: The `id` of the source/target *normalized* entities.
    b.  `relation_type`: Assign a type using **ONLY ONE** of the allowed values: {allowed_relation_types_str}.
    c.  `strength`: Assign an **INTEGER score from 1 (lowest) to 5 (highest)** based on the clarity/strength of the connection in the passage.
    d.  `description`: Sentence or short phrase from the passage describing the relationship, OR a concise summary of the logical link (e.g., "производит", "регулирует", "является частью").
    e.  `topic`: The *main RP Topic title* you identified in Step 1.
    f.  `subtopic`: The *most relevant subtopic* listed under the chosen main topic that this relationship pertains to. Use null if no specific subtopic fits well.
6.  **Format Output:** Structure the result STRICTLY as a single JSON object with keys 'entities' (list of entity objects) and 'relationships' (list of relationship objects). 

**Definition of an Entity (Node):**
An entity MUST be a specific educational concept, term, person, place, event, or object. AVOID common words, vague descriptions, section titles.
- Good Examples: 'Митохондрия', 'Закон Ома', 'Петр I'.
- Bad Examples: 'Важность', 'Основные характеристики', 'Этап'.

**Normalization Requirement (CRITICAL - FOLLOW STRICTLY):**
- **Goal:** Represent each core concept with only ONE node.
- **Synonyms:** If the text uses synonyms (e.g., 'шапероны', 'белки теплового шока'), choose the **single most appropriate or frequently used term** from the passage as the canonical name. **DO NOT create separate nodes for synonyms.**
- **Hierarchy/Generalization:** If the text mentions specific types (e.g., 'Гемоглобин А', 'Гемоглобин F'), normalize them to the **broader concept** (e.g., 'Гемоглобин') UNLESS the distinction between types is the main point of the passage. **DO NOT create separate nodes for minor variations if the broader concept exists.**
- **Canonical Name:** Use this single, normalized name for the `name` field and for generating the `id`.

**Self-Check Before Output:**
- Have I represented all synonyms (like шапероны/БТШ) with a single node?
- Have I generalized specific types (like Гемоглобин А/F) to the main concept where appropriate?
- Have I included key logical links (like Митохондрия/АТФ relationship) if implied by the text?

**Example JSON Format (Illustrative):**
    {{
      "entities": [
    {{ "id": "node_шапероны", "name": "Шапероны", ... }}, // NOT separate node for БТШ
    {{ "id": "node_гемоглобин", "name": "Гемоглобин", ... }} // NOT separate nodes for A/F
      ],
      "relationships": [
    {{ "source": "node_митохондрия", "target": "node_атф", "relation_type": "logical", "description": "производит", ... }}
      ]
    }}

Textbook Passage:
    ---
    {chapter_text}
    ---

Return ONLY the single, valid JSON object adhering to ALL instructions, especially the normalization and logical link requirements.
    """

    try:
        # Updated logger message
        logger.info(f"Sending request to LLM for combined context determination and extraction (Subject: {subject}, Grade: {grade})...")
        completion = client.chat.completions.create(
            model="anthropic/claude-3.7-sonnet",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant. Determine the most relevant RP Topic for the provided text passage, then extract SPECIFIC EDUCATIONAL ENTITIES and relationships, associating each with the determined RP topic and the most relevant subtopic from that topic. Follow all instructions precisely, including normalization and JSON output format.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            # max_tokens=8000,
        )
        response_content = completion.choices[0].message.content
        logger.info("Received response from LLM.")

        # --- Parse Initial JSON --- 
        raw_data = None
        cleaned_response = response_content.strip()
        try:
            raw_data = json.loads(cleaned_response)
            logger.info("Successfully parsed initial LLM response JSON directly.")
        except json.JSONDecodeError as e:
            logger.warning(f"Direct JSON parsing failed: {e}. Attempting cleaning heuristics...")
            logger.debug(f"Response content before cleaning:\n{cleaned_response}")
            # Heuristic 1: Remove markdown code block fences
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()

            # Heuristic 2: Find the first '{' and last '}'
            json_start = -1
            json_end = -1
            first_brace = cleaned_response.find('{')
            first_bracket = cleaned_response.find('[')
            
            if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
                json_start = first_brace
                json_end = cleaned_response.rfind('}')
            elif first_bracket != -1:
                json_start = first_bracket
                json_end = cleaned_response.rfind(']')

            if json_start != -1 and json_end != -1:
                json_str = cleaned_response[json_start : json_end + 1]
                logger.debug(f"Attempting to parse cleaned JSON string:\n{json_str}")
                try:
                    raw_data = json.loads(json_str)
                    logger.info("Successfully parsed initial LLM response JSON after cleaning.")
                except json.JSONDecodeError as e_clean:
                    logger.error(f"JSON parsing failed even after cleaning: {e_clean}", exc_info=True)
                    logger.error(f"Original response causing failure:\n{response_content}")
                    logger.error(f"Cleaned response causing failure:\n{json_str}")
                    raw_data = None # Ensure raw_data is None on failure
            else:
                 # Log the cleaned response if no JSON structure found
                 logger.error(f"Could not find valid JSON object structure in initial LLM response: {cleaned_response}")
                 raw_data = None # Ensure raw_data is None on failure
        
        # Check if raw_data was successfully populated after all attempts
        if raw_data is None:
             logger.error("Initial JSON parsing failed completely (direct and after cleaning).")
             return None
             
        # --- ADDED: Second LLM Call (Refinement) ---
        # logger.info("Calling refinement LLM pass...")
        # refined_data = refine_llm_extraction(client, raw_data, subject, grade)
        #
        # if refined_data is None:
        #     logger.warning("Refinement LLM pass failed. Proceeding with initial extraction data.")
        #     # Decide how to handle refinement failure: proceed with raw_data or fail?
        #     # For now, proceed with raw_data but log a warning.
        #     final_data_dict = raw_data
        # else:
        #     logger.info("Refinement LLM pass successful. Using refined data.")
        #     final_data_dict = refined_data
        # --- END Second LLM Call ---

        # Always use the initial extraction data when refinement is disabled
        final_data_dict = raw_data

        # --- Process FINAL data (Convert refined dict to Node/Edge objects) --- #
        # Use final_data_dict instead of raw_data for processing below
        try:
            nodes = []
            edges = []
            # Prepare Enum maps safely
            node_type_map = {nt.value: nt for nt in NodeType} if isinstance(NodeType, type(Enum)) else {}
            relation_type_map = {rt.value: rt for rt in RelationType} if isinstance(RelationType, type(Enum)) else {}
            default_node_type = NodeType.CONCEPT if has_node_class else 'concept'
            default_relation_type = RelationType.LOGICAL if has_edge_class else 'logical'
            DEFAULT_IMPORTANCE = 3 # Default importance score
            DEFAULT_STRENGTH = 3 # Default strength score

            # Convert entities
            if 'entities' in final_data_dict and isinstance(final_data_dict['entities'], list):
                 logger.info(f"Processing {len(final_data_dict['entities'])} refined entities.")
                 for entity_dict in final_data_dict['entities']:
                    try:
                        node_type_str = entity_dict.get('type')
                        node_type = node_type_map.get(node_type_str, default_node_type) if node_type_str else default_node_type
                        if node_type_str and node_type_str not in node_type_map:
                            logger.warning(f"Unknown entity type '{node_type_str}' received from LLM. Using default: {default_node_type}")

                        # Validate and clean media list
                        node_media = entity_dict.get('media')
                        if isinstance(node_media, list):
                            node_media = [str(item) for item in node_media if isinstance(item, (str, int, float))]
                        else:
                             node_media = None # Ensure it's None if not a list

                        # Extract topic/subtopic determined by LLM for THIS entity
                        llm_topic = entity_dict.get('topic')
                        llm_subtopic = entity_dict.get('subtopic')
                        
                        # Create the Metadata object for this node using LLM context
                        node_metadata_obj = Metadata(
                            subject=subject, # Overall subject
                            grade=grade,     # Overall grade
                            topic=str(llm_topic) if llm_topic else DEFAULT_TOPIC, # Use LLM topic or default
                            subtopic=str(llm_subtopic) if llm_subtopic else None # Use LLM subtopic
                        )

                        # --- Importance Validation ---
                        importance = DEFAULT_IMPORTANCE
                        raw_importance = entity_dict.get('importance')
                        if isinstance(raw_importance, int):
                            if 1 <= raw_importance <= 5:
                                importance = raw_importance
                            else:
                                logger.warning(f"Entity '{entity_dict.get('id')}' importance '{raw_importance}' out of range [1, 5]. Using default {DEFAULT_IMPORTANCE}.")
                        elif raw_importance is not None: # Log if not None and not int
                            logger.warning(f"Entity '{entity_dict.get('id')}' importance '{raw_importance}' is not an integer. Using default {DEFAULT_IMPORTANCE}.")
                        # --- End Importance Validation ---

                        node_data = {
                            'id': str(entity_dict['id']),
                            'title': str(entity_dict['name']),
                            'type': node_type,
                            'importance': importance, # Use validated importance
                            'order': 0, # ADDED default value for required 'order' field
                            'summary': str(entity_dict.get('context', '')),
                            'meta': [asdict(node_metadata_obj)] if node_metadata_obj else [],
                            'date': str(entity_dict['date']) if entity_dict.get('date') is not None else None,
                            'geo': str(entity_dict['geo']) if entity_dict.get('geo') is not None else None,
                            'media': node_media,
                            'chunks': []
                        }
                        nodes.append(Node.from_dict(node_data) if has_node_class else node_data)

                    except (KeyError, TypeError, ValueError) as e:
                        logger.warning(f"Skipping entity due to invalid data: {e}. Data: {entity_dict}", exc_info=True)
            else:
                logger.warning("No 'entities' list found or invalid format in refined data.")

            # Convert relationships
            if 'relationships' in final_data_dict and isinstance(final_data_dict['relationships'], list):
                 logger.info(f"Processing {len(final_data_dict['relationships'])} refined relationships.")
                 for i, rel_dict in enumerate(final_data_dict['relationships']):
                    try:
                        rel_type_str = rel_dict.get('relation_type')
                        relation_type = relation_type_map.get(rel_type_str, default_relation_type) if rel_type_str else default_relation_type
                        if rel_type_str and rel_type_str not in relation_type_map:
                            logger.warning(f"Unknown relation type '{rel_type_str}' received from LLM. Using default: {default_relation_type}")

                        # --- Strength Validation ---
                        strength = DEFAULT_STRENGTH
                        raw_strength = rel_dict.get('strength')
                        if isinstance(raw_strength, int):
                             if 1 <= raw_strength <= 5:
                                 strength = raw_strength
                             else:
                                 logger.warning(f"Relationship '{rel_dict.get('source')}->{rel_dict.get('target')}' strength '{raw_strength}' out of range [1, 5]. Using default {DEFAULT_STRENGTH}.")
                        elif raw_strength is not None: # Log if not None and not int
                             logger.warning(f"Relationship '{rel_dict.get('source')}->{rel_dict.get('target')}' strength '{raw_strength}' is not an integer. Using default {DEFAULT_STRENGTH}.")
                        # --- End Strength Validation ---

                        # Extract topic/subtopic determined by LLM for THIS relationship
                        llm_topic = rel_dict.get('topic')
                        llm_subtopic = rel_dict.get('subtopic')

                        # Create the Metadata object for this edge using LLM context
                        edge_metadata_obj = Metadata(
                            subject=subject,
                            grade=grade,
                            topic=str(llm_topic) if llm_topic else DEFAULT_TOPIC, # Use LLM topic or default
                            subtopic=str(llm_subtopic) if llm_subtopic else None
                        )

                        edge_data = {
                            'id': f"edge_{uuid.uuid4()}_{i}",
                            'source_id': str(rel_dict['source']),
                            'target_id': str(rel_dict['target']),
                            'description': str(rel_dict.get('description', '')),
                            'relation_type': relation_type,
                            'strength': strength, # Use validated strength
                            'meta': [asdict(edge_metadata_obj)] if edge_metadata_obj else [], 
                            'chunks': []
                        }
                        edges.append(Edge.from_dict(edge_data) if has_edge_class else edge_data)

                    except (KeyError, TypeError, ValueError) as e:
                        logger.warning(f"Skipping relationship due to invalid data: {e}. Data: {rel_dict}", exc_info=True)
            else:
                logger.warning("No 'relationships' list found or invalid format in refined data.")

            if not nodes:
                logger.warning("LLM response processed, but no valid nodes were extracted for this chunk.")
                # Return empty dict, as processing technically finished but yielded nothing
                return {"nodes": [], "edges": []}

            logger.info(f"Successfully parsed {len(nodes)} nodes and {len(edges)} edges from final (refined) data.")
            return {"nodes": nodes, "edges": edges}

        except Exception as e_process: # Catch errors during Node/Edge creation
            logger.error(f"An error occurred while processing the final (refined) LLM data: {e_process}", exc_info=True)
            st.error(f"Error processing LLM data structure. See logs.")
            return None

    except openai.APIError as e:
        logger.error(f"OpenRouter API Error: {e}")
        st.error(f"LLM API Error: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred calling LLM: {e}", exc_info=True)
        st.error(f"Unexpected error during LLM call. See logs.")
        return None 