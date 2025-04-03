import streamlit as st
# --- Configuration ---
st.set_page_config(layout="wide", page_title="Textbook Chapter Analyzer")

import networkx as nx
from pyvis.network import Network
import json
import openai # Keep for AuthenticationError if needed by client init
from graph_structures import Node, Edge, Metadata, NodeType, RelationType # Import the structures
import uuid # Import uuid to generate unique IDs for edges
from typing import List, Dict, Any
import os # Import os for path joining
# Import the new storage functions
from graph_storage import save_graph_to_json, load_graph_from_json
# Import the main pipeline function (commented out since we don't use it now)
# from pipeline import build_graph_from_files
import logging

# PDF Parsing отключен, так как не используется
# try:
#     import fitz  # PyMuPDF
# except ImportError:
#     st.error("PyMuPDF is not installed. Please install it: pip install pymupdf")
#     fitz = None # Set to None if import fails

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Configure logging (optional, could be done in pipeline too)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory and file paths
GRAPH_DIR = "saved_graphs" # Directory to save/load graphs
NODES_FILE = os.path.join(GRAPH_DIR, "nodes.json")
EDGES_FILE = os.path.join(GRAPH_DIR, "edges.json")

# Ensure the save directory exists
os.makedirs(GRAPH_DIR, exist_ok=True)

# --- Helper Functions ---

def create_graph(structured_data: dict, min_importance: int = 1):
    """
    Creates a NetworkX graph from structured data (lists of Node and Edge objects)
    and filters nodes by minimum importance. Assigns visual attributes based on type/strength.
    Returns the NetworkX graph object ONLY.
    """
    G = nx.DiGraph()

    if not structured_data or not isinstance(structured_data, dict):
        st.warning("Invalid structured data received.")
        return G # Return empty graph

    nodes: List[Node] = structured_data.get('nodes', [])
    edges: List[Edge] = structured_data.get('edges', [])

    if not nodes:
        st.warning("No nodes found in the structured data.")
        return G # Return empty graph

    valid_nodes = set()
    node_objects_map = {node.id: node for node in nodes}

    # --- Visual Mappings ---
    node_color_map = {
        NodeType.CONCEPT: '#1f78b4', # Blue
        NodeType.PERSON: '#33a02c',   # Green
        NodeType.EVENT: '#ff7f00'    # Orange
    }
    default_node_color = '#aaaaaa' # Grey for unknown

    edge_color_map = {
        RelationType.LOGICAL: '#888888',     # Grey
        RelationType.CAUSAL: '#e31a1c',       # Red
        RelationType.STRUCTURAL: '#1f78b4',  # Blue
        RelationType.COMPARATIVE: '#984ea3', # Purple
        RelationType.REFERENTIAL: '#4daf4a'  # Green
    }
    default_edge_color = '#cccccc' # Light grey for unknown
    # --- End Visual Mappings ---

    # Add nodes that meet the importance threshold
    for node in nodes:
        if node.importance >= min_importance:
            # Build base tooltip info
            node_tooltip_parts = [
                f"<b>Name:</b> {node.title}",
                f"<b>Type:</b> {node.type.value if hasattr(node.type, 'value') else node.type}",
                f"<b>Importance:</b> {node.importance}",
                # Removed order from tooltip for brevity, still used for layout
                # f"<b>Order:</b> {node.order}" 
            ]
            
            # Add RP Context from metadata to tooltip
            rp_contexts_str = ""
            if node.meta:
                unique_contexts = set() # Avoid duplicates in tooltip
                context_lines = []
                # Limit number of contexts shown in tooltip to avoid excessive length
                max_tooltip_contexts = 5 
                contexts_shown = 0
                for meta_item in node.meta:
                    if contexts_shown >= max_tooltip_contexts: 
                        context_lines.append("- ... (more)")
                        break 
                    meta_subject = getattr(meta_item, 'subject', 'N/A')
                    meta_grade = getattr(meta_item, 'grade', 'N/A')
                    meta_topic = getattr(meta_item, 'topic', 'N/A')
                    meta_subtopic = getattr(meta_item, 'subtopic', None)
                    # Shortened context representation for tooltip
                    context_repr = f"T: {meta_topic[:30] + '...' if len(meta_topic) > 30 else meta_topic}" 
                    if meta_subtopic:
                         context_repr += f" / S: {meta_subtopic[:25] + '...' if len(meta_subtopic) > 25 else meta_subtopic}"
                    # context_repr += f" ({meta_subject} G{meta_grade})" # Removed Subj/Grade for brevity
                    if context_repr not in unique_contexts:
                        context_lines.append(f"- {context_repr}")
                        unique_contexts.add(context_repr)
                        contexts_shown += 1
                        
                if context_lines:
                    rp_contexts_str = "<hr><b>RP Contexts:</b><br>" + "<br>".join(context_lines)
            
            # Add Summary/Context to tooltip (shortened)
            summary_str = ""
            if node.summary:
                 summary_preview = node.summary.replace('\n', ' ')[:150] # Show first 150 chars preview
                 summary_str = f"<hr><b>Summary:</b><br>{summary_preview}..."
            
            # Combine all parts for the final tooltip (title attribute)
            node_tooltip = "<br>".join(node_tooltip_parts) + rp_contexts_str + summary_str
            
            node_color = node_color_map.get(node.type, default_node_color)

            G.add_node(
                node.id,
                label=node.title, # Visible label remains the title
                title=node_tooltip, # Enhanced tooltip
                importance=node.importance,
                order=node.order,
                size=10 + node.importance * 2,
                value=node.importance,
                # context=node.summary, # Context is now in the tooltip
                color=node_color,
                meta_list=node.meta # Keep full meta list for filtering
            )
            valid_nodes.add(node.id)

    # Add edges where both source and target nodes are valid
    for edge in edges:
        if edge.source_id in valid_nodes and edge.target_id in valid_nodes:
            edge_color = edge_color_map.get(edge.relation_type, default_edge_color)
            edge_width = max(1, edge.strength * 1.5)
            # Tooltip remains detailed
            edge_tooltip = f"Type: {edge.relation_type.value}<br>Strength: {edge.strength}<br>Desc: {edge.description or 'N/A'}"

            G.add_edge(
                edge.source_id,
                edge.target_id,
                title=edge_tooltip, # Detailed tooltip
                label=edge.relation_type.value if hasattr(edge.relation_type, 'value') else str(edge.relation_type), # CHANGED: Use relation type for visible label
                color=edge_color,
                width=edge_width,
                strength=edge.strength
            )

    # Calculate positions based on order
    pos = {}
    if G.nodes:
        try:
            sorted_nodes_with_data = sorted(
                G.nodes(data=True),
                key=lambda item: (item[1].get('order', 0), item[0])
            )
        except KeyError as e:
            st.error(f"Error sorting nodes for layout: Missing attribute {e}. Check node data.")
            return G

        if not sorted_nodes_with_data:
             return G

        orders = [data.get('order', 0) for _, data in sorted_nodes_with_data]
        min_order = min(orders) if orders else 0
        max_order = max(orders) if orders else 0
        order_span = max(1, max_order - min_order)

        x_scale = 1500
        y_levels = [-250, 0, 250, -125, 125, -375, 375]
        num_y_levels = len(y_levels)

        for index, (node_id, data) in enumerate(sorted_nodes_with_data):
            order = data.get('order', 0)
            x_pos = (order - min_order) / order_span * x_scale if order_span > 0 else 0.5 * x_scale
            y_pos = y_levels[index % num_y_levels]
            pos[node_id] = (x_pos, y_pos)

    return G

def visualize_graph(G, min_strength_filter: int = 1, active_filters: dict = None):
    """Visualizes the NetworkX graph using Pyvis, applying strength and metadata filters."""
    if not G or not G.nodes:
        return ""

    net = Network(notebook=True, cdn_resources='in_line', height='700px', width='100%', directed=isinstance(G, nx.DiGraph))

    # --- Helper function to check if a node passes metadata filters ---
    def check_meta_filters(node_meta_list: List[Metadata], filters: dict) -> bool:
        if not filters: # No filters active
            return True
        if not node_meta_list: # Node has no metadata, cannot pass filters
            return False

        # Check if *any* of the node's metadata entries match *all* active filters
        for meta_item in node_meta_list:
            passes_all = True
            # Check Subject Filter
            selected_subjects = filters.get('subject')
            if selected_subjects and meta_item.subject not in selected_subjects:
                passes_all = False

            # Check Grade Filter
            selected_grades = filters.get('grade')
            if passes_all and selected_grades and meta_item.grade not in selected_grades:
                passes_all = False

            # Check Topic Filter
            selected_topics = filters.get('topic')
            if passes_all and selected_topics and meta_item.topic not in selected_topics:
                passes_all = False

            # Check Subtopic Filter (New)
            selected_subtopics = filters.get('subtopic')
            if passes_all and selected_subtopics:
                # Node's subtopic must be in the selected list.
                # If meta_item.subtopic is None, it cannot match an active subtopic filter.
                if meta_item.subtopic is None or meta_item.subtopic not in selected_subtopics:
                    passes_all = False

            if passes_all: # Found a metadata entry matching all active filters
                return True

        return False # No metadata entry matched all filters
    # --- End helper function ---

    visible_nodes = set()

    # Add nodes, applying metadata filters
    for node_id, data in G.nodes(data=True):
        node_meta = data.get('meta_list', [])
        if check_meta_filters(node_meta, active_filters):
            base_title = data.get('title', '') # Title is already enhanced in create_graph
            node_color = data.get('color', '#aaaaaa') # Ensure color is retrieved
            
            net.add_node(
                node_id,
                label=data.get('label'),
                title=base_title,
                size=data.get('size'),
                value=data.get('value'),
                color=node_color, # Pass the retrieved color
                physics=True
            )
            visible_nodes.add(node_id)

    # Add edges, applying strength filter AND checking if both nodes are visible
    for edge in G.edges(data=True):
        source, target, data = edge
        edge_strength = data.get('strength', 0)
        if edge_strength >= min_strength_filter and source in visible_nodes and target in visible_nodes:
            net.add_edge(
                source,
                target,
                title=data.get('title'),
                color=data.get('color'),
                width=data.get('width')
            )

    # Configure physics/layout using Pyvis options
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=150, spring_strength=0.08, damping=0.4)
    # Or use Barnes Hut for potentially faster layout on larger graphs:
    # net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=250, spring_strength=0.04, damping=0.09, overlap=0)
    
    # Enable physics options in the UI for tweaking
    net.show_buttons(filter_=[ "physics" ])
    
    try:
        html_path = "graph.html"
        net.save_graph(html_path)
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except Exception as e:
        st.error(f"Error generating graph visualization: {e}")
        return ""

# PDF Text Extraction Helper отключен, так как не используется
# def extract_text_from_pdf(file_content: bytes) -> str:
#     """Extracts text from PDF file content using PyMuPDF."""
#     ...

# --- Streamlit App UI ---
st.title("📚 Meta-Textbook Graph Builder")

st.markdown("Upload a Work Program (RP) file and a Book file (PDF, TXT, MD) to build the knowledge graph.")

# # --- File Uploaders (Accept PDF) ---
# upload_col1, upload_col2 = st.columns(2)
# with upload_col1:
#     # Allow PDF, TXT, MD
#     rp_file = st.file_uploader("Upload Work Program (RP)", type=["pdf", "txt", "md"], key="rp_uploader", disabled=True)
# with upload_col2:
#     # Allow PDF, TXT, MD
#     book_file = st.file_uploader("Upload Book File", type=["pdf", "txt", "md"], key="book_uploader", disabled=True)

# # --- API Key Inputs ---
# key_col1, key_col2 = st.columns(2)
# with key_col1:
#     openrouter_api_key = st.text_input("OpenRouter API Key (for Chat Model):", type="password", key="api_key_input_or")
# with key_col2:
#     # Added input for direct OpenAI API key for embeddings
#     openai_api_key_embed = st.text_input("OpenAI API Key (for Embeddings):", type="password", key="api_key_input_oai")

# # --- Processing Button ---
# process_button_clicked = st.button("🚀 Process Files & Build Graph", key="process_button", disabled=True)

# --- Filtering Controls (Now separate from main input) ---
st.sidebar.title("Display Options")
min_importance = st.sidebar.slider("Minimum Node Importance:", min_value=1, max_value=10, value=1, step=1, key="importance_slider")
min_strength = st.sidebar.slider("Minimum Edge Strength:", min_value=1, max_value=5, value=1, step=1, key="strength_slider")

# --- Save/Load Buttons (Moved to Sidebar) ---
st.sidebar.markdown("--- ")
save_button_clicked = st.sidebar.button("💾 Save Graph to JSON", key="save_button")
load_button_clicked = st.sidebar.button("📂 Load Graph from JSON", key="load_button")

# Placeholder for graph
graph_placeholder = st.empty()
graph_placeholder.markdown("<p style='text-align: center; color: grey;'>Graph will appear here after loading.</p>", unsafe_allow_html=True)

# --- Logic ---
# Handle Process button click - ОТКЛЮЧЕНО, т.к. кнопки загрузки файлов отключены
# """
# if process_button_clicked:
#     # --- Input Validation ---
#     if not openrouter_api_key:
#         st.warning("Please enter your OpenRouter API key (for Chat Model).")
#     elif not openai_api_key_embed:
#         st.warning("Please enter your OpenAI API key (for Embeddings).")
#     elif not rp_file:
#         st.warning("Please upload the Work Program (RP) file.")
#     elif not book_file:
#         st.warning("Please upload the Book file.")
#     elif fitz is None and (rp_file.type == "application/pdf" or book_file.type == "application/pdf"):
#          st.error("PyMuPDF is required to process PDF files but is not installed or failed to import.")
#     else:
#         # --- File Reading and Text Extraction ---
#         rp_content = ""
#         book_content = ""
#         valid_files = True

#         try:
#             logger.info(f"Reading RP file: {rp_file.name} (Type: {rp_file.type})")
#             rp_bytes = rp_file.read()
#             if rp_file.type == "application/pdf":
#                 rp_content = extract_text_from_pdf(rp_bytes)
#                 if not rp_content:
#                     st.error(f"Could not extract text from RP PDF: {rp_file.name}")
#                     valid_files = False
#             else: # Assume text-based (txt, md)
#                 rp_content = rp_bytes.decode("utf-8")
#         except Exception as e:
#             st.error(f"Error reading RP file {rp_file.name}: {e}")
#             logger.error(f"Error reading RP file {rp_file.name}: {e}", exc_info=True)
#             valid_files = False

#         if valid_files:
#             try:
#                 logger.info(f"Reading Book file: {book_file.name} (Type: {book_file.type})")
#                 book_bytes = book_file.read()
#                 if book_file.type == "application/pdf":
#                     book_content = extract_text_from_pdf(book_bytes)
#                     if not book_content:
#                         st.error(f"Could not extract text from Book PDF: {book_file.name}")
#                         valid_files = False
#                 else: # Assume text-based (txt, md)
#                     book_content = book_bytes.decode("utf-8")
#             except Exception as e:
#                 st.error(f"Error reading Book file {book_file.name}: {e}")
#                 logger.error(f"Error reading Book file {book_file.name}: {e}", exc_info=True)
#                 valid_files = False

#         # --- Pipeline Execution ---
#         if valid_files and rp_content and book_content:
#             with st.spinner("Processing files and building graph... This may take a while!"):
#                 try:
#                     logger.info("Calling main pipeline function...")
#                     # Pass both API keys to the pipeline function
#                     structured_data = build_graph_from_files(
#                         rp_content,
#                         book_content,
#                         openrouter_api_key=openrouter_api_key, # Key for chat model via OpenRouter
#                         openai_api_key=openai_api_key_embed   # Key for embeddings via direct OpenAI
#                     )

#                     # Check pipeline result (returns None on major internal errors now)
#                     if structured_data is None:
#                         st.error("Pipeline execution failed. Check logs for details.")
#                         st.session_state['analysis_complete'] = False
#                     elif structured_data.get("nodes") or structured_data.get("edges"): # Check if it has data
#                         st.session_state['structured_data'] = structured_data
#                         st.session_state['analysis_complete'] = True
#                         st.success("Processing complete! Displaying graph.")
#                         st.rerun()
#                     else: # Pipeline finished but returned empty dict
#                         st.warning("Pipeline finished but returned no graph data (Nodes or Edges).")
#                         st.session_state['structured_data'] = {"nodes": [], "edges": []} # Store empty data
#                         st.session_state['analysis_complete'] = True # Allow displaying empty state
#                         st.rerun()

#                 except Exception as e:
#                     st.error(f"An unexpected error occurred during pipeline execution: {e}")
#                     logger.error(f"Pipeline execution failed: {e}", exc_info=True)
#                     st.session_state['analysis_complete'] = False
#         elif valid_files: # Files were read OK but content extraction failed (e.g., empty PDF text)
#              st.warning("Could not get valid text content from one or both files.")
# """

# Handle Save/Load buttons (Now checks the sidebar button state)
if save_button_clicked:
    if 'structured_data' in st.session_state and st.session_state['structured_data']:
        save_graph_to_json(st.session_state['structured_data'], NODES_FILE, EDGES_FILE)
        # Add success message for save
        st.sidebar.success(f"Graph saved to {GRAPH_DIR}")
    else:
        st.sidebar.warning("No graph data to save.")

if load_button_clicked:
    loaded_graph_data = load_graph_from_json(NODES_FILE, EDGES_FILE)
    if loaded_graph_data and (loaded_graph_data.get("nodes") or loaded_graph_data.get("edges")):
        st.session_state['structured_data'] = loaded_graph_data
        st.session_state['analysis_complete'] = True
        # Add success message for load
        st.sidebar.success(f"Graph loaded from {GRAPH_DIR}")
        st.rerun()
    else:
        # Message if files not found or empty
        st.sidebar.warning(f"Could not load graph from {GRAPH_DIR}.")

# Display graph if analysis complete (even if data is empty)
if st.session_state.get('analysis_complete', False):
    if 'structured_data' in st.session_state:
        # --- Metadata Filtering UI (Moved to Sidebar) ---
        st.sidebar.markdown("--- ")
        st.sidebar.markdown("#### Filters")
        structured_graph_data = st.session_state['structured_data']
        all_nodes: List[Node] = structured_graph_data.get('nodes', [])
        
        # Collect all unique metadata values from all nodes
        all_subjects = set()
        all_grades = set()
        all_topics = set()
        all_subtopics = set()
        for node in all_nodes:
            if node.meta:
                for m in node.meta:
                    if hasattr(m, 'subject') and m.subject: all_subjects.add(m.subject)
                    if hasattr(m, 'grade') and m.grade: all_grades.add(m.grade)
                    if hasattr(m, 'topic') and m.topic: all_topics.add(m.topic)
                    # Only add non-None/non-empty subtopics to the filter options
                    if hasattr(m, 'subtopic') and m.subtopic: all_subtopics.add(m.subtopic)

        subjects = sorted(list(all_subjects))
        grades = sorted(list(all_grades))
        topics = sorted(list(all_topics))
        subtopics = sorted(list(all_subtopics))

        # Create multiselect widgets
        selected_subjects = st.sidebar.multiselect("Filter by Subject:", options=subjects, default=subjects, key="subject_filter")
        selected_grades = st.sidebar.multiselect("Filter by Grade:", options=grades, default=grades, key="grade_filter")
        selected_topics = st.sidebar.multiselect("Filter by Topic:", options=topics, default=topics, key="topic_filter")
        # Add subtopic filter
        selected_subtopics = st.sidebar.multiselect("Filter by Subtopic:", options=subtopics, default=subtopics, key="subtopic_filter")

        # Update active_filters dictionary
        active_filters = {
            'subject': selected_subjects,
            'grade': selected_grades,
            'topic': selected_topics,
            'subtopic': selected_subtopics # Add subtopic selections
        }
        # --- End Metadata Filtering UI ---

        # Get filter values from sidebar sliders
        current_min_importance = st.session_state.get('importance_slider', 1)
        current_min_strength = st.session_state.get('strength_slider', 1)

        # Create Graph (applies importance filter)
        G = create_graph(structured_graph_data, current_min_importance)

        if G:
            # Visualize Graph (applies strength and metadata filters)
            graph_html = visualize_graph(G, current_min_strength, active_filters)
            if graph_html:
                with graph_placeholder.container():
                    st.components.v1.html(graph_html, height=750, scrolling=True)
            else:
                 graph_placeholder.error("Could not generate graph visualization.")
        else:
             graph_placeholder.warning("No graph data to display based on the current filters.")
    else:
        graph_placeholder.warning("Analysis complete, but no data found in session state.")

# --- Display Node Details (check session state) ---
st.markdown("--- ")
st.subheader("Node Details")
if st.session_state.get('analysis_complete', False) and 'structured_data' in st.session_state and st.session_state['structured_data'] and st.session_state['structured_data'].get('nodes'):
    nodes_list: List[Node] = st.session_state['structured_data']['nodes']
    # Use title for display name, ID for mapping (ensure unique display names if needed)
    node_options = {f"{node.title} (ID: {node.id[:8]}...)": node.id for node in sorted(nodes_list, key=lambda x: x.title)}
    node_map = {node.id: node for node in nodes_list}

    selected_node_display_name = st.selectbox("Select Node to View Details:", options=list(node_options.keys()), index=None, key="node_select_detail")

    if selected_node_display_name:
        selected_node_id = node_options[selected_node_display_name]
        selected_node = node_map.get(selected_node_id)
        if selected_node:
            # --- Use a container for a card-like appearance --- 
            with st.container(border=True): # Added border=True
                st.markdown(f"**Title:** {selected_node.title}")
                st.markdown(f"**ID:** `{selected_node.id}`") # Display full ID
                st.markdown(f"**Type:** {selected_node.type.value if hasattr(selected_node.type, 'value') else selected_node.type}")
                st.markdown(f"**Importance:** {selected_node.importance}")
                st.markdown(f"**Order:** {selected_node.order if selected_node.order > 0 else 'N/A'}") # Handle order 0
                if selected_node.date: st.markdown(f"**Date:** {selected_node.date}")
                if selected_node.geo: st.markdown(f"**Geo:** {selected_node.geo}")
                if selected_node.media:
                    st.markdown("**Media:**")
                    if isinstance(selected_node.media, list):
                        for item in selected_node.media:
                            if isinstance(item, (str, int, float)):
                                st.markdown(f" - `{str(item)}`")
                            else:
                                logger.warning(f"Node {selected_node.id} has unexpected media item type: {type(item)}. Item: {item}")
                                st.markdown(f" - `[Unsupported Media Item Type: {type(item)}]`")
                    else:
                         logger.warning(f"Node {selected_node.id} has non-list media attribute: {selected_node.media}")
                         st.markdown(f" - `[Invalid Media Attribute Format]`")

                st.markdown("**Summary/Context:**")
                # Use st.text_area for potentially long summaries
                st.text_area("", value=selected_node.summary, height=150, disabled=True, key=f"summary_{selected_node.id}")

                # Updated Metadata Display
                if selected_node.meta:
                    st.markdown("**Associated RP Contexts:**")
                    # Use columns for better layout if many contexts
                    cols = st.columns(2) # Adjust number of columns as needed
                    col_idx = 0
                    unique_contexts = set() # Avoid displaying identical contexts multiple times
                    for i, meta_item in enumerate(selected_node.meta):
                        meta_subject = getattr(meta_item, 'subject', 'N/A')
                        meta_grade = getattr(meta_item, 'grade', 'N/A')
                        meta_topic = getattr(meta_item, 'topic', 'N/A')
                        meta_subtopic = getattr(meta_item, 'subtopic', None)
                        context_str = f"Subj: {meta_subject}, Grade: {meta_grade}, Topic: {meta_topic}" + (f", Subtopic: {meta_subtopic}" if meta_subtopic else "")
                        if context_str not in unique_contexts:
                            with cols[col_idx % len(cols)]:
                                 st.markdown(f"- {context_str}")
                            col_idx += 1
                            unique_contexts.add(context_str)
                else:
                    st.markdown("**Associated RP Contexts:** None")
        else:
             st.info("Selected node data not found.")
else:
    st.info("Process files or load a graph to see node details here.")
# --- End Display Node Details ---

st.markdown("--- ")
st.markdown("Built with Streamlit, OpenRouter, NetworkX, and Pyvis.") 