"""Functions for saving and loading the graph data (nodes, edges) to/from JSON files."""

import json
import os
from typing import List, Dict, Any
import streamlit as st # Keep streamlit for st.success/error/warning

# Import base structures needed for type hinting and loading
# Assuming graph_structures.py is in the same directory
try:
    from graph_structures import Node, Edge
except ImportError:
    # Handle cases where execution might happen differently (e.g. tests)
    # This might need adjustment based on project structure/execution context
    print("Warning: Could not import Node/Edge from graph_structures. Assuming basic dict handling for loading.")
    Node = dict
    Edge = dict


# --- Save Function ---
def save_graph_to_json(graph_data: Dict[str, List[Any]], nodes_path: str, edges_path: str):
    """Saves graph nodes and edges to separate JSON files."""
    try:
        nodes_list = graph_data.get('nodes', [])
        edges_list = graph_data.get('edges', [])

        # Convert nodes to dicts using the method if available
        nodes_dicts = [node.to_dict() if hasattr(node, 'to_dict') else node for node in nodes_list]
        # Convert edges to dicts using the method if available
        edges_dicts = [edge.to_dict() if hasattr(edge, 'to_dict') else edge for edge in edges_list]

        # Ensure directory exists
        os.makedirs(os.path.dirname(nodes_path), exist_ok=True)
        os.makedirs(os.path.dirname(edges_path), exist_ok=True)

        # Save nodes
        with open(nodes_path, 'w', encoding='utf-8') as f:
            json.dump(nodes_dicts, f, ensure_ascii=False, indent=2)

        # Save edges
        with open(edges_path, 'w', encoding='utf-8') as f:
            json.dump(edges_dicts, f, ensure_ascii=False, indent=2)

        st.success(f"Graph saved successfully to {os.path.basename(nodes_path)} and {os.path.basename(edges_path)} in {os.path.dirname(nodes_path)}")

    except Exception as e:
        st.error(f"Error saving graph: {e}")

# --- Load Function ---
def load_graph_from_json(nodes_path: str, edges_path: str) -> Dict[str, List[Any]]:
    """Loads graph nodes and edges from JSON files."""
    loaded_data = {"nodes": [], "edges": []}
    try:
        # Load nodes
        if os.path.exists(nodes_path):
            with open(nodes_path, 'r', encoding='utf-8') as f:
                nodes_dicts = json.load(f)
                # Use from_dict if Node class has it, otherwise keep as dict
                if hasattr(Node, 'from_dict'):
                     loaded_data["nodes"] = [Node.from_dict(d) for d in nodes_dicts]
                else:
                     loaded_data["nodes"] = nodes_dicts # Keep as dicts if class/method missing
        else:
            st.warning(f"Nodes file not found: {nodes_path}")

        # Load edges
        if os.path.exists(edges_path):
            with open(edges_path, 'r', encoding='utf-8') as f:
                edges_dicts = json.load(f)
                 # Use from_dict if Edge class has it, otherwise keep as dict
                if hasattr(Edge, 'from_dict'):
                    loaded_data["edges"] = [Edge.from_dict(d) for d in edges_dicts]
                else:
                    loaded_data["edges"] = edges_dicts # Keep as dicts if class/method missing
        else:
            st.warning(f"Edges file not found: {edges_path}")

        if loaded_data["nodes"] or loaded_data["edges"]:
            st.success("Graph loaded successfully.")
        return loaded_data

    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from file: {e}")
        return {"nodes": [], "edges": []}
    except Exception as e:
        st.error(f"Error loading graph: {e}")
        return {"nodes": [], "edges": []} 