from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal, Union, Dict, Any, Type
from enum import Enum
import uuid # Keep uuid import if needed elsewhere, though not strictly needed for these methods

# --- Enums for controlled vocabularies ---

class NodeType(Enum):
    """Type of a node (entity) in the graph."""
    CONCEPT = "concept"
    PERSON = "person"
    EVENT = "event"
    LOCATION = "location"
    ORGANIZATION = "organization"
    TERM = "term"

class RelationType(Enum):
    """Type of relationship (edge) between nodes."""
    LOGICAL = "logical"
    CAUSAL = "causal"
    STRUCTURAL = "structural"
    COMPARATIVE = "comparative"
    REFERENTIAL = "referential"

# --- Metadata Structure ---

@dataclass
class Metadata:
    """Metadata associating nodes/edges with educational context."""
    subject: str  # e.g., "Биология"
    grade: str    # e.g., "10"
    topic: str    # e.g., "Биология клетки"
    subtopic: Optional[str] = None # e.g., "Основные положения современной клеточной теории"

    def to_dict(self) -> Dict[str, Any]:
        """Converts Metadata object to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls: Type['Metadata'], data: Dict[str, Any]) -> 'Metadata':
        """Creates Metadata object from a dictionary."""
        return cls(**data)

# --- Node (Vertex) Structure ---

@dataclass
class Node:
    """Represents a node (vertex) in the knowledge graph."""
    id: str  # Unique identifier (UUID recommended, represented as string)
    type: NodeType # Type of the entity
    title: str # Primary name/label of the node (e.g., "Фотосинтез")
    summary: str # Short explanation/definition
    # Fields added for compatibility with current app.py logic
    importance: int # Importance score (1-10), used for filtering/sizing
    order: int # Sequential order of appearance, used for layout
    # --- Original fields from spec ---
    chunks: List[str] = field(default_factory=list) # List of source chunk IDs providing definition/context
    meta: List[Metadata] = field(default_factory=list) # List of metadata contexts
    # Optional fields
    date: Optional[str] = None # Date or date range associated with the node (e.g., "17.08.1771")
    geo: Optional[str] = None # Geographic location or coordinates
    media: Optional[List[str]] = field(default_factory=list) # List of paths/URLs to associated media (images, etc.)

    def to_dict(self) -> Dict[str, Any]:
        """Converts Node object to a dictionary suitable for JSON."""
        data = asdict(self)
        data['type'] = self.type.value # Convert enum to string
        data['meta'] = [m.to_dict() for m in self.meta] # Convert list of Metadata objects
        return data

    @classmethod
    def from_dict(cls: Type['Node'], data: Dict[str, Any]) -> 'Node':
        """Creates Node object from a dictionary."""
        data['type'] = NodeType(data['type']) # Convert string back to enum
        data['meta'] = [Metadata.from_dict(m) for m in data.get('meta', [])] # Convert list of dicts back to Metadata objects
        # Handle optional fields that might be missing if saved from an older version
        data.setdefault('chunks', [])
        data.setdefault('date', None)
        data.setdefault('geo', None)
        data.setdefault('media', [])
        return cls(**data)

# --- Edge (Relationship) Structure ---

@dataclass
class Edge:
    """Represents an edge (relationship) between two nodes in the graph."""
    id: str # Unique identifier (UUID recommended, represented as string)
    source_id: str # ID of the source node
    target_id: str # ID of the target node
    relation_type: RelationType # Type of the relationship
    strength: int # Strength of the relationship (1-5)
    # Field added for compatibility with current app.py logic
    description: Optional[str] = None # Original description from LLM, used for edge label/title
    # --- Original fields from spec ---
    chunks: List[str] = field(default_factory=list) # List of source chunk IDs supporting the relationship
    meta: List[Metadata] = field(default_factory=list) # List of metadata contexts relevant to this edge

    def __post_init__(self):
        # Validate strength is within the allowed range
        if not 1 <= self.strength <= 5:
            raise ValueError(f"Edge strength must be between 1 and 5, got {self.strength}")

    def to_dict(self) -> Dict[str, Any]:
        """Converts Edge object to a dictionary suitable for JSON."""
        data = asdict(self)
        data['relation_type'] = self.relation_type.value # Convert enum to string
        data['meta'] = [m.to_dict() for m in self.meta] # Convert list of Metadata objects
        return data

    @classmethod
    def from_dict(cls: Type['Edge'], data: Dict[str, Any]) -> 'Edge':
        """Creates Edge object from a dictionary."""
        data['relation_type'] = RelationType(data['relation_type']) # Convert string back to enum
        data['meta'] = [Metadata.from_dict(m) for m in data.get('meta', [])] # Convert list of dicts back to Metadata objects
        # Handle optional fields
        data.setdefault('description', None)
        data.setdefault('chunks', [])
        # Validate strength after creation
        instance = cls(**data)
        return instance
