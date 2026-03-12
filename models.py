from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class NodeTypeEnum(str, Enum):
    CONCEPT = "concept"
    PERSON = "person"
    EVENT = "event"
    LOCATION = "location"
    ORGANIZATION = "organization"
    TERM = "term"
    PROCESS = "process"
    OBJECT = "object"
    PLACE = "place"
    WORK = "work"
    THEORY = "theory"


class RelationTypeEnum(str, Enum):
    LOGICAL = "logical"
    CAUSAL = "causal"
    STRUCTURAL = "structural"
    COMPARATIVE = "comparative"
    REFERENTIAL = "referential"
    IS_A = "is_a"
    PART_OF = "part_of"
    EXPLAINS = "explains"
    EXAMPLE_OF = "example_of"
    CONTRASTS_WITH = "contrasts_with"
    LEADS_TO = "leads_to"
    ANALOGOUS_TO = "analogous_to"


class MetadataModel(BaseModel):
    subject: str
    grade: str
    topic: str
    subtopic: Optional[str] = None


class NodeModel(BaseModel):
    id: str
    type: NodeTypeEnum
    title: str
    summary: str
    importance: int = Field(ge=1, le=10)
    order: int
    chunks: List[str] = []
    meta: List[MetadataModel] = []
    date: Optional[str] = None
    geo: Optional[str] = None
    description: Optional[str] = None
    media: List[str] = []


class EdgeModel(BaseModel):
    id: str
    source_id: str
    target_id: str
    relation_type: RelationTypeEnum
    strength: int = Field(ge=1, le=5)
    description: Optional[str] = None
    chunks: List[str] = []
    meta: List[MetadataModel] = []


class GraphModel(BaseModel):
    nodes: List[NodeModel]
    edges: List[EdgeModel]


class NodeCreateModel(BaseModel):
    type: NodeTypeEnum
    title: str
    summary: str
    importance: int = Field(ge=1, le=10)
    order: Optional[int] = None
    chunks: List[str] = []
    meta: List[MetadataModel] = []
    date: Optional[str] = None
    geo: Optional[str] = None
    description: Optional[str] = None
    media: List[str] = []


class EdgeCreateModel(BaseModel):
    source_id: str
    target_id: str
    relation_type: RelationTypeEnum
    strength: int = Field(ge=1, le=5)
    description: Optional[str] = None
    chunks: List[str] = []
    meta: List[MetadataModel] = []


class SearchParams(BaseModel):
    query: str
    metadata_filter: Optional[Dict[str, str]] = None
    type_filter: Optional[List[NodeTypeEnum]] = None


class ErrorResponse(BaseModel):
    detail: str


class SearchResultItem(BaseModel):
    id: str
    label: str
    type: NodeTypeEnum


class SearchResultsModel(BaseModel):
    results: List[SearchResultItem]


# Модели для генерации вопросов
class GenerateQuestionsRequest(BaseModel):
    node_id: str


class GenerateQuestionsResponse(BaseModel):
    success: bool
    questions: List[str] 