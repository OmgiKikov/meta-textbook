from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Body, Path, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Dict, Any, Optional
import os
import json
import uuid
import logging
import re  # для parse_questions
from openai import OpenAI  # OpenRouter пример клиента
from dotenv import load_dotenv  # для загрузки переменных окружения
from fastapi.staticfiles import StaticFiles  # для сервинга статических файлов
from pydantic import BaseModel

# Импортируем модули из существующего проекта
from graph_structures import Node, Edge, Metadata, NodeType, RelationType
from graph_storage import save_graph_to_json, load_graph_from_json
from models import NodeModel, EdgeModel, GraphModel, NodeCreateModel, EdgeCreateModel, SearchParams, ErrorResponse, NodeTypeEnum, RelationTypeEnum, MetadataModel, SearchResultItem, SearchResultsModel, GenerateQuestionsRequest, GenerateQuestionsResponse
from pipeline import build_graph_from_files  # Импорт функции сборки графа из pipeline
try:
    import fitz  # PyMuPDF для обработки PDF
except ImportError:
    fitz = None
from langchain_gigachat import GigaChat
from langchain.schema import HumanMessage, SystemMessage
from giga_api_call.utils import GigaChatAPIWrapper
import time

giga = GigaChatAPIWrapper()

def chat_with_gigachat(giga, system_prompt, user_message):
        messages = [
            SystemMessage(
                content=system_prompt
            ),
            HumanMessage(
                content=user_message
            )
        ]

        res = giga.chat.invoke(messages, temperature=0.1)
        return res.content

def chat_with_gigachat_user(giga, user_message):
    try:
        messages = [
            HumanMessage(
                content=user_message
            )
        ]

        # Устанавливаем таймаут 20 секунд
        start_time = time.time()
        res = giga.chat.invoke(messages, temperature=0.1)
        
        # Проверяем, не превышен ли таймаут
        elapsed_time = time.time() - start_time
        if elapsed_time > 20:
            logger.warning(f"Запрос к GigaChat занял {elapsed_time:.2f} секунд")
        
        # Проверяем корректность ответа
        if not res or not hasattr(res, 'content') or not res.content:
            logger.error("GigaChat вернул пустой ответ")
            raise ValueError("Пустой ответ от GigaChat")
            
        return res.content
    except Exception as e:
        logger.error(f"Ошибка при вызове GigaChat: {str(e)}")
        # Пробуем реинициализировать клиент
        try:
            logger.info("Пробуем переинициализировать клиент GigaChat")
            # Пробуем обновить токен
            giga.authorize()
            giga.create_chat_instance()
            
            # Повторяем запрос
            messages = [
                HumanMessage(
                    content=user_message
                )
            ]
            res = giga.chat.invoke(messages, temperature=0.1)
            return res.content
        except Exception as retry_error:
            logger.error(f"Повторная ошибка при вызове GigaChat: {str(retry_error)}")
            raise ValueError(f"Не удалось получить ответ от GigaChat: {str(e)}, повторная попытка: {str(retry_error)}")

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Meta Textbook API", 
              description="API для работы с графами знаний",
              version="1.0.0")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить все источники (в продакшн стоит настроить конкретные домены)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем статику (изображения, файлы фронтенда)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Конфигурация ---
GRAPH_DIR = "biology_graph"
NODES_FILE = os.path.join(GRAPH_DIR, "nodes.json")
EDGES_FILE = os.path.join(GRAPH_DIR, "edges.json")
IMAGES_DIR = "static/biology-images"  # Директория для изображений
# Путь к файлу с контекстами буллетов
chunk_dict_path = os.path.join(GRAPH_DIR, "new_dict_chunk.json")
# Загружаем данные из JSON-файла
with open(chunk_dict_path, 'r', encoding='utf-8') as f:
    chunk_dict = json.load(f)

# Ensure the directories exist
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs("static/physics-images", exist_ok=True)
os.makedirs("static/biology-images", exist_ok=True)
os.makedirs("static/history-images", exist_ok=True)

# Загрузка переменных окружения и настройка OpenRouter клиента
load_dotenv()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
openrouter_api_key_env = os.getenv("OPENROUTER_API_KEY")
openrouter_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=openrouter_api_key_env)

# Кеш для сгенерированных вопросов: node_id -> List[str]
saved_questions: Dict[str, List[str]] = {}

# Вспомогательные функции

def convert_node_model_to_node(node_model: NodeCreateModel) -> Node:
    """Конвертирует Pydantic модель в объект Node."""
    node_id = str(uuid.uuid4())
    
    # Конвертирует метаданные из Pydantic модели в объекты Metadata
    meta_list = []
    for meta_model in node_model.meta:
        meta = Metadata(
            subject=meta_model.subject,
            grade=meta_model.grade,
            topic=meta_model.topic,
            subtopic=meta_model.subtopic
        )
        meta_list.append(meta)
    
    # Конвертирует тип из строки в enum NodeType
    node_type = NodeType(node_model.type.value)
    
    # Если order не указан, генерируем новый (можно заменить на более умную логику)
    order = node_model.order if node_model.order is not None else 0
    
    return Node(
        id=node_id,
        type=node_type,
        title=node_model.title,
        summary=node_model.summary,
        importance=node_model.importance,
        order=order,
        chunks=node_model.chunks,
        meta=meta_list,
        date=node_model.date,
        geo=node_model.geo,
        description=node_model.description,
        media=node_model.media
    )

def convert_edge_model_to_edge(edge_model: EdgeCreateModel) -> Edge:
    """Конвертирует Pydantic модель в объект Edge."""
    edge_id = str(uuid.uuid4())
    
    # Конвертирует метаданные из Pydantic модели в объекты Metadata
    meta_list = []
    for meta_model in edge_model.meta:
        meta = Metadata(
            subject=meta_model.subject,
            grade=meta_model.grade,
            topic=meta_model.topic,
            subtopic=meta_model.subtopic
        )
        meta_list.append(meta)
    
    # Конвертирует тип из строки в enum RelationType
    relation_type = RelationType(edge_model.relation_type.value)
    
    return Edge(
        id=edge_id,
        source_id=edge_model.source_id,
        target_id=edge_model.target_id,
        relation_type=relation_type,
        strength=edge_model.strength,
        description=edge_model.description,
        chunks=edge_model.chunks,
        meta=meta_list
    )

def extract_text_from_pdf(file_content: bytes) -> str:
    """Извлекает текст из PDF файла используя PyMuPDF."""
    text = ""
    if not fitz:
        logger.error("PDF обработка требует PyMuPDF. Установите: pip install pymupdf")
        return ""
    try:
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        logger.error(f"Ошибка чтения PDF файла: {e}")
        return ""
    return text

# Функция для парсинга ответов из LLM в список вопросов
def parse_questions(text: str) -> List[str]:
    """Извлекает вопросы из текста, где каждый вопрос начинается с номера и точки"""
    # Убираем \n и прочие лишние символы
    cleaned_text = text.replace("\\n", "\n")
    # Ищем строки, начинающиеся с чисел и точки
    matches = re.findall(r'^\d+\.\s+(.*?)(?=\n\d+\.|\Z)', cleaned_text, re.DOTALL | re.MULTILINE)
    return [q.strip().replace('\n', ' ') for q in matches]

# Простой тестовый эндпоинт
@app.get("/")
async def root():
    return {"message": "Meta Textbook API is running"}

# Эндпоинт для получения данных графа
@app.get("/api/graph/data", response_model=GraphModel)
async def get_graph_data():
    try:
        graph_data = load_graph_from_json(NODES_FILE, EDGES_FILE)
        # Преобразуем объекты Node и Edge в словари
        result = {
            "nodes": [node.to_dict() for node in graph_data["nodes"]],
            "edges": [edge.to_dict() for edge in graph_data["edges"]]
        }
        return result
    except Exception as e:
        logger.error(f"Error getting graph data: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting graph data: {str(e)}")

# Эндпоинты для работы с узлами (nodes)

# Эндпоинт для получения информации о конкретном узле
@app.get("/api/graph/node/{node_id}", response_model=NodeModel)
async def get_node(node_id: str = Path(..., description="Идентификатор узла")):
    try:
        graph_data = load_graph_from_json(NODES_FILE, EDGES_FILE)
        
        # Ищем узел по ID
        for node in graph_data["nodes"]:
            if node.id == node_id:
                return node.to_dict()
        
        # Если узел не найден, возвращаем 404
        raise HTTPException(status_code=404, detail=f"Node with ID {node_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting node with ID {node_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting node: {str(e)}")

# Эндпоинт для создания нового узла
@app.post("/api/graph/node", response_model=NodeModel, status_code=201)
async def create_node(node_data: NodeCreateModel):
    try:
        # Загружаем текущий граф
        graph_data = load_graph_from_json(NODES_FILE, EDGES_FILE)
        
        # Создаем новый узел
        new_node = convert_node_model_to_node(node_data)
        
        # Добавляем узел в граф
        graph_data["nodes"].append(new_node)
        
        # Сохраняем обновленный граф
        success = save_graph_to_json(graph_data, NODES_FILE, EDGES_FILE)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save graph data")
        
        # Возвращаем созданный узел
        return new_node.to_dict()
    except Exception as e:
        logger.error(f"Error creating node: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating node: {str(e)}")

# Эндпоинт для обновления узла
@app.put("/api/graph/node/{node_id}", response_model=NodeModel)
async def update_node(
    node_data: NodeModel,
    node_id: str = Path(..., description="Идентификатор узла")
):
    try:
        # Загружаем текущий граф
        graph_data = load_graph_from_json(NODES_FILE, EDGES_FILE)
        
        # Ищем узел для обновления
        node_found = False
        for i, node in enumerate(graph_data["nodes"]):
            if node.id == node_id:
                # Конвертируем метаданные из Pydantic модели в объекты Metadata
                meta_list = []
                for meta_model in node_data.meta:
                    meta = Metadata(
                        subject=meta_model.subject,
                        grade=meta_model.grade,
                        topic=meta_model.topic,
                        subtopic=meta_model.subtopic
                    )
                    meta_list.append(meta)
                
                # Обновляем узел
                node_type = NodeType(node_data.type.value)
                updated_node = Node(
                    id=node_id,
                    type=node_type,
                    title=node_data.title,
                    summary=node_data.summary,
                    importance=node_data.importance,
                    order=node_data.order,
                    chunks=node_data.chunks,
                    meta=meta_list,
                    date=node_data.date,
                    geo=node_data.geo,
                    description=node_data.description,
                    media=node_data.media
                )
                
                graph_data["nodes"][i] = updated_node
                node_found = True
                break
        
        if not node_found:
            raise HTTPException(status_code=404, detail=f"Node with ID {node_id} not found")
        
        # Сохраняем обновленный граф
        success = save_graph_to_json(graph_data, NODES_FILE, EDGES_FILE)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save graph data")
        
        # Возвращаем обновленный узел
        return updated_node.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating node with ID {node_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating node: {str(e)}")

# Эндпоинт для удаления узла
@app.delete("/api/graph/node/{node_id}", status_code=204)
async def delete_node(node_id: str = Path(..., description="Идентификатор узла")):
    try:
        # Загружаем текущий граф
        graph_data = load_graph_from_json(NODES_FILE, EDGES_FILE)
        
        # Ищем узел для удаления
        node_found = False
        for i, node in enumerate(graph_data["nodes"]):
            if node.id == node_id:
                # Удаляем узел из списка
                graph_data["nodes"].pop(i)
                node_found = True
                break
        
        if not node_found:
            raise HTTPException(status_code=404, detail=f"Node with ID {node_id} not found")
        
        # Также нужно удалить все связи, связанные с этим узлом
        edges_to_keep = []
        for edge in graph_data["edges"]:
            if edge.source_id != node_id and edge.target_id != node_id:
                edges_to_keep.append(edge)
        
        graph_data["edges"] = edges_to_keep
        
        # Сохраняем обновленный граф
        success = save_graph_to_json(graph_data, NODES_FILE, EDGES_FILE)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save graph data")
        
        # Возвращаем 204 No Content
        return Response(status_code=204)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting node with ID {node_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting node: {str(e)}")

# Эндпоинты для работы со связями (edges)

# Эндпоинт для получения информации о конкретной связи
@app.get("/api/graph/edge/{edge_id}", response_model=EdgeModel)
async def get_edge(edge_id: str = Path(..., description="Идентификатор связи")):
    try:
        graph_data = load_graph_from_json(NODES_FILE, EDGES_FILE)
        
        # Ищем связь по ID
        for edge in graph_data["edges"]:
            if edge.id == edge_id:
                return edge.to_dict()
        
        # Если связь не найдена, возвращаем 404
        raise HTTPException(status_code=404, detail=f"Edge with ID {edge_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting edge with ID {edge_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting edge: {str(e)}")

# Эндпоинт для создания новой связи
@app.post("/api/graph/edge", response_model=EdgeModel, status_code=201)
async def create_edge(edge_data: EdgeCreateModel):
    try:
        # Загружаем текущий граф
        graph_data = load_graph_from_json(NODES_FILE, EDGES_FILE)
        
        # Проверяем, существуют ли указанные узлы
        source_exists = False
        target_exists = False
        
        for node in graph_data["nodes"]:
            if node.id == edge_data.source_id:
                source_exists = True
            if node.id == edge_data.target_id:
                target_exists = True
            
            if source_exists and target_exists:
                break
        
        if not source_exists:
            raise HTTPException(status_code=404, detail=f"Source node with ID {edge_data.source_id} not found")
        if not target_exists:
            raise HTTPException(status_code=404, detail=f"Target node with ID {edge_data.target_id} not found")
        
        # Создаем новую связь
        new_edge = convert_edge_model_to_edge(edge_data)
        
        # Добавляем связь в граф
        graph_data["edges"].append(new_edge)
        
        # Сохраняем обновленный граф
        success = save_graph_to_json(graph_data, NODES_FILE, EDGES_FILE)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save graph data")
        
        # Возвращаем созданную связь
        return new_edge.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating edge: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating edge: {str(e)}")

# Эндпоинт для обновления связи
@app.put("/api/graph/edge/{edge_id}", response_model=EdgeModel)
async def update_edge(
    edge_data: EdgeModel,
    edge_id: str = Path(..., description="Идентификатор связи")
):
    try:
        # Загружаем текущий граф
        graph_data = load_graph_from_json(NODES_FILE, EDGES_FILE)
        
        # Проверяем, существуют ли указанные узлы
        source_exists = False
        target_exists = False
        
        for node in graph_data["nodes"]:
            if node.id == edge_data.source_id:
                source_exists = True
            if node.id == edge_data.target_id:
                target_exists = True
            
            if source_exists and target_exists:
                break
        
        if not source_exists:
            raise HTTPException(status_code=404, detail=f"Source node with ID {edge_data.source_id} not found")
        if not target_exists:
            raise HTTPException(status_code=404, detail=f"Target node with ID {edge_data.target_id} not found")
        
        # Ищем связь для обновления
        edge_found = False
        for i, edge in enumerate(graph_data["edges"]):
            if edge.id == edge_id:
                # Конвертируем метаданные из Pydantic модели в объекты Metadata
                meta_list = []
                for meta_model in edge_data.meta:
                    meta = Metadata(
                        subject=meta_model.subject,
                        grade=meta_model.grade,
                        topic=meta_model.topic,
                        subtopic=meta_model.subtopic
                    )
                    meta_list.append(meta)
                
                # Обновляем связь
                relation_type = RelationType(edge_data.relation_type.value)
                updated_edge = Edge(
                    id=edge_id,
                    source_id=edge_data.source_id,
                    target_id=edge_data.target_id,
                    relation_type=relation_type,
                    strength=edge_data.strength,
                    description=edge_data.description,
                    chunks=edge_data.chunks,
                    meta=meta_list
                )
                
                graph_data["edges"][i] = updated_edge
                edge_found = True
                break
        
        if not edge_found:
            raise HTTPException(status_code=404, detail=f"Edge with ID {edge_id} not found")
        
        # Сохраняем обновленный граф
        success = save_graph_to_json(graph_data, NODES_FILE, EDGES_FILE)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save graph data")
        
        # Возвращаем обновленную связь
        return updated_edge.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating edge with ID {edge_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating edge: {str(e)}")

# Эндпоинт для удаления связи
@app.delete("/api/graph/edge/{edge_id}", status_code=204)
async def delete_edge(edge_id: str = Path(..., description="Идентификатор связи")):
    try:
        # Загружаем текущий граф
        graph_data = load_graph_from_json(NODES_FILE, EDGES_FILE)
        
        # Ищем связь для удаления
        edge_found = False
        for i, edge in enumerate(graph_data["edges"]):
            if edge.id == edge_id:
                # Удаляем связь из списка
                graph_data["edges"].pop(i)
                edge_found = True
                break
        
        if not edge_found:
            raise HTTPException(status_code=404, detail=f"Edge with ID {edge_id} not found")
        
        # Сохраняем обновленный граф
        success = save_graph_to_json(graph_data, NODES_FILE, EDGES_FILE)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save graph data")
        
        # Возвращаем 204 No Content
        return Response(status_code=204)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting edge with ID {edge_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting edge: {str(e)}")

# Эндпоинт для поиска узлов по названию
@app.get("/api/graph/search", response_model=SearchResultsModel)
async def search_nodes(query: str = Query(..., description="Текст для поиска узлов")):
    # Загружаем граф
    graph_data = load_graph_from_json(NODES_FILE, EDGES_FILE)
    # Фильтруем узлы по подстроке названия
    results = []
    for node in graph_data.get("nodes", []):
        if query.lower() in node.title.lower():
            results.append(SearchResultItem(
                id=node.id,
                label=node.title,
                type=NodeTypeEnum(node.type.value)
            ))
    return SearchResultsModel(results=results)

# Добавляю endpoint для загрузки файлов и построения графа
@app.post("/process")
async def process_files(
    openrouter_api_key: str = Form(..., description="API ключ OpenRouter"),
    openai_api_key: str = Form(..., description="API ключ OpenAI"),
    rp_file: UploadFile = File(..., description="Файл РП"),
    book_file: UploadFile = File(..., description="Файл книги")
):
    try:
        # Чтение и извлечение текста из RP
        rp_bytes = await rp_file.read()
        if rp_file.content_type == "application/pdf":
            rp_content = extract_text_from_pdf(rp_bytes)
            if not rp_content:
                raise HTTPException(status_code=400, detail=f"Не удалось извлечь текст из PDF: {rp_file.filename}")
        else:
            rp_content = rp_bytes.decode("utf-8")
        # Чтение и извлечение текста из книги
        book_bytes = await book_file.read()
        if book_file.content_type == "application/pdf":
            book_content = extract_text_from_pdf(book_bytes)
            if not book_content:
                raise HTTPException(status_code=400, detail=f"Не удалось извлечь текст из PDF: {book_file.filename}")
        else:
            book_content = book_bytes.decode("utf-8")

        # Построение графа
        structured_data = build_graph_from_files(
            rp_content,
            book_content,
            openrouter_api_key=openrouter_api_key,
            openai_api_key=openai_api_key
        )
        if not structured_data:
            raise HTTPException(status_code=500, detail="Ошибка выполнения pipeline")

        # Сериализация данных для ответа
        serializable_data = {
            "nodes": [
                {
                    "id": node.id,
                    "title": node.title,
                    "type": node.type.value if hasattr(node.type, "value") else str(node.type),
                    "importance": node.importance,
                    "order": node.order,
                    "meta": [
                        {"subject": m.subject, "grade": m.grade, "topic": m.topic, "subtopic": m.subtopic}
                        for m in node.meta
                    ] if node.meta else []
                }
                for node in structured_data.get("nodes", [])
            ],
            "edges": [
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relation_type": edge.relation_type.value if hasattr(edge.relation_type, "value") else str(edge.relation_type),
                    "strength": edge.strength,
                    "description": edge.description
                }
                for edge in structured_data.get("edges", [])
            ]
        }

        # Сохранение графа
        if not save_graph_to_json(structured_data, NODES_FILE, EDGES_FILE):
            raise HTTPException(status_code=500, detail="Не удалось сохранить граф")

        return {"success": True, "message": "Обработка завершена", "data": serializable_data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /process: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Эндпоинт для получения опций метаданных
@app.get("/api/graph/metadata_options")
async def get_metadata_options():
    try:
        # Логируем вызов эндпоинта
        logger.info("Запрос метаданных для фильтров")
        
        graph_data = load_graph_from_json(NODES_FILE, EDGES_FILE)
        if not graph_data or "nodes" not in graph_data:
            logger.warning("Не удалось загрузить данные графа или граф пуст")
            # Возвращаем фиксированные данные вместо ошибки
            return {
                "success": True,
                "options": {
                    "subjects": ["Математика", "Физика", "Биология", "История России", "Информатика"],
                    "grades": ["5 класс", "6 класс", "7 класс", "8 класс"],
                    "topics": ["Алгебра", "Геометрия", "Механика", "Электричество"],
                    "subtopics": ["Уравнения", "Тригонометрия", "Вектор", "Магнитное поле"]
                }
            }
        
        all_subjects = set()
        all_grades = set()
        all_topics = set()
        all_subtopics = set()
        
        # Словари для подсчета количества узлов и связей по темам и подтемам
        topic_node_counts = {}
        subtopic_node_counts = {}
        topic_edge_counts = {}
        subtopic_edge_counts = {}
        
        # Логируем количество узлов
        logger.info(f"Количество узлов для анализа метаданных: {len(graph_data['nodes'])}")
        
        # Счетчик узлов с метаданными
        nodes_with_meta = 0
        
        # Собираем темы и подтемы из узлов и считаем их количество
        for node in graph_data["nodes"]:
            if node.meta:
                nodes_with_meta += 1
                for m in node.meta:
                    if hasattr(m, 'subject') and m.subject: all_subjects.add(m.subject)
                    if hasattr(m, 'grade') and m.grade: all_grades.add(m.grade)
                    
                    # Подсчет узлов для тем
                    if hasattr(m, 'topic') and m.topic: 
                        all_topics.add(m.topic)
                        if m.topic in topic_node_counts:
                            topic_node_counts[m.topic] += 1
                        else:
                            topic_node_counts[m.topic] = 1
                    
                    # Подсчет узлов для подтем
                    if hasattr(m, 'subtopic') and m.subtopic: 
                        all_subtopics.add(m.subtopic)
                        if m.subtopic in subtopic_node_counts:
                            subtopic_node_counts[m.subtopic] += 1
                        else:
                            subtopic_node_counts[m.subtopic] = 1
        
        # Подсчет связей для тем и подтем
        for edge in graph_data.get("edges", []):
            # Найдем тему и подтему источника
            source_topic = None
            source_subtopic = None
            target_topic = None
            target_subtopic = None
            
            # Найдем узлы источника и цели
            source_node = next((n for n in graph_data["nodes"] if n.id == edge.source_id), None)
            target_node = next((n for n in graph_data["nodes"] if n.id == edge.target_id), None)
            
            # Получим тему и подтему из метаданных узлов
            if source_node and source_node.meta:
                for m in source_node.meta:
                    if hasattr(m, 'topic'): source_topic = m.topic
                    if hasattr(m, 'subtopic'): source_subtopic = m.subtopic
            
            if target_node and target_node.meta:
                for m in target_node.meta:
                    if hasattr(m, 'topic'): target_topic = m.topic
                    if hasattr(m, 'subtopic'): target_subtopic = m.subtopic
            
            # Увеличиваем счетчики связей для тем и подтем
            if source_topic:
                if source_topic in topic_edge_counts:
                    topic_edge_counts[source_topic] += 1
                else:
                    topic_edge_counts[source_topic] = 1
                    
            if target_topic and target_topic != source_topic:
                if target_topic in topic_edge_counts:
                    topic_edge_counts[target_topic] += 1
                else:
                    topic_edge_counts[target_topic] = 1
                    
            if source_subtopic:
                if source_subtopic in subtopic_edge_counts:
                    subtopic_edge_counts[source_subtopic] += 1
                else:
                    subtopic_edge_counts[source_subtopic] = 1
                    
            if target_subtopic and target_subtopic != source_subtopic:
                if target_subtopic in subtopic_edge_counts:
                    subtopic_edge_counts[target_subtopic] += 1
                else:
                    subtopic_edge_counts[target_subtopic] = 1
        
        # Фильтруем темы и подтемы с учетом минимального количества узлов и связей
        filtered_topics = [topic for topic in all_topics 
                          if topic_node_counts.get(topic, 0) >= 5 and topic_edge_counts.get(topic, 0) >= 4]
        
        filtered_subtopics = [subtopic for subtopic in all_subtopics 
                             if subtopic_node_counts.get(subtopic, 0) >= 5 and subtopic_edge_counts.get(subtopic, 0) >= 4]
        
        # Логируем результаты анализа
        logger.info(f"Узлов с метаданными: {nodes_with_meta}")
        logger.info(f"Найдено предметов: {len(all_subjects)}, классов: {len(all_grades)}, всего тем: {len(all_topics)}, всего подтем: {len(all_subtopics)}")
        logger.info(f"После фильтрации: тем: {len(filtered_topics)}, подтем: {len(filtered_subtopics)}")
        
        # Для отладки - показываем какие темы были отфильтрованы
        filtered_out_topics = all_topics - set(filtered_topics)
        filtered_out_subtopics = all_subtopics - set(filtered_subtopics)
        
        if filtered_out_topics:
            logger.info(f"Отфильтрованные темы (меньше 5 узлов или 4 связей): {', '.join(filtered_out_topics)}")
            for topic in filtered_out_topics:
                logger.info(f"Тема '{topic}': узлов - {topic_node_counts.get(topic, 0)}, связей - {topic_edge_counts.get(topic, 0)}")
        
        if filtered_out_subtopics:
            logger.info(f"Отфильтрованные подтемы (меньше 5 узлов или 4 связей): {', '.join(filtered_out_subtopics)}")
            for subtopic in filtered_out_subtopics:
                logger.info(f"Подтема '{subtopic}': узлов - {subtopic_node_counts.get(subtopic, 0)}, связей - {subtopic_edge_counts.get(subtopic, 0)}")
        
        # Если нет данных метаданных, используем фиксированные
        if not all_subjects and not all_grades and not filtered_topics and not filtered_subtopics:
            logger.warning("Не найдено данных метаданных, используем фиксированные")
            return {
                "success": True,
                "options": {
                    "subjects": ["Математика", "Физика", "Биология", "История России", "Информатика"],
                    "grades": ["5 класс", "6 класс", "7 класс", "8 класс"],
                    "topics": ["Алгебра", "Геометрия", "Механика", "Электричество"],
                    "subtopics": ["Уравнения", "Тригонометрия", "Вектор", "Магнитное поле"]
                }
            }
        
        result = {
            "success": True,
            "options": {
                "subjects": sorted(all_subjects) or ["Биология"],
                "grades": sorted(all_grades) or ["5 класс"],
                "topics": sorted(filtered_topics) or ["Экология"],
                "subtopics": sorted(filtered_subtopics) or ["Экосистемы"]
            }
        }
        
        # Логируем итоговый результат
        logger.info(f"Возвращаем опции метаданных: {result}")
        
        return result
    except Exception as e:
        logger.error(f"Ошибка при получении опций метаданных: {e}", exc_info=True)
        # В случае ошибки возвращаем фиксированные данные
        return {
            "success": True,
            "options": {
                "subjects": ["Математика", "Физика", "Биология", "История России", "Информатика"],
                "grades": ["5 класс", "6 класс", "7 класс", "8 класс"],
                "topics": ["Алгебра", "Геометрия", "Механика", "Электричество"],
                "subtopics": ["Уравнения", "Тригонометрия", "Вектор", "Магнитное поле"]
            }
        }

# Эндпоинт для получения опций узлов
@app.get("/api/graph/node_options")
async def get_node_options():
    graph_data = load_graph_from_json(NODES_FILE, EDGES_FILE)
    if not graph_data or "nodes" not in graph_data:
        raise HTTPException(status_code=400, detail="Не удалось загрузить данные графа")
    node_options = [
        {"id": node.id, "title": f"{node.title} (ID: {node.id[:8]}...)"}
        for node in sorted(graph_data["nodes"], key=lambda n: n.title)
    ]
    return {"success": True, "options": node_options}

# Эндпоинт для сохранения графа
@app.post("/api/graph/save")
async def save_graph_endpoint():
    graph_data = load_graph_from_json(NODES_FILE, EDGES_FILE)
    if not graph_data:
        raise HTTPException(status_code=400, detail="Нет данных графа для сохранения")
    if not save_graph_to_json(graph_data, NODES_FILE, EDGES_FILE):
        raise HTTPException(status_code=500, detail="Ошибка при сохранении графа")
    return {"success": True, "message": f"Граф сохранен в {GRAPH_DIR}"}

# Эндпоинт для загрузки графа
@app.get("/api/graph/load")
async def load_graph_endpoint():
    graph_data = load_graph_from_json(NODES_FILE, EDGES_FILE)
    if not graph_data:
        raise HTTPException(status_code=400, detail=f"Не удалось загрузить граф из {GRAPH_DIR}")
    result = {
        "nodes": [node.to_dict() for node in graph_data["nodes"]],
        "edges": [edge.to_dict() for edge in graph_data["edges"]]
    }
    return {"success": True, "message": f"Граф загружен из {GRAPH_DIR}", "data": result}

# Эндпоинт для генерации вопросов
@app.post("/api/graph/generate_questions", response_model=GenerateQuestionsResponse)
async def generate_questions(request: GenerateQuestionsRequest):
    try:
        node_id = request.node_id
        logger.info(f"Запрос на генерацию вопросов для узла {node_id}")
        
        # Проверяем кеш
        if node_id in saved_questions:
            logger.info(f"Возвращаем вопросы из кеша для узла {node_id}")
            return GenerateQuestionsResponse(success=True, questions=saved_questions[node_id])
        
        # Загружаем граф
        graph_data = load_graph_from_json(NODES_FILE, EDGES_FILE)
        nodes = graph_data.get("nodes", [])
        
        # Находим узел
        node = next((n for n in nodes if n.id == node_id), None)
        if not node:
            logger.warning(f"Узел с ID {node_id} не найден")
            raise HTTPException(status_code=404, detail=f"Node with ID {node_id} not found")
        
        # Формируем контекст для LLM
        context = {
            "title": node.title,
            "description": node.description or "",
            "summary": node.summary or []
        }
        
        # Собираем связанные узлы
        related_titles = []
        for edge in graph_data.get("edges", []):
            if edge.source_id == node_id:
                target = next((n for n in nodes if n.id == edge.target_id), None)
                if target:
                    related_titles.append(target.title)
            elif edge.target_id == node_id:
                source = next((n for n in nodes if n.id == edge.source_id), None)
                if source:
                    related_titles.append(source.title)
        str_related = "\n".join(related_titles)
        
        # Системное сообщение для LLM
        system_prompt = f"""Твоя задача вывести 3 вопроса, которые смогли мы проверить знания ученика по данной теме '{context['title']}'. Ты можешь брать данные из:
1) Определение:
{context['description']}
2) Резюме:
{context['summary']}
3) Связи с этой темой:
{str_related}

Формат вывода:
1. Вопрос 1
2. Вопрос 2
3. Вопрос 3"""

        logger.info("Отправляем запрос к GigaChat")
        
        try:
            # Ставим таймаут на запрос
            start_time = time.time()
            response_content = chat_with_gigachat_user(giga, system_prompt)
            logger.info(f"Получен ответ от GigaChat за {time.time() - start_time:.2f} сек")
            
            # Парсим ответ на вопросы
            questions = parse_questions(response_content)
            
            # Проверяем, что вопросы успешно извлечены
            if not questions or len(questions) == 0:
                logger.warning(f"Не удалось извлечь вопросы из ответа: {response_content[:100]}...")
                # Создаем стандартные вопросы
                questions = [
                    f"Что такое {context['title']}?",
                    f"Каковы основные характеристики {context['title']}?",
                    f"Как {context['title']} связано с другими понятиями?",
                ]
            
            # Кэшируем результат
            saved_questions[node_id] = questions
            
            # Возвращаем первые 3 вопроса (или меньше, если получено меньше)
            return GenerateQuestionsResponse(success=True, questions=questions[:3])
        except Exception as e:
            logger.error(f"Ошибка при вызове GigaChat: {str(e)}")
            
            # В случае ошибки генерируем стандартные вопросы
            fallback_questions = [
                f"Что такое {context['title']}?",
                f"Каковы основные характеристики {context['title']}?",
                f"Как {context['title']} связано с другими понятиями?",
            ]
            
            # Кэшируем и возвращаем стандартные вопросы
            saved_questions[node_id] = fallback_questions
            return GenerateQuestionsResponse(success=True, questions=fallback_questions)
    except Exception as e:
        logger.error(f"Неперехваченная ошибка при генерации вопросов: {str(e)}")
        # Возвращаем общие вопросы в случае любой непредвиденной ошибки
        return GenerateQuestionsResponse(success=True, questions=[
            "Что такое этот концепт?",
            "Каковы его основные характеристики?",
            "Как он связан с другими понятиями?",
        ])

# Класс для фильтрации графа
class FilterRequest(BaseModel):
    subjects: Optional[List[str]] = []
    grades: Optional[List[str]] = []
    topics: Optional[List[str]] = []
    subtopics: Optional[List[str]] = []

# Эндпоинт для фильтрации графа
@app.post("/api/graph/filter", response_model=GraphModel)
async def filter_graph(filter_params: FilterRequest):
    try:
        # Загружаем весь граф
        graph_data = load_graph_from_json(NODES_FILE, EDGES_FILE)
        
        # Получаем все узлы и связи
        all_nodes = graph_data["nodes"]
        all_edges = graph_data["edges"]
        
        # Проверяем, есть ли фильтры
        has_filters = (
            filter_params.subjects and len(filter_params.subjects) > 0 or
            filter_params.grades and len(filter_params.grades) > 0 or
            filter_params.topics and len(filter_params.topics) > 0 or
            filter_params.subtopics and len(filter_params.subtopics) > 0
        )
        
        # Если фильтров нет, возвращаем весь граф
        if not has_filters:
            return {
                "nodes": [node.to_dict() for node in all_nodes],
                "edges": [edge.to_dict() for edge in all_edges]
            }
        
        # Словари для подсчета количества узлов и связей по темам и подтемам
        topic_node_counts = {}
        subtopic_node_counts = {}
        
        # Собираем темы и подтемы из узлов и считаем их количество
        for node in all_nodes:
            if node.meta:
                for m in node.meta:
                    # Подсчет узлов для тем
                    if hasattr(m, 'topic') and m.topic: 
                        if m.topic in topic_node_counts:
                            topic_node_counts[m.topic] += 1
                        else:
                            topic_node_counts[m.topic] = 1
                    
                    # Подсчет узлов для подтем
                    if hasattr(m, 'subtopic') and m.subtopic: 
                        if m.subtopic in subtopic_node_counts:
                            subtopic_node_counts[m.subtopic] += 1
                        else:
                            subtopic_node_counts[m.subtopic] = 1
        
        # Словари для подсчета связей
        topic_edge_counts = {}
        subtopic_edge_counts = {}
        
        # Подсчет связей для тем и подтем
        for edge in all_edges:
            # Найдем тему и подтему источника
            source_topic = None
            source_subtopic = None
            target_topic = None
            target_subtopic = None
            
            # Найдем узлы источника и цели
            source_node = next((n for n in all_nodes if n.id == edge.source_id), None)
            target_node = next((n for n in all_nodes if n.id == edge.target_id), None)
            
            # Получим тему и подтему из метаданных узлов
            if source_node and source_node.meta:
                for m in source_node.meta:
                    if hasattr(m, 'topic'): source_topic = m.topic
                    if hasattr(m, 'subtopic'): source_subtopic = m.subtopic
            
            if target_node and target_node.meta:
                for m in target_node.meta:
                    if hasattr(m, 'topic'): target_topic = m.topic
                    if hasattr(m, 'subtopic'): target_subtopic = m.subtopic
            
            # Увеличиваем счетчики связей для тем и подтем
            if source_topic:
                if source_topic in topic_edge_counts:
                    topic_edge_counts[source_topic] += 1
                else:
                    topic_edge_counts[source_topic] = 1
                    
            if target_topic and target_topic != source_topic:
                if target_topic in topic_edge_counts:
                    topic_edge_counts[target_topic] += 1
                else:
                    topic_edge_counts[target_topic] = 1
                    
            if source_subtopic:
                if source_subtopic in subtopic_edge_counts:
                    subtopic_edge_counts[source_subtopic] += 1
                else:
                    subtopic_edge_counts[source_subtopic] = 1
                    
            if target_subtopic and target_subtopic != source_subtopic:
                if target_subtopic in subtopic_edge_counts:
                    subtopic_edge_counts[target_subtopic] += 1
                else:
                    subtopic_edge_counts[target_subtopic] = 1
        
        # Фильтруем список тем и подтем с учетом минимального количества узлов и связей
        if filter_params.topics:
            original_topics = filter_params.topics.copy()
            filter_params.topics = [topic for topic in filter_params.topics 
                                  if topic_node_counts.get(topic, 0) >= 5 and topic_edge_counts.get(topic, 0) >= 4]
            
            # Логируем отфильтрованные темы
            if len(original_topics) != len(filter_params.topics):
                removed_topics = set(original_topics) - set(filter_params.topics)
                logger.info(f"Отфильтрованы темы из запроса: {', '.join(removed_topics)}")
                for topic in removed_topics:
                    logger.info(f"Тема '{topic}': узлов - {topic_node_counts.get(topic, 0)}, связей - {topic_edge_counts.get(topic, 0)}")
        
        if filter_params.subtopics:
            original_subtopics = filter_params.subtopics.copy()
            filter_params.subtopics = [subtopic for subtopic in filter_params.subtopics 
                                     if subtopic_node_counts.get(subtopic, 0) >= 5 and subtopic_edge_counts.get(subtopic, 0) >= 4]
            
            # Логируем отфильтрованные подтемы
            if len(original_subtopics) != len(filter_params.subtopics):
                removed_subtopics = set(original_subtopics) - set(filter_params.subtopics)
                logger.info(f"Отфильтрованы подтемы из запроса: {', '.join(removed_subtopics)}")
                for subtopic in removed_subtopics:
                    logger.info(f"Подтема '{subtopic}': узлов - {subtopic_node_counts.get(subtopic, 0)}, связей - {subtopic_edge_counts.get(subtopic, 0)}")
        
        # Фильтруем узлы по метаданным
        filtered_nodes = []
        filtered_node_ids = set()
        
        for node in all_nodes:
            # Пропускаем узлы без метаданных
            if not node.meta:
                continue
                
            # Проверяем каждый элемент метаданных узла
            for meta in node.meta:
                # Флаги для проверки соответствия фильтрам
                subject_match = not filter_params.subjects or (hasattr(meta, 'subject') and meta.subject in filter_params.subjects)
                grade_match = not filter_params.grades or (hasattr(meta, 'grade') and meta.grade in filter_params.grades)
                topic_match = not filter_params.topics or (hasattr(meta, 'topic') and meta.topic in filter_params.topics)
                subtopic_match = not filter_params.subtopics or (hasattr(meta, 'subtopic') and meta.subtopic in filter_params.subtopics)
                
                # Если узел соответствует всем активным фильтрам
                if subject_match and grade_match and topic_match and subtopic_match:
                    filtered_nodes.append(node)
                    filtered_node_ids.add(node.id)
                    break  # Прерываем цикл, если нашли соответствие
        
        # Фильтруем связи, оставляя только те, которые соединяют отфильтрованные узлы
        filtered_edges = [
            edge for edge in all_edges
            if edge.source_id in filtered_node_ids and edge.target_id in filtered_node_ids
        ]
        
        logger.info(f"Фильтрация: найдено {len(filtered_nodes)} узлов и {len(filtered_edges)} связей")
        
        # Возвращаем отфильтрованный граф
        return {
            "nodes": [node.to_dict() for node in filtered_nodes],
            "edges": [edge.to_dict() for edge in filtered_edges]
        }
    except Exception as e:
        logger.error(f"Ошибка при фильтрации графа: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при фильтрации графа: {str(e)}")

# Добавляем модель для запроса контекста буллета
class BulletContextRequest(BaseModel):
    node_id: str
    chunk_id: str

# Добавляем модель для ответа с контекстом
class BulletContextResponse(BaseModel):
    success: bool
    context: str
    source: Optional[str] = None

# Добавляем эндпоинт для получения контекста буллета
@app.post("/api/graph/bullet_context", response_model=BulletContextResponse)
async def get_bullet_context(request: BulletContextRequest):
    try:
        
        # Получаем контекст по node_id и chunk_id
        node_chunks = chunk_dict.get(request.node_id, {})
        id_ch = int(request.chunk_id.split(".")[-1])
        sorted_list = sorted(list(node_chunks.keys()), reverse=False)
        chunk_id = sorted_list[id_ch]
        print(sorted_list)
        print(chunk_id)
        context = node_chunks.get(chunk_id, None)
        print(context)
        
        # Если контекст не найден
        if not context:
            return BulletContextResponse(
                success=False, 
                context=f"Контекст не найден для узла {request.node_id} и чанка {request.chunk_id}"
            )
        
        # Определяем источник (можно расширить в будущем)
        source = ""
        if isinstance(context, dict) and "source" in context:
            source = context.get("source", "")
            context_text = context.get("text", str(context))
        else:
            context_text = str(context)
        
        return BulletContextResponse(
            success=True,
            context=context_text,
            source=source
        )
    except Exception as e:
        logger.error(f"Неперехваченная ошибка при получении контекста буллета: {str(e)}")
        return BulletContextResponse(
            success=False,
            context=f"Ошибка при получении контекста: {str(e)}"
        )

# Класс для запроса на переключение графа
class SwitchGraphRequest(BaseModel):
    graph_name: str  # "biology_graph", "physics_graph" или "history_graph"

# Эндпоинт для переключения текущего графа
@app.post("/api/graph/switch")
async def switch_graph(request: SwitchGraphRequest):
    try:
        global GRAPH_DIR, NODES_FILE, EDGES_FILE, chunk_dict_path, chunk_dict, IMAGES_DIR
        
        # Проверяем, что указанный граф существует
        if request.graph_name not in ["biology_graph", "physics_graph", "history_graph"]:
            raise HTTPException(status_code=400, detail=f"Указанный граф {request.graph_name} не существует")
        
        if not os.path.isdir(request.graph_name):
            raise HTTPException(status_code=400, detail=f"Директория графа {request.graph_name} не найдена")
        
        # Устанавливаем новые пути к файлам графа
        GRAPH_DIR = request.graph_name
        NODES_FILE = os.path.join(GRAPH_DIR, "nodes.json")
        EDGES_FILE = os.path.join(GRAPH_DIR, "edges.json")
        chunk_dict_path = os.path.join(GRAPH_DIR, "new_dict_chunk.json")
        
        # Устанавливаем соответствующую директорию для изображений
        if GRAPH_DIR == "physics_graph":
            IMAGES_DIR = "static/physics-images"
        elif GRAPH_DIR == "biology_graph":
            IMAGES_DIR = "static/biology-images"
        elif GRAPH_DIR == "history_graph":
            IMAGES_DIR = "static/history-images"
        
        # Убедимся, что директория существует
        os.makedirs(IMAGES_DIR, exist_ok=True)
        
        # Перезагружаем chunk_dict
        try:
            with open(chunk_dict_path, 'r', encoding='utf-8') as f:
                chunk_dict = json.load(f)
            logger.info(f"Загружены данные чанков из {chunk_dict_path}")
        except Exception as e:
            logger.warning(f"Не удалось загрузить данные чанков из {chunk_dict_path}: {e}")
            chunk_dict = {}  # Используем пустой словарь в случае ошибки
        
        # Возвращаем успешный ответ
        return {
            "success": True, 
            "message": f"Переключено на граф {GRAPH_DIR}",
            "metadata": {
                "graph_dir": GRAPH_DIR,
                "nodes_file": NODES_FILE,
                "edges_file": EDGES_FILE,
                "images_dir": IMAGES_DIR,
                "chunks_loaded": len(chunk_dict) > 0
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при переключении графа: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при переключении графа: {str(e)}")

# Эндпоинт для получения текущего графа
@app.get("/api/graph/current")
async def get_current_graph():
    return {
        "success": True,
        "graph_name": GRAPH_DIR,
        "metadata": {
            "graph_dir": GRAPH_DIR,
            "nodes_file": NODES_FILE,
            "edges_file": EDGES_FILE,
            "images_dir": IMAGES_DIR
        }
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False) 