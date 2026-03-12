# Meta-Textbook

Интерактивная образовательная платформа с визуализацией знаний в виде графа и майндмэпов.

## Описание проекта

Meta-Textbook — веб-приложение для изучения различных предметов через интерактивные графы знаний и майндмэпы. Система использует AI (OpenRouter, GigaChat) для генерации структурированного представления учебного материала и интерактивного диалога с пользователем.

Поддерживаемые предметы: биология, физика, история.

## Основные функции

- Визуализация знаний в виде интерактивного графа (D3.js)
- Майндмэпы с возможностью расширения и углубления (Markmap)
- Чат с AI-ассистентом по теме
- Генерация графов знаний из PDF-учебников и рабочих программ
- Поддержка разных уровней сложности (школьный, университетский, продвинутый)
- REST API для работы с графами знаний

## Технический стек

### Фронтенд
- Next.js 15 / React 19
- TypeScript
- Tailwind CSS
- D3.js (визуализация графа)
- Markmap (майндмэпы)
- Vercel AI SDK

### Бэкенд
- FastAPI (Python)
- OpenRouter API (доступ к LLM)
- GigaChat API (Sber)
- ChromaDB (векторное хранилище)
- PyMuPDF (обработка PDF)

## Установка и запуск

### Требования
- Node.js 18+
- Python 3.10+
- Docker (опционально)

### Установка зависимостей

```bash
git clone https://github.com/OmgiKikov/meta-textbook.git
cd meta-textbook
```

#### Бэкенд (Python)
```bash
python -m venv .venv
source .venv/bin/activate  # На Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### Фронтенд (Next.js)
```bash
cd frontend
pnpm install  # или npm install
```

### Настройка окружения

1. Создайте файл `.env` в корне проекта:
```
OPENROUTER_API_KEY=your_openrouter_api_key
OPENAI_API_KEY=your_openai_api_key
SECRET_KEY=your_gigachat_secret
SCOPE=your_gigachat_scope
```

2. Создайте файл `.env.local` в директории `frontend/`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Запуск

#### Режим разработки

1. Бэкенд:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. Фронтенд:
```bash
cd frontend
pnpm dev
```

#### Docker

```bash
docker-compose up -d --build
```

Бэкенд: `http://localhost:1000`, Фронтенд: `http://localhost:3000`

## Структура проекта

```
├── main.py                 # FastAPI сервер, API эндпоинты
├── pipeline.py             # Пайплайн построения графа из PDF/текста
├── llm_interface.py        # Интерфейс к LLM (OpenRouter)
├── graph_structures.py     # Структуры данных графа (Node, Edge)
├── graph_storage.py        # Сохранение/загрузка графов в JSON
├── models.py               # Pydantic-модели для API
├── parsers.py              # Парсинг текста
├── utils.py                # Утилиты (чанкинг текста)
├── giga_api_call/          # Интеграция с GigaChat API
├── frontend/               # Next.js фронтенд
│   ├── app/                # Страницы и API-маршруты
│   ├── components/         # React-компоненты
│   ├── lib/                # Вспомогательные функции
│   └── styles/             # CSS
├── biology_graph/          # Граф знаний: биология
├── physics_graph/          # Граф знаний: физика
├── history_graph/          # Граф знаний: история
├── saved_graphs/           # Сохранённые графы
├── Dockerfile              # Docker: бэкенд
├── Dockerfile.server       # Docker: сервер
└── docker-compose.yml      # Docker Compose конфигурация
```
