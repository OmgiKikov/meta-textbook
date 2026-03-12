# Meta-Textbook

Интерактивная образовательная платформа для изучения школьных предметов через графы знаний, майндмэпы и AI-чат.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Next.js](https://img.shields.io/badge/Next.js-15-black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Что это

Meta-Textbook строит интерактивные графы знаний из учебников и рабочих программ с помощью AI. Пользователь видит структуру предмета в виде графа, может углубляться в любой узел, открывать майндмэпы и задавать вопросы AI-ассистенту.

**Предметы:** биология, физика, история (готовые графы включены в репозиторий).

### Ключевые возможности

- **Граф знаний** — интерактивная визуализация связей между понятиями (D3.js)
- **Майндмэпы** — древовидное представление тем с возможностью раскрытия (Markmap)
- **AI-чат** — диалог с ассистентом по выбранной теме (GigaChat / OpenRouter)
- **Генерация графов** — автоматическое построение из PDF-учебников и рабочих программ
- **Уровни сложности** — школьный, университетский, продвинутый

---

## Технический стек

| Слой | Технологии |
|------|-----------|
| **Фронтенд** | Next.js 15, React 19, TypeScript, Tailwind CSS, Radix UI |
| **Визуализация** | D3.js (граф), Markmap (майндмэпы), Recharts (графики) |
| **Бэкенд** | FastAPI, Uvicorn, Pydantic |
| **AI / LLM** | GigaChat (Sber), OpenRouter (GPT и др.) |
| **Обработка данных** | ChromaDB (эмбеддинги), PyMuPDF (PDF), NetworkX |
| **Деплой** | Docker, Docker Compose |

---

## Быстрый старт

### Требования

- **Python** 3.10+
- **Node.js** 18+ и **pnpm**
- **Docker** (опционально, для деплоя)

### 1. Клонирование

```bash
git clone https://github.com/OmgiKikov/meta-textbook.git
cd meta-textbook
```

### 2. Настройка переменных окружения

Создайте файл **`.env`** в корне проекта:

```env
# === OpenRouter (основной LLM для генерации графов) ===
OPENROUTER_API_KEY=your_openrouter_api_key

# === GigaChat (Sber, AI-чат с пользователем) ===
SECRET_KEY=your_gigachat_secret_key
SCOPE=GIGACHAT_API_PERS
GIGA_HOST=https://gigachat.devices.sberbank.ru/api/v1/chat/completions
GIGA_MODEL=GigaChat
OAUTH_URL=https://ngw.devices.sberbank.ru:9443/api/v2/oauth
API_URL=https://gigachat.devices.sberbank.ru/api/v1/
```

Создайте файл **`frontend/.env.local`**:

```env
# === Адрес бэкенда ===
NEXT_PUBLIC_API_URL=http://localhost:8000

# === OpenAI / OpenRouter (для чата и майндмэпов на фронте) ===
OPENAI_API_KEY=your_openai_or_openrouter_api_key
```

### 3. Установка и запуск

#### Вариант A: Локально (разработка)

```bash
# Бэкенд
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

```bash
# Фронтенд (в отдельном терминале)
cd frontend
pnpm install
pnpm dev
```

Или одной командой (macOS):
```bash
./run.sh
```

| Сервис | URL |
|--------|-----|
| Фронтенд | http://localhost:3000 |
| Бэкенд API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |

#### Вариант B: Docker

```bash
./deploy.sh
# → http://localhost:1000

# Или через Docker Compose
docker-compose up -d --build
# → http://localhost:1000
```

---

## Структура проекта

```
meta-textbook/
│
├── main.py                  # FastAPI-сервер, все API-эндпоинты
├── pipeline.py              # Пайплайн: PDF → граф знаний
├── llm_interface.py         # Вызовы LLM через OpenRouter
├── graph_structures.py      # Модели данных: Node, Edge, Metadata
├── graph_storage.py         # Сериализация графов в JSON
├── models.py                # Pydantic-схемы для API
├── parsers.py               # Парсинг текста
├── utils.py                 # Утилиты (чанкинг текста)
│
├── giga_api_call/           # Обёртка над GigaChat API
│   ├── main.py              # CLI-чатбот (пример)
│   └── utils.py             # Авторизация, токены, запросы
│
├── frontend/                # Next.js-приложение
│   ├── app/                 # Страницы и API-роуты
│   │   └── api/             # chat, mindmap, node-edges, node-image
│   ├── components/          # React-компоненты
│   │   ├── d3-graph.tsx     # Визуализация графа (D3)
│   │   ├── chat-panel.tsx   # Чат с AI
│   │   ├── mindmap.tsx      # Майндмэп
│   │   └── ui/              # Radix UI компоненты
│   ├── lib/                 # API-клиент, утилиты, промпты
│   └── styles/              # CSS
│
├── biology_graph/           # Готовый граф: Биология
├── physics_graph/           # Готовый граф: Физика
├── history_graph/           # Готовый граф: История
├── saved_graphs/            # Сохранённые пользовательские графы
│
├── Dockerfile               # Сборка: фронт + бэк в одном образе
├── docker-compose.yml       # Docker Compose конфигурация
├── deploy.sh                # Скрипт деплоя (Docker)
├── run.sh                   # Быстрый запуск (macOS)
└── requirements.txt         # Python-зависимости
```

---

## API

Бэкенд предоставляет REST API (документация доступна на `/docs`):

| Метод | Эндпоинт | Описание |
|-------|----------|----------|
| `GET` | `/api/graph/{subject}` | Получить граф по предмету |
| `GET` | `/api/graph/{subject}/nodes` | Список узлов графа |
| `GET` | `/api/graph/{subject}/edges` | Список связей графа |
| `GET` | `/api/graph/{subject}/node/{id}` | Детали узла |
| `POST` | `/api/graph/{subject}/search` | Поиск по графу |
| `POST` | `/api/graph/generate_questions` | Генерация вопросов по теме |
| `POST` | `/api/chat` | Чат с AI по теме |
| `POST` | `/api/build_graph` | Построить граф из PDF |

---

## Переменные окружения (справочник)

### Бэкенд (`.env`)

| Переменная | Обязательна | Описание |
|-----------|:-----------:|----------|
| `OPENROUTER_API_KEY` | да | Ключ OpenRouter для генерации графов |
| `SECRET_KEY` | да | Секретный ключ GigaChat (Sber ID) |
| `SCOPE` | да | OAuth scope GigaChat (`GIGACHAT_API_PERS` или `GIGACHAT_API_CORP`) |
| `GIGA_HOST` | да | Эндпоинт GigaChat API |
| `GIGA_MODEL` | да | Модель GigaChat (`GigaChat`, `GigaChat-Pro` и др.) |
| `OAUTH_URL` | да | URL для получения OAuth-токена GigaChat |
| `API_URL` | да | Базовый URL GigaChat API |

### Фронтенд (`frontend/.env.local`)

| Переменная | Обязательна | Описание |
|-----------|:-----------:|----------|
| `NEXT_PUBLIC_API_URL` | да | URL бэкенда (по умолчанию `http://localhost:8000`) |
| `OPENAI_API_KEY` | да | Ключ для AI-чата и генерации майндмэпов на фронте |
