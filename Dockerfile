FROM node:20-alpine AS frontend-builder

# Установка pnpm
RUN npm install -g pnpm

WORKDIR /app/frontend

# Копирование package.json и pnpm-lock.yaml
COPY frontend/package.json frontend/pnpm-lock.yaml ./

# Копирование всех исходных файлов 
COPY frontend/ ./

# Установка зависимостей
RUN pnpm install

# Сборка фронтенда (вывод подробной информации для диагностики)
RUN pnpm build || (echo "Build failed with error" && ls -la && exit 1)

# Основной образ с Python
FROM python:3.12-slim

WORKDIR /app

# Копирование бэкенд файлов
COPY requirements.txt .
COPY *.py ./
COPY utils/ ./utils/
COPY saved_graphs/ ./saved_graphs/ 
COPY saved_pdf/ ./saved_pdf/
COPY static/ ./static/

# Установка зависимостей бэкенда
RUN pip install --no-cache-dir -r requirements.txt

# Установка Node.js и необходимых инструментов
RUN apt-get update && apt-get install -y \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g pnpm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Копирование фронтенда из предыдущего этапа
COPY --from=frontend-builder /app/frontend/.next /app/frontend/.next
COPY --from=frontend-builder /app/frontend/public /app/frontend/public 
COPY --from=frontend-builder /app/frontend/package.json /app/frontend/
COPY --from=frontend-builder /app/frontend/pnpm-lock.yaml /app/frontend/
COPY --from=frontend-builder /app/frontend/next.config.mjs /app/frontend/

# Рабочая директория фронтенда
WORKDIR /app/frontend

# Установка только производственных зависимостей
RUN pnpm install --production

# Вернуться в корневую директорию
WORKDIR /app

# Создаем скрипт для запуска обоих сервисов
RUN echo '#!/bin/bash\n\
# Запуск бэкенда в фоне на порту 8000 (внутренний)\n\
uvicorn main:app --host 0.0.0.0 --port 8000 &\n\
\n\
# Ждем запуска бэкенда\n\
sleep 5\n\
\n\
# Переходим в директорию фронтенда и запускаем его на порту 1000 (внешний)\n\
cd /app/frontend && NEXT_PUBLIC_API_URL=http://localhost:8000 pnpm start -p 1000\n\
' > /app/start.sh

RUN chmod +x /app/start.sh

# Открываем порт 1000 для фронтенда и 8000 для бэкенда
EXPOSE 1000 8000

# Запускаем оба сервиса
CMD ["/app/start.sh"]
