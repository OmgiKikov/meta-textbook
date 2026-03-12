FROM node:20-alpine AS frontend-builder

RUN npm install -g pnpm

WORKDIR /app/frontend

COPY frontend/package.json frontend/pnpm-lock.yaml ./
COPY frontend/ ./

RUN pnpm install
RUN pnpm build

# ---

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g pnpm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY *.py ./
COPY giga_api_call/ ./giga_api_call/
COPY biology_graph/ ./biology_graph/
COPY physics_graph/ ./physics_graph/
COPY history_graph/ ./history_graph/
COPY saved_graphs/ ./saved_graphs/

COPY --from=frontend-builder /app/frontend/.next /app/frontend/.next
COPY --from=frontend-builder /app/frontend/public /app/frontend/public
COPY --from=frontend-builder /app/frontend/package.json /app/frontend/
COPY --from=frontend-builder /app/frontend/pnpm-lock.yaml /app/frontend/
COPY --from=frontend-builder /app/frontend/next.config.mjs /app/frontend/

WORKDIR /app/frontend
RUN pnpm install --production
WORKDIR /app

RUN printf '#!/bin/bash\nuvicorn main:app --host 0.0.0.0 --port 8000 &\nsleep 3\ncd /app/frontend && pnpm start -p 1000\n' > /app/start.sh \
    && chmod +x /app/start.sh

EXPOSE 1000 8000

CMD ["/app/start.sh"]
