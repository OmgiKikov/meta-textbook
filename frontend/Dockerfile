FROM node:20-alpine AS builder

# Установка pnpm
RUN npm install -g pnpm

WORKDIR /app

# Копирование файлов package.json и pnpm-lock.yaml
COPY package.json pnpm-lock.yaml ./

# Установка зависимостей
RUN pnpm install

# Копирование исходных файлов
COPY . .

# Сборка приложения
RUN pnpm build

# Этап запуска
FROM node:20-alpine AS runner

WORKDIR /app

# Установка pnpm
RUN npm install -g pnpm

# Копирование необходимых файлов
COPY --from=builder /app/package.json ./
COPY --from=builder /app/pnpm-lock.yaml ./
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/next.config.js ./

# Установка только производственных зависимостей
RUN pnpm install --production

# Expose port 1000
EXPOSE 1000

# Запуск приложения
CMD ["pnpm", "start", "-p", "1000"] 