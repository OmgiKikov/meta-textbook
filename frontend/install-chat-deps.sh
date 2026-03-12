#!/bin/bash

# Скрипт для установки зависимостей чата

echo "Установка зависимостей для компонентов чата..."

# Используем pnpm для установки зависимостей
pnpm add @ai-sdk/openai @microsoft/fetch-event-source ai fast-json-patch framer-motion markmap-lib markmap-view openai

echo "Зависимости установлены успешно!"
echo "Для запуска приложения выполните: pnpm run dev" 