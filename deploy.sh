#!/bin/bash

echo "Останавливаем и удаляем существующий контейнер..."
docker stop meta-textbook 2>/dev/null || true
docker rm meta-textbook 2>/dev/null || true

echo "Сборка контейнера..."
docker build -t meta-textbook:latest .

echo "Запуск контейнера..."
docker run -d --name meta-textbook \
  -p 1000:1000 \
  --restart unless-stopped \
  meta-textbook:latest

echo "Контейнер запущен!"
echo "Приложение доступно по адресу: http://localhost:1000"
echo ""
echo "Для просмотра логов: docker logs -f meta-textbook"
echo "Для остановки: docker stop meta-textbook"
echo "Для удаления: docker rm meta-textbook" 