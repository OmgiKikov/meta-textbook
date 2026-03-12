#!/bin/bash

# Остановить все контейнеры, если они запущены
docker-compose down

# Собрать и запустить контейнеры
docker-compose up --build -d

echo "Контейнеры запущены!"
echo "Бэкенд доступен по адресу: http://localhost:1000"
echo "Фронтенд доступен по адресу: http://localhost:3000"

# Вывести логи (можно раскомментировать, если нужно)
# docker-compose logs -f 