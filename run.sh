#!/bin/bash

# Display startup message
echo "Starting Meta Textbook Application..."
echo "========================================"

# Create necessary directories
mkdir -p saved_graphs static/images

# Start backend server in a separate terminal window
echo "Starting backend server..."
osascript -e 'tell app "Terminal" to do script "cd '$(pwd)' && python main.py"'

# Navigate to frontend directory and start the dev server
echo "Starting frontend server..."
cd frontend
pnpm install
pnpm run dev

# Note: To stop both servers, press Ctrl+C in this terminal and close the backend server terminal window 