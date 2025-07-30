#!/bin/bash
# Startup script for the emissions API

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting the emissions prediction API..."
uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}