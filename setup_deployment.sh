#!/bin/bash
set -e

echo "🚀 Setting up complete Prefect deployment..."

# Set Prefect API URL
export PREFECT_API_URL=http://127.0.0.1:4200/api
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api

# Start Prefect server
echo "📦 Starting Prefect server..."
prefect server start &
SERVER_PID=$!

# Start MLflow
echo "📦 Starting MLflow..."
docker-compose up -d mlflow-server

# Wait for server
echo "⏳ Waiting for Prefect server..."
sleep 5
until curl -f http://localhost:4200/api/health > /dev/null 2>&1; do
    sleep 2
done

# Create work pool
echo "🏊 Creating work pool..."
prefect work-pool create default --type process || true

# Deploy flows
echo "🚀 Deploying flows..."
python deploy.py

# Start worker
echo "👷 Starting worker..."
prefect worker start --pool default &

echo "✅ System ready!"
echo "🌐 Prefect UI: http://localhost:4200"
echo "📊 MLflow UI: http://localhost:5000"
echo "Press Ctrl+C to stop"
wait
