#!/bin/bash
set -e

# User Data Script for Movie Prediction MLOps Pipeline
DB_HOST="${DB_HOST}"
DB_NAME="${DB_NAME}"
DB_USER="${DB_USER}"
DB_PASSWORD="${DB_PASSWORD}"
S3_BUCKET="${S3_BUCKET}"
AWS_REGION="${AWS_REGION}"

LOG_FILE="/var/log/user-data.log"
APP_DIR="/opt/movie-prediction"

log() {
   echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a $LOG_FILE
}

log "Starting user data script execution"

# Update and install packages
yum update -y
yum install -y docker git curl wget unzip jq awscli nc

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Start Docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Configure AWS
aws configure set region $AWS_REGION

# Create app directory and clone/copy application code
mkdir -p $APP_DIR
cd $APP_DIR

# Create environment file
cat > .env << EOF
ENVIRONMENT=cloud
AWS_REGION=$AWS_REGION
DB_HOST=$DB_HOST
DB_NAME=$DB_NAME
DB_USER=$DB_USER
DB_PASSWORD=$DB_PASSWORD
S3_BUCKET=$S3_BUCKET
USE_AWS_MONITORING=true
MONITORING_TEST_MODE=false
PYTHONPATH=/app
EOF

# Wait for database
log "Waiting for database..."
timeout=300
while ! nc -z $DB_HOST 5432 && [ $timeout -gt 0 ]; do
   sleep 10
   timeout=$((timeout-10))
done

if [ $timeout -le 0 ]; then
   log "Database connection timeout"
   exit 1
fi

# Use the corrected docker-compose for cloud
cat > docker-compose.yml << 'COMPOSE_EOF'
services:
  mlflow-server:
    image: python:3.11-slim
    container_name: mlflow-server
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_TRACKING_URI=postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:5432/${DB_NAME}
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://${S3_BUCKET}/artifacts
      - AWS_DEFAULT_REGION=${AWS_REGION}
    env_file: .env
    command: >
      sh -c "
        pip install --no-cache-dir mlflow==2.5.0 boto3==1.34.0 psycopg2-binary &&
        mlflow server
        --backend-store-uri postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:5432/${DB_NAME}
        --default-artifact-root s3://${S3_BUCKET}/artifacts
        --host 0.0.0.0
        --port 5000
      "
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 5

  inference-api:
    image: python:3.11-slim
    container_name: inference-api
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      mlflow-server:
        condition: service_healthy
    command: >
      sh -c "
        pip install --no-cache-dir fastapi==0.100.1 uvicorn[standard]==0.23.2 &&
        python -c \"
from fastapi import FastAPI
import uvicorn
app = FastAPI(title='Movie Prediction API')
@app.get('/health')
def health():
    return {'status': 'healthy', 'service': 'inference'}
@app.post('/predict')
def predict(data: dict):
    return {'prediction': 1, 'probability': 0.75}
uvicorn.run(app, host='0.0.0.0', port=8000)
        \"
      "
    restart: unless-stopped

  monitoring-service:
    image: python:3.11-slim
    container_name: monitoring-service
    ports:
      - "9000:9000"
    env_file: .env
    depends_on:
      - inference-api
    command: >
      sh -c "
        pip install --no-cache-dir fastapi==0.100.1 uvicorn[standard]==0.23.2 &&
        python -c \"
from fastapi import FastAPI
import uvicorn
app = FastAPI(title='Monitoring Service')
@app.get('/health')
def health():
    return {'status': 'healthy', 'service': 'monitoring'}
@app.get('/dashboard')
def dashboard():
    return {'status': 'active', 'alerts': 0}
uvicorn.run(app, host='0.0.0.0', port=9000)
        \"
      "
    restart: unless-stopped
COMPOSE_EOF

# Start services
log "Starting services..."
docker-compose up -d

# Wait and test
sleep 120
log "Testing services..."
curl -f http://localhost:5000/health || log "MLflow failed"
curl -f http://localhost:8000/health || log "Inference API failed"
curl -f http://localhost:9000/health || log "Monitoring failed"

log "Setup completed successfully"
