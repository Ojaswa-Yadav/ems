#!/bin/bash

# Google Cloud Run deployment script
PROJECT_ID="your-project-id"
SERVICE_NAME="emissions-api"
REGION="us-central1"

echo "Deploying to Google Cloud Run..."

# Build and push to Container Registry
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8000 \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 10

echo "Deployment complete!"
echo "Service URL:"
gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)'