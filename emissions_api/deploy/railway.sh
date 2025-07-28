#!/bin/bash

# Railway deployment script
echo "Deploying to Railway..."

# Install Railway CLI if not present
if ! command -v railway &> /dev/null; then
    echo "Installing Railway CLI..."
    npm install -g @railway/cli
fi

# Login to Railway (will prompt for auth)
railway login

# Create new project or link existing
railway init

# Deploy the application
railway up

echo "Deployment complete! Check Railway dashboard for URL."