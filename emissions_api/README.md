# Carbon Emissions Prediction API

A FastAPI-based service for predicting carbon emissions (Scope 1, 2, and 3) for companies based on historical data and sector analysis.

## Features

- RESTful API for carbon emissions prediction
- Sector-based modeling and intensity calculations
- Data validation and error handling
- Docker containerization
- Health check endpoints

## Quick Start

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API:
```bash
uvicorn main:app --reload
```

3. Access the API documentation at `http://localhost:8000/docs`

### Docker Deployment

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

2. The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Predict emissions for companies

## Request Format

```json
{
  "companies": [
    {
      "ticker": "AAPL",
      "name": "Apple Inc.",
      "sector": "Technology",
      "market_value": 2500000000,
      "revenue_2023": 365000000,
      "scope1_2022": 1500000,
      "scope2_2022": 2500000,
      "scope3_2022": 15000000
    }
  ],
  "target_year": 2024
}
```

## Response Format

```json
[
  {
    "ticker": "AAPL",
    "name": "Apple Inc.",
    "sector": "Technology",
    "predicted_scope1": 1600000,
    "predicted_scope2": 2600000,
    "predicted_scope3": 15500000,
    "confidence_score": 0.8
  }
]
```

## Cloud Deployment Options

### 1. Google Cloud Run
- Serverless, pay-per-use
- Automatic scaling
- Easy CI/CD integration

### 2. AWS ECS/Fargate
- Container orchestration
- Load balancing
- Auto-scaling

### 3. Azure Container Instances
- Quick deployment
- Integrated with Azure services
- Cost-effective for moderate traffic

### 4. Railway/Render
- Simple deployment
- GitHub integration
- Good for startups/small projects