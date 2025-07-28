from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Optional
import logging

app = FastAPI(title="Carbon Emissions Prediction API", version="1.0.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompanyData(BaseModel):
    ticker: str
    name: str
    sector: str
    market_value: Optional[float] = None
    revenue_2023: Optional[float] = None

    @validator('ticker', 'name', 'sector')
    def validate_required_strings(cls, v):
        if not v or not v.strip():
            raise ValueError('Field cannot be empty')
        return v.strip()

class PredictionRequest(BaseModel):
    companies: List[CompanyData]
    target_year: int = 2024

class PredictionResponse(BaseModel):
    ticker: str
    name: str
    sector: str
    predicted_scope1: float
    predicted_scope2: float
    predicted_scope3: float
    confidence_score: float = 0.75

@app.get("/")
def root():
    return {"message": "Carbon Emissions Prediction API", "version": "1.0.0", "status": "active"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "emissions-api"}

@app.post("/predict", response_model=List[PredictionResponse])
def predict_emissions(request: PredictionRequest):
    try:
        predictions = []
        
        sector_multipliers = {
            "Energy": {"scope1": 1200, "scope2": 800, "scope3": 2000},
            "Materials": {"scope1": 1000, "scope2": 600, "scope3": 1800},
            "Industrials": {"scope1": 800, "scope2": 500, "scope3": 1500},
            "Information Technology": {"scope1": 150, "scope2": 100, "scope3": 700},
            "Financials": {"scope1": 100, "scope2": 80, "scope3": 600},
        }
        
        for company in request.companies:
            multipliers = sector_multipliers.get(company.sector, {
                "scope1": 500, "scope2": 300, "scope3": 1000
            })
            
            base_factor = 1.0
            if company.market_value:
                base_factor = max(0.1, min(10.0, company.market_value / 1000000))
            elif company.revenue_2023:
                base_factor = max(0.1, min(10.0, company.revenue_2023 / 1000000))
            
            predictions.append(PredictionResponse(
                ticker=company.ticker,
                name=company.name,
                sector=company.sector,
                predicted_scope1=round(multipliers["scope1"] * base_factor, 2),
                predicted_scope2=round(multipliers["scope2"] * base_factor, 2),
                predicted_scope3=round(multipliers["scope3"] * base_factor, 2),
                confidence_score=0.75
            ))
        
        return predictions
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")