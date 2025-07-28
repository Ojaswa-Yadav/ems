from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Optional
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Carbon Emissions Prediction API", version="1.0.0")

class CompanyData(BaseModel):
    ticker: str
    name: str
    sector: str
    market_value: Optional[float] = None
    revenue_2023: Optional[float] = None
    revenue_2022: Optional[float] = None
    revenue_2021: Optional[float] = None
    scope1_2023: Optional[float] = None
    scope2_2023: Optional[float] = None
    scope3_2023: Optional[float] = None
    scope1_2022: Optional[float] = None
    scope2_2022: Optional[float] = None
    scope3_2022: Optional[float] = None
    scope1_2021: Optional[float] = None
    scope2_2021: Optional[float] = None
    scope3_2021: Optional[float] = None

    @validator('ticker', 'name', 'sector')
    def validate_required_strings(cls, v):
        if not v or not v.strip():
            raise ValueError('Field cannot be empty')
        return v.strip()

class PredictionRequest(BaseModel):
    companies: List[CompanyData]
    target_year: int = 2024

    @validator('companies')
    def validate_companies_list(cls, v):
        if not v:
            raise ValueError('At least one company must be provided')
        return v

class PredictionResponse(BaseModel):
    ticker: str
    name: str
    sector: str
    predicted_scope1: float
    predicted_scope2: float
    predicted_scope3: float
    confidence_score: float

@app.get("/")
async def root():
    return {"message": "Carbon Emissions Prediction API", "version": "1.0.0", "status": "online"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=List[PredictionResponse])
async def predict_emissions(request: PredictionRequest):
    try:
        predictions = []
        
        for company in request.companies:
            # Simple prediction logic based on sector and size
            sector_multipliers = {
                "Technology": {"scope1": 1000, "scope2": 1500, "scope3": 8000},
                "Finance": {"scope1": 500, "scope2": 800, "scope3": 3000},
                "Healthcare": {"scope1": 1200, "scope2": 1800, "scope3": 5000},
                "Energy": {"scope1": 5000, "scope2": 3000, "scope3": 15000},
                "Manufacturing": {"scope1": 3000, "scope2": 2500, "scope3": 12000},
                "Retail": {"scope1": 800, "scope2": 1200, "scope3": 6000}
            }
            
            # Default multipliers if sector not found
            multipliers = sector_multipliers.get(company.sector, 
                                               {"scope1": 1000, "scope2": 1500, "scope3": 5000})
            
            # Calculate size factor based on market value or revenue
            size_factor = 1.0
            if company.market_value:
                size_factor = np.log1p(company.market_value) / 20
            elif company.revenue_2023:
                size_factor = np.log1p(company.revenue_2023) / 18
            elif company.revenue_2022:
                size_factor = np.log1p(company.revenue_2022) / 18
            
            # Generate predictions
            pred_scope1 = max(0, multipliers["scope1"] * size_factor)
            pred_scope2 = max(0, multipliers["scope2"] * size_factor)
            pred_scope3 = max(0, multipliers["scope3"] * size_factor)
            
            predictions.append(PredictionResponse(
                ticker=company.ticker,
                name=company.name,
                sector=company.sector,
                predicted_scope1=round(pred_scope1, 2),
                predicted_scope2=round(pred_scope2, 2),
                predicted_scope3=round(pred_scope3, 2),
                confidence_score=0.75
            ))
        
        return predictions
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)