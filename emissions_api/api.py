from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Optional
import logging

app = FastAPI(title="Carbon Emissions Prediction API", version="1.0.0")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompanyData(BaseModel):
    ticker: str
    name: str
    sector: str
    asset_class: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    market_value: Optional[float] = None
    notional_value: Optional[float] = None
    shares: Optional[float] = None
    scope1_2023: Optional[float] = None
    scope2_2023: Optional[float] = None
    scope3_2023: Optional[float] = None
    scope1_2022: Optional[float] = None
    scope2_2022: Optional[float] = None
    scope3_2022: Optional[float] = None
    scope1_2021: Optional[float] = None
    scope2_2021: Optional[float] = None
    scope3_2021: Optional[float] = None
    revenue_2023: Optional[float] = None
    revenue_2022: Optional[float] = None
    revenue_2021: Optional[float] = None

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
        if len(v) > 1000:
            raise ValueError('Maximum 1000 companies per request')
        return v

class PredictionResponse(BaseModel):
    ticker: str
    name: str
    sector: str
    predicted_scope1: float
    predicted_scope2: float
    predicted_scope3: float
    confidence_score: Optional[float] = None

@app.get("/")
async def root():
    return {"message": "Carbon Emissions Prediction API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=List[PredictionResponse])
async def predict_emissions(request: PredictionRequest):
    try:
        predictions = []
        
        # Simple sector-based prediction logic
        sector_multipliers = {
            "Energy": {"scope1": 1200, "scope2": 800, "scope3": 2000},
            "Materials": {"scope1": 1000, "scope2": 600, "scope3": 1800},
            "Industrials": {"scope1": 800, "scope2": 500, "scope3": 1500},
            "Consumer Discretionary": {"scope1": 400, "scope2": 300, "scope3": 1200},
            "Consumer Staples": {"scope1": 600, "scope2": 400, "scope3": 1400},
            "Health Care": {"scope1": 200, "scope2": 150, "scope3": 800},
            "Financials": {"scope1": 100, "scope2": 80, "scope3": 600},
            "Information Technology": {"scope1": 150, "scope2": 100, "scope3": 700},
            "Communication Services": {"scope1": 300, "scope2": 200, "scope3": 900},
            "Utilities": {"scope1": 1500, "scope2": 1000, "scope3": 2200},
            "Real Estate": {"scope1": 500, "scope2": 350, "scope3": 1100}
        }
        
        for company in request.companies:
            # Get sector multipliers or use default
            multipliers = sector_multipliers.get(company.sector, {
                "scope1": 500, "scope2": 300, "scope3": 1000
            })
            
            # Simple prediction based on market value or revenue
            base_factor = 1.0
            if company.market_value:
                base_factor = max(0.1, min(10.0, company.market_value / 1000000))  # Normalize to millions
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

# For Vercel serverless deployment
app.add_middleware(
    lambda request, call_next: call_next(request),
)