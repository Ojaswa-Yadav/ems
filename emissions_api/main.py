from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import logging
try:
    from models import (
        clean_numeric_data, calculate_sector_intensity, 
        fill_missing_values_using_sector_intensity, idw_interpolation, 
        prepare_features
    )
except ImportError:
    # Fallback: include functions directly in main.py
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import RobustScaler
    
    def clean_numeric_data(df, columns):
        for col in columns:
            if col in df.columns:
                mask = df[col].astype(str).str.contains(',', na=False)
                df.loc[mask, col] = df.loc[mask, col].astype(str).str.replace(',', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def calculate_sector_intensity(df, sector_col, emissions_cols, revenue_col):
        try:
            df[emissions_cols] = df[emissions_cols].apply(pd.to_numeric, errors='coerce')
            df[revenue_col] = pd.to_numeric(df[revenue_col], errors='coerce')
            df['total_emissions'] = df[emissions_cols].sum(axis=1)
            sector_totals = df.groupby(sector_col).agg(
                total_emissions=('total_emissions', 'sum'),
                total_revenue=(revenue_col, 'sum')
            )
            sector_totals['Intensity'] = sector_totals.apply(
                lambda row: row['total_emissions'] / row['total_revenue'] if row['total_revenue'] > 0 else 0, axis=1
            )
            return sector_totals[['Intensity']].reset_index()
        except Exception as e:
            logger.error(f"Error in calculate_sector_intensity: {str(e)}")
            return pd.DataFrame(columns=[sector_col, 'Intensity'])
    
    def fill_missing_values_using_sector_intensity(df, sector_col, emissions_cols, sector_intensities_df, revenue_col):
        try:
            for col in emissions_cols:
                if col in df.columns and not sector_intensities_df.empty:
                    for sector, intensity in sector_intensities_df.values:
                        missing_rows = (df[sector_col] == sector) & (df[col].isnull())
                        df.loc[missing_rows, col] = intensity * df.loc[missing_rows, revenue_col]
            return df
        except Exception as e:
            logger.error(f"Error in filling values using sector intensity: {str(e)}")
            return df
    
    def idw_interpolation(df, sector_col, emissions_cols, weight_col):
        try:
            for col in emissions_cols:
                if col not in df.columns:
                    continue
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce')
                for sector in df[sector_col].unique():
                    sector_data = df[df[sector_col] == sector]
                    known_data = sector_data[~sector_data[col].isnull()]
                    missing_data = sector_data[sector_data[col].isnull()]
                    if not missing_data.empty and not known_data.empty:
                        weights = 1 / (known_data[weight_col] + 1e-6)
                        weights /= weights.sum()
                        for idx in missing_data.index:
                            df.loc[idx, col] = (known_data[col] * weights).sum()
            return df
        except Exception as e:
            logger.error(f"Error in IDW interpolation: {str(e)}")
            return df
    
    def prepare_features(df, emissions_cols, revenue_col, sector_col):
        df = clean_numeric_data(df, emissions_cols + [revenue_col, "Shares"])
        scaler = RobustScaler()
        for col in emissions_cols + [revenue_col]:
            if col in df.columns:
                df[f'{col}_robust'] = scaler.fit_transform(df[[col]].fillna(0))
        for col in emissions_cols + [revenue_col]:
            if col in df.columns:
                df[f'log_{col}'] = np.log1p(df[col].fillna(0).clip(lower=1e-6))
        return df

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

    @validator('market_value', 'notional_value', 'shares', 'scope1_2023', 'scope2_2023', 'scope3_2023',
               'scope1_2022', 'scope2_2022', 'scope3_2022', 'scope1_2021', 'scope2_2021', 'scope3_2021',
               'revenue_2023', 'revenue_2022', 'revenue_2021')
    def validate_positive_numbers(cls, v):
        if v is not None and v < 0:
            raise ValueError('Numeric values must be non-negative')
        return v

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

    @validator('target_year')
    def validate_target_year(cls, v):
        if v < 2020 or v > 2030:
            raise ValueError('Target year must be between 2020 and 2030')
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
        # Convert input data to DataFrame
        df = pd.DataFrame([company.dict() for company in request.companies])
        
        # Process predictions
        predictions = process_emissions_predictions(df, request.target_year)
        
        return predictions
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def process_emissions_predictions(df: pd.DataFrame, target_year: int) -> List[PredictionResponse]:
    """
    Main function to process emissions predictions
    """
    try:
        # Rename columns to match the original script format
        df = df.rename(columns={
            'ticker': 'Ticker',
            'name': 'Name',
            'sector': 'Sector',
            'asset_class': 'Asset Class',
            'country': 'Country of Headquarters',
            'region': 'Geographical Region',
            'market_value': 'market value',
            'notional_value': 'Notional Value',
            'shares': 'Shares',
            'scope1_2023': 'Carbon Emissions Scope 1 (2023)',
            'scope2_2023': 'Carbon Emissions Scope 2 (2023)',
            'scope3_2023': 'Carbon Emissions Scope 3 (2023)',
            'scope1_2022': 'Carbon Emissions Scope 1 (2022)',
            'scope2_2022': 'Carbon Emissions Scope 2 (2022)',
            'scope3_2022': 'Carbon Emissions Scope 3 (2022)',
            'scope1_2021': 'Carbon Emissions Scope 1 (2021)',
            'scope2_2021': 'Carbon Emissions Scope 2 (2021)',
            'scope3_2021': 'Carbon Emissions Scope 3 (2021)',
            'revenue_2023': 'Revenue (2023)',
            'revenue_2022': 'Revenue (2022)',
            'revenue_2021': 'Revenue (2021)'
        })

        # Clean numeric data
        numeric_cols = [
            'Carbon Emissions Scope 1 (2023)', 'Carbon Emissions Scope 2 (2023)', 'Carbon Emissions Scope 3 (2023)',
            'Carbon Emissions Scope 1 (2022)', 'Carbon Emissions Scope 2 (2022)', 'Carbon Emissions Scope 3 (2022)',
            'Carbon Emissions Scope 1 (2021)', 'Carbon Emissions Scope 2 (2021)', 'Carbon Emissions Scope 3 (2021)',
            'Revenue (2023)', 'Revenue (2022)', 'Revenue (2021)', 'market value', 'Notional Value', 'Shares'
        ]
        df = clean_numeric_data(df, numeric_cols)

        # Reshape data into long format for processing
        df_long = df.melt(
            id_vars=[
                'Ticker', 'Name', 'Sector', 'Asset Class', 'Country of Headquarters',
                'Geographical Region', 'market value', 'Notional Value', 'Shares'
            ],
            value_vars=[col for col in numeric_cols if col not in ['market value', 'Notional Value', 'Shares']],
            var_name='Metric_Year',
            value_name='Value'
        )

        # Extract year and metric
        df_long['Year'] = df_long['Metric_Year'].str.extract(r'(\d{4})').astype(int)
        df_long['Metric'] = df_long['Metric_Year'].str.extract(r'(Scope \d|Revenue)').fillna('Unknown')
        df_long.drop(columns=['Metric_Year'], inplace=True)

        # Pivot back to wide format
        df_pivot = df_long.pivot_table(
            index=[col for col in df_long.columns if col not in ['Metric', 'Value']],
            columns='Metric',
            values='Value'
        ).reset_index()

        # Prepare features
        emissions_cols = ['Scope 1', 'Scope 2', 'Scope 3']
        revenue_col = 'Revenue'
        sector_col = 'Sector'

        # Calculate sector intensity and fill missing values
        sector_intensities = calculate_sector_intensity(df_pivot, sector_col, emissions_cols, revenue_col)
        df_pivot = fill_missing_values_using_sector_intensity(
            df_pivot, sector_col, emissions_cols, sector_intensities, revenue_col
        )
        df_pivot = idw_interpolation(df_pivot, sector_col, emissions_cols, 'market value')
        df_pivot = prepare_features(df_pivot, emissions_cols, revenue_col, sector_col)

        # Generate predictions using simplified model
        predictions = generate_simple_predictions(df_pivot, emissions_cols)
        
        # Format response
        response = []
        for _, row in predictions.iterrows():
            response.append(PredictionResponse(
                ticker=row['Ticker'],
                name=row['Name'],
                sector=row['Sector'],
                predicted_scope1=max(0, row.get('Predicted_Scope 1', 0)),
                predicted_scope2=max(0, row.get('Predicted_Scope 2', 0)),
                predicted_scope3=max(0, row.get('Predicted_Scope 3', 0)),
                confidence_score=0.8  # Placeholder confidence score
            ))
        
        return response

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise e

def generate_simple_predictions(df: pd.DataFrame, emissions_cols: List[str]) -> pd.DataFrame:
    """
    Generate predictions using sector averages and trends
    """
    predictions = df.copy()
    
    for col in emissions_cols:
        # Use sector median as baseline prediction
        sector_medians = df.groupby('Sector')[col].median()
        predictions[f'Predicted_{col}'] = predictions['Sector'].map(sector_medians).fillna(0)
        
        # Adjust based on company size (market value)
        if 'market value' in predictions.columns:
            market_value = predictions['market value'].fillna(predictions['market value'].median())
            size_factor = np.log1p(market_value) / np.log1p(market_value.median())
            predictions[f'Predicted_{col}'] *= size_factor
    
    return predictions

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)