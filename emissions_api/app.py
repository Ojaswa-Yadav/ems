from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
from main import main

app = FastAPI(title="Carbon Emissions Prediction API", version="1.0.0")

class EmissionsData(BaseModel):
    data: List[Dict[str, Any]]

def clean_for_json(obj):
    """Clean data to make it JSON serializable"""
    if isinstance(obj, dict):
        return {key: clean_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj) or (isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj))):
        return 0.0
    else:
        return obj

# Add CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "Carbon Emissions Prediction API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict_emissions(file: UploadFile = File(...), return_csv: bool = False):
    try:
        contents = await file.read()
        contents_str = contents.decode("utf-8")
        input_df = pd.read_csv(StringIO(contents_str))
        
        final_predictions, results = main(input_df)
        
        if final_predictions is None:
            return JSONResponse(status_code=500, content={"error": results.get("error", "Unknown error")})
        
        if return_csv:
            output = BytesIO()
            final_predictions.to_csv(output, index=False)
            output.seek(0)
            return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=predictions.csv"})
        
        response = {
            "predictions": final_predictions.to_dict(orient="records"),
            "metrics": results
        }
        
        # Clean the response to ensure JSON compatibility
        clean_response = clean_for_json(response)
        return JSONResponse(content=clean_response)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict-json")
async def predict_emissions_json(data: EmissionsData):
    try:
        input_df = pd.DataFrame(data.data)
        
        final_predictions, results = main(input_df)
        
        if final_predictions is None:
            return JSONResponse(status_code=500, content={"error": results.get("error", "Unknown error")})
        
        response = {
            "predictions": final_predictions.to_dict(orient="records"),
            "metrics": results
        }
        
        # Clean the response to ensure JSON compatibility
        clean_response = clean_for_json(response)
        return JSONResponse(content=clean_response)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# For local running: uvicorn app:app --reload