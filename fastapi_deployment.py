"""
Production-Ready FastAPI Deployment for NYC Taxi Tip Prediction
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
import numpy as np
import joblib
import logging
from datetime import datetime
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NYC Taxi Tip Prediction API",
    description="Predict taxi tip amounts using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Pydantic Models ====================

class TripData(BaseModel):
    """
    Input data for a single trip prediction
    """
    pickup_datetime: str = Field(..., example="2024-01-15 14:30:00")
    dropoff_datetime: str = Field(..., example="2024-01-15 14:45:00")
    passenger_count: int = Field(..., ge=1, le=6, example=2)
    trip_distance: float = Field(..., gt=0, lt=100, example=5.2)
    pickup_longitude: float = Field(..., ge=-74.5, le=-73.5, example=-73.98)
    pickup_latitude: float = Field(..., ge=40.5, le=41.0, example=40.75)
    dropoff_longitude: float = Field(..., ge=-74.5, le=-73.5, example=-73.95)
    dropoff_latitude: float = Field(..., ge=40.5, le=41.0, example=40.77)
    fare_amount: float = Field(..., gt=0, lt=500, example=15.50)
    payment_type: int = Field(..., ge=1, le=6, example=1)
    rate_code_id: Optional[int] = Field(1, ge=1, le=6, example=1)
    vendor_id: Optional[int] = Field(1, example=1)
    extra: Optional[float] = Field(0.0, example=0.5)
    mta_tax: Optional[float] = Field(0.5, example=0.5)
    tolls_amount: Optional[float] = Field(0.0, example=0.0)
    improvement_surcharge: Optional[float] = Field(0.3, example=0.3)
    congestion_surcharge: Optional[float] = Field(0.0, example=2.5)
    airport_fee: Optional[float] = Field(0.0, example=0.0)
    store_and_fwd_flag: Optional[str] = Field('N', example='N')
    
    @validator('pickup_datetime', 'dropoff_datetime', pre=True)
    def validate_datetime(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
            return v
        except ValueError:
            raise ValueError('Datetime must be in format: YYYY-MM-DD HH:MM:SS')
    
    class Config:
        schema_extra = {
            "example": {
                "pickup_datetime": "2024-01-15 14:30:00",
                "dropoff_datetime": "2024-01-15 14:45:00",
                "passenger_count": 2,
                "trip_distance": 5.2,
                "pickup_longitude": -73.98,
                "pickup_latitude": 40.75,
                "dropoff_longitude": -73.95,
                "dropoff_latitude": 40.77,
                "fare_amount": 15.50,
                "payment_type": 1,
                "rate_code_id": 1,
                "vendor_id": 1,
                "extra": 0.5,
                "mta_tax": 0.5,
                "tolls_amount": 0.0,
                "improvement_surcharge": 0.3,
                "congestion_surcharge": 2.5,
                "airport_fee": 0.0,
                "store_and_fwd_flag": "N"
            }
        }


class BatchPredictionRequest(BaseModel):
    """
    Batch prediction request
    """
    trips: List[TripData]


class PredictionResponse(BaseModel):
    """
    Prediction response
    """
    predicted_tip: float = Field(..., description="Predicted tip amount in USD")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="95% confidence interval")
    fare_amount: float = Field(..., description="Trip fare amount")
    tip_percentage: float = Field(..., description="Predicted tip as percentage of fare")
    model_version: str = Field(..., description="Model version used")


class BatchPredictionResponse(BaseModel):
    """
    Batch prediction response
    """
    predictions: List[PredictionResponse]
    total_trips: int


class HealthResponse(BaseModel):
    """
    Health check response
    """
    status: str
    model_loaded: bool
    version: str
    timestamp: str


# ==================== Model Loading ====================

class ModelService:
    """
    Service for loading and managing ML models
    """
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.model_version = "1.0.0"
        self.loaded = False
        
    def load_model(self):
        """
        Load the trained model and preprocessor
        """
        try:
            self.model = joblib.load('models/random_forest.pkl')
            self.preprocessor = joblib.load('preprocessor.pkl')
            self.loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_input(self, trip_data: TripData) -> np.ndarray:
        """
        Preprocess input data for prediction
        """
        # Convert to DataFrame
        import pandas as pd
        
        data_dict = trip_data.dict()
        data_dict['pickup_datetime'] = pd.to_datetime(data_dict['pickup_datetime'])
        data_dict['dropoff_datetime'] = pd.to_datetime(data_dict['dropoff_datetime'])
        data_dict['total_amount'] = (
            data_dict['fare_amount'] + 
            data_dict.get('extra', 0) +
            data_dict.get('mta_tax', 0) +
            data_dict.get('tolls_amount', 0) +
            data_dict.get('improvement_surcharge', 0) +
            data_dict.get('congestion_surcharge', 0) +
            data_dict.get('airport_fee', 0)
        )
        
        df = pd.DataFrame([data_dict])
        
        # Use preprocessor to transform
        X = self.preprocessor.transform(df)
        
        return X
    
    def predict(self, X: np.ndarray) -> Dict:
        """
        Make prediction
        """
        prediction = self.model.predict(X)[0]
        
        # Calculate confidence interval (simplified)
        # In production, use proper uncertainty estimation
        confidence = {
            'lower': max(0, prediction - 1.5),
            'upper': prediction + 1.5
        }
        
        return {
            'prediction': float(prediction),
            'confidence_interval': confidence
        }


# Initialize model service
model_service = ModelService()


# ==================== Startup Event ====================

@app.on_event("startup")
async def startup_event():
    """
    Load model on startup
    """
    logger.info("Starting up API...")
    try:
        model_service.load_model()
        logger.info("API ready to serve predictions")
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise


# ==================== API Endpoints ====================

@app.get("/", response_model=Dict)
async def root():
    """
    Root endpoint
    """
    return {
        "message": "NYC Taxi Tip Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    return HealthResponse(
        status="healthy" if model_service.loaded else "unhealthy",
        model_loaded=model_service.loaded,
        version=model_service.model_version,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_tip(trip_data: TripData):
    """
    Predict tip amount for a single trip
    
    Returns the predicted tip amount, confidence interval, and related metrics.
    """
    try:
        # Preprocess input
        X = model_service.preprocess_input(trip_data)
        
        # Make prediction
        result = model_service.predict(X)
        
        # Calculate metrics
        predicted_tip = result['prediction']
        tip_percentage = (predicted_tip / trip_data.fare_amount) * 100
        
        return PredictionResponse(
            predicted_tip=round(predicted_tip, 2),
            confidence_interval=result['confidence_interval'],
            fare_amount=trip_data.fare_amount,
            tip_percentage=round(tip_percentage, 2),
            model_version=model_service.model_version
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict tip amounts for multiple trips
    
    Accepts a list of trips and returns predictions for all.
    """
    try:
        predictions = []
        
        for trip_data in request.trips:
            X = model_service.preprocess_input(trip_data)
            result = model_service.predict(X)
            
            predicted_tip = result['prediction']
            tip_percentage = (predicted_tip / trip_data.fare_amount) * 100
            
            predictions.append(PredictionResponse(
                predicted_tip=round(predicted_tip, 2),
                confidence_interval=result['confidence_interval'],
                fare_amount=trip_data.fare_amount,
                tip_percentage=round(tip_percentage, 2),
                model_version=model_service.model_version
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_trips=len(predictions)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """
    Get model information
    """
    return {
        "model_type": "Random Forest Regressor",
        "version": model_service.model_version,
        "features": model_service.preprocessor.feature_names if model_service.preprocessor else [],
        "loaded": model_service.loaded
    }


# ==================== Run Server ====================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )