from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import pickle
from statsmodels.tsa.arima.model import ARIMA
from pydantic import BaseModel
from typing import List, Dict
from enum import Enum
import logging
import os
import sys

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cryptocurrency Price Prediction API",
    description="API for predicting cryptocurrency prices using various models",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelType(str, Enum):
    lstm = "lstm"
    nn = "nn"
    gru = "gru"
    arima = "arima"

class CoinType(str, Enum):
    btc = "btc"
    eth = "eth"
    xrp = "xrp"
    bnb = "bnb"
    sol = "sol"

# Model configurations with absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    ModelType.lstm: {
        'model': os.path.join(BASE_DIR, 'models', 'LSTM_price_prediction.keras'),
        'scaler': os.path.join(BASE_DIR, 'models', 'LSTM_price_scaler.pkl')
    },
    ModelType.nn: {
        'model': os.path.join(BASE_DIR, 'models', 'NN_price_prediction.keras'),
        'scaler': os.path.join(BASE_DIR, 'models', 'NN_price_scaler.pkl')
    },
    ModelType.gru: {
        'model': os.path.join(BASE_DIR, 'models', 'gru_model.h5'),
        'scaler': None
    },
    ModelType.arima: {
        'model': os.path.join(BASE_DIR, 'ARIMA_price_prediction.pkl'),
        'scaler': None
    }
}

class PredictionPoint(BaseModel):
    date: str
    price: float

class PredictionResponse(BaseModel):
    model: str
    coin: str
    generated_at: str
    predictions: List[PredictionPoint]

@app.on_event("startup")
async def startup_event():
    """Verify model files exist on startup"""
    logger.info("Checking model files...")
    for model_type, paths in MODEL_PATHS.items():
        if paths['model'] and not os.path.exists(paths['model']):
            logger.error(f"Model file not found: {paths['model']}")
        if paths['scaler'] and not os.path.exists(paths['scaler']):
            logger.error(f"Scaler file not found: {paths['scaler']}")
    logger.info("Model file check complete")

def predict_deep_learning(model, data, scaler, prediction_days=60, future_days=30):
    """Helper function for deep learning models (LSTM, NN, GRU)"""
    try:
        logger.debug("Starting deep learning prediction")
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
        
        test_start = dt.datetime(2020,1,1)
        test_end = dt.datetime.now() + dt.timedelta(days=prediction_days)
        
        logger.debug(f"Downloading test data from {test_start} to {test_end}")
        test_data = yf.download(f"{data.index.name}-USD", start=test_start, end=test_end)
        
        total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
        
        model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
        model_inputs = model_inputs.reshape(-1, 1)
        model_inputs = scaler.fit_transform(model_inputs)

        future_predictions = []
        last_sequence = model_inputs[-prediction_days:]

        logger.debug("Making predictions")
        for i in range(future_days):
            current_sequence = last_sequence.reshape((1, prediction_days, 1))
            next_pred = model.predict(current_sequence, verbose=0)
            next_pred = scaler.inverse_transform(next_pred)[0][0]
            future_predictions.append(next_pred)
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = scaler.transform([[next_pred]])[0][0]
        
        return future_predictions
    except Exception as e:
        logger.error(f"Error in deep learning prediction: {str(e)}")
        raise

@app.get("/predict/{model_type}/{coin}", response_model=PredictionResponse)
async def predict(model_type: ModelType, coin: CoinType):
    """Predict cryptocurrency prices"""
    try:
        logger.info(f"Starting prediction for {coin.value} using {model_type.value} model")
        
        # Load cryptocurrency data
        start = dt.datetime(2016,1,1)
        end = dt.datetime.now()
        ticker = f"{coin.value.upper()}-USD"
        
        logger.debug(f"Downloading data for {ticker}")
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        
        data.index.name = coin.value.upper()

        # Check if model files exist
        model_path = MODEL_PATHS[model_type]['model']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if model_type == ModelType.arima:
            logger.debug("Using ARIMA model")
            future_predictions = predict_arima(model_path, data)
        else:
            logger.debug(f"Loading {model_type.value} model")
            model = load_model(model_path)
            
            scaler_path = MODEL_PATHS[model_type]['scaler']
            if scaler_path and os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            else:
                scaler = MinMaxScaler(feature_range=(0,1))
            
            future_predictions = predict_deep_learning(model, data, scaler)

        start_date = dt.datetime.now()
        future_dates = pd.date_range(start=start_date, periods=31)[1:]

        predictions = PredictionResponse(
            model=model_type.value,
            coin=coin.value.upper(),
            generated_at=dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            predictions=[
                PredictionPoint(
                    date=date.strftime('%Y-%m-%d'),
                    price=float(price)
                )
                for date, price in zip(future_dates, future_predictions)
            ]
        )
        
        logger.info("Prediction completed successfully")
        return predictions
    
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in prediction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if we can access the models directory
        if not os.path.exists(os.path.join(BASE_DIR, 'models')):
            return {
                "status": "warning",
                "message": "Models directory not found",
                "base_dir": BASE_DIR
            }
        
        # List available model files
        model_files = os.listdir(os.path.join(BASE_DIR, 'models'))
        
        return {
            "status": "healthy",
            "base_dir": BASE_DIR,
            "available_models": model_files
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "base_dir": BASE_DIR
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)