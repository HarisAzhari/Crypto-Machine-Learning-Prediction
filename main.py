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

# Enums for validation
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

# Model configurations
MODEL_PATHS = {
    ModelType.lstm: {'model': 'models/LSTM_price_prediction.keras', 'scaler': 'models/LSTM_price_scaler.pkl'},
    ModelType.nn: {'model': 'models/NN_price_prediction.keras', 'scaler': 'models/NN_price_scaler.pkl'},
    ModelType.gru: {'model': 'models/gru_model.h5', 'scaler': None},
    ModelType.arima: {'model': 'models/ARIMA_price_prediction.pkl', 'scaler': None}
}

# Pydantic models for response
class PredictionPoint(BaseModel):
    date: str
    price: float

class PredictionResponse(BaseModel):
    model: str
    coin: str
    generated_at: str
    predictions: List[PredictionPoint]

def predict_deep_learning(model, data, scaler, prediction_days=60, future_days=30):
    """Helper function for deep learning models (LSTM, NN, GRU)"""
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    
    # Prepare test data
    test_start = dt.datetime(2020,1,1)
    test_end = dt.datetime.now() + dt.timedelta(days=prediction_days)
    test_data = yf.download(f"{data.index.name}-USD", start=test_start, end=test_end)
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)

    # Calculate future predictions
    future_predictions = []
    last_sequence = model_inputs[-prediction_days:]

    for _ in range(future_days):
        current_sequence = last_sequence.reshape((1, prediction_days, 1))
        next_pred = model.predict(current_sequence, verbose=0)
        next_pred = scaler.inverse_transform(next_pred)[0][0]
        future_predictions.append(next_pred)
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = scaler.transform([[next_pred]])[0][0]
    
    return future_predictions

def predict_arima(model_path, data, future_days=30):
    """Helper function for ARIMA predictions"""
    try:
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)
        
        # Create and fit a new ARIMA model with the same order
        model = ARIMA(data['Close'], order=loaded_model.order)
        fitted_model = model.fit()
        
        # Make future predictions
        future_predictions = fitted_model.forecast(steps=future_days)
        return future_predictions.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in ARIMA prediction: {str(e)}")

@app.get("/predict/{model_type}/{coin}", response_model=PredictionResponse)
async def predict(model_type: ModelType, coin: CoinType):
    """
    Predict cryptocurrency prices
    
    - **model_type**: Type of model to use (lstm, nn, gru, arima)
    - **coin**: Cryptocurrency to predict (btc, eth, xrp, bnb, sol)
    """
    try:
        # Load cryptocurrency data
        start = dt.datetime(2016,1,1)
        end = dt.datetime.now()
        ticker = f"{coin.value.upper()}-USD"
        data = yf.download(ticker, start=start, end=end)
        data.index.name = coin.value.upper()

        # Get predictions based on model type
        if model_type == ModelType.arima:
            future_predictions = predict_arima(
                MODEL_PATHS[model_type]['model'],
                data
            )
        else:
            # Load model and scaler for deep learning models
            try:
                model = load_model(MODEL_PATHS[model_type]['model'])
                scaler = MinMaxScaler(feature_range=(0,1))
                
                future_predictions = predict_deep_learning(
                    model,
                    data,
                    scaler
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error loading model or making predictions: {str(e)}"
                )

        # Generate dates for predictions
        start_date = dt.datetime.now()
        future_dates = pd.date_range(start=start_date, periods=31)[1:]

        # Create response
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
        
        return predictions
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Model file not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)