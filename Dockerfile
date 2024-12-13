# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for numpy and ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all model files and the main application
COPY LSTM_price_prediction.keras .
COPY LSTM_price_scaler.pkl .
COPY NN_price_prediction.keras .
COPY NN_price_scaler.pkl .
COPY gru_model.h5 .
COPY ARIMA_price_prediction.pkl .
COPY main.py .

# Expose the port the app runs on
EXPOSE 5001

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]