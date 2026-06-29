# ============================================================
#  Dockerfile — Stock Predictor ML Training Pipeline
#  Builds a container that:
#    1. Installs all Python dependencies
#    2. Runs transfer learning (parent + child model training)
#    3. Exports ONNX models and scaler params
#    4. Serves MLflow UI on port 5000
#
#  Usage:
#    docker build -t stock-predictor .
#    docker run -p 5000:5000 stock-predictor
# ============================================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching — only reinstalls if requirements change)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    mlflow \
    yfinance \
    xgboost \
    onnxmltools \
    onnxruntime \
    scikit-learn \
    pandas \
    numpy \
    matplotlib \
    seaborn

# Copy source code
COPY python/ ./python/

# Create output directory for models and plots
RUN mkdir -p /app/models /app/plots

# Set MLflow tracking URI to SQLite inside container
ENV MLFLOW_TRACKING_URI=sqlite:///app/mlflow.db

# Expose MLflow UI port
EXPOSE 5000

# Default command: train then serve MLflow UI
CMD ["sh", "-c", \
     "python python/transfer_learning.py && \
      mlflow ui --backend-store-uri sqlite:///app/mlflow.db \
                --host 0.0.0.0 \
                --port 5000"]
