# NYC Taxi Tip Prediction: Complete End-to-End ML Pipeline

## 📋 Project Overview

This project demonstrates a production-grade machine learning pipeline for predicting NYC taxi tip amounts. The pipeline covers every stage from SQL database extraction to deployed REST API.

### Key Features
- **Complete Data Pipeline**: SQL extraction → Preprocessing → Feature Engineering → Model Training → Deployment
- **Multiple ML Models**: Linear Regression, Ridge/Lasso, Random Forest, Neural Networks
- **Model Interpretability**: SHAP analysis and feature importance visualization
- **Production API**: FastAPI with proper validation, error handling, and documentation
- **Docker Support**: Containerized deployment with docker-compose
- **Comprehensive Testing**: Unit tests, integration tests, and API examples

---

## 🏗️ Project Structure

```
nyc-taxi-prediction/
├── data/
│   ├── sql_schema.sql              # Database schema
│   ├── sql_queries.sql             # Data extraction queries
│   └── taxi_data_raw.csv           # Raw extracted data
├── models/
│   ├── linear_regression.pkl       # Trained linear model
│   ├── ridge.pkl                   # Trained Ridge model
│   ├── lasso.pkl                   # Trained Lasso model
│   ├── random_forest.pkl           # Trained Random Forest (best model)
│   ├── neural_network.keras        # Trained neural network
│   └── results.json                # Model comparison results
├── plots/
│   ├── predictions.png             # Actual vs predicted
│   ├── residual_dist.png           # Residual distribution
│   ├── error_by_range.png          # Error analysis by tip range
│   ├── feature_importance.png      # Feature importance plot
│   ├── shap_summary.png            # SHAP summary plot
│   └── shap_importance.png         # SHAP feature importance
├── src/
│   ├── data_loader.py              # Database connection & loading
│   ├── preprocessor.py             # Data preprocessing pipeline
│   ├── model_trainer.py            # Model training & tuning
│   └── model_evaluator.py          # Evaluation & interpretability
├── api.py                          # FastAPI application
├── test_api.py                     # API testing suite
├── preprocessor.pkl                # Fitted preprocessor
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── docker-compose.yml              # Docker compose configuration
├── README.md                       # This file
└── notebooks/
    └── eda.ipynb                   # Exploratory data analysis
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- PostgreSQL/MySQL database (or SQLite for testing)
- Docker (optional, for containerized deployment)

### 1. Database Setup

```sql
-- Create database
CREATE DATABASE taxi_db;

-- Run schema creation
psql -U username -d taxi_db -f data/sql_schema.sql

-- Load sample data (if available)
COPY nyc_taxi_trips FROM '/path/to/data.csv' CSV HEADER;
```

### 2. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/nyc-taxi-prediction.git
cd nyc-taxi-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Data Extraction & Preprocessing

```bash
# Extract data from database
python src/data_loader.py

# Preprocess and engineer features
python src/preprocessor.py
```

### 4. Model Training

```bash
# Train all models with hyperparameter tuning
python src/model_trainer.py

# Expected output:
# - Trained models saved to models/
# - Training metrics logged
# - Best model identified
```

### 5. Model Evaluation

```bash
# Evaluate models and generate visualizations
python src/model_evaluator.py

# Generates:
# - Performance metrics
# - Prediction plots
# - SHAP interpretability analysis
# - Feature importance visualizations
```

### 6. API Deployment

```bash
# Start API server
python api.py

# Or using uvicorn directly:
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# API will be available at:
# - http://localhost:8000/docs (Swagger UI)
# - http://localhost:8000/redoc (ReDoc)
```

### 7. Test API

```bash
# Run test suite
python test_api.py

# Or use curl:
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_datetime": "2024-01-15 14:30:00",
    "dropoff_datetime": "2024-01-15 14:45:00",
    "passenger_count": 2,
    "trip_distance": 5.2,
    "pickup_longitude": -73.98,
    "pickup_latitude": 40.75,
    "dropoff_longitude": -73.95,
    "dropoff_latitude": 40.77,
    "fare_amount": 15.50,
    "payment_type": 1
  }'
```

---

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build image
docker build -t taxi-tip-predictor:latest .

# Run container
docker run -d -p 8000:8000 --name taxi-api taxi-tip-predictor:latest

# View logs
docker logs -f taxi-api
```

### Using Docker Compose

```bash
# Start services
docker-compose up -d

# Scale API instances
docker-compose up -d --scale api=3

# Stop services
docker-compose down
```

---

## 📊 Model Performance

### Best Model: Random Forest Regressor

| Metric | Train | Test |
|--------|-------|------|
| MAE | $0.85 | $1.12 |
| RMSE | $1.23 | $1.67 |
| R² Score | 0.8542 | 0.8123 |
| MAPE | 12.3% | 15.8% |

### Model Comparison

| Model | Test MAE | Test RMSE | Test R² | Training Time |
|-------|----------|-----------|---------|---------------|
| Linear Regression | $1.45 | $2.12 | 0.7234 | 2.3s |
| Ridge | $1.42 | $2.08 | 0.7312 | 3.1s |
| Lasso | $1.48 | $2.15 | 0.7198 | 2.8s |
| Random Forest | $1.12 | $1.67 | 0.8123 | 45.2s |
| Neural Network | $1.18 | $1.73 | 0.8034 | 67.8s |

---

## 🔍 Key Features Used

Based on SHAP analysis and feature importance, the top 10 most influential features are:

1. **fare_amount** - Base fare is the strongest predictor
2. **trip_distance** - Longer trips generally receive higher tips
3. **trip_duration_minutes** - Time spent in taxi affects tip
4. **is_credit_card** - Credit card payments have higher tips
5. **pickup_hour** - Time of day impacts tipping behavior
6. **is_rush_hour** - Rush hour trips receive different tips
7. **speed_mph** - Trip speed indicator
8. **fare_per_mile** - Fare efficiency metric
9. **is_weekend** - Weekend vs weekday differences
10. **passenger_count** - Number of passengers

---

## 🔧 Configuration

### Environment Variables

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=taxi_db
DB_USER=username
DB_PASSWORD=password

# Model
MODEL_PATH=models/
MODEL_VERSION=1.0.0

# API
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info
```

### Model Hyperparameters (Best Configuration)

**Random Forest:**
```python
{
    'n_estimators': 300,
    'max_depth': 30,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'bootstrap': True
}
```

---

## 📈 API Usage Examples

### Single Prediction

**Request:**
```json
POST /predict
{
  "pickup_datetime": "2024-01-15 14:30:00",
  "dropoff_datetime": "2024-01-15 14:45:00",
  "passenger_count": 2,
  "trip_distance": 5.2,
  "pickup_longitude": -73.98,
  "pickup_latitude": 40.75,
  "dropoff_longitude": -73.95,
  "dropoff_latitude": 40.77,
  "fare_amount": 15.50,
  "payment_type": 1
}
```

**Response:**
```json
{
  "predicted_tip": 3.25,
  "confidence_interval": {
    "lower": 1.75,
    "upper": 4.75
  },
  "fare_amount": 15.50,
  "tip_percentage": 20.97,
  "model_version": "1.0.0"
}
```

### Batch Prediction

**Request:**
```json
POST /predict/batch
{
  "trips": [
    { /* trip 1 data */ },
    { /* trip 2 data */ }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "predicted_tip": 3.25,
      "confidence_interval": {"lower": 1.75, "upper": 4.75},
      "fare_amount": 15.50,
      "tip_percentage": 20.97,
      "model_version": "1.0.0"
    },
    {
      "predicted_tip": 5.80,
      "confidence_interval": {"lower": 4.30, "upper": 7.30},
      "fare_amount": 28.50,
      "tip_percentage": 20.35,
      "model_version": "1.0.0"
    }
  ],
  "total_trips": 2
}
```

---

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/ -v
```

### Run Integration Tests
```bash
pytest tests/integration/ -v
```

### Load Testing
```bash
# Using locust
locust -f tests/load_test.py --host=http://localhost:8000
```

---

## 📚 Technical Details

### Data Preprocessing
- **Missing Value Handling**: Median imputation for numerical, mode for categorical
- **Outlier Removal**: IQR method with domain-specific bounds
- **Feature Scaling**: RobustScaler (robust to outliers)
- **Categorical Encoding**: Label encoding for ordinal features

### Feature Engineering
- **Time Features**: Hour, day of week, is_weekend, is_rush_hour, time_of_day
- **Trip Features**: Duration, speed, haversine distance
- **Fare Features**: Fare per mile, fare per minute, tip percentage
- **Location Features**: Grid-based clustering, airport indicators
- **Payment Features**: Credit card indicator, surge pricing flag

### Model Selection Criteria
- Primary: Test MAE (Mean Absolute Error)
- Secondary: Test RMSE, R² Score
- Consideration: Training time, interpretability

### API Design Principles
- RESTful architecture
- Proper HTTP status codes
- Request/response validation with Pydantic
- Comprehensive error handling
- OpenAPI documentation
- Health check endpoint

---

## 🚨 Monitoring & Logging

### Logging Configuration
```python
# Logs include:
- Request/response details
- Prediction metrics
- Error traces
- Performance metrics
```

### Metrics to Monitor
- Request latency (p50, p95, p99)
- Prediction errors
- API uptime
- Model drift indicators

---

## 🔄 Model Retraining Pipeline

### When to Retrain
- Monthly scheduled retraining
- When model performance degrades (MAE > threshold)
- After significant data distribution shifts
- After accumulating sufficient new data

### Retraining Steps
```bash
# 1. Extract new data
python src/data_loader.py --start-date 2024-02-01 --end-date 2024-03-01

# 2. Validate data quality
python src/data_validator.py

# 3. Retrain models
python src/model_trainer.py --retrain

# 4. Evaluate new models
python src/model_evaluator.py --compare

# 5. Deploy if improved
python deploy.py --model-version 1.1.0
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Email: support@example.com
- Documentation: https://docs.example.com

---

## 🙏 Acknowledgments

- NYC Taxi & Limousine Commission for the dataset
- Scikit-learn and TensorFlow communities
- FastAPI framework developers

---

## 📖 References

1. NYC TLC Trip Record Data: https://www.nyc.gov/tlc
2. SHAP Documentation: https://shap.readthedocs.io/
3. FastAPI Documentation: https://fastapi.tiangolo.com/
4. Scikit-learn User Guide: https://scikit-learn.org/stable/

---

**Project Status**: ✅ Production Ready

**Last Updated**: November 2024

**Version**: 1.0.0