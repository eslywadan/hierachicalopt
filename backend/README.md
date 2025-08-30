# LSTM Backend Service

Flask-based backend service for high-performance LSTM model training and prediction.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python run.py
```

The server will start at `http://localhost:5000`

## 📡 API Endpoints

### Health Check
- **GET** `/health` - Check if the service is running

### Data Generation
- **POST** `/api/data/generate` - Generate synthetic training data

### Model Training
- **POST** `/api/model/train` - Train LSTM model
- **GET** `/api/models` - List all trained models
- **DELETE** `/api/model/{model_id}` - Delete a trained model

### Predictions
- **POST** `/api/model/predict` - Make predictions using trained model

### Validation
- **POST** `/api/model/validate` - Validate production parameters using Little's Law

## 🧠 Model Architecture

The LSTM service uses a dual-layer LSTM architecture:
- **Layer 1**: Configurable LSTM units (default: 64)
- **Layer 2**: Configurable LSTM units (default: 32)
- **Dense Layers**: 64 → 32 → 3 (WIP, Throughput, Cycle Time)
- **Dropout**: Configurable rate for regularization
- **Optimizer**: Adam with configurable learning rate

## 📊 Features

### Data Generation
- Synthetic manufacturing data generation
- Realistic seasonality and noise modeling
- Multiple plants, applications, and panel sizes
- Little's Law compliance validation

### Model Training
- TensorFlow/Keras backend for optimal performance
- Configurable model architecture
- Real-time training progress (planned)
- Model persistence using native Keras format (.keras)
- Backward compatibility with legacy HDF5 models (.h5)
- Performance metrics (R², RMSE, MAE)

### Predictions
- Multi-day predictions
- Confidence estimation
- Little's Law validation
- Production feasibility checks

### Validation Services
- Little's Law compliance checking
- Production parameter validation
- Optimization recommendations
- Feasibility analysis

## 🔧 Configuration

### Training Configuration
```json
{
  "lstmUnits1": 64,
  "lstmUnits2": 32,
  "dropoutRate": 0.2,
  "sequenceLength": 10,
  "epochs": 50,
  "batchSize": 32,
  "learningRate": 0.00001,
  "trainTestSplit": 0.8
}
```

### Data Configuration
```json
{
  "plants": ["Plant_A", "Plant_B", "Plant_C"],
  "applications": ["Automotive", "Consumer_Electronics", "Industrial"],
  "panelSizes": ["Small", "Medium", "Large", "Extra_Large"],
  "historicalDays": 545,
  "baseWIP": 100,
  "baseThroughput": 50,
  "seasonality": 0.2,
  "noiseLevel": 0.1
}
```

## 📁 Project Structure

```
backend/
├── app.py                 # Main Flask application
├── run.py                 # Server startup script
├── requirements.txt       # Python dependencies
├── models/
│   ├── lstm_service.py    # LSTM model training and prediction
│   ├── data_generator.py  # Synthetic data generation
│   └── __init__.py
├── utils/
│   ├── validation.py      # Little's Law validation services
│   └── __init__.py
└── saved_models/          # Model persistence directory
```

## 🔍 Usage Examples

### Angular Frontend Integration
The Angular frontend at `http://localhost:4200/operation-model/backend` provides a complete UI for:
- Backend health monitoring
- Training configuration
- Model training with progress tracking
- Prediction requests
- Model management
- Activity logging

### API Usage Examples

#### Generate Training Data
```bash
curl -X POST http://localhost:5000/api/data/generate \
  -H "Content-Type: application/json" \
  -d '{
    "historicalDays": 365,
    "baseWIP": 100,
    "baseThroughput": 50
  }'
```

#### Train Model
```bash
curl -X POST http://localhost:5000/api/model/train \
  -H "Content-Type: application/json" \
  -d '{
    "epochs": 30,
    "batchSize": 32,
    "learningRate": 0.00001
  }'
```

#### Make Prediction
```bash
curl -X POST http://localhost:5000/api/model/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "your-model-id",
    "plant": "Plant_A",
    "application": "Automotive",
    "panelSize": "Medium",
    "currentWIP": 120,
    "plannedThroughput": 60,
    "predictionDays": 30
  }'
```

## ⚡ Performance Benefits

### vs. TensorFlow.js (Frontend)
- **Training Speed**: 5-10x faster using native TensorFlow
- **Model Complexity**: Support for larger, more complex models
- **Memory Usage**: Better memory management for large datasets
- **Scalability**: Can handle multiple concurrent requests
- **GPU Support**: Optional GPU acceleration available

### Optimizations
- Model caching for fast predictions
- Efficient data preprocessing pipelines
- Optimized TensorFlow configurations
- Background training support (planned)
- Model compression (planned)

## 🛠️ Development

### Adding New Features
1. Add service methods in appropriate modules
2. Create API endpoints in `app.py`
3. Update Angular service for frontend integration
4. Add tests and documentation

### Debugging
- Set `debug=True` in Flask app for detailed error messages
- Check logs for training progress and errors
- Use `/health` endpoint to verify service status

## 📈 Monitoring

The service provides comprehensive logging for:
- Training progress and metrics
- API request/response cycles
- Error tracking and debugging
- Model performance monitoring

## 🔒 Security Notes

This is a development service. For production deployment:
- Add authentication and authorization
- Implement rate limiting
- Add input validation and sanitization
- Use production WSGI server (gunicorn)
- Add HTTPS support