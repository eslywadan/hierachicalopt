"""
LSTM Model Service
Handles training, prediction, and management of LSTM models for manufacturing operations
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import uuid
import time
import logging
from typing import List, Dict, Any, Tuple
import pickle
import os

# Check TensorFlow/Keras version
TF_VERSION = tf.__version__
KERAS_MAJOR_VERSION = int(tf.keras.__version__.split('.')[0]) if hasattr(tf.keras, '__version__') else 2

logger = logging.getLogger(__name__)

class LSTMModelService:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.model_metadata = {}
        
        # Create model storage directory
        self.model_dir = "saved_models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        logger.info("🧠 LSTM Model Service initialized")
        logger.info(f"📦 TensorFlow version: {TF_VERSION}")
        logger.info(f"📦 Keras version: {tf.keras.__version__ if hasattr(tf.keras, '__version__') else 'integrated'}")

    def train_model(self, training_data: List[Dict], config: Dict) -> Dict[str, Any]:
        """
        Train LSTM model with the provided data and configuration
        """
        start_time = time.time()
        model_id = str(uuid.uuid4())
        
        logger.info(f"🏋️ Training LSTM model {model_id} with {len(training_data)} data points")
        
        try:
            # Convert training data to DataFrame
            df = pd.DataFrame(training_data)
            
            # Prepare features and targets
            X, y, feature_scaler, target_scaler, encoders = self._prepare_data(df, config)
            
            # Create sequences for LSTM
            X_seq, y_seq = self._create_sequences(X, y, config['sequence_length'])
            
            # Split data
            split_idx = int(len(X_seq) * config['train_test_split'])
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
            
            logger.info(f"📊 Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
            
            # Build LSTM model
            model = self._build_lstm_model(
                input_shape=(config['sequence_length'], X_train.shape[2]),
                config=config
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                validation_data=(X_test, y_test),
                verbose=1,
                shuffle=False  # Important for time series data
            )
            
            # Evaluate model
            y_pred = model.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred, target_scaler)
            
            # Save model and associated data
            self._save_model(model_id, model, feature_scaler, target_scaler, encoders, config, metrics)
            
            training_time = time.time() - start_time
            
            logger.info(f"✅ Model {model_id} trained successfully in {training_time:.2f}s")
            logger.info(f"📈 Model metrics: R²={metrics['r2_score']:.4f}, RMSE={metrics['rmse']:.4f}")
            
            return {
                'model_id': model_id,
                'history': {
                    'loss': [float(x) for x in history.history['loss']],
                    'val_loss': [float(x) for x in history.history['val_loss']],
                    'epochs': len(history.history['loss'])
                },
                'metrics': metrics,
                'training_time': training_time,
                'model_info': {
                    'input_shape': list(X_train.shape[1:]),
                    'output_shape': y_train.shape[1:],
                    'total_params': model.count_params(),
                    'config': config,
                    'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error training model {model_id}: {str(e)}")
            raise e

    def predict(self, model_id: str, prediction_request: Dict) -> Dict[str, Any]:
        """
        Make predictions using a trained model
        """
        if model_id not in self.models:
            self._load_model(model_id)
        
        logger.info(f"🔮 Making prediction with model {model_id}")
        
        try:
            model = self.models[model_id]
            feature_scaler = self.scalers[f"{model_id}_features"]
            target_scaler = self.scalers[f"{model_id}_targets"]
            encoders = self.encoders[model_id]
            config = self.model_metadata[model_id]['config']
            
            # Prepare input data
            input_features = self._prepare_prediction_input(
                prediction_request, encoders, feature_scaler, config
            )
            
            # Make prediction
            prediction = model.predict(input_features)
            
            # Inverse transform prediction
            prediction_scaled = target_scaler.inverse_transform(prediction)
            
            # Generate predictions for multiple days
            predictions = []
            for i in range(prediction_request['prediction_days']):
                pred_point = {
                    'day': i + 1,
                    'predicted_wip': float(prediction_scaled[0, 0]),
                    'predicted_throughput': float(prediction_scaled[0, 1]),
                    'predicted_cycle_time': float(prediction_scaled[0, 2]) if prediction_scaled.shape[1] > 2 else 0.0,
                    'confidence': 0.85  # Simple confidence estimation
                }
                predictions.append(pred_point)
            
            # Validate predictions using Little's Law
            validation = self._validate_predictions(predictions)
            
            # Little's Law analysis
            little_law_analysis = self._analyze_littles_law(predictions)
            
            return {
                'predictions': predictions,
                'validation': validation,
                'little_law_analysis': little_law_analysis
            }
            
        except Exception as e:
            logger.error(f"❌ Error making prediction: {str(e)}")
            raise e

    def _prepare_data(self, df: pd.DataFrame, config: Dict) -> Tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler, Dict]:
        """
        Prepare data for LSTM training
        """
        logger.info("🔄 Preparing data for LSTM training...")
        
        # Encode categorical variables
        encoders = {}
        categorical_features = ['plant', 'application', 'panel_size']
        
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
                encoders[feature] = le
        
        # Select features for training
        feature_columns = [
            'day', 'wip', 'throughput', 'cycle_time', 
            'finished_goods', 'semi_finished_goods'
        ]
        
        # Add encoded categorical features
        for feature in categorical_features:
            if f'{feature}_encoded' in df.columns:
                feature_columns.append(f'{feature}_encoded')
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        X = df[available_features].values
        
        # Target variables (what we want to predict)
        target_columns = ['wip', 'throughput', 'cycle_time']
        available_targets = [col for col in target_columns if col in df.columns]
        
        y = df[available_targets].values
        
        # Scale features and targets
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        X_scaled = feature_scaler.fit_transform(X)
        y_scaled = target_scaler.fit_transform(y)
        
        logger.info(f"📊 Features shape: {X_scaled.shape}, Targets shape: {y_scaled.shape}")
        
        return X_scaled, y_scaled, feature_scaler, target_scaler, encoders

    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        """
        logger.info(f"🔗 Creating sequences with length {sequence_length}")
        
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)

    def _build_lstm_model(self, input_shape: Tuple[int, int], config: Dict) -> tf.keras.Model:
        """
        Build LSTM model architecture
        """
        logger.info(f"🏗️ Building LSTM model with input shape: {input_shape}")
        
        model = Sequential([
            LSTM(
                config['lstm_units_1'], 
                return_sequences=True,
                input_shape=input_shape
            ),
            Dropout(config['dropout_rate']),
            
            LSTM(
                config['lstm_units_2'], 
                return_sequences=False
            ),
            Dropout(config['dropout_rate']),
            
            Dense(64, activation='relu'),
            Dropout(config['dropout_rate']),
            
            Dense(32, activation='relu'),
            
            # Output layer - adjust based on target dimensions
            Dense(3)  # wip, throughput, cycle_time
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"🏗️ Model built with {model.count_params()} parameters")
        
        return model

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, target_scaler: StandardScaler) -> Dict[str, float]:
        """
        Calculate model performance metrics
        """
        # Inverse transform for real-scale metrics
        y_true_scaled = target_scaler.inverse_transform(y_true)
        y_pred_scaled = target_scaler.inverse_transform(y_pred)
        
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_true_scaled, y_pred_scaled))),
            'mae': float(mean_absolute_error(y_true_scaled, y_pred_scaled)),
            'r2_score': float(r2_score(y_true_scaled, y_pred_scaled))
        }
        
        return metrics

    def _save_model(self, model_id: str, model: tf.keras.Model, feature_scaler: StandardScaler, 
                   target_scaler: StandardScaler, encoders: Dict, config: Dict, metrics: Dict):
        """
        Save model and associated data
        """
        model_path = os.path.join(self.model_dir, f"{model_id}.keras")
        
        # Save Keras model - format is automatically inferred from .keras extension in Keras 3
        model.save(model_path)
        
        # Save scalers and encoders
        scalers_path = os.path.join(self.model_dir, f"{model_id}_scalers.pkl")
        with open(scalers_path, 'wb') as f:
            pickle.dump({
                'feature_scaler': feature_scaler,
                'target_scaler': target_scaler,
                'encoders': encoders
            }, f)
        
        # Store in memory
        self.models[model_id] = model
        self.scalers[f"{model_id}_features"] = feature_scaler
        self.scalers[f"{model_id}_targets"] = target_scaler
        self.encoders[model_id] = encoders
        self.model_metadata[model_id] = {
            'config': config,
            'metrics': metrics,
            'model_path': model_path,
            'scalers_path': scalers_path
        }
        
        logger.info(f"💾 Model {model_id} saved successfully")

    def _load_model(self, model_id: str):
        """
        Load model from disk
        """
        if model_id not in self.model_metadata:
            raise ValueError(f"Model {model_id} not found in metadata")
        
        metadata = self.model_metadata[model_id]
        model_path = metadata['model_path']
        
        # Check if model exists with .keras extension, otherwise try .h5 for backward compatibility
        if not os.path.exists(model_path) and model_path.endswith('.keras'):
            # Try legacy .h5 format
            legacy_path = model_path.replace('.keras', '.h5')
            if os.path.exists(legacy_path):
                logger.warning(f"Loading legacy HDF5 model. Consider re-saving in native Keras format.")
                model_path = legacy_path
        
        # Load Keras model
        self.models[model_id] = tf.keras.models.load_model(model_path)
        
        # Load scalers and encoders
        with open(metadata['scalers_path'], 'rb') as f:
            data = pickle.load(f)
            self.scalers[f"{model_id}_features"] = data['feature_scaler']
            self.scalers[f"{model_id}_targets"] = data['target_scaler']
            self.encoders[model_id] = data['encoders']
        
        logger.info(f"📂 Model {model_id} loaded successfully")

    def _prepare_prediction_input(self, request: Dict, encoders: Dict, feature_scaler: StandardScaler, config: Dict) -> np.ndarray:
        """
        Prepare input data for prediction
        """
        # Create dummy sequence data (simplified approach)
        sequence_length = config['sequence_length']
        
        # Create base feature vector
        features = [
            1,  # day
            request.get('current_wip', 100),
            request.get('planned_throughput', 50),
            request.get('current_wip', 100) / max(request.get('planned_throughput', 50), 1),  # cycle_time
            request.get('target_production', 1000),  # finished_goods
            request.get('current_wip', 100) * 0.5  # semi_finished_goods
        ]
        
        # Add encoded categorical features
        for feature_name, encoder in encoders.items():
            if feature_name in request:
                try:
                    encoded_value = encoder.transform([request[feature_name]])[0]
                    features.append(encoded_value)
                except ValueError:
                    # Use first class if value not seen during training
                    features.append(0)
        
        # Create sequence
        features_array = np.array(features)
        sequence = np.tile(features_array, (sequence_length, 1))
        
        # Scale features
        sequence_scaled = feature_scaler.transform(sequence)
        
        # Reshape for LSTM input
        input_data = sequence_scaled.reshape(1, sequence_length, -1)
        
        return input_data

    def _validate_predictions(self, predictions: List[Dict]) -> Dict[str, Any]:
        """
        Validate predictions for feasibility
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'recommendations': []
        }
        
        for pred in predictions:
            wip = pred['predicted_wip']
            throughput = pred['predicted_throughput']
            cycle_time = pred['predicted_cycle_time']
            
            # Check for negative values
            if wip < 0 or throughput < 0 or cycle_time < 0:
                validation['is_valid'] = False
                validation['warnings'].append(f"Day {pred['day']}: Negative values predicted")
            
            # Check Little's Law compliance (WIP = Throughput × Cycle Time)
            expected_wip = throughput * cycle_time
            if abs(wip - expected_wip) > wip * 0.2:  # 20% tolerance
                validation['warnings'].append(
                    f"Day {pred['day']}: Little's Law deviation - "
                    f"WIP={wip:.1f}, Expected={expected_wip:.1f}"
                )
        
        return validation

    def _analyze_littles_law(self, predictions: List[Dict]) -> Dict[str, Any]:
        """
        Analyze Little's Law compliance in predictions
        """
        compliance_scores = []
        
        for pred in predictions:
            wip = pred['predicted_wip']
            throughput = pred['predicted_throughput']
            cycle_time = pred['predicted_cycle_time']
            
            if throughput > 0:
                expected_wip = throughput * cycle_time
                compliance = 1 - abs(wip - expected_wip) / max(wip, expected_wip)
                compliance_scores.append(max(0, compliance))
            else:
                compliance_scores.append(0)
        
        avg_compliance = np.mean(compliance_scores) if compliance_scores else 0
        
        return {
            'average_compliance': float(avg_compliance),
            'daily_compliance': [float(score) for score in compliance_scores],
            'analysis': {
                'excellent': sum(1 for s in compliance_scores if s > 0.9),
                'good': sum(1 for s in compliance_scores if 0.7 <= s <= 0.9),
                'fair': sum(1 for s in compliance_scores if 0.5 <= s < 0.7),
                'poor': sum(1 for s in compliance_scores if s < 0.5)
            }
        }