"""
Parallel LSTM Model Service
Handles parallel training of plant-specific LSTM models for manufacturing operations
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
from typing import List, Dict, Any, Tuple, Optional
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PlantModelInfo:
    """Information about a plant-specific model"""
    model_id: str
    plant: str
    created_at: datetime
    metrics: Dict[str, float]
    config: Dict[str, Any]
    training_time: float
    data_points: int
    status: str  # 'training', 'completed', 'failed'

class ParallelLSTMService:
    def __init__(self, max_workers: int = 4):
        """
        Initialize Parallel LSTM Service
        
        Args:
            max_workers: Maximum number of parallel training processes
        """
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.model_metadata = {}
        self.plant_models = {}  # Maps plant -> list of model IDs
        self.training_status = {}  # Track training status for each plant
        self.max_workers = max_workers
        self.lock = threading.Lock()
        
        # Create model storage directory
        self.model_dir = "saved_models"
        self.plant_model_dir = os.path.join(self.model_dir, "plant_specific")
        os.makedirs(self.plant_model_dir, exist_ok=True)
        
        logger.info("🚀 Parallel LSTM Model Service initialized")
        logger.info(f"🔧 Max parallel workers: {max_workers}")
        logger.info(f"📦 TensorFlow version: {tf.__version__}")
    
    def train_plant_models_parallel(self, training_data: List[Dict], config: Dict, 
                                   plants: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train separate models for each plant in parallel
        
        Args:
            training_data: Complete training dataset
            config: LSTM configuration
            plants: List of plants to train (None = all plants in data)
        
        Returns:
            Dictionary with training results for each plant
        """
        start_time = time.time()
        df = pd.DataFrame(training_data)
        
        # Determine plants to train
        if plants is None:
            plants = df['plant'].unique().tolist() if 'plant' in df.columns else []
        
        if not plants:
            raise ValueError("No plants found in training data")
        
        logger.info(f"🏭 Starting parallel training for {len(plants)} plants: {plants}")
        
        # Initialize status for each plant
        for plant in plants:
            self.training_status[plant] = {
                'status': 'pending',
                'progress': 0,
                'message': 'Waiting to start...'
            }
        
        results = {}
        failed_plants = []
        
        # Use ThreadPoolExecutor for parallel training
        # Note: For true parallel processing with TensorFlow, consider using separate processes
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(plants))) as executor:
            # Submit training tasks for each plant
            future_to_plant = {
                executor.submit(self._train_single_plant_model, 
                              plant, 
                              df[df['plant'] == plant].to_dict('records'),
                              config): plant
                for plant in plants
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_plant):
                plant = future_to_plant[future]
                try:
                    result = future.result()
                    results[plant] = result
                    self.training_status[plant]['status'] = 'completed'
                    self.training_status[plant]['progress'] = 100
                    self.training_status[plant]['message'] = 'Training completed successfully'
                    logger.info(f"✅ Plant {plant} training completed")
                except Exception as e:
                    logger.error(f"❌ Plant {plant} training failed: {str(e)}")
                    failed_plants.append(plant)
                    results[plant] = {
                        'success': False,
                        'error': str(e)
                    }
                    self.training_status[plant]['status'] = 'failed'
                    self.training_status[plant]['message'] = f'Training failed: {str(e)}'
        
        total_time = time.time() - start_time
        
        # Compile overall results
        overall_results = {
            'success': len(failed_plants) == 0,
            'total_plants': len(plants),
            'successful_plants': len(plants) - len(failed_plants),
            'failed_plants': failed_plants,
            'total_training_time': total_time,
            'average_time_per_plant': total_time / len(plants) if plants else 0,
            'plant_results': results,
            'training_status': self.training_status
        }
        
        logger.info(f"🎉 Parallel training completed in {total_time:.2f}s")
        logger.info(f"📊 Success rate: {overall_results['successful_plants']}/{overall_results['total_plants']}")
        
        return overall_results
    
    def _train_single_plant_model(self, plant: str, plant_data: List[Dict], 
                                 config: Dict) -> Dict[str, Any]:
        """
        Train a model for a single plant
        
        Args:
            plant: Plant identifier
            plant_data: Training data for this plant
            config: LSTM configuration
        
        Returns:
            Training results for this plant
        """
        start_time = time.time()
        model_id = f"{plant}_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"🏭 Training model for plant {plant} (ID: {model_id}) with {len(plant_data)} samples")
        
        # Update status
        with self.lock:
            self.training_status[plant] = {
                'status': 'training',
                'progress': 10,
                'message': f'Processing {len(plant_data)} samples...'
            }
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(plant_data)
            
            # Prepare data
            X, y, feature_scaler, target_scaler, encoders = self._prepare_plant_data(df, config, plant)
            
            # Update progress
            with self.lock:
                self.training_status[plant]['progress'] = 30
                self.training_status[plant]['message'] = 'Creating sequences...'
            
            # Create sequences
            X_seq, y_seq = self._create_sequences(X, y, config['sequence_length'])
            
            if len(X_seq) < 10:
                raise ValueError(f"Insufficient data for plant {plant}: only {len(X_seq)} sequences")
            
            # Split data
            split_idx = int(len(X_seq) * config['train_test_split'])
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
            
            # Update progress
            with self.lock:
                self.training_status[plant]['progress'] = 40
                self.training_status[plant]['message'] = f'Training on {len(X_train)} samples...'
            
            # Build model
            model = self._build_plant_lstm_model(
                input_shape=(config['sequence_length'], X_train.shape[2]),
                config=config,
                plant=plant
            )
            
            # Custom callback to update progress
            progress_callback = PlantTrainingCallback(plant, self.training_status, self.lock)
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                validation_data=(X_test, y_test),
                verbose=0,  # Reduce output in parallel training
                callbacks=[progress_callback],
                shuffle=False
            )
            
            # Update progress
            with self.lock:
                self.training_status[plant]['progress'] = 90
                self.training_status[plant]['message'] = 'Evaluating model...'
            
            # Evaluate model
            y_pred = model.predict(X_test, verbose=0)
            metrics = self._calculate_metrics(y_test, y_pred, target_scaler)
            
            # Save model
            self._save_plant_model(model_id, plant, model, feature_scaler, 
                                 target_scaler, encoders, config, metrics)
            
            # Store plant model mapping
            with self.lock:
                if plant not in self.plant_models:
                    self.plant_models[plant] = []
                self.plant_models[plant].append(model_id)
            
            training_time = time.time() - start_time
            
            # Create plant model info
            model_info = PlantModelInfo(
                model_id=model_id,
                plant=plant,
                created_at=datetime.now(),
                metrics=metrics,
                config=config,
                training_time=training_time,
                data_points=len(plant_data),
                status='completed'
            )
            
            # Store metadata
            with self.lock:
                self.model_metadata[model_id] = model_info
            
            logger.info(f"✅ Plant {plant} model trained: R²={metrics['r2_score']:.4f}, "
                       f"RMSE={metrics['rmse']:.2f}, Time={training_time:.2f}s")
            
            return {
                'success': True,
                'model_id': model_id,
                'plant': plant,
                'metrics': metrics,
                'training_time': training_time,
                'data_points': len(plant_data),
                'history': {
                    'loss': [float(x) for x in history.history['loss'][-10:]],  # Last 10 epochs
                    'val_loss': [float(x) for x in history.history['val_loss'][-10:]]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error training plant {plant} model: {str(e)}")
            with self.lock:
                self.training_status[plant]['status'] = 'failed'
                self.training_status[plant]['message'] = str(e)
            raise
    
    def _prepare_plant_data(self, df: pd.DataFrame, config: Dict, plant: str) -> Tuple:
        """
        Prepare data for a specific plant
        """
        logger.info(f"🔄 Preparing data for plant {plant}...")
        
        # Encode categorical variables (excluding plant since it's fixed)
        encoders = {}
        categorical_features = ['application', 'panel_size']
        
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
                encoders[feature] = le
        
        # Select features
        feature_columns = [
            'day', 'wip', 'throughput', 'cycle_time', 
            'finished_goods', 'semi_finished_goods'
        ]
        
        # Add encoded features
        for feature in categorical_features:
            if f'{feature}_encoded' in df.columns:
                feature_columns.append(f'{feature}_encoded')
        
        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features].values
        
        # Target variables
        target_columns = ['wip', 'throughput', 'cycle_time']
        available_targets = [col for col in target_columns if col in df.columns]
        y = df[available_targets].values
        
        # Scale features and targets
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        X_scaled = feature_scaler.fit_transform(X)
        y_scaled = target_scaler.fit_transform(y)
        
        return X_scaled, y_scaled, feature_scaler, target_scaler, encoders
    
    def _build_plant_lstm_model(self, input_shape: Tuple[int, int], 
                               config: Dict, plant: str) -> tf.keras.Model:
        """
        Build LSTM model for a specific plant
        """
        logger.info(f"🏗️ Building LSTM model for plant {plant} with shape: {input_shape}")
        
        model = Sequential([
            LSTM(
                config['lstm_units_1'], 
                return_sequences=True,
                input_shape=input_shape,
                name=f'{plant}_lstm_1'
            ),
            Dropout(config['dropout_rate']),
            
            LSTM(
                config['lstm_units_2'], 
                return_sequences=False,
                name=f'{plant}_lstm_2'
            ),
            Dropout(config['dropout_rate']),
            
            Dense(64, activation='relu', name=f'{plant}_dense_1'),
            Dropout(config['dropout_rate']),
            
            Dense(32, activation='relu', name=f'{plant}_dense_2'),
            
            Dense(3, name=f'{plant}_output')  # wip, throughput, cycle_time
        ], name=f'plant_{plant}_model')
        
        optimizer = Adam(learning_rate=config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _save_plant_model(self, model_id: str, plant: str, model: tf.keras.Model,
                         feature_scaler: StandardScaler, target_scaler: StandardScaler,
                         encoders: Dict, config: Dict, metrics: Dict):
        """
        Save plant-specific model
        """
        # Create plant directory
        plant_dir = os.path.join(self.plant_model_dir, plant)
        os.makedirs(plant_dir, exist_ok=True)
        
        model_path = os.path.join(plant_dir, f"{model_id}.keras")
        model.save(model_path)
        
        # Save scalers and metadata
        metadata_path = os.path.join(plant_dir, f"{model_id}_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'feature_scaler': feature_scaler,
                'target_scaler': target_scaler,
                'encoders': encoders,
                'config': config,
                'metrics': metrics,
                'plant': plant,
                'model_id': model_id,
                'created_at': datetime.now().isoformat()
            }, f)
        
        logger.info(f"💾 Plant {plant} model {model_id} saved successfully")
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, 
                         sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          target_scaler: StandardScaler) -> Dict[str, float]:
        """Calculate model performance metrics"""
        y_true_scaled = target_scaler.inverse_transform(y_true)
        y_pred_scaled = target_scaler.inverse_transform(y_pred)
        
        return {
            'rmse': float(np.sqrt(mean_squared_error(y_true_scaled, y_pred_scaled))),
            'mae': float(mean_absolute_error(y_true_scaled, y_pred_scaled)),
            'r2_score': float(r2_score(y_true_scaled, y_pred_scaled))
        }
    
    def get_best_model_for_plant(self, plant: str) -> Optional[str]:
        """
        Get the best performing model for a specific plant
        
        Args:
            plant: Plant identifier
        
        Returns:
            Model ID of the best model or None
        """
        if plant not in self.plant_models:
            return None
        
        best_model_id = None
        best_r2 = -float('inf')
        
        for model_id in self.plant_models[plant]:
            if model_id in self.model_metadata:
                metrics = self.model_metadata[model_id].metrics
                if metrics.get('r2_score', -float('inf')) > best_r2:
                    best_r2 = metrics['r2_score']
                    best_model_id = model_id
        
        return best_model_id
    
    def predict_with_plant_model(self, plant: str, prediction_request: Dict) -> Dict[str, Any]:
        """
        Make predictions using the best model for a specific plant
        
        Args:
            plant: Plant identifier
            prediction_request: Prediction parameters
        
        Returns:
            Prediction results
        """
        model_id = self.get_best_model_for_plant(plant)
        
        if not model_id:
            raise ValueError(f"No trained model found for plant {plant}")
        
        logger.info(f"🔮 Making prediction for plant {plant} using model {model_id}")
        
        # Load model if not in memory
        if model_id not in self.models:
            self._load_plant_model(plant, model_id)
        
        try:
            model = self.models[model_id]
            feature_scaler = self.scalers[f"{model_id}_features"]
            target_scaler = self.scalers[f"{model_id}_targets"]
            encoders = self.encoders[model_id]
            
            # Get model metadata for configuration
            if model_id in self.model_metadata:
                config = self.model_metadata[model_id].config
            else:
                # Use default config if metadata not available
                config = {'sequence_length': 10}
            
            # Prepare input data for plant-specific prediction
            input_features = self._prepare_plant_prediction_input(
                prediction_request, encoders, feature_scaler, config, plant
            )
            
            # Make prediction
            prediction = model.predict(input_features, verbose=0)
            
            # Inverse transform prediction
            prediction_scaled = target_scaler.inverse_transform(prediction)
            
            # Generate predictions for multiple days
            predictions = []
            prediction_days = prediction_request.get('prediction_days', 30)
            
            for i in range(prediction_days):
                pred_point = {
                    'day': i + 1,
                    'plant': plant,
                    'predicted_wip': float(prediction_scaled[0, 0]),
                    'predicted_throughput': float(prediction_scaled[0, 1]),
                    'predicted_cycle_time': float(prediction_scaled[0, 2]) if prediction_scaled.shape[1] > 2 else 0.0,
                    'confidence': 0.85  # Simple confidence estimation
                }
                predictions.append(pred_point)
            
            # Validate predictions using Little's Law
            validation = self._validate_plant_predictions(predictions)
            
            # Little's Law analysis
            little_law_analysis = self._analyze_plant_littles_law(predictions)
            
            return {
                'plant': plant,
                'model_id': model_id,
                'predictions': predictions,
                'validation': validation,
                'little_law_analysis': little_law_analysis
            }
            
        except Exception as e:
            logger.error(f"❌ Error making prediction for plant {plant}: {str(e)}")
            raise e
    
    def _load_plant_model(self, plant: str, model_id: str):
        """Load a plant-specific model from disk"""
        plant_dir = os.path.join(self.plant_model_dir, plant)
        model_path = os.path.join(plant_dir, f"{model_id}.keras")
        
        if not os.path.exists(model_path):
            # Try .h5 for backward compatibility
            model_path = os.path.join(plant_dir, f"{model_id}.h5")
        
        self.models[model_id] = tf.keras.models.load_model(model_path)
        
        # Load metadata
        metadata_path = os.path.join(plant_dir, f"{model_id}_metadata.pkl")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            self.scalers[f"{model_id}_features"] = metadata['feature_scaler']
            self.scalers[f"{model_id}_targets"] = metadata['target_scaler']
            self.encoders[model_id] = metadata['encoders']
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status for all plants"""
        with self.lock:
            return self.training_status.copy()
    
    def get_plant_models_summary(self) -> Dict[str, Any]:
        """Get summary of all plant models"""
        summary = {}
        
        for plant, model_ids in self.plant_models.items():
            plant_summary = {
                'total_models': len(model_ids),
                'models': []
            }
            
            for model_id in model_ids:
                if model_id in self.model_metadata:
                    info = self.model_metadata[model_id]
                    plant_summary['models'].append({
                        'model_id': model_id,
                        'created_at': info.created_at.isoformat(),
                        'r2_score': info.metrics.get('r2_score', 0),
                        'rmse': info.metrics.get('rmse', 0),
                        'training_time': info.training_time,
                        'data_points': info.data_points
                    })
            
            # Sort by R² score
            plant_summary['models'].sort(key=lambda x: x['r2_score'], reverse=True)
            
            if plant_summary['models']:
                plant_summary['best_model'] = plant_summary['models'][0]['model_id']
                plant_summary['best_r2_score'] = plant_summary['models'][0]['r2_score']
            
            summary[plant] = plant_summary
        
        return summary


class PlantTrainingCallback(tf.keras.callbacks.Callback):
    """Custom callback to update training progress for a specific plant"""
    
    def __init__(self, plant: str, training_status: Dict, lock: threading.Lock):
        super().__init__()
        self.plant = plant
        self.training_status = training_status
        self.lock = lock
    
    def on_epoch_end(self, epoch, logs=None):
        """Update progress after each epoch"""
        with self.lock:
            progress = 40 + int((epoch + 1) / self.params['epochs'] * 40)  # 40-80% range
            self.training_status[self.plant]['progress'] = min(progress, 80)
            self.training_status[self.plant]['message'] = f"Epoch {epoch + 1}/{self.params['epochs']}"
            
            if logs:
                self.training_status[self.plant]['current_loss'] = float(logs.get('loss', 0))
                self.training_status[self.plant]['current_val_loss'] = float(logs.get('val_loss', 0))

    def _prepare_plant_prediction_input(self, request: Dict, encoders: Dict, 
                                      feature_scaler: StandardScaler, config: Dict, 
                                      plant: str) -> np.ndarray:
        """
        Prepare input data for plant-specific prediction
        """
        # Create dummy sequence data (simplified approach)
        sequence_length = config.get('sequence_length', 10)
        
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
            if feature_name == 'plant':
                # Use the specific plant for this prediction
                try:
                    encoded_value = encoder.transform([plant])[0]
                    features.append(encoded_value)
                except ValueError:
                    # Use first class if plant not seen during training
                    features.append(0)
            elif feature_name in request:
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
    
    def _validate_plant_predictions(self, predictions: List[Dict]) -> Dict[str, Any]:
        """
        Validate plant-specific predictions for feasibility
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
                validation['warnings'].append(f"Day {pred['day']}: Negative values predicted for {pred['plant']}")
            
            # Check Little's Law compliance (WIP = Throughput × Cycle Time)
            expected_wip = throughput * cycle_time
            if abs(wip - expected_wip) > wip * 0.2:  # 20% tolerance
                validation['warnings'].append(
                    f"Day {pred['day']} ({pred['plant']}): Little's Law deviation - "
                    f"WIP={wip:.1f}, Expected={expected_wip:.1f}"
                )
        
        return validation
    
    def _analyze_plant_littles_law(self, predictions: List[Dict]) -> Dict[str, Any]:
        """
        Analyze Little's Law compliance in plant-specific predictions
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
            'plant': predictions[0]['plant'] if predictions else None,
            'analysis': {
                'excellent': sum(1 for s in compliance_scores if s > 0.9),
                'good': sum(1 for s in compliance_scores if 0.7 <= s <= 0.9),
                'fair': sum(1 for s in compliance_scores if 0.5 <= s < 0.7),
                'poor': sum(1 for s in compliance_scores if s < 0.5)
            }
        }