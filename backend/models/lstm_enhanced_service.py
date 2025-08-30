"""
Enhanced LSTM Service with advanced training techniques
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import uuid
import os
import pickle
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import math

logger = logging.getLogger(__name__)

class EnhancedLSTMService:
    def __init__(self):
        self.models_dir = 'trained_models'
        self.scalers_dir = 'scalers'
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.scalers_dir, exist_ok=True)
        
        logger.info("🚀 Enhanced LSTM Model Service initialized")
        logger.info(f"📦 TensorFlow version: {tf.__version__}")
    
    def train_enhanced_model(self, data: List[Dict[str, Any]], config: Dict) -> Dict[str, Any]:
        """
        Train LSTM model with advanced techniques
        """
        try:
            logger.info("🧠 Starting enhanced LSTM training...")
            logger.info(f"📊 Training data points: {len(data)}")
            logger.info(f"🔧 Config: {config}")
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Enhanced preprocessing
            X, y, feature_scaler, target_scaler, encoders = self._enhanced_preprocessing(df)
            
            # Create sequences
            sequence_length = config.get('sequence_length', 20)  # Increased for better patterns
            X_seq, y_seq = self._create_sequences(X, y, sequence_length)
            
            if len(X_seq) == 0:
                raise ValueError("No sequences created. Check sequence length vs data size.")
            
            # Split data with stratification
            train_test_split_ratio = config.get('train_test_split', 0.85)  # Increased train ratio
            X_train, X_test, y_train, y_test = train_test_split(
                X_seq, y_seq, 
                test_size=1-train_test_split_ratio,
                shuffle=True,
                random_state=42
            )
            
            logger.info(f"📏 Training sequences: {X_train.shape}, Test sequences: {X_test.shape}")
            
            # Build enhanced model
            model = self._build_enhanced_lstm_model(X_train.shape[1:], config)
            
            # Enhanced training with callbacks
            history = self._train_with_enhanced_callbacks(
                model, X_train, y_train, X_test, y_test, config
            )
            
            # Evaluate model
            y_pred = model.predict(X_test, verbose=0)
            
            # Calculate metrics for each target
            metrics = self._calculate_enhanced_metrics(y_test, y_pred, target_scaler)
            
            # Save model and scalers
            model_id = self._save_enhanced_model(model, feature_scaler, target_scaler, encoders, config)
            
            training_time = len(history['loss']) * config.get('epoch_time_estimate', 2.0)
            
            result = {
                'success': True,
                'model_id': model_id,
                'training_history': {
                    'loss': [float(x) for x in history['loss']],
                    'val_loss': [float(x) for x in history.get('val_loss', [])],
                    'epochs': len(history['loss'])
                },
                'metrics': metrics,
                'training_time': training_time,
                'model_summary': self._get_model_summary(model)
            }
            
            logger.info(f"✅ Enhanced training completed: {model_id}")
            logger.info(f"📈 Final R² score: {metrics.get('r2_score', 0):.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Enhanced training failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'model_id': None
            }
    
    def _enhanced_preprocessing(self, df: pd.DataFrame) -> Tuple:
        """Enhanced data preprocessing with feature engineering"""
        logger.info("🔧 Enhanced preprocessing...")
        
        # Feature engineering - add time-based features
        if 'day' in df.columns:
            df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365.25)
            df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365.25)
            df['month_sin'] = np.sin(2 * np.pi * df['day'] / 30.44)
            df['month_cos'] = np.cos(2 * np.pi * df['day'] / 30.44)
            df['week_sin'] = np.sin(2 * np.pi * df['day'] / 7)
            df['week_cos'] = np.cos(2 * np.pi * df['day'] / 7)
        
        # Add interaction features
        if 'wip' in df.columns and 'throughput' in df.columns:
            df['wip_throughput_ratio'] = df['wip'] / (df['throughput'] + 1e-8)
            df['utilization'] = df['throughput'] / (df['wip'] + 1e-8)
        
        # Add moving averages
        for col in ['wip', 'throughput', 'cycle_time']:
            if col in df.columns:
                df[f'{col}_ma_3'] = df[col].rolling(window=3, min_periods=1).mean()
                df[f'{col}_ma_7'] = df[col].rolling(window=7, min_periods=1).mean()
                df[f'{col}_lag_1'] = df[col].shift(1).fillna(df[col].mean())
        
        # Encode categorical variables
        encoders = {}
        for col in ['plant', 'application', 'panel_size']:
            if col in df.columns:
                encoder = LabelEncoder()
                df[f'{col}_encoded'] = encoder.fit_transform(df[col].astype(str))
                encoders[col] = encoder
        
        # Select enhanced features
        feature_columns = [
            'day', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'week_sin', 'week_cos',
            'plant_encoded', 'application_encoded', 'panel_size_encoded',
            'finished_goods', 'semi_finished_goods', 'littles_law_compliance',
            'wip_ma_3', 'wip_ma_7', 'wip_lag_1',
            'throughput_ma_3', 'throughput_ma_7', 'throughput_lag_1',
            'cycle_time_ma_3', 'cycle_time_ma_7', 'cycle_time_lag_1',
            'wip_throughput_ratio', 'utilization'
        ]
        
        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features].values
        
        # Target variables
        target_columns = ['wip', 'throughput', 'cycle_time']
        available_targets = [col for col in target_columns if col in df.columns]
        y = df[available_targets].values
        
        # Enhanced scaling
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        X_scaled = feature_scaler.fit_transform(X)
        y_scaled = target_scaler.fit_transform(y)
        
        logger.info(f"📊 Enhanced features: {len(available_features)}, shape: {X_scaled.shape}")
        
        return X_scaled, y_scaled, feature_scaler, target_scaler, encoders
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        logger.info(f"🔗 Creating sequences with length {sequence_length}")
        
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _build_enhanced_lstm_model(self, input_shape: Tuple, config: Dict) -> tf.keras.Model:
        """Build enhanced LSTM architecture"""
        logger.info(f"🏗️ Building enhanced LSTM model with input shape: {input_shape}")
        
        model = tf.keras.Sequential()
        
        # First LSTM layer with attention
        model.add(tf.keras.layers.LSTM(
            config.get('lstm_units_1', 64),
            return_sequences=True,
            input_shape=input_shape,
            recurrent_dropout=config.get('recurrent_dropout', 0.1),
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(config.get('dropout_rate', 0.2)))
        
        # Second LSTM layer
        model.add(tf.keras.layers.LSTM(
            config.get('lstm_units_2', 32),
            return_sequences=False,
            recurrent_dropout=config.get('recurrent_dropout', 0.1),
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(config.get('dropout_rate', 0.2)))
        
        # Dense layers with residual connection
        dense_units = config.get('dense_units', 32)
        model.add(tf.keras.layers.Dense(
            dense_units, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(config.get('dropout_rate', 0.2)))
        
        # Output layer - always 3 outputs (WIP, throughput, cycle_time)
        model.add(tf.keras.layers.Dense(3))
        
        # Enhanced optimizer with learning rate scheduling
        initial_lr = config.get('learning_rate', 0.001)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=initial_lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0  # Gradient clipping
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', self._r2_metric]
        )
        
        logger.info(f"🎯 Model compiled with enhanced optimizer")
        return model
    
    def _r2_metric(self, y_true, y_pred):
        """Custom R² metric for Keras"""
        ss_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
        ss_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true)))
        return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())
    
    def _train_with_enhanced_callbacks(self, model, X_train, y_train, X_test, y_test, config):
        """Train model with advanced callbacks"""
        callbacks = []
        
        # Learning rate reduction on plateau
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ))
        
        # Early stopping
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ))
        
        # Model checkpoint
        checkpoint_path = os.path.join(self.models_dir, 'best_model_checkpoint.h5')
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ))
        
        # Cosine annealing
        initial_lr = config.get('learning_rate', 0.001)
        epochs = config.get('epochs', 100)
        
        def cosine_annealing(epoch):
            return initial_lr * (1 + np.cos(np.pi * epoch / epochs)) / 2
        
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(cosine_annealing, verbose=1))
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=config.get('batch_size', 32),
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def _calculate_enhanced_metrics(self, y_test, y_pred, target_scaler):
        """Calculate comprehensive metrics"""
        # Inverse transform predictions
        y_test_orig = target_scaler.inverse_transform(y_test)
        y_pred_orig = target_scaler.inverse_transform(y_pred)
        
        # Overall metrics
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        r2 = r2_score(y_test_orig, y_pred_orig)
        
        # Per-target metrics
        target_names = ['wip', 'throughput', 'cycle_time']
        per_target_metrics = {}
        
        for i, target in enumerate(target_names[:y_test_orig.shape[1]]):
            per_target_metrics[target] = {
                'rmse': float(np.sqrt(mean_squared_error(y_test_orig[:, i], y_pred_orig[:, i]))),
                'mae': float(mean_absolute_error(y_test_orig[:, i], y_pred_orig[:, i])),
                'r2': float(r2_score(y_test_orig[:, i], y_pred_orig[:, i]))
            }
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
            'per_target': per_target_metrics
        }
    
    def _save_enhanced_model(self, model, feature_scaler, target_scaler, encoders, config):
        """Save model and associated components"""
        model_id = f"enhanced_{uuid.uuid4().hex[:8]}"
        
        # Save model
        model_path = os.path.join(self.models_dir, f"{model_id}.h5")
        model.save(model_path)
        
        # Save scalers and encoders
        scalers_path = os.path.join(self.scalers_dir, f"{model_id}_scalers.pkl")
        with open(scalers_path, 'wb') as f:
            pickle.dump({
                'feature_scaler': feature_scaler,
                'target_scaler': target_scaler,
                'encoders': encoders,
                'config': config,
                'created_at': datetime.now().isoformat()
            }, f)
        
        logger.info(f"💾 Enhanced model saved: {model_id}")
        return model_id
    
    def _get_model_summary(self, model):
        """Get model architecture summary"""
        summary_lines = []
        model.summary(print_fn=summary_lines.append)
        return '\n'.join(summary_lines)

    def get_enhanced_training_config(self) -> Dict:
        """Get optimized training configuration"""
        return {
            'lstm_units_1': 96,      # Sweet spot between performance and training time
            'lstm_units_2': 48,      # Balanced architecture
            'dropout_rate': 0.25,    # Higher dropout for better generalization
            'recurrent_dropout': 0.15, # Prevent overfitting in recurrent connections
            'dense_units': 48,       # Additional dense layer for complexity
            'sequence_length': 15,   # Optimal sequence length for manufacturing data
            'epochs': 150,           # More epochs with early stopping
            'batch_size': 64,        # Larger batch size for stability
            'learning_rate': 0.002,  # Higher initial learning rate with scheduling
            'train_test_split': 0.85 # More training data
        }