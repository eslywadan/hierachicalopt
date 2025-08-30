"""
Flask Backend Service for LSTM Model Training
Provides REST API endpoints for hierarchical optimization LSTM models
"""
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import logging
from datetime import datetime

# Import our LSTM model services
from models.lstm_service import LSTMModelService
from models.lstm_parallel_service import ParallelLSTMService
from models.lstm_enhanced_service import EnhancedLSTMService
from models.data_generator import DataGenerator
from models.master_data_service import MasterDataService
from utils.validation import ValidationService
from utils.model_visualization import model_visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure CORS - Very permissive for development
CORS(app, origins="*", allow_headers="*", methods="*")

# Initialize services
master_data_service = MasterDataService()
data_generator = DataGenerator(master_data_service)
lstm_service = LSTMModelService()
parallel_lstm_service = ParallelLSTMService(max_workers=4)  # Parallel training service
enhanced_lstm_service = EnhancedLSTMService()  # Enhanced training service
validation_service = ValidationService()

# Global model cache
model_cache = {}

# Add after_request handler to ensure CORS headers are always added
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

@app.route('/health', methods=['GET', 'OPTIONS'])
@cross_origin()
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.1'  # Updated version to verify new code is running
    })

@app.route('/debug/test-generator')
def debug_test_generator():
    """Debug endpoint to test data generator directly"""
    try:
        config = {
            'plants': ['Taiwan_Fab1'],
            'applications': ['Commercial Display'], 
            'panel_sizes': ['32"'],
            'historical_days': 1
        }
        
        params = {
            'base_wip': 100,
            'base_throughput': 50,
            'seasonality': 0.2,
            'noise_level': 0.1
        }
        
        logger.info("🔍 DEBUG: Testing data generator directly")
        result = data_generator.generate_data(config, params)
        
        return jsonify({
            'success': True,
            'data_points': len(result),
            'config': config,
            'params': params,
            'sample': result[0] if result else None,
            'generator_type': str(type(data_generator)),
            'master_service_type': str(type(master_data_service))
        })
        
    except Exception as e:
        logger.error(f"❌ Debug test failed: {str(e)}")
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/data/generate', methods=['POST'])
def generate_training_data():
    """Generate synthetic training data for LSTM model"""
    try:
        data = request.json
        logger.info(f"🔍 Received request data: {data}")
        
        if not data:
            logger.error("❌ No data received in request")
            return jsonify({
                'success': False,
                'error': 'No data provided in request'
            }), 400
        
        # Extract configuration
        config = {
            'plants': data.get('plants', ['Plant_A', 'Plant_B', 'Plant_C']),
            'applications': data.get('applications', ['Automotive', 'Consumer_Electronics', 'Industrial']),
            'panel_sizes': data.get('panelSizes', ['Small', 'Medium', 'Large', 'Extra_Large']),
            'historical_days': data.get('historicalDays', 545)
        }
        
        generator_params = {
            'base_wip': data.get('baseWIP', data.get('base_wip', 100)),
            'base_throughput': data.get('baseThroughput', data.get('base_throughput', 50)),
            'seasonality': data.get('seasonality', 0.2),
            'noise_level': data.get('noiseLevel', data.get('noise_level', 0.1))
        }
        
        logger.info(f"📊 Config: {config}")
        logger.info(f"⚙️ Params: {generator_params}")
        
        # Generate data
        logger.info(f"🚀 About to call data_generator.generate_data()")
        training_data = data_generator.generate_data(config, generator_params)
        
        logger.info(f"✅ Generated {len(training_data)} data points")
        
        return jsonify({
            'success': True,
            'data_points': len(training_data),
            'data': training_data[:100],  # Return first 100 points for preview
            'summary': {
                'total_points': len(training_data),
                'plants': config['plants'],
                'date_range': {
                    'start': training_data[0]['date'] if training_data else None,
                    'end': training_data[-1]['date'] if training_data else None
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating training data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model/train/parallel/plants', methods=['POST'])
def train_plant_models_parallel():
    """Train separate LSTM models for each plant in parallel"""
    try:
        data = request.json
        logger.info("🏭 Starting parallel plant-specific model training...")
        model_visualizer.add_training_log("🏭 Starting parallel plant-specific model training", "INFO")
        
        # Extract configuration
        lstm_config = {
            'lstm_units_1': data.get('lstmUnits1', 64),
            'lstm_units_2': data.get('lstmUnits2', 32),
            'dropout_rate': data.get('dropoutRate', 0.2),
            'sequence_length': data.get('sequenceLength', 10),
            'epochs': data.get('epochs', 50),
            'batch_size': data.get('batchSize', 32),
            'learning_rate': data.get('learningRate', 0.00001),
            'train_test_split': data.get('trainTestSplit', 0.8)
        }
        
        # Get plants to train
        plants_to_train = data.get('plants', None)  # None means all plants
        model_visualizer.add_training_log(f"Training plants: {plants_to_train or 'all available plants'}", "INFO")
        
        # Get training data
        training_data = data.get('trainingData')
        if not training_data:
            # Generate training data if not provided
            model_visualizer.add_training_log("Generating parallel training data...", "INFO")
            config = {
                'plants': plants_to_train or ['Plant_A', 'Plant_B', 'Plant_C'],
                'applications': ['Automotive', 'Consumer_Electronics', 'Industrial'],
                'panel_sizes': ['Small', 'Medium', 'Large', 'Extra_Large'],
                'historical_days': data.get('historicalDays', 545)
            }
            
            generator_params = {
                'base_wip': data.get('baseWIP', 100),
                'base_throughput': data.get('baseThroughput', 50),
                'seasonality': data.get('seasonality', 0.2),
                'noise_level': data.get('noiseLevel', 0.1)
            }
            
            training_data = data_generator.generate_data(config, generator_params)
            model_visualizer.add_training_log(f"Generated {len(training_data)} data points for parallel training", "SUCCESS")
        
        # Train models in parallel
        model_visualizer.add_training_log("Starting parallel training execution...", "INFO")
        result = parallel_lstm_service.train_plant_models_parallel(
            training_data, 
            lstm_config, 
            plants_to_train
        )
        
        logger.info(f"✅ Parallel training completed: {result['successful_plants']}/{result['total_plants']} successful")
        model_visualizer.add_training_log(f"✅ Parallel training completed: {result['successful_plants']}/{result['total_plants']} plants", "SUCCESS")
        
        return jsonify({
            'success': result['success'],
            'total_plants': result['total_plants'],
            'successful_plants': result['successful_plants'],
            'failed_plants': result['failed_plants'],
            'training_time': result['total_training_time'],
            'average_time_per_plant': result['average_time_per_plant'],
            'plant_results': result['plant_results']
        })
        
    except Exception as e:
        logger.error(f"Error in parallel plant training: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model/plants/status', methods=['GET'])
def get_plant_training_status():
    """Get current training status for all plants"""
    try:
        status = parallel_lstm_service.get_training_status()
        return jsonify({
            'success': True,
            'status': status
        })
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model/plants/summary', methods=['GET'])
def get_plant_models_summary():
    """Get summary of all plant-specific models"""
    try:
        summary = parallel_lstm_service.get_plant_models_summary()
        return jsonify({
            'success': True,
            'summary': summary
        })
    except Exception as e:
        logger.error(f"Error getting plant models summary: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model/plant/<plant>/best', methods=['GET'])
def get_best_plant_model(plant):
    """Get the best model for a specific plant"""
    try:
        model_id = parallel_lstm_service.get_best_model_for_plant(plant)
        if model_id:
            return jsonify({
                'success': True,
                'plant': plant,
                'model_id': model_id,
                'metadata': parallel_lstm_service.model_metadata.get(model_id)
            })
        else:
            return jsonify({
                'success': False,
                'error': f'No model found for plant {plant}'
            }), 404
    except Exception as e:
        logger.error(f"Error getting best model for plant {plant}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model/plant/<plant>/predict', methods=['POST'])
def predict_with_plant_model(plant):
    """Make predictions using the best model for a specific plant"""
    try:
        data = request.json
        logger.info(f"🔮 Making prediction for plant {plant}")
        
        prediction_request = {
            'plant': plant,
            'application': data.get('application'),
            'panel_size': data.get('panelSize'),
            'prediction_days': data.get('predictionDays', 30),
            'current_wip': data.get('currentWIP'),
            'planned_throughput': data.get('plannedThroughput'),
            'target_production': data.get('targetProduction')
        }
        
        # Use plant-specific model for prediction
        result = parallel_lstm_service.predict_with_plant_model(plant, prediction_request)
        
        return jsonify({
            'success': True,
            'plant': plant,
            'predictions': result.get('predictions', []),
            'validation': result.get('validation', {}),
            'little_law_analysis': result.get('little_law_analysis', {}),
            'model_id': result.get('model_id')
        })
        
    except Exception as e:
        logger.error(f"Error making plant-specific prediction for {plant}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model/train', methods=['POST'])
def train_lstm_model():
    """Train LSTM model with provided data and configuration"""
    try:
        data = request.json
        logger.info("Starting LSTM model training...")
        model_visualizer.add_training_log("Starting regular LSTM model training...", "INFO")
        
        # Extract training configuration
        lstm_config = {
            'lstm_units_1': data.get('lstmUnits1', 64),
            'lstm_units_2': data.get('lstmUnits2', 32),
            'dropout_rate': data.get('dropoutRate', 0.2),
            'sequence_length': data.get('sequenceLength', 10),
            'epochs': data.get('epochs', 50),
            'batch_size': data.get('batchSize', 32),
            'learning_rate': data.get('learningRate', 0.00001),
            'train_test_split': data.get('trainTestSplit', 0.8)
        }
        
        model_visualizer.add_training_log(f"Training config: {lstm_config['epochs']} epochs, {lstm_config['batch_size']} batch size", "INFO")
        
        # Get training data
        training_data = data.get('trainingData')
        if not training_data:
            # Generate default training data if not provided
            model_visualizer.add_training_log("Generating training data...", "INFO")
            config = {
                'plants': ['Plant_A', 'Plant_B', 'Plant_C'],
                'applications': ['Automotive', 'Consumer_Electronics', 'Industrial'],
                'panel_sizes': ['Small', 'Medium', 'Large', 'Extra_Large'],
                'historical_days': 545
            }
            
            generator_params = {
                'base_wip': 100,
                'base_throughput': 50,
                'seasonality': 0.2,
                'noise_level': 0.1
            }
            
            training_data = data_generator.generate_data(config, generator_params)
            model_visualizer.add_training_log(f"Generated {len(training_data)} training data points", "SUCCESS")
        
        # Train model
        model_visualizer.add_training_log("Starting model training...", "INFO")
        result = lstm_service.train_model(training_data, lstm_config)
        
        # Cache the trained model
        model_id = result['model_id']
        model_cache[model_id] = result['model_info']
        
        logger.info(f"Model training completed. Model ID: {model_id}")
        model_visualizer.add_training_log(f"Training completed: {model_id[:8]}...", "SUCCESS")
        model_visualizer.add_training_log(f"R² Score: {result['metrics'].get('r2_score', 0):.4f}", "INFO")
        
        return jsonify({
            'success': True,
            'model_id': model_id,
            'training_history': result['history'],
            'metrics': result['metrics'],
            'training_time': result['training_time'],
            'model_summary': result['model_info']
        })
        
    except Exception as e:
        logger.error(f"Error training LSTM model: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model/train/enhanced', methods=['POST'])
def train_enhanced_model():
    """Train enhanced LSTM model with advanced techniques"""
    try:
        data = request.json
        model_visualizer.add_training_log("🚀 Starting enhanced LSTM training with advanced techniques", "INFO")
        
        # Generate training data if not provided
        training_data = data.get('trainingData')
        if not training_data:
            model_visualizer.add_training_log("Generating enhanced training data...", "INFO")
            # Use provided configuration or default
            config = {
                'plants': data.get('plants', ['Taiwan_Fab1', 'China_Fab1', 'Korea_Fab1']),
                'applications': data.get('applications', ['Commercial Display', 'Consumer TV', 'Gaming Monitor']),
                'panel_sizes': data.get('panelSizes', ['15.6"', '27"', '43"']),
                'historical_days': data.get('historicalDays', 365)
            }
            
            params = {
                'base_wip': data.get('baseWIP', 100),
                'base_throughput': data.get('baseThroughput', 50),
                'seasonality': data.get('seasonality', 0.2),
                'noise_level': data.get('noiseLevel', 0.1)
            }
            
            training_data = data_generator.generate_data(config, params)
            logger.info(f"🏭 Generated {len(training_data)} training data points for enhanced model")
            model_visualizer.add_training_log(f"Generated {len(training_data)} data points for enhanced training", "SUCCESS")
        
        # Get enhanced training configuration
        enhanced_config = enhanced_lstm_service.get_enhanced_training_config()
        
        # Override with user parameters if provided
        for key in ['lstm_units_1', 'lstm_units_2', 'dropout_rate', 'sequence_length', 'epochs', 'batch_size', 'learning_rate']:
            if key in data:
                enhanced_config[key] = data[key]
        
        logger.info(f"🚀 Starting enhanced LSTM training with {len(training_data)} data points")
        model_visualizer.add_training_log(f"Enhanced training config: {enhanced_config['epochs']} epochs, BatchNorm + L2 reg", "INFO")
        
        # Train enhanced model
        result = enhanced_lstm_service.train_enhanced_model(training_data, enhanced_config)
        
        if result['success']:
            # Cache the model result for predictions
            model_cache[result['model_id']] = {
                'model_type': 'enhanced',
                'result': result,
                'training_data_size': len(training_data),
                'config': enhanced_config,
                'created_at': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Enhanced model training completed: {result['model_id']}")
            model_visualizer.add_training_log(f"✅ Enhanced training completed: {result['model_id'][:8]}...", "SUCCESS")
            if result.get('metrics', {}).get('r2_score'):
                model_visualizer.add_training_log(f"Enhanced R² Score: {result['metrics']['r2_score']:.4f}", "SUCCESS")
        
        return jsonify({
            'success': result['success'],
            'model_id': result['model_id'],
            'training_history': result.get('training_history', {}),
            'metrics': result.get('metrics', {}),
            'training_time': result.get('training_time', 0),
            'model_summary': result.get('model_summary', ''),
            'data_points': len(training_data),
            'enhanced_features': True
        })
        
    except Exception as e:
        logger.error(f"Error training enhanced LSTM model: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model/train/enhanced/config', methods=['GET'])
def get_enhanced_config():
    """Get optimized enhanced training configuration"""
    try:
        config = enhanced_lstm_service.get_enhanced_training_config()
        return jsonify({
            'success': True,
            'config': config,
            'description': {
                'lstm_units_1': 'First LSTM layer units (96 for balanced performance)',
                'lstm_units_2': 'Second LSTM layer units (48 for efficiency)', 
                'dropout_rate': 'Dropout rate (0.25 for better generalization)',
                'recurrent_dropout': 'Recurrent dropout (0.15 to prevent overfitting)',
                'sequence_length': 'Sequence length (15 optimal for manufacturing)',
                'epochs': 'Training epochs (150 with early stopping)',
                'batch_size': 'Batch size (64 for stability)',
                'learning_rate': 'Learning rate (0.002 with scheduling)'
            }
        })
    except Exception as e:
        logger.error(f"Error getting enhanced config: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model/predict', methods=['POST'])
def predict():
    """Make predictions using trained LSTM model"""
    try:
        data = request.json
        model_id = data.get('model_id')
        
        if not model_id or model_id not in model_cache:
            return jsonify({
                'success': False,
                'error': 'Model not found. Please train a model first.'
            }), 400
        
        prediction_request = {
            'plant': data.get('plant'),
            'application': data.get('application'),
            'panel_size': data.get('panelSize'),
            'prediction_days': data.get('predictionDays', 30),
            'current_wip': data.get('currentWIP'),
            'planned_throughput': data.get('plannedThroughput'),
            'target_production': data.get('targetProduction')
        }
        
        # Make prediction
        result = lstm_service.predict(model_id, prediction_request)
        
        return jsonify({
            'success': True,
            'predictions': result['predictions'],
            'validation': result['validation'],
            'little_law_analysis': result['little_law_analysis']
        })
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model/validate', methods=['POST'])
def validate_production():
    """Validate production parameters using Little's Law"""
    try:
        data = request.json
        
        validation_request = {
            'wip': data.get('wip'),
            'throughput': data.get('throughput'),
            'cycle_time': data.get('cycleTime'),
            'target_production': data.get('targetProduction')
        }
        
        result = validation_service.validate_littles_law(validation_request)
        
        return jsonify({
            'success': True,
            'validation': result
        })
        
    except Exception as e:
        logger.error(f"Error validating production: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models', methods=['GET'])
def list_models():
    """List all trained models"""
    return jsonify({
        'success': True,
        'models': [
            {
                'model_id': model_id,
                'info': model_info
            }
            for model_id, model_info in model_cache.items()
        ]
    })

@app.route('/api/model/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a trained model from cache"""
    if model_id in model_cache:
        del model_cache[model_id]
        return jsonify({
            'success': True,
            'message': f'Model {model_id} deleted successfully'
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Model not found'
        }), 404

@app.route('/api/masterdata/config', methods=['GET'])
def get_master_data_config():
    """Get synchronized master data configuration"""
    try:
        config = master_data_service.get_synchronized_config()
        logger.info("📋 Retrieved synchronized master data configuration")
        
        return jsonify({
            'success': True,
            'config': config,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting master data config: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/masterdata/summary', methods=['GET'])
def get_master_data_summary():
    """Get master data summary for dashboard"""
    try:
        summary = master_data_service.get_master_data_summary()
        logger.info("📊 Retrieved master data summary")
        
        return jsonify({
            'success': True,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting master data summary: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/masterdata/validate', methods=['POST'])
def validate_data_consistency():
    """Validate data consistency across all sources"""
    try:
        logger.info("🔍 Starting data consistency validation")
        
        report = master_data_service.validate_data_consistency()
        
        logger.info(f"✅ Data consistency validation completed: {report.total_issues} issues found")
        
        return jsonify({
            'success': True,
            'report': {
                'timestamp': report.timestamp.isoformat(),
                'total_issues': report.total_issues,
                'is_consistent': report.is_consistent,
                'plant_issues': report.plant_issues,
                'product_issues': report.product_issues,
                'panel_size_issues': report.panel_size_issues,
                'demand_data_issues': report.demand_data_issues,
                'plant_data_issues': report.plant_data_issues,
                'recommendations': report.recommendations
            }
        })
        
    except Exception as e:
        logger.error(f"Error validating data consistency: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/masterdata/export', methods=['GET'])
def export_master_data():
    """Export master data configuration as JSON"""
    try:
        export_data = master_data_service.export_master_data_config()
        
        logger.info("📤 Exported master data configuration")
        
        return jsonify({
            'success': True,
            'export_data': export_data
        })
        
    except Exception as e:
        logger.error(f"Error exporting master data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/masterdata/reload', methods=['POST'])
def reload_master_data():
    """Reload master data from CSV files"""
    try:
        logger.info("🔄 Reloading master data from CSV files")
        
        # Reinitialize master data service
        global master_data_service
        master_data_service = MasterDataService()
        
        config = master_data_service.get_synchronized_config()
        
        logger.info("✅ Master data reloaded successfully")
        
        return jsonify({
            'success': True,
            'message': 'Master data reloaded successfully',
            'config': config,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error reloading master data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model/<model_id>/visualize', methods=['GET'])
def visualize_model_architecture(model_id):
    """Visualize model architecture and parameters"""
    try:
        logger.info(f"🎨 Creating visualization for model {model_id}")
        
        # Try to load model from different services
        model = None
        model_name = f"Model_{model_id[:8]}"
        training_history = {}
        
        # Check if it's a cached model
        if model_id in model_cache:
            model_info = model_cache[model_id]
            if hasattr(model_info, 'get') and 'model' in model_info:
                model = model_info['model']
                training_history = model_info.get('training_history', {})
        
        # Check enhanced models
        if not model:
            try:
                import os
                import tensorflow as tf
                model_path = os.path.join('trained_models', f"{model_id}.h5")
                if os.path.exists(model_path):
                    model = tf.keras.models.load_model(model_path)
                    model_name = f"Enhanced_Model_{model_id[:8]}"
            except Exception as e:
                logger.warning(f"Could not load enhanced model: {str(e)}")
        
        # Check parallel models
        if not model:
            try:
                for plant in ['Taiwan_Fab1', 'China_Fab1', 'Korea_Fab1']:
                    plant_model_id = parallel_lstm_service.get_best_model_for_plant(plant)
                    if plant_model_id and plant_model_id.startswith(model_id[:8]):
                        model_path = os.path.join('saved_models', 'plant_specific', plant, f"{plant_model_id}.keras")
                        if os.path.exists(model_path):
                            model = tf.keras.models.load_model(model_path)
                            model_name = f"{plant}_Model_{plant_model_id[:8]}"
                            break
            except Exception as e:
                logger.warning(f"Could not load plant model: {str(e)}")
        
        if not model:
            return jsonify({
                'success': False,
                'error': f'Model {model_id} not found or could not be loaded'
            }), 404
        
        # Create visualization
        visualization_result = model_visualizer.visualize_model_architecture(
            model, 
            model_name,
            save_path=f"visualizations/model_arch_{model_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        
        return jsonify({
            'success': True,
            'model_id': model_id,
            'visualization': visualization_result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error visualizing model {model_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/training/status/dashboard', methods=['GET'])
def get_training_dashboard():
    """Get comprehensive training status dashboard"""
    try:
        logger.info("📊 Generating training status dashboard")
        
        # Collect training status from all services
        status_data = {
            'enhanced_training': {},
            'parallel_training': {},
            'regular_training': {},
            'overall_status': 'idle'
        }
        
        # Get parallel training status
        try:
            parallel_status = parallel_lstm_service.get_training_status()
            if parallel_status.get('success'):
                status_data['parallel_training'] = parallel_status.get('status', {})
        except Exception as e:
            logger.warning(f"Could not get parallel training status: {str(e)}")
        
        # Get plant models summary
        try:
            plant_summary = parallel_lstm_service.get_plant_models_summary()
            status_data['plant_models'] = plant_summary
        except Exception as e:
            logger.warning(f"Could not get plant models summary: {str(e)}")
        
        # Mock training history for demonstration (in real scenario, this would come from training callbacks)
        sample_training_history = {
            'loss': [2.5, 1.8, 1.2, 0.9, 0.7, 0.5, 0.4],
            'val_loss': [2.8, 2.0, 1.4, 1.0, 0.8, 0.6, 0.5],
            'mae': [1.2, 0.9, 0.7, 0.5, 0.4, 0.3, 0.25],
            'r2_score': [0.1, 0.3, 0.5, 0.65, 0.75, 0.82, 0.87]
        }
        
        current_status = {
            'status': 'training',
            'current_epoch': 7,
            'total_epochs': 50,
            'current_batch': 150,
            'total_batches': 8144,
            'elapsed_minutes': 45.5,
            'estimated_minutes': 180,
            'learning_rate': 0.002,
            'model_id': 'enhanced_training_current'
        }
        
        # Create training dashboard
        dashboard_result = model_visualizer.create_training_status_dashboard(
            sample_training_history,
            current_status,
            save_path=f"visualizations/training_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        
        return jsonify({
            'success': True,
            'dashboard': dashboard_result,
            'status_data': status_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error creating training dashboard: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/training/logs', methods=['GET'])
def get_training_logs():
    """Get recent training logs"""
    try:
        # Get recent training logs
        logs = list(model_visualizer.training_logs)[-50:]  # Last 50 logs
        
        return jsonify({
            'success': True,
            'logs': logs,
            'count': len(logs),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting training logs: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/training/logs', methods=['POST'])
def add_training_log():
    """Add a training log entry"""
    try:
        data = request.json
        message = data.get('message', 'No message')
        level = data.get('level', 'INFO')
        
        model_visualizer.add_training_log(message, level)
        
        return jsonify({
            'success': True,
            'message': 'Log entry added',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error adding training log: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model/<model_id>/report', methods=['GET'])
def export_model_report(model_id):
    """Export comprehensive model report"""
    try:
        logger.info(f"📋 Exporting comprehensive report for model {model_id}")
        
        # Try to load the model (similar logic as visualization endpoint)
        model = None
        model_name = f"Model_{model_id[:8]}"
        training_history = {}
        
        # Check if it's a cached model
        if model_id in model_cache:
            model_info = model_cache[model_id]
            if hasattr(model_info, 'get') and 'model' in model_info:
                model = model_info['model']
                training_history = model_info.get('training_history', {})
        
        if not model:
            # Try to load from enhanced models
            try:
                import os
                import tensorflow as tf
                model_path = os.path.join('trained_models', f"{model_id}.h5")
                if os.path.exists(model_path):
                    model = tf.keras.models.load_model(model_path)
                    model_name = f"Enhanced_Model_{model_id[:8]}"
            except Exception as e:
                logger.warning(f"Could not load enhanced model: {str(e)}")
        
        if not model:
            return jsonify({
                'success': False,
                'error': f'Model {model_id} not found'
            }), 404
        
        # Create reports directory
        os.makedirs('reports', exist_ok=True)
        
        # Export comprehensive report
        report_path = model_visualizer.export_model_report(
            model, model_name, training_history, 'reports'
        )
        
        return jsonify({
            'success': True,
            'model_id': model_id,
            'report_path': report_path,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error exporting model report for {model_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(_error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.route('/visualizations/<filename>')
def serve_visualization(filename):
    """Serve visualization images"""
    try:
        from flask import send_from_directory
        import os
        
        visualizations_dir = os.path.join(os.getcwd(), 'visualizations')
        if not os.path.exists(visualizations_dir):
            return jsonify({'error': 'Visualizations directory not found'}), 404
        
        file_path = os.path.join(visualizations_dir, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'Visualization file not found'}), 404
        
        return send_from_directory(visualizations_dir, filename)
        
    except Exception as e:
        logger.error(f"Error serving visualization {filename}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(_error):
    return jsonify({
        'success': False,
        'error': 'Resource not found'
    }), 404

@app.errorhandler(500)
def internal_error(_error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Flask LSTM Backend Service...")
    logger.info("📡 CORS enabled for Angular frontend")
    logger.info("🧠 LSTM model services initialized")
    logger.info("⚠️  Note: Using port 5001 to avoid macOS AirPlay conflict")
    
    # Start the Flask development server (port 5001 to avoid macOS AirPlay on 5000)
    app.run(
        host='localhost',
        port=5001,
        debug=True,
        threaded=True
    )