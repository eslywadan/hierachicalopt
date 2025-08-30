"""
Simplified Flask Backend with proper CORS configuration
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Configure CORS - VERY PERMISSIVE for development
CORS(app, 
     origins="*",  # Allow all origins in development
     allow_headers=["Content-Type"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

# Simple test to ensure CORS works
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "CORS is working!", "timestamp": datetime.now().isoformat()})

@app.route('/health', methods=['GET'])
def health():
    logger.info("Health check requested")
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'cors': 'enabled'
    })

@app.route('/api/models', methods=['GET'])
def list_models():
    return jsonify({
        'success': True,
        'models': []
    })

@app.route('/api/data/generate', methods=['POST'])
def generate_data():
    return jsonify({
        'success': True,
        'message': 'Data generation endpoint ready',
        'data_points': 0
    })

@app.route('/api/model/train', methods=['POST'])
def train_model():
    return jsonify({
        'success': True,
        'message': 'Training endpoint ready',
        'model_id': 'test-model-001'
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Simplified Flask Backend with CORS enabled")
    print("📡 Server will be available at: http://localhost:5000")
    print("🌐 CORS is configured to allow requests from any origin")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)