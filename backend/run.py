#!/usr/bin/env python3
"""
Flask Backend Server Startup Script
Run this script to start the LSTM backend service
"""
import os
import sys
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import tensorflow
        import numpy
        import pandas
        import sklearn
        logger.info("✅ All required dependencies are available")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("Please install requirements: pip install -r requirements.txt")
        return False

def start_server():
    """Start the Flask development server"""
    if not check_dependencies():
        return
    
    logger.info("🚀 Starting Flask LSTM Backend Service...")
    logger.info("📡 Server will be available at: http://localhost:5001")
    logger.info("🧠 API endpoints will be available at: http://localhost:5001/api")
    logger.info("⚠️  Note: Using port 5001 to avoid macOS AirPlay conflict on port 5000")
    logger.info("💡 Use Ctrl+C to stop the server")
    
    try:
        from app import app
        app.run(
            host='localhost',
            port=5001,
            debug=True,
            threaded=True,
            use_reloader=True
        )
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")

if __name__ == '__main__':
    start_server()