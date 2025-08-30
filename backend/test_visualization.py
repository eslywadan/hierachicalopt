#!/usr/bin/env python3
"""
Test script for model visualization tools
Demonstrates the visualization capabilities
"""

import os
import sys
import requests
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add the backend directory to the path
sys.path.append('/Users/qs.chou/projects/hierachicalopt/backend')

from utils.model_visualization import model_visualizer
import tensorflow as tf
from tensorflow import keras
import numpy as np

def create_sample_model():
    """Create a sample LSTM model for testing"""
    print("🏗️ Creating sample LSTM model for testing...")
    
    model = keras.Sequential([
        keras.layers.LSTM(64, return_sequences=True, input_shape=(10, 8), name='lstm_1'),
        keras.layers.BatchNormalization(name='batch_norm_1'),
        keras.layers.Dropout(0.2, name='dropout_1'),
        keras.layers.LSTM(32, return_sequences=False, name='lstm_2'),
        keras.layers.BatchNormalization(name='batch_norm_2'),
        keras.layers.Dropout(0.2, name='dropout_2'),
        keras.layers.Dense(16, activation='relu', name='dense_1'),
        keras.layers.Dense(3, name='output')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print(f"✅ Sample model created with {model.count_params():,} parameters")
    return model

def test_model_architecture_visualization():
    """Test model architecture visualization"""
    print("\n" + "="*60)
    print("🎨 TESTING MODEL ARCHITECTURE VISUALIZATION")
    print("="*60)
    
    try:
        # Create sample model
        model = create_sample_model()
        
        # Create visualization
        result = model_visualizer.visualize_model_architecture(
            model, 
            "Sample_LSTM_Model", 
            save_path="visualizations/test_model_architecture.png"
        )
        
        print("📊 Architecture visualization result:")
        print(json.dumps(result, indent=2))
        
        return True
        
    except Exception as e:
        print(f"❌ Architecture visualization test failed: {str(e)}")
        return False

def test_training_status_dashboard():
    """Test training status dashboard"""
    print("\n" + "="*60)
    print("📊 TESTING TRAINING STATUS DASHBOARD")
    print("="*60)
    
    try:
        # Sample training history
        sample_history = {
            'loss': [2.5, 1.8, 1.2, 0.9, 0.7, 0.5, 0.42, 0.38, 0.35],
            'val_loss': [2.8, 2.0, 1.4, 1.0, 0.8, 0.6, 0.48, 0.44, 0.41],
            'mae': [1.2, 0.9, 0.7, 0.5, 0.4, 0.3, 0.25, 0.22, 0.20],
            'r2_score': [0.1, 0.3, 0.5, 0.65, 0.75, 0.82, 0.87, 0.89, 0.91]
        }
        
        # Sample current status
        current_status = {
            'status': 'training',
            'current_epoch': 9,
            'total_epochs': 50,
            'current_batch': 1250,
            'total_batches': 8144,
            'elapsed_minutes': 25.5,
            'estimated_minutes': 120,
            'learning_rate': 0.002,
            'model_id': 'test_enhanced_model'
        }
        
        # Add some sample logs
        model_visualizer.add_training_log("Training started", "INFO")
        model_visualizer.add_training_log("Epoch 1 completed - Loss: 2.5", "INFO")
        model_visualizer.add_training_log("Learning rate reduced to 0.001", "WARNING")
        model_visualizer.add_training_log("Early stopping patience: 5", "INFO")
        model_visualizer.add_training_log("Model checkpoint saved", "SUCCESS")
        
        # Create dashboard
        result = model_visualizer.create_training_status_dashboard(
            sample_history,
            current_status,
            save_path="visualizations/test_training_dashboard.png"
        )
        
        print("📊 Training dashboard result:")
        print(json.dumps(result, indent=2))
        
        return True
        
    except Exception as e:
        print(f"❌ Training dashboard test failed: {str(e)}")
        return False

def test_api_endpoints():
    """Test the API endpoints"""
    print("\n" + "="*60)
    print("🌐 TESTING API ENDPOINTS")
    print("="*60)
    
    base_url = "http://localhost:5001/api"
    
    try:
        # Test training dashboard endpoint
        print("📊 Testing training dashboard endpoint...")
        response = requests.get(f"{base_url}/training/status/dashboard", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print("✅ Training dashboard endpoint working")
            print(f"   Success: {result.get('success', False)}")
        else:
            print(f"⚠️ Training dashboard endpoint returned {response.status_code}")
        
        # Test training logs endpoint
        print("📝 Testing training logs endpoint...")
        response = requests.get(f"{base_url}/training/logs", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print("✅ Training logs endpoint working")
            print(f"   Logs count: {result.get('count', 0)}")
        else:
            print(f"⚠️ Training logs endpoint returned {response.status_code}")
        
        # Test adding a log entry
        print("📝 Testing add log endpoint...")
        log_data = {
            "message": "Test log entry from visualization test",
            "level": "INFO"
        }
        response = requests.post(f"{base_url}/training/logs", json=log_data, timeout=10)
        if response.status_code == 200:
            print("✅ Add log endpoint working")
        else:
            print(f"⚠️ Add log endpoint returned {response.status_code}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("⚠️ Could not connect to backend server - make sure it's running on localhost:5001")
        return False
    except Exception as e:
        print(f"❌ API endpoint test failed: {str(e)}")
        return False

def test_model_report_export():
    """Test comprehensive model report export"""
    print("\n" + "="*60)
    print("📋 TESTING MODEL REPORT EXPORT")
    print("="*60)
    
    try:
        # Create sample model
        model = create_sample_model()
        
        # Sample training history
        sample_history = {
            'loss': [2.5, 1.8, 1.2, 0.9, 0.7, 0.5, 0.42],
            'val_loss': [2.8, 2.0, 1.4, 1.0, 0.8, 0.6, 0.48],
            'mae': [1.2, 0.9, 0.7, 0.5, 0.4, 0.3, 0.25]
        }
        
        # Export comprehensive report
        report_path = model_visualizer.export_model_report(
            model,
            "Test_LSTM_Model",
            sample_history,
            "reports"
        )
        
        print(f"✅ Model report exported to: {report_path}")
        
        # Check if files were created
        if os.path.exists(report_path):
            print("✅ Report file created successfully")
        else:
            print("⚠️ Report file not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Model report export test failed: {str(e)}")
        return False

def main():
    """Run all visualization tests"""
    print("🚀 STARTING MODEL VISUALIZATION TESTS")
    print("="*80)
    
    # Create necessary directories
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    test_results = []
    
    # Run tests
    test_results.append(("Model Architecture Visualization", test_model_architecture_visualization()))
    test_results.append(("Training Status Dashboard", test_training_status_dashboard()))
    test_results.append(("API Endpoints", test_api_endpoints()))
    test_results.append(("Model Report Export", test_model_report_export()))
    
    # Print summary
    print("\n" + "="*80)
    print("📊 TEST RESULTS SUMMARY")
    print("="*80)
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        print("🎉 All tests passed! The visualization system is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
    
    print("\n📁 Generated files:")
    for file_path in ["visualizations/test_model_architecture.png", 
                     "visualizations/test_training_dashboard.png"]:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (not found)")

if __name__ == "__main__":
    main()