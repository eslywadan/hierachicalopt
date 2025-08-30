#!/usr/bin/env python3
"""
Demonstration of the model visualization capabilities
Shows how to use the visualization tools for model analysis and training monitoring
"""

import os
import sys
import json
import time
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set matplotlib backend to prevent display issues
import matplotlib
matplotlib.use('Agg')

def demo_visualization_capabilities():
    """Demonstrate the model visualization capabilities"""
    print("🎨 MODEL VISUALIZATION SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    try:
        from utils.model_visualization import ModelVisualizer
        import tensorflow as tf
        from tensorflow import keras
        import numpy as np
        
        # Create visualizer instance
        visualizer = ModelVisualizer()
        
        print("✅ Successfully imported visualization module")
        
        # Create sample LSTM model
        print("\n🏗️ Creating sample Enhanced LSTM model...")
        
        model = keras.Sequential([
            keras.layers.LSTM(96, return_sequences=True, input_shape=(15, 24), 
                            dropout=0.25, recurrent_dropout=0.15, name='enhanced_lstm_1'),
            keras.layers.BatchNormalization(name='batch_norm_1'),
            keras.layers.LSTM(48, return_sequences=False, 
                            dropout=0.25, recurrent_dropout=0.15, name='enhanced_lstm_2'),
            keras.layers.BatchNormalization(name='batch_norm_2'),
            keras.layers.Dense(48, activation='relu', 
                             kernel_regularizer=keras.regularizers.l2(0.001), name='dense_enhanced'),
            keras.layers.Dropout(0.25, name='final_dropout'),
            keras.layers.Dense(3, name='output_layer')
        ])
        
        # Compile with enhanced optimizer
        optimizer = keras.optimizers.Adam(
            learning_rate=0.002,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        print(f"✅ Enhanced model created with {model.count_params():,} parameters")
        
        # Create directories
        os.makedirs('visualizations', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
        # 1. Visualize Model Architecture
        print("\n📊 Creating model architecture visualization...")
        
        arch_result = visualizer.visualize_model_architecture(
            model, 
            "Enhanced_LSTM_Manufacturing_Model",
            save_path="visualizations/enhanced_model_architecture.png"
        )
        
        if arch_result['visualization_created']:
            print("✅ Model architecture visualization created successfully")
            print(f"   - Total Parameters: {arch_result['model_info']['total_params']:,}")
            print(f"   - Trainable Parameters: {arch_result['model_info']['trainable_params']:,}")
            print(f"   - Input Shape: {arch_result['model_info']['input_shape']}")
            print(f"   - Output Shape: {arch_result['model_info']['output_shape']}")
        else:
            print("❌ Failed to create architecture visualization")
        
        # 2. Simulate Training Progress and Create Dashboard
        print("\n📈 Simulating enhanced training progress...")
        
        # Add realistic training logs
        visualizer.add_training_log("Enhanced LSTM training started with 96/48 architecture", "INFO")
        visualizer.add_training_log("Learning rate: 0.002 with cosine annealing scheduler", "INFO")
        visualizer.add_training_log("Batch normalization and L2 regularization enabled", "INFO")
        visualizer.add_training_log("Early stopping patience set to 10 epochs", "INFO")
        
        # Simulate realistic enhanced training metrics (showing improvement over time)
        enhanced_training_history = {
            'loss': [4.2, 2.8, 1.9, 1.4, 1.0, 0.7, 0.52, 0.42, 0.36, 0.32, 0.29, 0.26, 0.24],
            'val_loss': [4.5, 3.1, 2.2, 1.6, 1.2, 0.85, 0.62, 0.48, 0.41, 0.37, 0.34, 0.31, 0.28],
            'mae': [1.8, 1.3, 0.95, 0.72, 0.58, 0.47, 0.39, 0.33, 0.29, 0.26, 0.24, 0.22, 0.20],
            'r2_score': [0.02, 0.25, 0.48, 0.65, 0.75, 0.82, 0.87, 0.89, 0.91, 0.92, 0.93, 0.94, 0.95]
        }
        
        # Current enhanced training status
        current_status = {
            'status': 'training',
            'current_epoch': 13,
            'total_epochs': 150,
            'current_batch': 2847,
            'total_batches': 8144,
            'elapsed_minutes': 67.3,
            'estimated_minutes': 245,
            'learning_rate': 0.001854,  # Reduced due to scheduling
            'model_id': 'enhanced_manufacturing_model_2025',
            'best_val_loss': 0.28,
            'early_stopping_patience': 10,
            'current_patience': 3
        }
        
        # Add more realistic training logs
        for epoch in range(1, 14):
            loss_val = enhanced_training_history['loss'][epoch-1]
            val_loss_val = enhanced_training_history['val_loss'][epoch-1]
            r2_val = enhanced_training_history['r2_score'][epoch-1]
            
            if epoch <= 5:
                level = "INFO"
            elif epoch <= 10:
                level = "SUCCESS" if val_loss_val < enhanced_training_history['val_loss'][epoch-2] else "WARNING"
            else:
                level = "SUCCESS"
                
            visualizer.add_training_log(
                f"Epoch {epoch:2d}/150 - Loss: {loss_val:.4f}, Val_Loss: {val_loss_val:.4f}, R²: {r2_val:.4f}", 
                level
            )
        
        visualizer.add_training_log("Learning rate reduced to 0.001854 by cosine scheduler", "INFO")
        visualizer.add_training_log("Model checkpoint saved - best validation loss so far", "SUCCESS")
        visualizer.add_training_log("Enhanced features performing well - R² = 0.95", "SUCCESS")
        
        # Create training status dashboard
        dashboard_result = visualizer.create_training_status_dashboard(
            enhanced_training_history,
            current_status,
            save_path="visualizations/enhanced_training_dashboard.png"
        )
        
        if dashboard_result['dashboard_created']:
            print("✅ Training status dashboard created successfully")
            print(f"   - Current Epoch: {current_status['current_epoch']}/{current_status['total_epochs']}")
            print(f"   - Current R² Score: {enhanced_training_history['r2_score'][-1]:.4f}")
            print(f"   - Best Validation Loss: {min(enhanced_training_history['val_loss']):.4f}")
            print(f"   - Training Progress: {(current_status['current_epoch']/current_status['total_epochs']*100):.1f}%")
        else:
            print("❌ Failed to create training dashboard")
        
        # 3. Export Comprehensive Report
        print("\n📋 Exporting comprehensive model report...")
        
        report_path = visualizer.export_model_report(
            model,
            "Enhanced_LSTM_Manufacturing_Model",
            enhanced_training_history,
            "reports"
        )
        
        print(f"✅ Comprehensive report exported to: {report_path}")
        
        # 4. Display Summary
        print("\n" + "=" * 80)
        print("📊 VISUALIZATION SYSTEM SUMMARY")
        print("=" * 80)
        
        features = [
            "🎯 Model Architecture Visualization - Layer-by-layer breakdown with parameter counts",
            "📈 Training Progress Dashboard - Real-time metrics and progress tracking", 
            "📝 Training Logs Management - Centralized logging with different severity levels",
            "📊 Parameter Distribution Analysis - Visual breakdown of model complexity",
            "🔄 Input/Output Flow Diagrams - Data flow visualization through the network",
            "📋 Comprehensive Reporting - Exportable reports with visualizations and metrics",
            "⚡ Real-time Status Monitoring - Live training status and performance metrics",
            "🎨 Professional Visualizations - High-quality plots and diagrams"
        ]
        
        print("\n✅ IMPLEMENTED FEATURES:")
        for feature in features:
            print(f"   {feature}")
        
        # API Endpoints Available
        api_endpoints = [
            "GET  /api/model/<model_id>/visualize - Generate model architecture visualization",
            "GET  /api/training/status/dashboard - Get comprehensive training dashboard", 
            "GET  /api/training/logs - Retrieve recent training logs",
            "POST /api/training/logs - Add new training log entries",
            "GET  /api/model/<model_id>/report - Export comprehensive model report"
        ]
        
        print("\n🌐 AVAILABLE API ENDPOINTS:")
        for endpoint in api_endpoints:
            print(f"   {endpoint}")
        
        # Files Generated
        generated_files = [
            "visualizations/enhanced_model_architecture.png",
            "visualizations/enhanced_training_dashboard.png", 
            report_path
        ]
        
        print("\n📁 GENERATED FILES:")
        for file_path in generated_files:
            if os.path.exists(file_path):
                size_kb = os.path.getsize(file_path) / 1024
                print(f"   ✅ {file_path} ({size_kb:.1f} KB)")
            else:
                print(f"   ❌ {file_path} (not found)")
        
        print(f"\n🎉 Visualization system demonstration completed successfully!")
        print(f"🕒 Demonstration completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("   Make sure all required dependencies are installed:")
        print("   - matplotlib")
        print("   - seaborn") 
        print("   - tensorflow")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    success = demo_visualization_capabilities()
    sys.exit(0 if success else 1)