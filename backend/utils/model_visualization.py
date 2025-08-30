"""
Model Visualization and Training Status Module
Provides tools for visualizing model architecture and monitoring training progress
"""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI issues
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from datetime import datetime
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
import tensorflow as tf
from tensorflow import keras
import logging
import time
import threading
from collections import deque

logger = logging.getLogger(__name__)

class ModelVisualizer:
    """Visualizes model architecture, parameters, and training status"""
    
    def __init__(self):
        self.training_history = {}
        self.current_training_stats = {}
        self.training_logs = deque(maxlen=1000)
        
        # Initialize with some sample logs
        self._initialize_sample_logs()
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
    
    def _initialize_sample_logs(self):
        """Initialize with some sample logs to show in the dashboard"""
        sample_logs = [
            {"message": "LSTM Backend Service initialized", "level": "INFO"},
            {"message": "Backend API endpoints ready", "level": "INFO"},
            {"message": "Model visualization service started", "level": "INFO"},
            {"message": "Ready to accept training requests", "level": "SUCCESS"}
        ]
        
        for log in sample_logs:
            self.add_training_log(log["message"], log["level"])
        
    def visualize_model_architecture(self, model: tf.keras.Model, model_name: str = "LSTM Model", 
                                   save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a comprehensive visualization of the model architecture
        
        Args:
            model: Keras model to visualize
            model_name: Name of the model
            save_path: Path to save the visualization
            
        Returns:
            Dictionary containing model information and visualization data
        """
        try:
            # Extract model information
            model_info = self._extract_model_info(model, model_name)
            
            # Create architecture visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{model_name} - Architecture Overview', fontsize=16, fontweight='bold')
            
            # 1. Layer Architecture Diagram
            self._draw_layer_architecture(ax1, model, model_info)
            
            # 2. Parameter Distribution
            self._plot_parameter_distribution(ax2, model_info)
            
            # 3. Model Summary Table
            self._create_summary_table(ax3, model_info)
            
            # 4. Input/Output Flow Diagram
            self._draw_io_flow_diagram(ax4, model_info)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"📊 Model visualization saved to {save_path}")
            
            return {
                'model_info': model_info,
                'visualization_created': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error creating model visualization: {str(e)}")
            return {
                'error': str(e),
                'visualization_created': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def _extract_model_info(self, model: tf.keras.Model, model_name: str) -> Dict[str, Any]:
        """Extract comprehensive information from the model"""
        
        # Get model summary as string
        summary_lines = []
        model.summary(print_fn=summary_lines.append)
        
        # Extract layer information
        layers_info = []
        total_params = 0
        trainable_params = 0
        
        for i, layer in enumerate(model.layers):
            layer_params = layer.count_params()
            layer_trainable = sum([tf.size(w).numpy() for w in layer.trainable_weights])
            
            layer_info = {
                'index': i,
                'name': layer.name,
                'type': layer.__class__.__name__,
                'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'Unknown',
                'params': layer_params,
                'trainable_params': int(layer_trainable) if layer_trainable else 0,
                'config': self._safe_get_config(layer)
            }
            
            layers_info.append(layer_info)
            total_params += layer_params
            trainable_params += layer_info['trainable_params']
        
        # Get input and output shapes
        input_shape = model.input_shape if hasattr(model, 'input_shape') else None
        output_shape = model.output_shape if hasattr(model, 'output_shape') else None
        
        return {
            'name': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': total_params - trainable_params,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'layers': layers_info,
            'summary': '\n'.join(summary_lines),
            'optimizer': str(model.optimizer) if hasattr(model, 'optimizer') else 'Unknown',
            'loss': str(model.loss) if hasattr(model, 'loss') else 'Unknown',
            'metrics': [str(m) for m in model.metrics] if hasattr(model, 'metrics') else []
        }
    
    def _safe_get_config(self, layer) -> Dict[str, Any]:
        """Safely get layer configuration"""
        try:
            config = layer.get_config()
            # Filter out non-serializable items
            safe_config = {}
            for key, value in config.items():
                try:
                    json.dumps(value)  # Test if serializable
                    safe_config[key] = value
                except (TypeError, ValueError):
                    safe_config[key] = str(value)
            return safe_config
        except Exception:
            return {'error': 'Config not available'}
    
    def _draw_layer_architecture(self, ax, model, model_info):
        """Draw the layer architecture diagram"""
        ax.set_title('Layer Architecture', fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, len(model_info['layers']) + 1)
        
        # Color mapping for different layer types
        color_map = {
            'LSTM': '#3498db',
            'Dense': '#e74c3c',
            'Dropout': '#f39c12',
            'BatchNormalization': '#2ecc71',
            'Input': '#9b59b6',
            'Activation': '#e67e22'
        }
        
        for i, layer_info in enumerate(model_info['layers']):
            y_pos = len(model_info['layers']) - i
            layer_type = layer_info['type']
            
            # Choose color
            color = color_map.get(layer_type, '#95a5a6')
            
            # Draw layer box
            rect = FancyBboxPatch(
                (1, y_pos - 0.4), 8, 0.8,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor='black',
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add layer text
            ax.text(5, y_pos, f"{layer_info['name']}\n{layer_type}\nParams: {layer_info['params']:,}",
                   ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            
            # Draw connection arrows
            if i < len(model_info['layers']) - 1:
                ax.arrow(5, y_pos - 0.5, 0, -0.5, head_width=0.2, head_length=0.1,
                        fc='black', ec='black')
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    def _plot_parameter_distribution(self, ax, model_info):
        """Plot parameter distribution across layers"""
        ax.set_title('Parameter Distribution by Layer', fontweight='bold')
        
        layer_names = [layer['name'] for layer in model_info['layers'] if layer['params'] > 0]
        layer_params = [layer['params'] for layer in model_info['layers'] if layer['params'] > 0]
        
        if layer_params:
            colors = plt.cm.Set3(np.linspace(0, 1, len(layer_params)))
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(layer_params, labels=layer_names, autopct='%1.1f%%',
                                             colors=colors, startangle=90)
            
            # Improve text readability
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(8)
        else:
            ax.text(0.5, 0.5, 'No trainable parameters', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
    
    def _create_summary_table(self, ax, model_info):
        """Create a summary information table"""
        ax.set_title('Model Summary', fontweight='bold')
        ax.axis('off')
        
        # Prepare table data
        table_data = [
            ['Property', 'Value'],
            ['Model Name', model_info['name']],
            ['Total Parameters', f"{model_info['total_params']:,}"],
            ['Trainable Parameters', f"{model_info['trainable_params']:,}"],
            ['Non-trainable Parameters', f"{model_info['non_trainable_params']:,}"],
            ['Input Shape', str(model_info['input_shape'])],
            ['Output Shape', str(model_info['output_shape'])],
            ['Total Layers', str(len(model_info['layers']))],
            ['Optimizer', model_info['optimizer'][:50] + '...' if len(model_info['optimizer']) > 50 else model_info['optimizer']],
            ['Loss Function', model_info['loss']]
        ]
        
        # Create table
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                        cellLoc='left', loc='center', colWidths=[0.4, 0.6])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Header styling
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(len(table_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
    
    def _draw_io_flow_diagram(self, ax, model_info):
        """Draw input/output flow diagram"""
        ax.set_title('Input/Output Flow', fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # Input box
        input_box = FancyBboxPatch(
            (1, 7), 3, 1.5,
            boxstyle="round,pad=0.1",
            facecolor='#3498db',
            edgecolor='black',
            alpha=0.7
        )
        ax.add_patch(input_box)
        ax.text(2.5, 7.75, f"INPUT\n{model_info['input_shape']}", ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')
        
        # Model processing box
        model_box = FancyBboxPatch(
            (3, 4), 4, 2,
            boxstyle="round,pad=0.1",
            facecolor='#2ecc71',
            edgecolor='black',
            alpha=0.7
        )
        ax.add_patch(model_box)
        ax.text(5, 5, f"MODEL PROCESSING\n{len(model_info['layers'])} Layers\n{model_info['trainable_params']:,} Parameters",
               ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # Output box
        output_box = FancyBboxPatch(
            (6, 1), 3, 1.5,
            boxstyle="round,pad=0.1",
            facecolor='#e74c3c',
            edgecolor='black',
            alpha=0.7
        )
        ax.add_patch(output_box)
        ax.text(7.5, 1.75, f"OUTPUT\n{model_info['output_shape']}", ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')
        
        # Draw arrows
        ax.arrow(2.5, 6.8, 1.5, -1.5, head_width=0.2, head_length=0.2, fc='black', ec='black')
        ax.arrow(5.5, 3.8, 1.5, -1.5, head_width=0.2, head_length=0.2, fc='black', ec='black')
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    def create_training_status_dashboard(self, training_history: Dict[str, Any], 
                                       current_status: Dict[str, Any],
                                       save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a comprehensive training status dashboard
        
        Args:
            training_history: Dictionary containing training history
            current_status: Current training status information
            save_path: Path to save the dashboard
            
        Returns:
            Dictionary containing dashboard information
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle('Training Status Dashboard', fontsize=16, fontweight='bold')
            
            # 1. Loss and Metrics Over Time
            self._plot_training_metrics(ax1, training_history)
            
            # 2. Current Training Progress
            self._plot_training_progress(ax2, current_status)
            
            # 3. Training Statistics Summary
            self._create_training_summary(ax3, training_history, current_status)
            
            # 4. Recent Training Logs
            self._display_training_logs(ax4)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"📊 Training dashboard saved to {save_path}")
            
            return {
                'dashboard_created': True,
                'timestamp': datetime.now().isoformat(),
                'status': current_status,
                'history_points': len(training_history.get('loss', []))
            }
            
        except Exception as e:
            logger.error(f"❌ Error creating training dashboard: {str(e)}")
            return {
                'error': str(e),
                'dashboard_created': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def _plot_training_metrics(self, ax, training_history):
        """Plot training metrics over time"""
        ax.set_title('Training Metrics Over Time', fontweight='bold')
        
        if not training_history:
            ax.text(0.5, 0.5, 'No training history available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return
        
        # Plot loss
        if 'loss' in training_history:
            epochs = range(1, len(training_history['loss']) + 1)
            ax.plot(epochs, training_history['loss'], 'b-', label='Training Loss', linewidth=2)
        
        if 'val_loss' in training_history:
            epochs = range(1, len(training_history['val_loss']) + 1)
            ax.plot(epochs, training_history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
        
        # Add other metrics if available
        for metric_name in ['mae', 'mse', 'r2_score']:
            if metric_name in training_history:
                epochs = range(1, len(training_history[metric_name]) + 1)
                ax.plot(epochs, training_history[metric_name], label=metric_name.upper(), alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_training_progress(self, ax, current_status):
        """Plot current training progress"""
        ax.set_title('Current Training Progress', fontweight='bold')
        
        # Create progress indicators
        progress_data = [
            ('Epochs', current_status.get('current_epoch', 0), current_status.get('total_epochs', 100)),
            ('Batch', current_status.get('current_batch', 0), current_status.get('total_batches', 100)),
            ('Time', current_status.get('elapsed_minutes', 0), current_status.get('estimated_minutes', 60))
        ]
        
        y_positions = [2.5, 1.5, 0.5]
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        for i, (label, current, total) in enumerate(progress_data):
            progress = (current / total) * 100 if total > 0 else 0
            
            # Progress bar background
            bar_bg = FancyBboxPatch(
                (1, y_positions[i] - 0.15), 8, 0.3,
                boxstyle="round,pad=0.02",
                facecolor='lightgray',
                edgecolor='gray'
            )
            ax.add_patch(bar_bg)
            
            # Progress bar fill
            if progress > 0:
                bar_fill = FancyBboxPatch(
                    (1, y_positions[i] - 0.15), 8 * (progress / 100), 0.3,
                    boxstyle="round,pad=0.02",
                    facecolor=colors[i],
                    edgecolor=colors[i],
                    alpha=0.8
                )
                ax.add_patch(bar_fill)
            
            # Progress text
            ax.text(0.5, y_positions[i], f"{label}:", ha='right', va='center', fontweight='bold')
            ax.text(5, y_positions[i], f"{current}/{total} ({progress:.1f}%)", 
                   ha='center', va='center', fontweight='bold', color='white' if progress > 30 else 'black')
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    def _create_training_summary(self, ax, training_history, current_status):
        """Create training summary statistics"""
        ax.set_title('Training Summary', fontweight='bold')
        ax.axis('off')
        
        # Calculate summary statistics
        current_loss = training_history.get('loss', [0])[-1] if training_history.get('loss') else 'N/A'
        best_loss = min(training_history.get('loss', [float('inf')])) if training_history.get('loss') else 'N/A'
        current_val_loss = training_history.get('val_loss', [0])[-1] if training_history.get('val_loss') else 'N/A'
        
        # Prepare summary data
        summary_data = [
            ['Metric', 'Value'],
            ['Training Status', current_status.get('status', 'Unknown')],
            ['Current Epoch', f"{current_status.get('current_epoch', 0)}/{current_status.get('total_epochs', 0)}"],
            ['Current Loss', f"{current_loss:.6f}" if isinstance(current_loss, (int, float)) else str(current_loss)],
            ['Best Loss', f"{best_loss:.6f}" if isinstance(best_loss, (int, float)) else str(best_loss)],
            ['Current Val Loss', f"{current_val_loss:.6f}" if isinstance(current_val_loss, (int, float)) else str(current_val_loss)],
            ['Learning Rate', f"{current_status.get('learning_rate', 'N/A')}"],
            ['Elapsed Time', f"{current_status.get('elapsed_minutes', 0):.1f} min"],
            ['ETA', f"{current_status.get('eta_minutes', 0):.1f} min"],
            ['Model ID', current_status.get('model_id', 'N/A')]
        ]
        
        # Create table
        table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
                        cellLoc='left', loc='center', colWidths=[0.5, 0.5])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Header styling
        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(summary_data)):
            for j in range(len(summary_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
    
    def _display_training_logs(self, ax):
        """Display recent training logs"""
        ax.set_title('Recent Training Logs', fontweight='bold')
        ax.axis('off')
        
        if not self.training_logs:
            ax.text(0.5, 0.5, 'No training logs available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return
        
        # Get recent logs
        recent_logs = list(self.training_logs)[-10:]  # Last 10 logs
        
        log_text = ""
        for i, log_entry in enumerate(recent_logs):
            timestamp_raw = log_entry.get('timestamp', 'Unknown')
            message = log_entry.get('message', 'No message')
            
            # Format timestamp for display (show only time if it's a full ISO timestamp)
            if isinstance(timestamp_raw, str) and 'T' in timestamp_raw:
                try:
                    dt = datetime.fromisoformat(timestamp_raw.replace('Z', '+00:00'))
                    timestamp = dt.strftime('%H:%M:%S')
                except:
                    timestamp = timestamp_raw
            else:
                timestamp = str(timestamp_raw)
                
            log_text += f"{timestamp}: {message}\n"
        
        ax.text(0.05, 0.95, log_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='#f8f9fa', alpha=0.8))
    
    def add_training_log(self, message: str, level: str = 'INFO'):
        """Add a training log entry"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        }
        self.training_logs.append(log_entry)
        logger.info(f"📝 Training log: {message}")
    
    def update_training_status(self, status_update: Dict[str, Any]):
        """Update current training status"""
        self.current_training_stats.update(status_update)
        self.add_training_log(f"Status update: {status_update}")
    
    def export_model_report(self, model: tf.keras.Model, model_name: str, 
                           training_history: Dict[str, Any], save_dir: str) -> str:
        """
        Export a comprehensive model report with visualizations
        
        Args:
            model: Keras model
            model_name: Name of the model
            training_history: Training history data
            save_dir: Directory to save the report
            
        Returns:
            Path to the generated report
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create model architecture visualization
        arch_path = f"{save_dir}/model_architecture_{timestamp}.png"
        self.visualize_model_architecture(model, model_name, arch_path)
        
        # Create training dashboard
        if training_history:
            dashboard_path = f"{save_dir}/training_dashboard_{timestamp}.png"
            self.create_training_status_dashboard(training_history, self.current_training_stats, dashboard_path)
        
        # Generate text report
        report_path = f"{save_dir}/model_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(f"Model Report: {model_name}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            
            # Model summary
            f.write("MODEL SUMMARY:\n")
            f.write("-" * 40 + "\n")
            model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write("\n")
            
            # Training history
            if training_history:
                f.write("TRAINING HISTORY:\n")
                f.write("-" * 40 + "\n")
                for key, values in training_history.items():
                    if isinstance(values, list) and values:
                        f.write(f"{key.upper()}:\n")
                        f.write(f"  Initial: {values[0]:.6f}\n")
                        f.write(f"  Final: {values[-1]:.6f}\n")
                        f.write(f"  Best: {min(values) if 'loss' in key else max(values):.6f}\n\n")
        
        logger.info(f"📊 Model report exported to {report_path}")
        return report_path


# Global instance for easy access
model_visualizer = ModelVisualizer()