#!/usr/bin/env python3
"""
Training Visualization Viewer
Simple script to access and view training visualizations
"""
import os
import webbrowser
from pathlib import Path

def show_available_visualizations():
    """Show all available training visualization files"""
    print("🎨 TRAINING VISUALIZATION VIEWER")
    print("=" * 60)
    
    # Check for existing visualization files
    viz_dir = Path("visualizations")
    reports_dir = Path("reports")
    
    if not viz_dir.exists():
        print("❌ Visualizations directory not found. Run demo_visualization.py first.")
        return
        
    print("📊 AVAILABLE VISUALIZATION FILES:")
    print("-" * 40)
    
    # List all PNG files in visualizations directory
    png_files = list(viz_dir.glob("*.png"))
    if png_files:
        for i, file_path in enumerate(png_files, 1):
            size_kb = file_path.stat().st_size / 1024
            print(f"{i}. {file_path.name} ({size_kb:.1f} KB)")
    else:
        print("   No visualization files found.")
    
    print("\n📋 AVAILABLE REPORTS:")
    print("-" * 40)
    
    # List all text files in reports directory
    txt_files = list(reports_dir.glob("*.txt")) if reports_dir.exists() else []
    if txt_files:
        for i, file_path in enumerate(txt_files, 1):
            size_kb = file_path.stat().st_size / 1024
            print(f"{i}. {file_path.name} ({size_kb:.1f} KB)")
    else:
        print("   No report files found.")
    
    print("\n🌐 VISUALIZATION API ENDPOINTS (when server is running):")
    print("-" * 40)
    print("   📈 Training Dashboard: http://localhost:5001/api/training/status/dashboard")
    print("   🏗️  Model Architecture: http://localhost:5001/api/model/<model_id>/visualize")
    print("   📝 Training Logs: http://localhost:5001/api/training/logs")
    
    print("\n💡 HOW TO USE:")
    print("-" * 40)
    print("1. 🖼️  View PNG files: Open them with any image viewer")
    print("2. 📄 View Reports: Open TXT files with any text editor")
    print("3. 🌐 API Access: Use curl or browser when server is running")
    print("4. 🚀 Generate New: Run demo_visualization.py or python run.py")
    
    # Show example API calls
    print("\n📡 EXAMPLE API CALLS:")
    print("-" * 40)
    print("# View training dashboard JSON:")
    print("curl -s http://localhost:5001/api/training/status/dashboard | jq")
    print("")
    print("# Get training logs:")
    print("curl -s http://localhost:5001/api/training/logs")
    print("")
    print("# View model architecture for a specific model:")
    print("curl -s http://localhost:5001/api/model/parallel_model_123/visualize")
    
    return png_files, txt_files

def open_visualization(file_path):
    """Open a visualization file in the default viewer"""
    try:
        if os.name == 'darwin':  # macOS
            os.system(f"open '{file_path}'")
        elif os.name == 'nt':     # Windows
            os.startfile(file_path)
        else:                     # Linux
            os.system(f"xdg-open '{file_path}'")
        print(f"✅ Opened: {file_path}")
    except Exception as e:
        print(f"❌ Error opening {file_path}: {e}")

def main():
    """Main function to show visualization options"""
    png_files, txt_files = show_available_visualizations()
    
    if not png_files and not txt_files:
        print("\n💡 To generate visualizations, run:")
        print("   python demo_visualization.py")
        return
    
    print(f"\n🎯 QUICK ACCESS:")
    print("-" * 40)
    
    # Auto-open the most recent files
    if png_files:
        latest_png = max(png_files, key=lambda p: p.stat().st_mtime)
        print(f"📊 Latest visualization: {latest_png.name}")
        
        choice = input("\nOpen latest visualization? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            open_visualization(str(latest_png))
    
    if txt_files:
        latest_txt = max(txt_files, key=lambda p: p.stat().st_mtime)
        print(f"📋 Latest report: {latest_txt.name}")
        
        choice = input("Open latest report? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            open_visualization(str(latest_txt))

if __name__ == "__main__":
    main()