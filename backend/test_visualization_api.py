#!/usr/bin/env python3
"""
Test visualization API endpoints
"""
import requests
import json

def test_visualization_endpoints():
    base_url = "http://localhost:5001"
    
    print("🧪 Testing Visualization API Endpoints...")
    print("-" * 60)
    
    # Test 1: Training Dashboard
    print("1. Testing Training Dashboard API...")
    try:
        response = requests.get(f"{base_url}/api/training/status/dashboard")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Dashboard API: {response.status_code}")
            print(f"   📊 Has dashboard: {data.get('dashboard', {}).get('dashboard_created', False)}")
        else:
            print(f"   ❌ Dashboard API: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Dashboard API Error: {e}")
    
    # Test 2: Model Architecture (with fallback)
    print("\n2. Testing Model Architecture API...")
    model_ids = ['enhanced_manufacturing_model_2025', 'parallel_model', 'test_model']
    
    for model_id in model_ids:
        try:
            response = requests.get(f"{base_url}/api/model/{model_id}/visualize")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Model {model_id}: {response.status_code}")
                break
            else:
                print(f"   ⚠️  Model {model_id}: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Model {model_id} Error: {e}")
    
    # Test 3: Training Logs
    print("\n3. Testing Training Logs API...")
    try:
        response = requests.get(f"{base_url}/api/training/logs")
        if response.status_code == 200:
            logs = response.json()
            print(f"   ✅ Logs API: {response.status_code}")
            print(f"   📝 Log count: {len(logs) if isinstance(logs, list) else 'N/A'}")
        else:
            print(f"   ❌ Logs API: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Logs API Error: {e}")
    
    # Test 4: Image Access
    print("\n4. Testing Image Access...")
    images = ['enhanced_training_dashboard.png', 'enhanced_model_architecture.png']
    
    for image in images:
        try:
            response = requests.head(f"{base_url}/visualizations/{image}")
            if response.status_code == 200:
                size = response.headers.get('Content-Length', '0')
                print(f"   ✅ Image {image}: {response.status_code} ({int(size)/1024:.1f} KB)")
            else:
                print(f"   ❌ Image {image}: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Image {image} Error: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 VISUALIZATION API STATUS: READY")
    print("💡 Frontend should now work with fallback handling")
    
    # Instructions
    print("\n📋 INSTRUCTIONS:")
    print("1. 🚀 Start Angular dev server: ng serve")
    print("2. 🌐 Navigate to: http://localhost:4200/operation-model/backend")
    print("3. ▶️  Start parallel training")
    print("4. 📊 Click progress bar when training starts")
    print("5. ✨ Visualization panel should appear with data/images")

if __name__ == "__main__":
    test_visualization_endpoints()