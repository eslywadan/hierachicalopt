#!/usr/bin/env python3
"""
Simple test script to verify Flask server and CORS configuration
"""
import requests
import json

def test_backend():
    base_url = "http://localhost:5001"
    
    print("🧪 Testing Flask Backend Server...")
    print("-" * 50)
    
    # Test health endpoint
    try:
        print("1. Testing /health endpoint...")
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Headers: {dict(response.headers)}")
        print(f"   Response: {response.json()}")
        print("   ✅ Health check passed!")
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Cannot connect to server at {base_url}")
        print("   Please ensure the Flask server is running:")
        print("   cd backend && python run.py")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    print("-" * 50)
    print("✅ Backend server is running!")
    
    # Check CORS headers
    print("\n2. Checking CORS headers...")
    cors_headers = response.headers.get('Access-Control-Allow-Origin')
    if cors_headers:
        print(f"   ✅ CORS is configured: {cors_headers}")
    else:
        print("   ⚠️  CORS headers not found in response")
    
    return True

if __name__ == "__main__":
    test_backend()