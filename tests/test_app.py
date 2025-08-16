#!/usr/bin/env python3
"""
Tests for the Voice Distress Detection System
"""

import pytest
import requests
import tempfile
import os

def test_server_running():
    """Test if the server is running"""
    try:
        response = requests.get("http://127.0.0.1:8000/", timeout=5)
        assert response.status_code == 200
        assert "Voice Distress Detection" in response.text
    except requests.exceptions.ConnectionError:
        pytest.skip("Server not running")

def test_upload_endpoint():
    """Test the upload endpoint"""
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("test content")
        temp_path = f.name
    
    try:
        # Test upload
        with open(temp_path, 'rb') as f:
            files = {"file": ("test.txt", f, "text/plain")}
            response = requests.post("http://127.0.0.1:8000/voice-check", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "transcript" in data
        assert "distress" in data
        assert "label" in data
        
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def test_distress_detection():
    """Test distress detection logic"""
    from src.app import detect_distress_simple
    
    # Test distress keywords
    assert detect_distress_simple("help me") == "Distress"
    assert detect_distress_simple("I'm scared") == "Distress"
    assert detect_distress_simple("emergency") == "Distress"
    
    # Test safe content
    assert detect_distress_simple("hello world") == "Safe"
    assert detect_distress_simple("nice weather") == "Safe" 