#!/usr/bin/env python3
"""
Simple run script for the Voice Distress Detection System
"""

import subprocess
import sys
import os

def main():
    """Run the application"""
    print("🎤 Starting Voice Distress Detection System...")
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Virtual environment not detected")
        print("   Please activate your virtual environment first:")
        print("   source .venv/bin/activate")
        print()
    
    # Check if requirements are installed
    try:
        import fastapi
        import uvicorn
        import whisper
        print("✅ All dependencies are installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Please install requirements: pip install -r requirements.txt")
        return
    
    # Start the server
    try:
        print("🚀 Starting server at http://127.0.0.1:8000")
        print("   Press Ctrl+C to stop")
        print()
        
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "src.app:app", 
            "--reload", 
            "--host", "127.0.0.1", 
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main() 