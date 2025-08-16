#!/usr/bin/env python3
"""
Voice Distress Detection System
Main entry point for the FastAPI application
"""

import uvicorn
from src.app import app

if __name__ == "__main__":
    uvicorn.run(
        "src.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    ) 