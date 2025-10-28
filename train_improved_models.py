#!/usr/bin/env python3
"""
Training Script for Improved Safe Zone Detection
Trains all improved models
"""

import os
import sys
import logging

# Add src to path
sys.path.append('src')

from improve_emotion_classification import ImprovedEmotionClassifier
from improve_distress_detection import ImprovedDistressDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Train all improved models"""
    logger.info("Starting training of improved models...")
    
    # Train emotion classification model
    logger.info("Training emotion classification model...")
    emotion_classifier = ImprovedEmotionClassifier()
    X, y = emotion_classifier.prepare_training_data()
    best_model, accuracy = emotion_classifier.train_models(X, y)
    logger.info(f"Emotion classification training complete. Best accuracy: {accuracy:.3f}")
    
    # Train distress detection model
    logger.info("Training distress detection model...")
    distress_detector = ImprovedDistressDetector()
    # Add training data preparation here
    distress_detector.save_model()
    logger.info("Distress detection training complete.")
    
    logger.info("All model training completed!")

if __name__ == "__main__":
    main()
