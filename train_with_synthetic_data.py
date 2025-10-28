#!/usr/bin/env python3
"""
Training with Synthetic Data for Safe Zone Detection
Creates synthetic training data and trains improved models
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Add src to path
sys.path.append('src')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Generate synthetic training data for emotion classification"""
    
    def __init__(self):
        self.emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotions)}
        self.idx_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_idx.items()}
    
    def generate_synthetic_features(self, emotion: str, num_samples: int = 100) -> Tuple[List, List]:
        """Generate synthetic audio features for a given emotion"""
        features = []
        labels = []
        
        # Define feature ranges for each emotion
        emotion_ranges = {
            'neutral': {
                'mfcc_mean': (0, 10), 'mfcc_std': (5, 15),
                'pitch_mean': (140, 160), 'pitch_std': (10, 30),
                'rms_mean': (0.08, 0.12), 'rms_std': (0.02, 0.05),
                'tempo': (100, 120)
            },
            'calm': {
                'mfcc_mean': (0, 8), 'mfcc_std': (3, 12),
                'pitch_mean': (130, 150), 'pitch_std': (5, 20),
                'rms_mean': (0.06, 0.10), 'rms_std': (0.01, 0.03),
                'tempo': (80, 110)
            },
            'happy': {
                'mfcc_mean': (5, 15), 'mfcc_std': (8, 20),
                'pitch_mean': (160, 200), 'pitch_std': (15, 40),
                'rms_mean': (0.10, 0.18), 'rms_std': (0.03, 0.08),
                'tempo': (120, 160)
            },
            'sad': {
                'mfcc_mean': (0, 8), 'mfcc_std': (3, 10),
                'pitch_mean': (100, 140), 'pitch_std': (5, 25),
                'rms_mean': (0.04, 0.08), 'rms_std': (0.01, 0.04),
                'tempo': (60, 100)
            },
            'angry': {
                'mfcc_mean': (8, 20), 'mfcc_std': (10, 25),
                'pitch_mean': (150, 200), 'pitch_std': (20, 50),
                'rms_mean': (0.12, 0.25), 'rms_std': (0.05, 0.12),
                'tempo': (110, 150)
            },
            'fearful': {
                'mfcc_mean': (5, 18), 'mfcc_std': (8, 22),
                'pitch_mean': (160, 220), 'pitch_std': (25, 60),
                'rms_mean': (0.08, 0.20), 'rms_std': (0.04, 0.10),
                'tempo': (130, 180)
            },
            'disgust': {
                'mfcc_mean': (3, 12), 'mfcc_std': (5, 15),
                'pitch_mean': (120, 160), 'pitch_std': (10, 35),
                'rms_mean': (0.06, 0.14), 'rms_std': (0.02, 0.06),
                'tempo': (90, 130)
            },
            'surprised': {
                'mfcc_mean': (8, 18), 'mfcc_std': (10, 25),
                'pitch_mean': (170, 220), 'pitch_std': (20, 45),
                'rms_mean': (0.10, 0.20), 'rms_std': (0.04, 0.09),
                'tempo': (120, 170)
            }
        }
        
        ranges = emotion_ranges.get(emotion, emotion_ranges['neutral'])
        
        for _ in range(num_samples):
            # Generate features within the emotion's range
            feature_vector = []
            
            # MFCC features
            mfcc_mean = np.random.uniform(*ranges['mfcc_mean'])
            mfcc_std = np.random.uniform(*ranges['mfcc_std'])
            feature_vector.extend([mfcc_mean, mfcc_std, mfcc_mean + mfcc_std, mfcc_mean - mfcc_std])
            
            # Pitch features
            pitch_mean = np.random.uniform(*ranges['pitch_mean'])
            pitch_std = np.random.uniform(*ranges['pitch_std'])
            feature_vector.extend([pitch_mean, pitch_std, pitch_mean + pitch_std, pitch_mean - pitch_std])
            
            # RMS features
            rms_mean = np.random.uniform(*ranges['rms_mean'])
            rms_std = np.random.uniform(*ranges['rms_std'])
            feature_vector.extend([rms_mean, rms_std, rms_mean + rms_std, rms_mean - rms_std])
            
            # Spectral features
            spec_cent_mean = np.random.uniform(2000, 3000)
            spec_cent_std = np.random.uniform(100, 500)
            feature_vector.extend([spec_cent_mean, spec_cent_std])
            
            spec_rolloff_mean = np.random.uniform(3000, 5000)
            spec_rolloff_std = np.random.uniform(200, 800)
            feature_vector.extend([spec_rolloff_mean, spec_rolloff_std])
            
            # ZCR features
            zcr_mean = np.random.uniform(0.01, 0.1)
            zcr_std = np.random.uniform(0.005, 0.05)
            feature_vector.extend([zcr_mean, zcr_std])
            
            # Tempo and speech rate
            tempo = np.random.uniform(*ranges['tempo'])
            speech_rate = tempo + np.random.uniform(-20, 20)
            feature_vector.extend([tempo, speech_rate])
            
            features.append(feature_vector)
            labels.append(self.emotion_to_idx[emotion])
        
        return features, labels

class ImprovedEmotionClassifierSynthetic:
    """Improved emotion classifier using synthetic data"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotions)}
        self.idx_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_idx.items()}
        
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_accuracy = 0.0
    
    def generate_training_data(self, samples_per_emotion: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data"""
        logger.info("Generating synthetic training data...")
        
        generator = SyntheticDataGenerator()
        all_features = []
        all_labels = []
        
        for emotion in self.emotions:
            logger.info(f"Generating {samples_per_emotion} samples for {emotion}")
            features, labels = generator.generate_synthetic_features(emotion, samples_per_emotion)
            all_features.extend(features)
            all_labels.extend(labels)
        
        X = np.array(all_features)
        y = np.array(all_labels)
        
        logger.info(f"Generated {len(X)} samples with {X.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def train_models(self, X: np.ndarray, y: np.ndarray):
        """Train multiple ML models and select the best one"""
        logger.info("Training emotion classification models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to try
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'MLP': MLPClassifier(hidden_layer_sizes=(512, 256, 128), random_state=42, max_iter=1000)
        }
        
        # Train and evaluate models
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            logger.info(f"{name} CV scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Train on full training set
            model.fit(X_train_scaled, y_train)
            
            # Evaluate on test set
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"{name} Test Accuracy: {accuracy:.3f}")
            
            # Track best model
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model
                logger.info(f"New best model: {name} with accuracy {accuracy:.3f}")
        
        # Save best model and scaler
        joblib.dump(self.best_model, self.models_dir / "best_emotion_model.pkl")
        joblib.dump(self.scaler, self.models_dir / "emotion_scaler.pkl")
        
        # Generate detailed report
        y_pred_best = self.best_model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred_best, target_names=self.emotions)
        logger.info(f"Best model classification report:\n{report}")
        
        return self.best_model, self.best_accuracy

def main():
    """Main training function"""
    logger.info("Starting training with synthetic data...")
    
    classifier = ImprovedEmotionClassifierSynthetic()
    
    # Generate training data
    X, y = classifier.generate_training_data(samples_per_emotion=200)
    
    # Train models
    best_model, accuracy = classifier.train_models(X, y)
    
    logger.info(f"Training complete! Best accuracy: {accuracy:.3f}")
    
    # Test the model
    logger.info("Testing the trained model...")
    
    # Generate some test data
    generator = SyntheticDataGenerator()
    test_features, test_labels = generator.generate_synthetic_features('happy', 10)
    test_X = np.array(test_features)
    test_y = np.array(test_labels)
    
    # Test predictions
    test_X_scaled = classifier.scaler.transform(test_X)
    predictions = classifier.best_model.predict(test_X_scaled)
    
    logger.info(f"Test predictions: {[classifier.idx_to_emotion[p] for p in predictions[:5]]}")
    logger.info(f"Actual labels: {[classifier.idx_to_emotion[l] for l in test_y[:5]]}")
    
    logger.info("Synthetic data training completed successfully!")

if __name__ == "__main__":
    main()
