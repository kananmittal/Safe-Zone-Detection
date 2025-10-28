#!/usr/bin/env python3
"""
Improved Emotion Classification using Machine Learning
Replaces rule-based approach with trained ML model
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
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add src to path
sys.path.append('src')

from src.audio_processor import TorchAudioProcessor
from src.data_processor import DatasetProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionDataset(Dataset):
    """PyTorch Dataset for emotion classification"""
    
    def __init__(self, features, labels, scaler=None):
        self.features = features
        self.labels = labels
        self.scaler = scaler
        
        if scaler:
            self.features = scaler.transform(features)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])

class EmotionClassifier(nn.Module):
    """Neural Network for emotion classification"""
    
    def __init__(self, input_size, num_classes, hidden_sizes=[512, 256, 128]):
        super(EmotionClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ImprovedEmotionClassifier:
    """Improved emotion classification using multiple ML approaches"""
    
    def __init__(self, datasets_dir="datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.processed_dir = self.datasets_dir / "processed"
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        self.audio_processor = TorchAudioProcessor()
        self.data_processor = DatasetProcessor(str(datasets_dir))
        
        # Emotion mapping
        self.emotion_to_idx = {
            'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3,
            'angry': 4, 'fearful': 5, 'disgust': 6, 'surprised': 7
        }
        self.idx_to_emotion = {v: k for k, v in self.emotion_to_idx.items()}
        
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_accuracy = 0.0
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from all available datasets"""
        logger.info("Preparing training data...")
        
        all_features = []
        all_labels = []
        
        # Process RAVDESS dataset
        ravdess_dir = self.datasets_dir / "ravdess"
        if ravdess_dir.exists():
            logger.info("Processing RAVDESS dataset...")
            ravdess_features, ravdess_labels = self._process_ravdess_data(ravdess_dir)
            all_features.extend(ravdess_features)
            all_labels.extend(ravdess_labels)
        
        # Process TESS dataset
        tess_dir = self.datasets_dir / "tess"
        if tess_dir.exists():
            logger.info("Processing TESS dataset...")
            tess_features, tess_labels = self._process_tess_data(tess_dir)
            all_features.extend(tess_features)
            all_labels.extend(tess_labels)
        
        # Process IEMOCAP dataset
        iemocap_dir = self.datasets_dir / "IEMOCAP"
        if iemocap_dir.exists():
            logger.info("Processing IEMOCAP dataset...")
            iemocap_features, iemocap_labels = self._process_iemocap_data(iemocap_dir)
            all_features.extend(iemocap_features)
            all_labels.extend(iemocap_labels)
        
        if not all_features:
            raise ValueError("No training data found! Please ensure datasets are available.")
        
        X = np.array(all_features)
        y = np.array(all_labels)
        
        logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def _process_ravdess_data(self, ravdess_dir: Path) -> Tuple[List, List]:
        """Process RAVDESS dataset"""
        features = []
        labels = []
        
        emotion_map = {
            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        }
        
        for actor_dir in ravdess_dir.iterdir():
            if actor_dir.is_dir() and actor_dir.name.startswith('Actor_'):
                for audio_file in actor_dir.glob("*.wav"):
                    try:
                        # Extract features
                        result = self.audio_processor.process_audio(str(audio_file))
                        if 'features' in result:
                            features.append(self._extract_feature_vector(result['features']))
                            
                            # Get emotion from filename
                            filename = audio_file.name
                            parts = filename.replace('.wav', '').split('-')
                            if len(parts) >= 7:
                                emotion_code = parts[2]
                                emotion = emotion_map.get(emotion_code, 'neutral')
                                labels.append(self.emotion_to_idx[emotion])
                    except Exception as e:
                        logger.warning(f"Error processing {audio_file}: {e}")
                        continue
        
        return features, labels
    
    def _process_tess_data(self, tess_dir: Path) -> Tuple[List, List]:
        """Process TESS dataset"""
        features = []
        labels = []
        
        for emotion_dir in tess_dir.iterdir():
            if emotion_dir.is_dir() and emotion_dir.name.startswith(('OAF_', 'YAF_')):
                emotion = emotion_dir.name.split('_')[1].lower()
                if emotion in self.emotion_to_idx:
                    for audio_file in emotion_dir.glob("*.wav"):
                        try:
                            result = self.audio_processor.process_audio(str(audio_file))
                            if 'features' in result:
                                features.append(self._extract_feature_vector(result['features']))
                                labels.append(self.emotion_to_idx[emotion])
                        except Exception as e:
                            logger.warning(f"Error processing {audio_file}: {e}")
                            continue
        
        return features, labels
    
    def _process_iemocap_data(self, iemocap_dir: Path) -> Tuple[List, List]:
        """Process IEMOCAP dataset (simplified)"""
        features = []
        labels = []
        
        # This is a simplified version - in practice, you'd need to parse the complex IEMOCAP structure
        # For now, we'll skip this and focus on RAVDESS and TESS
        logger.info("IEMOCAP processing not implemented yet - skipping")
        return features, labels
    
    def _extract_feature_vector(self, features: Dict) -> List[float]:
        """Extract feature vector from features dictionary"""
        feature_vector = []
        
        # Key features for emotion classification
        key_features = [
            'mfcc_mean', 'mfcc_std', 'mfcc_max', 'mfcc_min',
            'pitch_mean', 'pitch_std', 'pitch_max', 'pitch_min',
            'rms_mean', 'rms_std', 'rms_max', 'rms_min',
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std',
            'zcr_mean', 'zcr_std',
            'tempo', 'speech_rate'
        ]
        
        for feature_name in key_features:
            if feature_name in features:
                value = features[feature_name]
                if isinstance(value, (list, np.ndarray)):
                    feature_vector.extend([np.mean(value), np.std(value)])
                else:
                    feature_vector.append(float(value))
            else:
                feature_vector.extend([0.0, 0.0])  # Default values
        
        return feature_vector
    
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
            
            # Save model
            self.models[name] = model
            
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
        report = classification_report(y_test, y_pred_best, target_names=list(self.emotion_to_idx.keys()))
        logger.info(f"Best model classification report:\n{report}")
        
        return self.best_model, self.best_accuracy
    
    def predict_emotion(self, audio_path: str) -> Dict:
        """Predict emotion for a single audio file"""
        try:
            # Extract features
            result = self.audio_processor.process_audio(audio_path)
            if 'features' not in result:
                return {'emotion': 'neutral', 'confidence': 0.0, 'error': 'No features extracted'}
            
            # Extract feature vector
            feature_vector = self._extract_feature_vector(result['features'])
            X = np.array([feature_vector])
            X_scaled = self.scaler.transform(X)
            
            # Predict
            if self.best_model is None:
                # Load saved model
                self.best_model = joblib.load(self.models_dir / "best_emotion_model.pkl")
                self.scaler = joblib.load(self.models_dir / "emotion_scaler.pkl")
            
            prediction = self.best_model.predict(X_scaled)[0]
            probabilities = self.best_model.predict_proba(X_scaled)[0]
            
            emotion = self.idx_to_emotion[prediction]
            confidence = float(np.max(probabilities))
            
            # Get emotion scores
            emotion_scores = {}
            for i, prob in enumerate(probabilities):
                emotion_scores[self.idx_to_emotion[i]] = float(prob)
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'emotion_scores': emotion_scores,
                'model_used': 'ML_Classifier'
            }
            
        except Exception as e:
            logger.error(f"Error in emotion prediction: {e}")
            return {'emotion': 'neutral', 'confidence': 0.0, 'error': str(e)}

def main():
    """Main training function"""
    logger.info("Starting improved emotion classification training...")
    
    classifier = ImprovedEmotionClassifier()
    
    # Prepare training data
    X, y = classifier.prepare_training_data()
    
    # Train models
    best_model, accuracy = classifier.train_models(X, y)
    
    logger.info(f"Training complete! Best accuracy: {accuracy:.3f}")
    
    # Test on a few samples
    logger.info("Testing on sample files...")
    ravdess_dir = Path("datasets/ravdess")
    if ravdess_dir.exists():
        for actor_dir in list(ravdess_dir.iterdir())[:2]:  # Test first 2 actors
            if actor_dir.is_dir():
                for audio_file in list(actor_dir.glob("*.wav"))[:1]:  # Test 1 file per actor
                    result = classifier.predict_emotion(str(audio_file))
                    logger.info(f"File: {audio_file.name} -> {result['emotion']} (confidence: {result['confidence']:.3f})")

if __name__ == "__main__":
    main()
