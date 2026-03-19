#!/usr/bin/env python3
"""
Advanced Emotion Recognition System
Comprehensive improvement for emotional recognition accuracy
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import joblib
import librosa
import soundfile as sf
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedAudioFeatureExtractor:
    """Advanced audio feature extraction for emotion recognition"""
    
    def __init__(self, sample_rate=16000, n_mfcc=13, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def extract_comprehensive_features(self, audio_path: str) -> Dict:
        """Extract comprehensive audio features for emotion recognition"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            features = {}
            
            # 1. MFCC Features (13 coefficients + deltas)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            features.update({
                'mfcc_mean': np.mean(mfcc, axis=1).tolist(),
                'mfcc_std': np.std(mfcc, axis=1).tolist(),
                'mfcc_delta_mean': np.mean(mfcc_delta, axis=1).tolist(),
                'mfcc_delta_std': np.std(mfcc_delta, axis=1).tolist(),
                'mfcc_delta2_mean': np.mean(mfcc_delta2, axis=1).tolist(),
                'mfcc_delta2_std': np.std(mfcc_delta2, axis=1).tolist()
            })
            
            # 2. Spectral Features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=self.hop_length)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=self.hop_length)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=self.hop_length)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
            
            features.update({
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_centroid_std': np.std(spectral_centroids),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'spectral_rolloff_std': np.std(spectral_rolloff),
                'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
                'spectral_bandwidth_std': np.std(spectral_bandwidth),
                'zcr_mean': np.mean(zero_crossing_rate),
                'zcr_std': np.std(zero_crossing_rate)
            })
            
            # 3. Rhythm Features
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, hop_length=self.hop_length)
            features['tempo'] = tempo
            
            # 4. Energy Features
            rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
            features.update({
                'rms_mean': np.mean(rms),
                'rms_std': np.std(rms),
                'rms_max': np.max(rms),
                'rms_min': np.min(rms)
            })
            
            # 5. Pitch Features
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, hop_length=self.hop_length)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features.update({
                    'pitch_mean': np.mean(pitch_values),
                    'pitch_std': np.std(pitch_values),
                    'pitch_max': np.max(pitch_values),
                    'pitch_min': np.min(pitch_values)
                })
            else:
                features.update({
                    'pitch_mean': 0.0,
                    'pitch_std': 0.0,
                    'pitch_max': 0.0,
                    'pitch_min': 0.0
                })
            
            # 6. Chroma Features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=self.hop_length)
            features.update({
                'chroma_mean': np.mean(chroma, axis=1).tolist(),
                'chroma_std': np.std(chroma, axis=1).tolist()
            })
            
            # 7. Tonnetz Features
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            features.update({
                'tonnetz_mean': np.mean(tonnetz, axis=1).tolist(),
                'tonnetz_std': np.std(tonnetz, axis=1).tolist()
            })
            
            # 8. Spectral Contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=self.hop_length)
            features.update({
                'spectral_contrast_mean': np.mean(spectral_contrast, axis=1).tolist(),
                'spectral_contrast_std': np.std(spectral_contrast, axis=1).tolist()
            })
            
            # 9. Mel Spectrogram Features
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=self.hop_length)
            features.update({
                'mel_spec_mean': np.mean(mel_spec, axis=1).tolist(),
                'mel_spec_std': np.std(mel_spec, axis=1).tolist()
            })
            
            # 10. Statistical Features
            features.update({
                'skewness': skew(audio),
                'kurtosis': kurtosis(audio),
                'duration': len(audio) / sr
            })
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {e}")
            return self._get_default_features()
    
    def _get_default_features(self) -> Dict:
        """Return default features if extraction fails"""
        return {
            'mfcc_mean': [0.0] * self.n_mfcc,
            'mfcc_std': [0.0] * self.n_mfcc,
            'mfcc_delta_mean': [0.0] * self.n_mfcc,
            'mfcc_delta_std': [0.0] * self.n_mfcc,
            'mfcc_delta2_mean': [0.0] * self.n_mfcc,
            'mfcc_delta2_std': [0.0] * self.n_mfcc,
            'spectral_centroid_mean': 0.0,
            'spectral_centroid_std': 0.0,
            'spectral_rolloff_mean': 0.0,
            'spectral_rolloff_std': 0.0,
            'spectral_bandwidth_mean': 0.0,
            'spectral_bandwidth_std': 0.0,
            'zcr_mean': 0.0,
            'zcr_std': 0.0,
            'tempo': 120.0,
            'rms_mean': 0.0,
            'rms_std': 0.0,
            'rms_max': 0.0,
            'rms_min': 0.0,
            'pitch_mean': 0.0,
            'pitch_std': 0.0,
            'pitch_max': 0.0,
            'pitch_min': 0.0,
            'chroma_mean': [0.0] * 12,
            'chroma_std': [0.0] * 12,
            'tonnetz_mean': [0.0] * 6,
            'tonnetz_std': [0.0] * 6,
            'spectral_contrast_mean': [0.0] * 7,
            'spectral_contrast_std': [0.0] * 7,
            'mel_spec_mean': [0.0] * 128,
            'mel_spec_std': [0.0] * 128,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'duration': 1.0
        }

class AdvancedEmotionClassifier:
    """Advanced emotion classifier with multiple approaches"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotions)}
        self.idx_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_idx.items()}
        
        self.feature_extractor = AdvancedAudioFeatureExtractor()
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.pca = None
        self.ensemble_model = None
        self.individual_models = {}
        
    def create_feature_vector(self, features: Dict) -> np.ndarray:
        """Create feature vector from extracted features"""
        feature_vector = []
        
        # Flatten all features
        for key, value in features.items():
            if isinstance(value, list):
                feature_vector.extend(value)
            else:
                feature_vector.append(float(value))
        
        return np.array(feature_vector)
    
    def generate_enhanced_synthetic_data(self, samples_per_emotion: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """Generate enhanced synthetic data with more realistic patterns"""
        logger.info("Generating enhanced synthetic data...")
        
        all_features = []
        all_labels = []
        
        # More realistic emotion patterns
        emotion_patterns = {
            'neutral': {
                'pitch_range': (120, 160), 'energy_range': (0.05, 0.12),
                'tempo_range': (90, 120), 'zcr_range': (0.01, 0.05),
                'spectral_centroid_range': (2000, 3000)
            },
            'calm': {
                'pitch_range': (100, 140), 'energy_range': (0.03, 0.08),
                'tempo_range': (70, 100), 'zcr_range': (0.005, 0.03),
                'spectral_centroid_range': (1800, 2500)
            },
            'happy': {
                'pitch_range': (160, 220), 'energy_range': (0.08, 0.20),
                'tempo_range': (120, 160), 'zcr_range': (0.02, 0.08),
                'spectral_centroid_range': (2500, 3500)
            },
            'sad': {
                'pitch_range': (80, 130), 'energy_range': (0.02, 0.06),
                'tempo_range': (60, 90), 'zcr_range': (0.005, 0.025),
                'spectral_centroid_range': (1500, 2200)
            },
            'angry': {
                'pitch_range': (140, 200), 'energy_range': (0.10, 0.25),
                'tempo_range': (110, 150), 'zcr_range': (0.03, 0.10),
                'spectral_centroid_range': (2500, 4000)
            },
            'fearful': {
                'pitch_range': (160, 250), 'energy_range': (0.05, 0.18),
                'tempo_range': (130, 180), 'zcr_range': (0.02, 0.12),
                'spectral_centroid_range': (3000, 4500)
            },
            'disgust': {
                'pitch_range': (100, 160), 'energy_range': (0.04, 0.12),
                'tempo_range': (80, 120), 'zcr_range': (0.01, 0.06),
                'spectral_centroid_range': (1800, 2800)
            },
            'surprised': {
                'pitch_range': (170, 230), 'energy_range': (0.08, 0.22),
                'tempo_range': (120, 170), 'zcr_range': (0.02, 0.09),
                'spectral_centroid_range': (2800, 3800)
            }
        }
        
        for emotion in self.emotions:
            logger.info(f"Generating {samples_per_emotion} samples for {emotion}")
            pattern = emotion_patterns[emotion]
            
            for _ in range(samples_per_emotion):
                # Generate features based on emotion patterns
                features = self._generate_emotion_features(emotion, pattern)
                feature_vector = self.create_feature_vector(features)
                
                all_features.append(feature_vector)
                all_labels.append(self.emotion_to_idx[emotion])
        
        X = np.array(all_features)
        y = np.array(all_labels)
        
        logger.info(f"Generated {len(X)} samples with {X.shape[1]} features")
        return X, y
    
    def _generate_emotion_features(self, emotion: str, pattern: Dict) -> Dict:
        """Generate realistic features for a specific emotion"""
        features = {}
        
        # Generate MFCC features
        mfcc_base = np.random.normal(0, 1, 13)
        features['mfcc_mean'] = mfcc_base.tolist()
        features['mfcc_std'] = (np.abs(mfcc_base) * 0.5).tolist()
        features['mfcc_delta_mean'] = (mfcc_base * 0.3).tolist()
        features['mfcc_delta_std'] = (np.abs(mfcc_base) * 0.2).tolist()
        features['mfcc_delta2_mean'] = (mfcc_base * 0.1).tolist()
        features['mfcc_delta2_std'] = (np.abs(mfcc_base) * 0.1).tolist()
        
        # Generate pitch features
        pitch_mean = np.random.uniform(*pattern['pitch_range'])
        features.update({
            'pitch_mean': pitch_mean,
            'pitch_std': np.random.uniform(10, 30),
            'pitch_max': pitch_mean + np.random.uniform(20, 50),
            'pitch_min': max(0, pitch_mean - np.random.uniform(20, 50))
        })
        
        # Generate energy features
        rms_mean = np.random.uniform(*pattern['energy_range'])
        features.update({
            'rms_mean': rms_mean,
            'rms_std': rms_mean * 0.3,
            'rms_max': rms_mean * np.random.uniform(1.5, 2.5),
            'rms_min': rms_mean * np.random.uniform(0.3, 0.7)
        })
        
        # Generate spectral features
        spec_cent = np.random.uniform(*pattern['spectral_centroid_range'])
        features.update({
            'spectral_centroid_mean': spec_cent,
            'spectral_centroid_std': spec_cent * 0.2,
            'spectral_rolloff_mean': spec_cent * np.random.uniform(1.2, 1.8),
            'spectral_rolloff_std': spec_cent * 0.3,
            'spectral_bandwidth_mean': spec_cent * np.random.uniform(0.8, 1.2),
            'spectral_bandwidth_std': spec_cent * 0.25
        })
        
        # Generate other features
        features.update({
            'zcr_mean': np.random.uniform(*pattern['zcr_range']),
            'zcr_std': np.random.uniform(0.005, 0.02),
            'tempo': np.random.uniform(*pattern['tempo_range']),
            'chroma_mean': np.random.uniform(0, 1, 12).tolist(),
            'chroma_std': np.random.uniform(0, 0.5, 12).tolist(),
            'tonnetz_mean': np.random.uniform(-1, 1, 6).tolist(),
            'tonnetz_std': np.random.uniform(0, 0.5, 6).tolist(),
            'spectral_contrast_mean': np.random.uniform(0, 2, 7).tolist(),
            'spectral_contrast_std': np.random.uniform(0, 0.5, 7).tolist(),
            'mel_spec_mean': np.random.uniform(0, 1, 128).tolist(),
            'mel_spec_std': np.random.uniform(0, 0.3, 128).tolist(),
            'skewness': np.random.uniform(-2, 2),
            'kurtosis': np.random.uniform(-1, 5),
            'duration': np.random.uniform(1.0, 5.0)
        })
        
        return features
    
    def train_advanced_models(self, X: np.ndarray, y: np.ndarray):
        """Train advanced ensemble of models"""
        logger.info("Training advanced emotion classification models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply PCA for dimensionality reduction
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        logger.info(f"PCA reduced features from {X_train_scaled.shape[1]} to {X_train_pca.shape[1]}")
        
        # Define individual models
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=8,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf', C=10, gamma='scale', probability=True,
                random_state=42
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(512, 256, 128, 64),
                activation='relu', solver='adam', alpha=0.001,
                learning_rate='adaptive', max_iter=1000, random_state=42
            )
        }
        
        # Train individual models
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_pca, y_train, cv=5, n_jobs=-1)
            logger.info(f"{name} CV scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Train on full training set
            model.fit(X_train_pca, y_train)
            
            # Evaluate on test set
            y_pred = model.predict(X_test_pca)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"{name} Test Accuracy: {accuracy:.3f}")
            
            # Store model
            self.individual_models[name] = model
        
        # Create ensemble model
        logger.info("Creating ensemble model...")
        self.ensemble_model = VotingClassifier(
            estimators=list(self.individual_models.items()),
            voting='soft'  # Use predicted probabilities
        )
        
        # Train ensemble
        self.ensemble_model.fit(X_train_pca, y_train)
        
        # Evaluate ensemble
        y_pred_ensemble = self.ensemble_model.predict(X_test_pca)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        
        logger.info(f"Ensemble Test Accuracy: {ensemble_accuracy:.3f}")
        
        # Generate detailed report
        report = classification_report(y_test, y_pred_ensemble, target_names=self.emotions)
        logger.info(f"Ensemble classification report:\n{report}")
        
        # Save models
        joblib.dump(self.ensemble_model, self.models_dir / "advanced_ensemble_model.pkl")
        joblib.dump(self.scaler, self.models_dir / "advanced_scaler.pkl")
        joblib.dump(self.pca, self.models_dir / "advanced_pca.pkl")
        
        for name, model in self.individual_models.items():
            joblib.dump(model, self.models_dir / f"advanced_{name.lower()}.pkl")
        
        return ensemble_accuracy
    
    def predict_emotion(self, audio_path: str) -> Dict:
        """Predict emotion for an audio file"""
        try:
            # Extract features
            features = self.feature_extractor.extract_comprehensive_features(audio_path)
            feature_vector = self.create_feature_vector(features)
            
            # Load models if not already loaded
            if self.ensemble_model is None:
                self.ensemble_model = joblib.load(self.models_dir / "advanced_ensemble_model.pkl")
                self.scaler = joblib.load(self.models_dir / "advanced_scaler.pkl")
                self.pca = joblib.load(self.models_dir / "advanced_pca.pkl")
            
            # Preprocess features
            X = np.array([feature_vector])
            X_scaled = self.scaler.transform(X)
            X_pca = self.pca.transform(X_scaled)
            
            # Predict
            prediction = self.ensemble_model.predict(X_pca)[0]
            probabilities = self.ensemble_model.predict_proba(X_pca)[0]
            
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
                'model_used': 'Advanced_Ensemble',
                'features_extracted': len(feature_vector)
            }
            
        except Exception as e:
            logger.error(f"Error in emotion prediction: {e}")
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'error': str(e),
                'model_used': 'Error'
            }

def main():
    """Main function to train and test advanced emotion recognition"""
    logger.info("🚀 Starting Advanced Emotion Recognition Training")
    logger.info("="*60)
    
    classifier = AdvancedEmotionClassifier()
    
    # Generate enhanced training data
    X, y = classifier.generate_enhanced_synthetic_data(samples_per_emotion=500)
    
    # Train models
    accuracy = classifier.train_advanced_models(X, y)
    
    logger.info(f"🎉 Training complete! Final accuracy: {accuracy:.3f}")
    
    # Test the model
    logger.info("🧪 Testing the trained model...")
    
    # Generate test data
    test_X, test_y = classifier.generate_enhanced_synthetic_data(samples_per_emotion=50)
    
    # Test predictions
    correct = 0
    total = len(test_X)
    
    for i in range(min(10, total)):  # Test first 10 samples
        # Create a dummy audio file for testing
        test_audio_path = f"test_audio_{i}.wav"
        
        # Generate synthetic audio data
        duration = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Generate different audio patterns based on emotion
        emotion_idx = test_y[i]
        emotion = classifier.idx_to_emotion[emotion_idx]
        
        if emotion == 'happy':
            audio = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
        elif emotion == 'sad':
            audio = 0.3 * np.sin(2 * np.pi * 220 * t)
        elif emotion == 'angry':
            audio = np.sin(2 * np.pi * 330 * t) + 0.8 * np.sin(2 * np.pi * 660 * t)
        else:
            audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Add some noise
        audio += 0.1 * np.random.randn(len(audio))
        
        # Save test audio
        sf.write(test_audio_path, audio, sample_rate)
        
        try:
            # Predict emotion
            result = classifier.predict_emotion(test_audio_path)
            predicted_emotion = result['emotion']
            expected_emotion = emotion
            
            if predicted_emotion == expected_emotion:
                correct += 1
            
            logger.info(f"  Sample {i+1}: Expected {expected_emotion}, Got {predicted_emotion} (Confidence: {result['confidence']:.3f})")
            
        except Exception as e:
            logger.error(f"Error testing sample {i+1}: {e}")
        finally:
            # Clean up test file
            if os.path.exists(test_audio_path):
                os.remove(test_audio_path)
    
    test_accuracy = correct / min(10, total)
    logger.info(f"Test accuracy: {test_accuracy:.3f} ({correct}/{min(10, total)})")
    
    logger.info("🎉 Advanced emotion recognition training completed!")

if __name__ == "__main__":
    main()
