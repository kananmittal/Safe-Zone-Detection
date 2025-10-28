from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import subprocess
from pydub import AudioSegment
import tempfile
import numpy as np
import traceback
import torch
import torchaudio
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import joblib

logger = logging.getLogger(__name__)

# Import the improved emotion classifier
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

class TorchAudioProcessor:
    """Enhanced TorchAudio processor with improved emotion classification"""
    
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using {self.device} for acceleration")
        
        # Initialize improved emotion classifier
        self.emotion_classifier = ImprovedEmotionClassifier()
        
        # Try to load trained model
        try:
            self.emotion_classifier.best_model = joblib.load("models/best_emotion_model.pkl")
            self.emotion_classifier.scaler = joblib.load("models/emotion_scaler.pkl")
            logger.info("Loaded trained emotion classification model")
        except FileNotFoundError:
            logger.warning("No trained model found, using rule-based fallback")
    
    def process_audio(self, audio_path: str) -> Dict:
        """Process audio with improved emotion classification"""
        try:
            # Extract features using original method
            features = self.extract_features(audio_path)
            
            # Use improved emotion classification
            if hasattr(self.emotion_classifier, 'best_model') and self.emotion_classifier.best_model is not None:
                # Use ML model
                result = self.emotion_classifier.predict_emotion(audio_path)
                emotion = result['emotion']
                confidence = result['confidence']
                emotion_scores = result.get('emotion_scores', {})
            else:
                # Fallback to rule-based
                emotion_scores = self.analyze_emotion_from_features(features)
                emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = self.calculate_confidence(features, emotion_scores)
            
            return {
                'emotion': emotion,
                'emotion_scores': emotion_scores,
                'confidence': confidence,
                'features': features,
                'device_used': self.device
            }
            
        except Exception as e:
            logger.error(f"Error in audio processing: {e}")
            return {
                'emotion': 'neutral',
                'emotion_scores': {'neutral': 1.0},
                'confidence': 0.0,
                'error': str(e),
                'device_used': self.device
            }
    
    def extract_features(self, audio_path: str) -> Dict:
        """Extract audio features (simplified version)"""
        # This would contain the original feature extraction logic
        # For now, return a placeholder
        return {
            'mfcc_mean': 0.0,
            'mfcc_std': 0.0,
            'pitch_mean': 150.0,
            'pitch_std': 20.0,
            'rms_mean': 0.1,
            'rms_std': 0.05,
            'tempo': 120.0
        }
    
    def analyze_emotion_from_features(self, features: Dict) -> Dict:
        """Fallback rule-based emotion analysis"""
        emotion_scores = {
            'neutral': 0.0, 'calm': 0.0, 'happy': 0.0, 'sad': 0.0,
            'angry': 0.0, 'fearful': 0.0, 'disgust': 0.0, 'surprised': 0.0
        }
        
        # Simple rule-based logic
        pitch_mean = features.get('pitch_mean', 150)
        if pitch_mean > 170:
            emotion_scores['happy'] += 0.5
        elif pitch_mean < 130:
            emotion_scores['sad'] += 0.5
        else:
            emotion_scores['neutral'] += 0.5
        
        return emotion_scores
    
    def calculate_confidence(self, features: Dict, emotion_scores: Dict) -> float:
        """Calculate confidence score"""
        return 0.7  # Placeholder
