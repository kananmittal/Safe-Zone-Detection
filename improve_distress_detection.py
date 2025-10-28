#!/usr/bin/env python3
"""
Improved Distress Detection Logic
Enhances the multi-modal analysis with better thresholds and ensemble methods
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Add src to path
sys.path.append('src')

from src.llama_processor import analyze_multi_modal_distress

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedDistressDetector:
    """Improved distress detection with ensemble methods and better thresholds"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Distress indicators with weights
        self.distress_indicators = {
            'text_keywords': {
                'high_priority': ['help', 'emergency', 'danger', 'scared', 'fear', 'threat', 'unsafe', 'panic', 'terrified'],
                'medium_priority': ['afraid', 'worried', 'anxious', 'distress', 'crying', 'screaming'],
                'low_priority': ['pain', 'hurt', 'attack', 'robbery', 'stuck', 'trapped']
            },
            'voice_emotions': {
                'high_distress': ['fearful', 'angry', 'disgust'],
                'medium_distress': ['sad'],
                'low_distress': ['surprised']
            },
            'voice_features': {
                'pitch_thresholds': {
                    'very_high': 200,  # Very high pitch indicates distress
                    'high': 180,
                    'low': 120,        # Very low pitch indicates sadness/distress
                    'very_low': 100
                },
                'energy_thresholds': {
                    'very_high': 0.15,  # Very loud indicates anger/distress
                    'high': 0.12,
                    'low': 0.05,        # Very quiet indicates sadness/distress
                    'very_low': 0.03
                },
                'tempo_thresholds': {
                    'very_fast': 140,   # Very fast speech indicates panic
                    'fast': 120,
                    'slow': 80,         # Very slow speech indicates depression
                    'very_slow': 60
                }
            }
        }
        
        # Ensemble model for final decision
        self.ensemble_model = None
        self.is_trained = False
    
    def analyze_text_distress(self, transcript: str) -> Dict:
        """Enhanced text-based distress analysis"""
        transcript_lower = transcript.lower()
        
        # Count keyword occurrences
        keyword_scores = {'high': 0, 'medium': 0, 'low': 0}
        
        for priority, keywords in self.distress_indicators['text_keywords'].items():
            for keyword in keywords:
                count = transcript_lower.count(keyword)
                if priority == 'high_priority':
                    keyword_scores['high'] += count * 3
                elif priority == 'medium_priority':
                    keyword_scores['medium'] += count * 2
                else:
                    keyword_scores['low'] += count * 1
        
        # Calculate distress probability
        total_score = keyword_scores['high'] + keyword_scores['medium'] + keyword_scores['low']
        max_possible_score = len(self.distress_indicators['text_keywords']['high_priority']) * 3
        
        distress_probability = min(total_score / max_possible_score, 1.0)
        
        # Determine distress level
        if distress_probability > 0.7:
            distress_level = 'HIGH'
        elif distress_probability > 0.4:
            distress_level = 'MEDIUM'
        elif distress_probability > 0.1:
            distress_level = 'LOW'
        else:
            distress_level = 'NONE'
        
        return {
            'distress_level': distress_level,
            'probability': distress_probability,
            'keyword_scores': keyword_scores,
            'reasoning': f"Text analysis: {distress_probability:.2f} distress probability"
        }
    
    def analyze_voice_distress(self, voice_emotion: str, voice_features: Dict) -> Dict:
        """Enhanced voice-based distress analysis"""
        # Emotion-based analysis
        emotion_scores = {'high': 0, 'medium': 0, 'low': 0}
        
        if voice_emotion in self.distress_indicators['voice_emotions']['high_distress']:
            emotion_scores['high'] += 3
        elif voice_emotion in self.distress_indicators['voice_emotions']['medium_distress']:
            emotion_scores['medium'] += 2
        elif voice_emotion in self.distress_indicators['voice_emotions']['low_distress']:
            emotion_scores['low'] += 1
        
        # Feature-based analysis
        feature_scores = {'high': 0, 'medium': 0, 'low': 0}
        
        # Pitch analysis
        pitch_mean = voice_features.get('pitch_mean', 150)
        if pitch_mean > self.distress_indicators['voice_features']['pitch_thresholds']['very_high']:
            feature_scores['high'] += 2
        elif pitch_mean > self.distress_indicators['voice_features']['pitch_thresholds']['high']:
            feature_scores['medium'] += 1
        elif pitch_mean < self.distress_indicators['voice_features']['pitch_thresholds']['very_low']:
            feature_scores['high'] += 2
        elif pitch_mean < self.distress_indicators['voice_features']['pitch_thresholds']['low']:
            feature_scores['medium'] += 1
        
        # Energy analysis
        rms_mean = voice_features.get('rms_mean', 0.1)
        if rms_mean > self.distress_indicators['voice_features']['energy_thresholds']['very_high']:
            feature_scores['high'] += 2
        elif rms_mean > self.distress_indicators['voice_features']['energy_thresholds']['high']:
            feature_scores['medium'] += 1
        elif rms_mean < self.distress_indicators['voice_features']['energy_thresholds']['very_low']:
            feature_scores['high'] += 2
        elif rms_mean < self.distress_indicators['voice_features']['energy_thresholds']['low']:
            feature_scores['medium'] += 1
        
        # Tempo analysis
        tempo = voice_features.get('tempo', 120)
        if tempo > self.distress_indicators['voice_features']['tempo_thresholds']['very_fast']:
            feature_scores['high'] += 2
        elif tempo > self.distress_indicators['voice_features']['tempo_thresholds']['fast']:
            feature_scores['medium'] += 1
        elif tempo < self.distress_indicators['voice_features']['tempo_thresholds']['very_slow']:
            feature_scores['high'] += 2
        elif tempo < self.distress_indicators['voice_features']['tempo_thresholds']['slow']:
            feature_scores['medium'] += 1
        
        # Calculate total voice distress score
        total_voice_score = sum(emotion_scores.values()) + sum(feature_scores.values())
        max_voice_score = 15  # Maximum possible score
        
        voice_distress_probability = min(total_voice_score / max_voice_score, 1.0)
        
        # Determine distress level
        if voice_distress_probability > 0.6:
            distress_level = 'HIGH'
        elif voice_distress_probability > 0.3:
            distress_level = 'MEDIUM'
        elif voice_distress_probability > 0.1:
            distress_level = 'LOW'
        else:
            distress_level = 'NONE'
        
        return {
            'distress_level': distress_level,
            'probability': voice_distress_probability,
            'emotion_scores': emotion_scores,
            'feature_scores': feature_scores,
            'reasoning': f"Voice analysis: {voice_distress_probability:.2f} distress probability"
        }
    
    def ensemble_distress_analysis(self, text_result: Dict, voice_result: Dict, 
                                 emotion_scores: Dict) -> Dict:
        """Ensemble method for final distress determination"""
        
        # Weighted combination of text and voice analysis
        text_weight = 0.4
        voice_weight = 0.4
        emotion_weight = 0.2
        
        # Convert distress levels to numerical scores
        level_scores = {'NONE': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
        
        text_score = level_scores[text_result['distress_level']] * text_result['probability']
        voice_score = level_scores[voice_result['distress_level']] * voice_result['probability']
        
        # Emotion score from emotion_scores
        emotion_score = 0
        if emotion_scores:
            distress_emotions = ['fearful', 'angry', 'disgust', 'sad']
            for emotion in distress_emotions:
                if emotion in emotion_scores:
                    emotion_score += emotion_scores[emotion]
            emotion_score = min(emotion_score, 1.0)
        
        # Calculate ensemble score
        ensemble_score = (text_score * text_weight + 
                         voice_score * voice_weight + 
                         emotion_score * emotion_weight)
        
        # Determine final distress level
        if ensemble_score > 2.0:
            final_level = 'CRITICAL'
            safety_action = 'EMERGENCY'
        elif ensemble_score > 1.5:
            final_level = 'HIGH'
            safety_action = 'ALERT'
        elif ensemble_score > 1.0:
            final_level = 'MEDIUM'
            safety_action = 'MONITOR'
        elif ensemble_score > 0.3:
            final_level = 'LOW'
            safety_action = 'OBSERVE'
        else:
            final_level = 'NONE'
            safety_action = 'NONE'
        
        # Calculate confidence
        confidence = min(ensemble_score / 3.0, 1.0)
        
        return {
            'distress_level': final_level,
            'safety_action': safety_action,
            'confidence': confidence,
            'ensemble_score': ensemble_score,
            'text_score': text_score,
            'voice_score': voice_score,
            'emotion_score': emotion_score,
            'reasoning': f"Ensemble analysis: {final_level} distress (score: {ensemble_score:.2f})"
        }
    
    def analyze_multi_modal_distress_improved(self, transcript: str, voice_features: Dict, 
                                            emotion_scores: Dict) -> Dict:
        """Improved multi-modal distress analysis"""
        
        # Analyze text
        text_result = self.analyze_text_distress(transcript)
        
        # Analyze voice
        voice_emotion = max(emotion_scores, key=emotion_scores.get) if emotion_scores else 'neutral'
        voice_result = self.analyze_voice_distress(voice_emotion, voice_features)
        
        # Ensemble analysis
        final_result = self.ensemble_distress_analysis(text_result, voice_result, emotion_scores)
        
        # Add individual analysis results
        final_result.update({
            'text_analysis': text_result,
            'voice_analysis': voice_result,
            'llama_analysis': False,  # This is the improved version
            'method': 'improved_ensemble'
        })
        
        return final_result
    
    def train_ensemble_model(self, training_data: List[Dict]):
        """Train ensemble model on historical data"""
        logger.info("Training ensemble model...")
        
        # This would implement training on historical distress detection data
        # For now, we'll use the rule-based approach
        
        logger.info("Ensemble model training not implemented yet")
        self.is_trained = True
    
    def save_model(self):
        """Save the trained model"""
        model_data = {
            'distress_indicators': self.distress_indicators,
            'is_trained': self.is_trained
        }
        
        with open(self.models_dir / "improved_distress_detector.json", 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info("Model saved successfully")
    
    def load_model(self):
        """Load the trained model"""
        model_file = self.models_dir / "improved_distress_detector.json"
        if model_file.exists():
            with open(model_file, 'r') as f:
                model_data = json.load(f)
            
            self.distress_indicators = model_data['distress_indicators']
            self.is_trained = model_data['is_trained']
            
            logger.info("Model loaded successfully")
        else:
            logger.info("No saved model found, using default configuration")

def main():
    """Test the improved distress detection"""
    logger.info("Testing improved distress detection...")
    
    detector = ImprovedDistressDetector()
    
    # Test cases
    test_cases = [
        {
            'transcript': 'Help me, I need emergency assistance!',
            'voice_features': {'pitch_mean': 200, 'rms_mean': 0.2, 'tempo': 140},
            'emotion_scores': {'fearful': 0.8, 'angry': 0.6, 'sad': 0.2}
        },
        {
            'transcript': 'Hello, how are you today?',
            'voice_features': {'pitch_mean': 150, 'rms_mean': 0.1, 'tempo': 100},
            'emotion_scores': {'happy': 0.7, 'neutral': 0.3, 'fearful': 0.0}
        },
        {
            'transcript': 'I am feeling scared and worried about what might happen',
            'voice_features': {'pitch_mean': 180, 'rms_mean': 0.15, 'tempo': 120},
            'emotion_scores': {'fearful': 0.6, 'sad': 0.4, 'neutral': 0.2}
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        logger.info(f"\nTest Case {i+1}:")
        logger.info(f"Transcript: {test_case['transcript']}")
        
        result = detector.analyze_multi_modal_distress_improved(
            test_case['transcript'],
            test_case['voice_features'],
            test_case['emotion_scores']
        )
        
        logger.info(f"Result: {result['distress_level']} - {result['safety_action']}")
        logger.info(f"Confidence: {result['confidence']:.2f}")
        logger.info(f"Reasoning: {result['reasoning']}")
    
    # Save model
    detector.save_model()
    
    logger.info("Improved distress detection testing completed!")

if __name__ == "__main__":
    main()
