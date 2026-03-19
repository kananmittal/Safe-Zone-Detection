#!/usr/bin/env python3
"""
Comprehensive Test for Improved Safe Zone Detection Models
Tests the trained models and improved components
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import joblib

# Add src to path
sys.path.append('src')

from improve_distress_detection import ImprovedDistressDetector
from train_with_synthetic_data import ImprovedEmotionClassifierSynthetic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedModelTester:
    """Test the improved models comprehensively"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.emotion_classifier = None
        self.distress_detector = ImprovedDistressDetector()
        
        # Load trained models
        self.load_trained_models()
    
    def load_trained_models(self):
        """Load the trained emotion classification model"""
        try:
            model_file = self.models_dir / "best_emotion_model.pkl"
            scaler_file = self.models_dir / "emotion_scaler.pkl"
            
            if model_file.exists() and scaler_file.exists():
                self.emotion_classifier = {
                    'model': joblib.load(model_file),
                    'scaler': joblib.load(scaler_file)
                }
                logger.info("✅ Loaded trained emotion classification model")
            else:
                logger.warning("❌ No trained model found, using synthetic data approach")
                self.emotion_classifier = ImprovedEmotionClassifierSynthetic()
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def test_emotion_classification(self) -> Dict:
        """Test emotion classification with synthetic data"""
        logger.info("🧠 Testing Emotion Classification...")
        
        # Generate test data
        emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        results = {
            'total_tests': 0,
            'correct_predictions': 0,
            'per_emotion_accuracy': {},
            'processing_times': []
        }
        
        for emotion in emotions:
            logger.info(f"  Testing {emotion}...")
            
            # Generate synthetic features for this emotion
            if hasattr(self.emotion_classifier, 'generate_training_data'):
                # Use the synthetic classifier
                generator = self.emotion_classifier
                test_features, test_labels = generator.generate_synthetic_features(emotion, 20)
                test_X = np.array(test_features)
                test_y = np.array(test_labels)
                
                # Predict
                start_time = time.time()
                test_X_scaled = generator.scaler.transform(test_X)
                predictions = generator.best_model.predict(test_X_scaled)
                processing_time = time.time() - start_time
                
                # Calculate accuracy
                correct = sum(1 for p, t in zip(predictions, test_y) if p == t)
                accuracy = correct / len(test_y)
                
                results['per_emotion_accuracy'][emotion] = accuracy
                results['total_tests'] += len(test_y)
                results['correct_predictions'] += correct
                results['processing_times'].append(processing_time)
                
                logger.info(f"    {emotion}: {accuracy:.2%} accuracy ({correct}/{len(test_y)})")
        
        # Calculate overall accuracy
        if results['total_tests'] > 0:
            results['overall_accuracy'] = results['correct_predictions'] / results['total_tests']
        else:
            results['overall_accuracy'] = 0.0
        
        results['avg_processing_time'] = np.mean(results['processing_times']) if results['processing_times'] else 0.0
        
        logger.info(f"  Overall Emotion Accuracy: {results['overall_accuracy']:.2%}")
        logger.info(f"  Average Processing Time: {results['avg_processing_time']:.3f}s")
        
        return results
    
    def test_distress_detection(self) -> Dict:
        """Test improved distress detection"""
        logger.info("🚨 Testing Distress Detection...")
        
        test_cases = [
            {
                'transcript': 'Help me, I need emergency assistance right now!',
                'voice_features': {'pitch_mean': 200, 'rms_mean': 0.2, 'tempo': 140},
                'emotion_scores': {'fearful': 0.8, 'angry': 0.6, 'sad': 0.2},
                'expected_distress': True,
                'description': 'High distress - emergency call'
            },
            {
                'transcript': 'Hello, how are you today? Everything is fine.',
                'voice_features': {'pitch_mean': 150, 'rms_mean': 0.1, 'tempo': 100},
                'emotion_scores': {'happy': 0.7, 'neutral': 0.3, 'fearful': 0.0},
                'expected_distress': False,
                'description': 'Normal conversation'
            },
            {
                'transcript': 'I am feeling scared and worried about what might happen',
                'voice_features': {'pitch_mean': 180, 'rms_mean': 0.15, 'tempo': 120},
                'emotion_scores': {'fearful': 0.6, 'sad': 0.4, 'neutral': 0.2},
                'expected_distress': True,
                'description': 'Moderate distress - worried'
            },
            {
                'transcript': 'This is a normal business meeting about quarterly reports',
                'voice_features': {'pitch_mean': 140, 'rms_mean': 0.08, 'tempo': 90},
                'emotion_scores': {'neutral': 0.8, 'calm': 0.2, 'fearful': 0.0},
                'expected_distress': False,
                'description': 'Business conversation'
            },
            {
                'transcript': 'I am in danger! Someone is following me!',
                'voice_features': {'pitch_mean': 220, 'rms_mean': 0.25, 'tempo': 160},
                'emotion_scores': {'fearful': 0.9, 'angry': 0.7, 'sad': 0.1},
                'expected_distress': True,
                'description': 'Critical distress - danger'
            }
        ]
        
        results = {
            'total_cases': len(test_cases),
            'correct_predictions': 0,
            'processing_times': [],
            'detailed_results': []
        }
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"  Test Case {i+1}: {test_case['description']}")
            
            start_time = time.time()
            
            # Analyze distress
            result = self.distress_detector.analyze_multi_modal_distress_improved(
                test_case['transcript'],
                test_case['voice_features'],
                test_case['emotion_scores']
            )
            
            processing_time = time.time() - start_time
            results['processing_times'].append(processing_time)
            
            # Check if prediction matches expected
            predicted_distress = result['distress_level'] in ['MEDIUM', 'HIGH', 'CRITICAL']
            expected_distress = test_case['expected_distress']
            is_correct = predicted_distress == expected_distress
            
            if is_correct:
                results['correct_predictions'] += 1
            
            # Store detailed result
            detailed_result = {
                'case': i+1,
                'description': test_case['description'],
                'transcript': test_case['transcript'],
                'expected_distress': expected_distress,
                'predicted_distress': predicted_distress,
                'distress_level': result['distress_level'],
                'safety_action': result['safety_action'],
                'confidence': result['confidence'],
                'is_correct': is_correct,
                'processing_time': processing_time
            }
            results['detailed_results'].append(detailed_result)
            
            status = "✅" if is_correct else "❌"
            logger.info(f"    {status} Expected: {expected_distress}, Got: {predicted_distress} ({result['distress_level']})")
            logger.info(f"    Safety Action: {result['safety_action']}, Confidence: {result['confidence']:.2f}")
        
        # Calculate accuracy
        results['accuracy'] = results['correct_predictions'] / results['total_cases']
        results['avg_processing_time'] = np.mean(results['processing_times'])
        
        logger.info(f"  Distress Detection Accuracy: {results['accuracy']:.2%}")
        logger.info(f"  Average Processing Time: {results['avg_processing_time']:.3f}s")
        
        return results
    
    def test_performance_improvements(self) -> Dict:
        """Test performance improvements"""
        logger.info("⚡ Testing Performance Improvements...")
        
        # Test emotion classification speed
        emotion_times = []
        for _ in range(10):
            start_time = time.time()
            # Simulate emotion classification
            if hasattr(self.emotion_classifier, 'generate_training_data'):
                generator = self.emotion_classifier
                test_features, _ = generator.generate_synthetic_features('neutral', 1)
                test_X = np.array(test_features)
                test_X_scaled = generator.scaler.transform(test_X)
                _ = generator.best_model.predict(test_X_scaled)
            emotion_times.append(time.time() - start_time)
        
        # Test distress detection speed
        distress_times = []
        for _ in range(5):
            start_time = time.time()
            self.distress_detector.analyze_multi_modal_distress_improved(
                "Test transcript",
                {'pitch_mean': 150, 'rms_mean': 0.1, 'tempo': 120},
                {'neutral': 0.8, 'happy': 0.2}
            )
            distress_times.append(time.time() - start_time)
        
        results = {
            'emotion_classification': {
                'avg_time': np.mean(emotion_times),
                'min_time': np.min(emotion_times),
                'max_time': np.max(emotion_times)
            },
            'distress_detection': {
                'avg_time': np.mean(distress_times),
                'min_time': np.min(distress_times),
                'max_time': np.max(distress_times)
            }
        }
        
        logger.info(f"  Emotion Classification: {results['emotion_classification']['avg_time']:.3f}s avg")
        logger.info(f"  Distress Detection: {results['distress_detection']['avg_time']:.3f}s avg")
        
        return results
    
    def generate_comprehensive_report(self, emotion_results: Dict, distress_results: Dict, performance_results: Dict):
        """Generate comprehensive test report"""
        logger.info("\n" + "="*60)
        logger.info("📊 COMPREHENSIVE IMPROVED MODEL TEST REPORT")
        logger.info("="*60)
        
        # Overall summary
        logger.info(f"\n🎯 OVERALL SUMMARY")
        logger.info(f"Emotion Classification: {emotion_results['overall_accuracy']:.2%} accuracy")
        logger.info(f"Distress Detection: {distress_results['accuracy']:.2%} accuracy")
        logger.info(f"Emotion Processing: {emotion_results['avg_processing_time']:.3f}s avg")
        logger.info(f"Distress Processing: {distress_results['avg_processing_time']:.3f}s avg")
        
        # Performance comparison
        logger.info(f"\n📈 PERFORMANCE COMPARISON")
        logger.info(f"Emotion Classification:")
        logger.info(f"  Current: {emotion_results['overall_accuracy']:.2%} (vs 10.42% original)")
        logger.info(f"  Improvement: +{(emotion_results['overall_accuracy'] - 0.1042) * 100:.1f}%")
        
        logger.info(f"Distress Detection:")
        logger.info(f"  Current: {distress_results['accuracy']:.2%} (vs 33.33% original)")
        logger.info(f"  Improvement: +{(distress_results['accuracy'] - 0.3333) * 100:.1f}%")
        
        # Detailed results
        logger.info(f"\n🔍 DETAILED DISTRESS DETECTION RESULTS")
        for result in distress_results['detailed_results']:
            status = "✅" if result['is_correct'] else "❌"
            logger.info(f"  {status} Case {result['case']}: {result['description']}")
            logger.info(f"    Expected: {result['expected_distress']}, Got: {result['predicted_distress']}")
            logger.info(f"    Level: {result['distress_level']}, Action: {result['safety_action']}")
        
        # Save detailed results
        comprehensive_results = {
            'emotion_classification': emotion_results,
            'distress_detection': distress_results,
            'performance': performance_results,
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'improvements': {
                'emotion_accuracy_improvement': (emotion_results['overall_accuracy'] - 0.1042) * 100,
                'distress_accuracy_improvement': (distress_results['accuracy'] - 0.3333) * 100
            }
        }
        
        with open('improved_model_test_results.json', 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        logger.info(f"\n💾 Detailed results saved to: improved_model_test_results.json")
        
        return comprehensive_results

def main():
    """Main test function"""
    logger.info("🚀 Starting Comprehensive Test of Improved Models")
    logger.info("="*50)
    
    tester = ImprovedModelTester()
    
    # Test emotion classification
    emotion_results = tester.test_emotion_classification()
    
    # Test distress detection
    distress_results = tester.test_distress_detection()
    
    # Test performance
    performance_results = tester.test_performance_improvements()
    
    # Generate comprehensive report
    comprehensive_results = tester.generate_comprehensive_report(
        emotion_results, distress_results, performance_results
    )
    
    logger.info("\n🎉 Comprehensive testing completed!")
    return comprehensive_results

if __name__ == "__main__":
    main()
