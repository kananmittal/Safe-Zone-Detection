#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
Tests the complete Safe Zone Detection pipeline and generates results matrix
"""

import os
import sys
import json
import time
import requests
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append('src')

from src.audio_processor import TorchAudioProcessor
from src.llama_processor import analyze_multi_modal_distress

class ModelEvaluator:
    def __init__(self):
        self.audio_processor = TorchAudioProcessor()
        self.results = []
        
    def test_audio_processor(self, test_files: List[Dict]) -> Dict:
        """Test the TorchAudio processor on test files"""
        print("🎵 Testing TorchAudio Processor...")
        
        results = {
            'total_files': len(test_files),
            'successful': 0,
            'failed': 0,
            'emotion_accuracy': {},
            'processing_times': [],
            'confidences': []
        }
        
        for i, test_file in enumerate(test_files):
            try:
                print(f"  Processing {i+1}/{len(test_files)}: {test_file['file']}")
                
                start_time = time.time()
                result = self.audio_processor.process_audio(test_file['path'])
                processing_time = time.time() - start_time
                
                expected_emotion = test_file.get('expected_emotion', 'unknown')
                detected_emotion = result['emotion']
                
                # Track accuracy
                if expected_emotion != 'unknown':
                    if expected_emotion not in results['emotion_accuracy']:
                        results['emotion_accuracy'][expected_emotion] = {'correct': 0, 'total': 0}
                    
                    results['emotion_accuracy'][expected_emotion]['total'] += 1
                    if expected_emotion == detected_emotion:
                        results['emotion_accuracy'][expected_emotion]['correct'] += 1
                
                results['successful'] += 1
                results['processing_times'].append(processing_time)
                results['confidences'].append(result['confidence'])
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                results['failed'] += 1
        
        return results
    
    def test_api_endpoint(self, test_files: List[Dict]) -> Dict:
        """Test the API endpoint with test files"""
        print("🌐 Testing API Endpoint...")
        
        results = {
            'total_files': len(test_files),
            'successful': 0,
            'failed': 0,
            'distress_accuracy': {'correct': 0, 'total': 0},
            'response_times': [],
            'api_errors': []
        }
        
        for i, test_file in enumerate(test_files):
            try:
                print(f"  Testing {i+1}/{len(test_files)}: {test_file['file']}")
                
                start_time = time.time()
                
                # Upload file to API
                with open(test_file['path'], 'rb') as f:
                    files = {"file": (test_file['file'], f, "audio/wav")}
                    response = requests.post("http://127.0.0.1:8000/voice-check", files=files, timeout=30)
                
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    results['successful'] += 1
                    results['response_times'].append(response_time)
                    
                    # Check distress detection accuracy if we have expected results
                    if 'expected_distress' in test_file:
                        expected_distress = test_file['expected_distress']
                        detected_distress = data.get('distress', False)
                        
                        results['distress_accuracy']['total'] += 1
                        if expected_distress == detected_distress:
                            results['distress_accuracy']['correct'] += 1
                else:
                    results['failed'] += 1
                    results['api_errors'].append(f"HTTP {response.status_code}: {response.text}")
                    
            except Exception as e:
                print(f"    ❌ Error: {e}")
                results['failed'] += 1
                results['api_errors'].append(str(e))
        
        return results
    
    def test_multi_modal_analysis(self, test_cases: List[Dict]) -> Dict:
        """Test the multi-modal analysis pipeline"""
        print("🧠 Testing Multi-Modal Analysis...")
        
        results = {
            'total_cases': len(test_cases),
            'successful': 0,
            'failed': 0,
            'distress_accuracy': {'correct': 0, 'total': 0},
            'processing_times': []
        }
        
        for i, test_case in enumerate(test_cases):
            try:
                print(f"  Analyzing {i+1}/{len(test_cases)}: {test_case.get('description', 'Unknown')}")
                
                start_time = time.time()
                
                # Simulate multi-modal analysis
                transcript = test_case.get('transcript', '')
                voice_features = test_case.get('voice_features', {})
                emotion_scores = test_case.get('emotion_scores', {})
                
                result = analyze_multi_modal_distress(transcript, voice_features, emotion_scores)
                
                processing_time = time.time() - start_time
                results['successful'] += 1
                results['processing_times'].append(processing_time)
                
                # Check accuracy if we have expected results
                if 'expected_distress' in test_case:
                    expected_distress = test_case['expected_distress']
                    detected_distress = result.get('distress_level', 'LOW') in ['MEDIUM', 'HIGH', 'CRITICAL']
                    
                    results['distress_accuracy']['total'] += 1
                    if expected_distress == detected_distress:
                        results['distress_accuracy']['correct'] += 1
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                results['failed'] += 1
        
        return results
    
    def generate_test_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate test data for evaluation"""
        print("📊 Generating Test Data...")
        
        # Audio test files (if available)
        audio_files = []
        datasets_dir = Path("datasets")
        
        # Check for RAVDESS dataset
        ravdess_dir = datasets_dir / "ravdess"
        if ravdess_dir.exists():
            for actor_dir in ravdess_dir.iterdir():
                if actor_dir.is_dir() and actor_dir.name.startswith('Actor_'):
                    for audio_file in list(actor_dir.glob("*.wav"))[:2]:  # Limit to 2 per actor
                        # Decode RAVDESS filename
                        filename = audio_file.name
                        parts = filename.replace('.wav', '').split('-')
                        if len(parts) >= 7:
                            emotion_code = parts[2]
                            emotion_map = {
                                '01': 'neutral', '02': 'calm', '03': 'happy',
                                '04': 'sad', '05': 'angry', '06': 'fearful',
                                '07': 'disgust', '08': 'surprised'
                            }
                            expected_emotion = emotion_map.get(emotion_code, 'unknown')
                            
                            audio_files.append({
                                'file': filename,
                                'path': str(audio_file),
                                'expected_emotion': expected_emotion,
                                'expected_distress': expected_emotion in ['angry', 'fearful', 'sad']
                            })
        
        # API test cases
        api_test_cases = [
            {
                'file': 'test_distress.wav',
                'path': 'test_distress.wav',
                'expected_distress': True,
                'description': 'Distress call test'
            },
            {
                'file': 'test_safe.wav', 
                'path': 'test_safe.wav',
                'expected_distress': False,
                'description': 'Safe conversation test'
            }
        ]
        
        # Multi-modal test cases
        multimodal_test_cases = [
            {
                'transcript': 'Help me, I need emergency assistance!',
                'voice_features': {'pitch_mean': 200, 'rms_mean': 0.2, 'tempo': 140},
                'emotion_scores': {'fear': 0.8, 'anger': 0.6, 'sadness': 0.2},
                'expected_distress': True,
                'description': 'High distress scenario'
            },
            {
                'transcript': 'Hello, how are you today?',
                'voice_features': {'pitch_mean': 150, 'rms_mean': 0.1, 'tempo': 100},
                'emotion_scores': {'happiness': 0.7, 'neutral': 0.3, 'fear': 0.0},
                'expected_distress': False,
                'description': 'Normal conversation'
            },
            {
                'transcript': 'I am feeling scared and worried',
                'voice_features': {'pitch_mean': 180, 'rms_mean': 0.15, 'tempo': 120},
                'emotion_scores': {'fear': 0.6, 'sadness': 0.4, 'neutral': 0.2},
                'expected_distress': True,
                'description': 'Moderate distress scenario'
            }
        ]
        
        return audio_files, api_test_cases, multimodal_test_cases
    
    def calculate_metrics(self, results: Dict) -> Dict:
        """Calculate comprehensive metrics from results"""
        metrics = {}
        
        # Overall success rate
        total_key = 'total_files' if 'total_files' in results else 'total_cases'
        if results[total_key] > 0:
            metrics['success_rate'] = results['successful'] / results[total_key]
            metrics['failure_rate'] = results['failed'] / results[total_key]
        
        # Processing performance
        if results.get('processing_times'):
            times = results['processing_times']
            metrics['avg_processing_time'] = np.mean(times)
            metrics['min_processing_time'] = np.min(times)
            metrics['max_processing_time'] = np.max(times)
            metrics['std_processing_time'] = np.std(times)
        
        # Accuracy metrics
        if 'emotion_accuracy' in results:
            emotion_acc = results['emotion_accuracy']
            total_correct = sum(stats['correct'] for stats in emotion_acc.values())
            total_predictions = sum(stats['total'] for stats in emotion_acc.values())
            
            if total_predictions > 0:
                metrics['overall_emotion_accuracy'] = total_correct / total_predictions
                
                # Per-emotion accuracy
                metrics['per_emotion_accuracy'] = {}
                for emotion, stats in emotion_acc.items():
                    if stats['total'] > 0:
                        metrics['per_emotion_accuracy'][emotion] = stats['correct'] / stats['total']
        
        if 'distress_accuracy' in results:
            distress_acc = results['distress_accuracy']
            if distress_acc['total'] > 0:
                metrics['distress_accuracy'] = distress_acc['correct'] / distress_acc['total']
        
        # Confidence metrics
        if results.get('confidences'):
            confidences = results['confidences']
            metrics['avg_confidence'] = np.mean(confidences)
            metrics['min_confidence'] = np.min(confidences)
            metrics['max_confidence'] = np.max(confidences)
        
        return metrics
    
    def generate_report(self, audio_results: Dict, api_results: Dict, multimodal_results: Dict):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE MODEL EVALUATION REPORT")
        print("="*60)
        
        # Calculate metrics for each component
        audio_metrics = self.calculate_metrics(audio_results)
        api_metrics = self.calculate_metrics(api_results)
        multimodal_metrics = self.calculate_metrics(multimodal_results)
        
        # Overall summary
        print(f"\n🎯 OVERALL SUMMARY")
        print(f"Audio Processor: {audio_results['successful']}/{audio_results['total_files']} successful")
        print(f"API Endpoint: {api_results['successful']}/{api_results['total_files']} successful")
        print(f"Multi-Modal: {multimodal_results['successful']}/{multimodal_results['total_cases']} successful")
        
        # Audio Processor Results
        print(f"\n🎵 AUDIO PROCESSOR RESULTS")
        if 'overall_emotion_accuracy' in audio_metrics:
            print(f"Overall Emotion Accuracy: {audio_metrics['overall_emotion_accuracy']:.2%}")
        
        if 'per_emotion_accuracy' in audio_metrics:
            print("Per-Emotion Accuracy:")
            for emotion, accuracy in audio_metrics['per_emotion_accuracy'].items():
                print(f"  {emotion}: {accuracy:.2%}")
        
        if 'avg_processing_time' in audio_metrics:
            print(f"Average Processing Time: {audio_metrics['avg_processing_time']:.2f}s")
        
        # API Results
        print(f"\n🌐 API ENDPOINT RESULTS")
        if 'distress_accuracy' in api_metrics:
            print(f"Distress Detection Accuracy: {api_metrics['distress_accuracy']:.2%}")
        
        if 'avg_processing_time' in api_metrics:
            print(f"Average Response Time: {api_metrics['avg_processing_time']:.2f}s")
        
        # Multi-Modal Results
        print(f"\n🧠 MULTI-MODAL ANALYSIS RESULTS")
        if 'distress_accuracy' in multimodal_metrics:
            print(f"Distress Detection Accuracy: {multimodal_metrics['distress_accuracy']:.2%}")
        
        if 'avg_processing_time' in multimodal_metrics:
            print(f"Average Processing Time: {multimodal_metrics['avg_processing_time']:.2f}s")
        
        # Performance Comparison with Targets
        print(f"\n🎯 PERFORMANCE vs TARGETS")
        targets = {
            'distress_accuracy': 0.95,  # 95% target
            'emotion_accuracy': 0.90,   # 90% target
            'processing_time': 0.1      # 100ms target
        }
        
        for metric, target in targets.items():
            if metric in audio_metrics:
                current = audio_metrics[metric]
                status = "✅" if current >= target else "❌"
                print(f"{status} {metric}: {current:.2%} (target: {target:.2%})")
            elif metric in api_metrics:
                current = api_metrics[metric]
                status = "✅" if current >= target else "❌"
                print(f"{status} {metric}: {current:.2%} (target: {target:.2%})")
        
        # Save detailed results
        detailed_results = {
            'audio_processor': {
                'results': audio_results,
                'metrics': audio_metrics
            },
            'api_endpoint': {
                'results': api_results,
                'metrics': api_metrics
            },
            'multimodal_analysis': {
                'results': multimodal_results,
                'metrics': multimodal_metrics
            },
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: evaluation_results.json")
        
        return detailed_results

def main():
    """Main evaluation function"""
    print("🚀 Starting Comprehensive Model Evaluation")
    print("="*50)
    
    evaluator = ModelEvaluator()
    
    # Generate test data
    audio_files, api_cases, multimodal_cases = evaluator.generate_test_data()
    
    print(f"📊 Test Data Generated:")
    print(f"  Audio files: {len(audio_files)}")
    print(f"  API test cases: {len(api_cases)}")
    print(f"  Multi-modal cases: {len(multimodal_cases)}")
    
    # Run evaluations
    audio_results = evaluator.test_audio_processor(audio_files)
    api_results = evaluator.test_api_endpoint(api_cases)
    multimodal_results = evaluator.test_multi_modal_analysis(multimodal_cases)
    
    # Generate comprehensive report
    detailed_results = evaluator.generate_report(audio_results, api_results, multimodal_results)
    
    print("\n🎉 Evaluation Complete!")
    return detailed_results

if __name__ == "__main__":
    main()
