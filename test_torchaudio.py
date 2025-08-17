#!/usr/bin/env python3
"""
Test script for TorchAudio Audio Processor with RAVDESS dataset
"""

import os
import sys
import json
from pathlib import Path
from src.audio_processor import TorchAudioProcessor
import time

# RAVDESS emotion mapping
EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm', 
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def decode_filename(filename):
    """Decode RAVDESS filename to extract emotion and metadata"""
    parts = filename.replace('.wav', '').split('-')
    if len(parts) >= 7:
        emotion_code = parts[2]
        intensity = parts[3]
        statement = parts[4]
        repetition = parts[5]
        actor = parts[6]
        
        emotion = EMOTION_MAP.get(emotion_code, 'unknown')
        
        return {
            'emotion': emotion,
            'intensity': intensity,
            'statement': statement,
            'repetition': repetition,
            'actor': actor,
            'emotion_code': emotion_code
        }
    return None

def test_single_file(audio_path, processor):
    """Test a single audio file"""
    try:
        print(f"\n🎵 Testing: {os.path.basename(audio_path)}")
        
        # Decode filename
        metadata = decode_filename(os.path.basename(audio_path))
        if metadata:
            print(f"   Expected emotion: {metadata['emotion']} (intensity: {metadata['intensity']})")
        
        # Process audio
        start_time = time.time()
        result = processor.process_audio(audio_path)
        processing_time = time.time() - start_time
        
        print(f"   Detected emotion: {result['emotion']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Processing time: {processing_time:.2f}s")
        print(f"   Device used: {result['device_used']}")
        
        # Show top emotion scores
        emotion_scores = result.get('emotion_scores', {})
        top_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"   Top emotions: {', '.join([f'{e[0]}: {e[1]:.2f}' for e in top_emotions])}")
        
        # Check if prediction matches expected
        if metadata and metadata['emotion'] in emotion_scores:
            expected_score = emotion_scores[metadata['emotion']]
            print(f"   Expected emotion score: {expected_score:.2f}")
        
        return {
            'file': os.path.basename(audio_path),
            'expected_emotion': metadata['emotion'] if metadata else 'unknown',
            'detected_emotion': result['emotion'],
            'confidence': result['confidence'],
            'processing_time': processing_time,
            'emotion_scores': emotion_scores,
            'device_used': result['device_used']
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {
            'file': os.path.basename(audio_path),
            'error': str(e)
        }

def test_emotion_category(emotion_code, processor, data_dir="Data", max_files=5):
    """Test multiple files of the same emotion"""
    print(f"\n🎭 Testing emotion category: {EMOTION_MAP.get(emotion_code, 'unknown')} ({emotion_code})")
    
    results = []
    file_count = 0
    
    # Search through all actor folders
    for actor_dir in sorted(os.listdir(data_dir)):
        if not actor_dir.startswith('Actor_'):
            continue
            
        actor_path = os.path.join(data_dir, actor_dir)
        if not os.path.isdir(actor_path):
            continue
            
        for filename in os.listdir(actor_path):
            if filename.endswith('.wav') and filename.split('-')[2] == emotion_code:
                if file_count >= max_files:
                    break
                    
                file_path = os.path.join(actor_path, filename)
                result = test_single_file(file_path, processor)
                results.append(result)
                file_count += 1
                
        if file_count >= max_files:
            break
    
    return results

def test_random_samples(processor, data_dir="Data", num_samples=10):
    """Test random samples from different emotions"""
    print(f"\n🎲 Testing {num_samples} random samples")
    
    all_files = []
    
    # Collect all audio files
    for actor_dir in os.listdir(data_dir):
        if not actor_dir.startswith('Actor_'):
            continue
            
        actor_path = os.path.join(data_dir, actor_dir)
        if not os.path.isdir(actor_path):
            continue
            
        for filename in os.listdir(actor_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(actor_path, filename)
                all_files.append(file_path)
    
    # Select random samples
    import random
    random.shuffle(all_files)
    selected_files = all_files[:num_samples]
    
    results = []
    for file_path in selected_files:
        result = test_single_file(file_path, processor)
        results.append(result)
    
    return results

def analyze_results(results):
    """Analyze test results"""
    print(f"\n📊 Analysis Results:")
    print(f"Total files tested: {len(results)}")
    
    # Filter out errors
    successful_results = [r for r in results if 'error' not in r]
    error_count = len(results) - len(successful_results)
    
    if error_count > 0:
        print(f"Errors: {error_count}")
    
    if not successful_results:
        print("No successful results to analyze")
        return
    
    # Accuracy analysis
    correct_predictions = 0
    total_predictions = 0
    
    emotion_accuracy = {}
    processing_times = []
    confidences = []
    
    for result in successful_results:
        if 'expected_emotion' in result and 'detected_emotion' in result:
            expected = result['expected_emotion']
            detected = result['detected_emotion']
            
            if expected not in emotion_accuracy:
                emotion_accuracy[expected] = {'correct': 0, 'total': 0}
            
            emotion_accuracy[expected]['total'] += 1
            total_predictions += 1
            
            if expected == detected:
                emotion_accuracy[expected]['correct'] += 1
                correct_predictions += 1
        
        if 'processing_time' in result:
            processing_times.append(result['processing_time'])
        
        if 'confidence' in result:
            confidences.append(result['confidence'])
    
    # Overall accuracy
    if total_predictions > 0:
        overall_accuracy = correct_predictions / total_predictions
        print(f"Overall accuracy: {overall_accuracy:.2%} ({correct_predictions}/{total_predictions})")
    
    # Per-emotion accuracy
    print(f"\nPer-emotion accuracy:")
    for emotion, stats in emotion_accuracy.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            print(f"  {emotion}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")
    
    # Performance metrics
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        print(f"\nPerformance:")
        print(f"  Average processing time: {avg_time:.2f}s")
        print(f"  Fastest: {min(processing_times):.2f}s")
        print(f"  Slowest: {max(processing_times):.2f}s")
    
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        print(f"  Average confidence: {avg_confidence:.2f}")
        print(f"  Highest confidence: {max(confidences):.2f}")
        print(f"  Lowest confidence: {min(confidences):.2f}")

def main():
    """Main test function"""
    print("🎵 TorchAudio Audio Processor Test Suite")
    print("=" * 50)
    
    # Initialize processor
    print("Initializing TorchAudio processor...")
    processor = TorchAudioProcessor()
    print(f"✅ Processor initialized on {processor.device}")
    
    # Check if Data directory exists
    data_dir = "Data"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found!")
        return
    
    # Test options
    print(f"\n📁 Found {data_dir} directory with RAVDESS dataset")
    print("\nChoose test option:")
    print("1. Test specific emotion (e.g., '03' for happy)")
    print("2. Test random samples")
    print("3. Test single file")
    print("4. Quick test (5 files per emotion)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        emotion_code = input("Enter emotion code (01-08): ").strip()
        if emotion_code in EMOTION_MAP:
            results = test_emotion_category(emotion_code, processor, data_dir)
            analyze_results(results)
        else:
            print("Invalid emotion code!")
    
    elif choice == "2":
        num_samples = input("Number of samples (default 10): ").strip()
        num_samples = int(num_samples) if num_samples.isdigit() else 10
        results = test_random_samples(processor, data_dir, num_samples)
        analyze_results(results)
    
    elif choice == "3":
        # List available files
        print("\nAvailable files:")
        file_count = 0
        for actor_dir in sorted(os.listdir(data_dir)):
            if actor_dir.startswith('Actor_'):
                actor_path = os.path.join(data_dir, actor_dir)
                if os.path.isdir(actor_path):
                    files = os.listdir(actor_path)
                    for filename in files[:3]:  # Show first 3 files per actor
                        metadata = decode_filename(filename)
                        if metadata:
                            print(f"  {filename} -> {metadata['emotion']}")
                            file_count += 1
                        if file_count >= 20:  # Limit display
                            break
                    if file_count >= 20:
                        break
        
        filename = input("\nEnter filename to test: ").strip()
        if filename:
            # Find the file
            file_path = None
            for actor_dir in os.listdir(data_dir):
                if actor_dir.startswith('Actor_'):
                    actor_path = os.path.join(data_dir, actor_dir)
                    if os.path.isdir(actor_path):
                        potential_path = os.path.join(actor_path, filename)
                        if os.path.exists(potential_path):
                            file_path = potential_path
                            break
            
            if file_path:
                result = test_single_file(file_path, processor)
                analyze_results([result])
            else:
                print("File not found!")
    
    elif choice == "4":
        print("\nRunning quick test on all emotions...")
        all_results = []
        for emotion_code in EMOTION_MAP.keys():
            results = test_emotion_category(emotion_code, processor, data_dir, max_files=2)
            all_results.extend(results)
        analyze_results(all_results)
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main() 