#!/usr/bin/env python3
"""
Debug script to analyze feature values from RAVDESS dataset
"""

import os
import numpy as np
from src.audio_processor import TorchAudioProcessor

def analyze_features_for_emotion(emotion_code, processor, data_dir="Data", num_files=3):
    """Analyze feature values for a specific emotion"""
    print(f"\n🔍 Analyzing features for emotion code: {emotion_code}")
    
    feature_values = {
        'pitch_mean': [],
        'rms_mean': [],
        'spectral_centroid_mean': [],
        'zcr_mean': [],
        'tempo': [],
        'mfcc_std': []
    }
    
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
                if file_count >= num_files:
                    break
                    
                file_path = os.path.join(actor_path, filename)
                print(f"  Processing: {filename}")
                
                try:
                    result = processor.process_audio(file_path)
                    features = result.get('features', {})
                    
                    for key in feature_values.keys():
                        if key in features:
                            feature_values[key].append(features[key])
                    
                    file_count += 1
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    
        if file_count >= num_files:
            break
    
    # Print statistics
    print(f"\n📊 Feature Statistics for {num_files} files:")
    for key, values in feature_values.items():
        if values:
            print(f"  {key}:")
            print(f"    Mean: {np.mean(values):.4f}")
            print(f"    Std:  {np.std(values):.4f}")
            print(f"    Min:  {np.min(values):.4f}")
            print(f"    Max:  {np.max(values):.4f}")
            print(f"    Values: {[f'{v:.4f}' for v in values]}")
        else:
            print(f"  {key}: No data")

def main():
    """Main debug function"""
    print("🔍 TorchAudio Feature Analysis")
    print("=" * 40)
    
    # Initialize processor
    processor = TorchAudioProcessor()
    
    # Analyze each emotion
    emotions = {
        '01': 'neutral',
        '02': 'calm', 
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    for code, name in emotions.items():
        analyze_features_for_emotion(code, processor, num_files=2)

if __name__ == "__main__":
    main() 