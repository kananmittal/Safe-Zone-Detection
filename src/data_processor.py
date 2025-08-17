#!/usr/bin/env python3
"""
Dataset Processor for Voice Distress Detection Fine-tuning
Handles RAVDESS, CREMA-D, TESS, and IEMOCAP datasets
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetProcessor:
    def __init__(self, datasets_dir: str = "datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.processed_dir = self.datasets_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        
        # Emotion mappings for different datasets
        self.emotion_mappings = {
            'ravdess': {
                '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
                '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
            },
            'crema': {
                'HAP': 'happy', 'SAD': 'sad', 'ANG': 'angry', 
                'FEA': 'fearful', 'DIS': 'disgust', 'NEU': 'neutral'
            },
            'tess': {
                'happy': 'happy', 'sad': 'sad', 'angry': 'angry',
                'fearful': 'fearful', 'disgust': 'disgust', 'neutral': 'neutral',
                'surprised': 'surprised'
            },
            'iemocap': {
                'hap': 'happy', 'sad': 'sad', 'ang': 'angry',
                'fea': 'fearful', 'dis': 'disgust', 'neu': 'neutral',
                'exc': 'excited', 'fru': 'frustrated'
            }
        }
        
        # Distress classification
        self.distress_emotions = ['angry', 'fearful', 'disgust', 'sad']
        self.safe_emotions = ['happy', 'neutral', 'calm', 'excited']
        
    def process_ravdess_dataset(self) -> pd.DataFrame:
        """Process RAVDESS dataset"""
        logger.info("Processing RAVDESS dataset...")
        
        data = []
        ravdess_dir = self.datasets_dir / "ravdess"
        
        for audio_file in ravdess_dir.rglob("*.wav"):
            try:
                # Parse filename: 03-01-05-01-02-01-16.wav
                # Format: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
                parts = audio_file.stem.split('-')
                if len(parts) >= 6:
                    emotion_code = parts[2]
                    intensity = parts[3]
                    statement = parts[4]
                    actor = parts[6]
                    
                    emotion = self.emotion_mappings['ravdess'].get(emotion_code, 'unknown')
                    is_distress = emotion in self.distress_emotions
                    
                    data.append({
                        'file_path': str(audio_file),
                        'dataset': 'ravdess',
                        'emotion': emotion,
                        'emotion_code': emotion_code,
                        'intensity': intensity,
                        'statement': statement,
                        'actor': actor,
                        'is_distress': is_distress,
                        'distress_level': 'HIGH' if is_distress else 'LOW',
                        'safety_action': 'ALERT' if is_distress else 'NONE'
                    })
            except Exception as e:
                logger.warning(f"Error processing {audio_file}: {e}")
        
        df = pd.DataFrame(data)
        logger.info(f"RAVDESS: {len(df)} samples processed")
        return df
    
    def process_crema_dataset(self) -> pd.DataFrame:
        """Process CREMA-D dataset"""
        logger.info("Processing CREMA-D dataset...")
        
        data = []
        crema_dir = self.datasets_dir / "crema"
        
        # Look for audio files in CREMA-D structure
        for audio_file in crema_dir.rglob("*.wav"):
            try:
                # Parse filename: 1001_DFA_ANG_XX.wav
                # Format: actor_emotion_intensity.wav
                filename = audio_file.stem
                parts = filename.split('_')
                
                if len(parts) >= 3:
                    actor = parts[0]
                    emotion_code = parts[2]
                    intensity = parts[3] if len(parts) > 3 else 'XX'
                    
                    emotion = self.emotion_mappings['crema'].get(emotion_code, 'unknown')
                    is_distress = emotion in self.distress_emotions
                    
                    data.append({
                        'file_path': str(audio_file),
                        'dataset': 'crema',
                        'emotion': emotion,
                        'emotion_code': emotion_code,
                        'intensity': intensity,
                        'actor': actor,
                        'is_distress': is_distress,
                        'distress_level': 'HIGH' if is_distress else 'LOW',
                        'safety_action': 'ALERT' if is_distress else 'NONE'
                    })
            except Exception as e:
                logger.warning(f"Error processing {audio_file}: {e}")
        
        df = pd.DataFrame(data)
        logger.info(f"CREMA-D: {len(df)} samples processed")
        return df
    
    def process_tess_dataset(self) -> pd.DataFrame:
        """Process TESS dataset"""
        logger.info("Processing TESS dataset...")
        
        data = []
        tess_dir = self.datasets_dir / "tess"
        
        # TESS structure: emotion_actor_phrase.wav
        for audio_file in tess_dir.rglob("*.wav"):
            try:
                filename = audio_file.stem
                parts = filename.split('_')
                
                if len(parts) >= 2:
                    emotion = parts[0].lower()
                    actor = parts[1] if len(parts) > 1 else 'unknown'
                    
                    # Map emotion to standard format
                    emotion = self.emotion_mappings['tess'].get(emotion, emotion)
                    is_distress = emotion in self.distress_emotions
                    
                    data.append({
                        'file_path': str(audio_file),
                        'dataset': 'tess',
                        'emotion': emotion,
                        'emotion_code': emotion.upper(),
                        'intensity': 'XX',
                        'actor': actor,
                        'is_distress': is_distress,
                        'distress_level': 'HIGH' if is_distress else 'LOW',
                        'safety_action': 'ALERT' if is_distress else 'NONE'
                    })
            except Exception as e:
                logger.warning(f"Error processing {audio_file}: {e}")
        
        df = pd.DataFrame(data)
        logger.info(f"TESS: {len(df)} samples processed")
        return df
    
    def process_iemocap_dataset(self) -> pd.DataFrame:
        """Process IEMOCAP dataset (if available)"""
        logger.info("Processing IEMOCAP dataset...")
        
        data = []
        iemocap_dir = self.datasets_dir / "iemocap"
        
        # Check if IEMOCAP is available
        if not iemocap_dir.exists():
            logger.warning("IEMOCAP dataset not found. Please download manually from https://sail.usc.edu/iemocap/")
            return pd.DataFrame()
        
        # Look for evaluation files
        for eval_file in iemocap_dir.rglob("*_eval.txt"):
            try:
                with open(eval_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            filename = parts[0]
                            emotion = parts[2].lower()
                            
                            # Find corresponding audio file
                            audio_file = iemocap_dir / f"{filename}.wav"
                            if audio_file.exists():
                                emotion = self.emotion_mappings['iemocap'].get(emotion, emotion)
                                is_distress = emotion in self.distress_emotions
                                
                                data.append({
                                    'file_path': str(audio_file),
                                    'dataset': 'iemocap',
                                    'emotion': emotion,
                                    'emotion_code': emotion.upper(),
                                    'intensity': 'XX',
                                    'actor': 'unknown',
                                    'is_distress': is_distress,
                                    'distress_level': 'HIGH' if is_distress else 'LOW',
                                    'safety_action': 'ALERT' if is_distress else 'NONE'
                                })
            except Exception as e:
                logger.warning(f"Error processing {eval_file}: {e}")
        
        df = pd.DataFrame(data)
        logger.info(f"IEMOCAP: {len(df)} samples processed")
        return df
    
    def create_training_dataset(self) -> pd.DataFrame:
        """Create combined training dataset"""
        logger.info("Creating combined training dataset...")
        
        # Process all available datasets
        datasets = []
        
        # RAVDESS
        ravdess_df = self.process_ravdess_dataset()
        if not ravdess_df.empty:
            datasets.append(ravdess_df)
        
        # CREMA-D
        crema_df = self.process_crema_dataset()
        if not crema_df.empty:
            datasets.append(crema_df)
        
        # TESS
        tess_df = self.process_tess_dataset()
        if not tess_df.empty:
            datasets.append(tess_df)
        
        # IEMOCAP
        iemocap_df = self.process_iemocap_dataset()
        if not iemocap_df.empty:
            datasets.append(iemocap_df)
        
        # Combine all datasets
        if datasets:
            combined_df = pd.concat(datasets, ignore_index=True)
            
            # Add training metadata
            combined_df['sample_id'] = range(len(combined_df))
            combined_df['split'] = 'train'  # Will be split later
            
            # Save combined dataset
            output_file = self.processed_dir / "combined_dataset.csv"
            combined_df.to_csv(output_file, index=False)
            
            # Create dataset statistics
            stats = self.create_dataset_statistics(combined_df)
            
            logger.info(f"Combined dataset created: {len(combined_df)} samples")
            logger.info(f"Dataset statistics:\n{stats}")
            
            return combined_df
        else:
            logger.error("No datasets found to process!")
            return pd.DataFrame()
    
    def create_dataset_statistics(self, df: pd.DataFrame) -> str:
        """Create dataset statistics"""
        stats = []
        stats.append("=== DATASET STATISTICS ===")
        stats.append(f"Total samples: {len(df)}")
        stats.append(f"Distress samples: {len(df[df['is_distress']])}")
        stats.append(f"Safe samples: {len(df[~df['is_distress']])}")
        stats.append(f"Distress ratio: {len(df[df['is_distress']])/len(df)*100:.1f}%")
        
        stats.append("\n=== BY DATASET ===")
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            distress_count = len(dataset_df[dataset_df['is_distress']])
            total_count = len(dataset_df)
            stats.append(f"{dataset.upper()}: {total_count} samples ({distress_count} distress)")
        
        stats.append("\n=== BY EMOTION ===")
        emotion_counts = df['emotion'].value_counts()
        for emotion, count in emotion_counts.items():
            distress_count = len(df[(df['emotion'] == emotion) & (df['is_distress'])])
            stats.append(f"{emotion}: {count} samples ({distress_count} distress)")
        
        return "\n".join(stats)
    
    def create_fine_tuning_data(self, df: pd.DataFrame) -> Dict:
        """Create data for Llama 3 fine-tuning"""
        logger.info("Creating fine-tuning data...")
        
        training_data = []
        
        for _, row in df.iterrows():
            # Create training prompt
            prompt = f"""Analyze this voice distress detection case:

TRANSCRIPT: "[Audio file: {row['file_path']}]"

VOICE FEATURES: [Extracted from audio analysis]

EMOTION ANALYSIS: {row['emotion']}: 80.0%

Based on the transcript, voice characteristics, and emotional indicators, determine if this person is in distress or danger.

Respond with:
DISTRESS_LEVEL: (LOW/MEDIUM/HIGH/CRITICAL)
CONFIDENCE: (0-100%)
REASONING: Brief explanation
SAFETY_ACTION: (NONE/MONITOR/ALERT/EMERGENCY)

Analysis:"""

            # Create expected response
            if row['is_distress']:
                response = f"""DISTRESS_LEVEL: HIGH
CONFIDENCE: 85%
REASONING: {row['emotion']} emotion detected with high intensity
SAFETY_ACTION: {row['safety_action']}"""
            else:
                response = f"""DISTRESS_LEVEL: LOW
CONFIDENCE: 90%
REASONING: {row['emotion']} emotion detected, no distress indicators
SAFETY_ACTION: {row['safety_action']}"""

            training_data.append({
                'prompt': prompt,
                'response': response,
                'file_path': row['file_path'],
                'emotion': row['emotion'],
                'is_distress': row['is_distress']
            })
        
        # Save training data
        training_file = self.processed_dir / "fine_tuning_data.json"
        with open(training_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        logger.info(f"Fine-tuning data created: {len(training_data)} samples")
        return training_data

def main():
    """Main function to process all datasets"""
    processor = DatasetProcessor()
    
    # Create combined dataset
    combined_df = processor.create_training_dataset()
    
    if not combined_df.empty:
        # Create fine-tuning data
        training_data = processor.create_fine_tuning_data(combined_df)
        
        print("\n🎯 Dataset Processing Complete!")
        print(f"📊 Total samples: {len(combined_df)}")
        print(f"🚨 Distress samples: {len(combined_df[combined_df['is_distress']])}")
        print(f"✅ Safe samples: {len(combined_df[~combined_df['is_distress']])}")
        print(f"📁 Output files:")
        print(f"   - Combined dataset: datasets/processed/combined_dataset.csv")
        print(f"   - Fine-tuning data: datasets/processed/fine_tuning_data.json")
        
        print("\n🚀 Ready for fine-tuning!")
    else:
        print("❌ No datasets processed. Please check dataset availability.")

if __name__ == "__main__":
    main() 