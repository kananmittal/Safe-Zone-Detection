#!/usr/bin/env python3
"""
Incremental Emotion Training on Real Datasets (RAVDESS -> CREMA -> IEMOCAP -> TESS)
- Extracts features via TorchAudioProcessor
- Trains scikit-learn models (RandomForest best-by-default)
- Saves a model after each dataset stage and a combined model

Usage:
  python train_emotion_datasets.py --limit-per-class 200 --save-dir models/real
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import numpy as np
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Add src to path
sys.path.append('src')
from src.audio_processor import TorchAudioProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMOTIONS = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']
EMOTION_TO_IDX = {e:i for i,e in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {i:e for e,i in EMOTION_TO_IDX.items()}

# Dataset helpers

def list_ravdess(root: Path, limit_per_class: int = 0) -> List[Tuple[str,str]]:
    """RAVDESS: filenames encode emotion code in position 3 (01..08)."""
    emotion_map = {
        '01': 'neutral','02':'calm','03':'happy','04':'sad',
        '05':'angry','06':'fearful','07':'disgust','08':'surprised'
    }
    pairs = []
    per_class = defaultdict(int)
    for actor_dir in sorted(root.glob('Actor_*')):
        for wav in actor_dir.glob('*.wav'):
            parts = wav.stem.split('-')
            if len(parts) >= 3:
                code = parts[2]
                label = emotion_map.get(code)
                if not label:
                    continue
                if limit_per_class and per_class[label] >= limit_per_class:
                    continue
                pairs.append((str(wav), label))
                per_class[label] += 1
    return pairs

def list_crema(root: Path, limit_per_class: int = 0) -> List[Tuple[str,str]]:
    """CREMA: filenames contain label token like _ANG_, _SAD_, _HAP_, _FEA_, _DIS_, _NEU_"""
    code_map = {
        'ANG':'angry','SAD':'sad','HAP':'happy','FEA':'fearful','DIS':'disgust','NEU':'neutral'
    }
    pairs = []
    per_class = defaultdict(int)
    for wav in root.glob('*.wav'):
        stem = wav.stem
        label = None
        for code, name in code_map.items():
            token = f"_{code}_"
            if token in stem:
                label = name
                break
        if not label:
            continue
        if limit_per_class and per_class[label] >= limit_per_class:
            continue
        pairs.append((str(wav), label))
        per_class[label] += 1
    return pairs

def list_tess(root: Path, limit_per_class: int = 0) -> List[Tuple[str,str]]:
    """TESS: emotion is encoded in directory names (OAF_angry, YAF_happy, etc.)."""
    dir_map = {
        'angry':'angry','disgust':'disgust','fear':'fearful','fearful':'fearful',
        'happy':'happy','neutral':'neutral','sad':'sad','pleasant_surprise':'surprised','pleasant_surprised':'surprised','Pleasant_surprise':'surprised','Pleasant_surprised':'surprised'
    }
    pairs = []
    per_class = defaultdict(int)
    for d in root.iterdir():
        if not d.is_dir():
            continue
        lower = d.name.lower()
        label = None
        for key, mapped in dir_map.items():
            if key in lower:
                label = mapped
                break
        if not label:
            continue
        for wav in d.glob('*.wav'):
            if limit_per_class and per_class[label] >= limit_per_class:
                continue
            pairs.append((str(wav), label))
            per_class[label] += 1
    return pairs

def list_iemocap(root: Path, limit_per_class: int = 0) -> List[Tuple[str,str]]:
    """IEMOCAP: heavy; use processed JSON if available for labels + relative paths."""
    processed = Path('datasets/processed/iemocap_train.json')
    if not processed.exists():
        processed = Path('datasets/processed/iemocap_combined.json')
    pairs = []
    per_class = defaultdict(int)
    if processed.exists():
        try:
            import json
            with open(processed, 'r') as f:
                data = json.load(f)
            for item in data:
                # expected fields: audio_path or path, and emotion
                audio_path = item.get('audio_path') or item.get('path')
                label = item.get('emotion') or item.get('label')
                if not audio_path or not label:
                    continue
                label = label.lower()
                # map IEMOCAP short codes
                map_short = {'hap':'happy','ang':'angry','sad':'sad','fea':'fearful','dis':'disgust','neu':'neutral','exc':'happy'}
                label = map_short.get(label, label)
                if label not in EMOTION_TO_IDX:
                    continue
                full = Path(audio_path)
                if not full.is_file():
                    # try relative to IEMOCAP root
                    cand = Path('datasets/IEMOCAP')/audio_path
                    if cand.is_file():
                        full = cand
                    else:
                        continue
                if limit_per_class and per_class[label] >= limit_per_class:
                    continue
                pairs.append((str(full), label))
                per_class[label] += 1
        except Exception as e:
            logger.warning(f"IEMOCAP processed parse failed: {e}")
    else:
        logger.info("IEMOCAP processed JSON not found; skipping.")
    return pairs

# Feature extraction

def extract_feature_vector(features: Dict) -> List[float]:
    keys = [
        'mfcc_mean','mfcc_std','mfcc_max','mfcc_min',
        'pitch_mean','pitch_std','pitch_max','pitch_min',
        'rms_mean','rms_std','rms_max','rms_min',
        'spectral_centroid_mean','spectral_centroid_std',
        'spectral_rolloff_mean','spectral_rolloff_std',
        'zcr_mean','zcr_std','tempo','speech_rate'
    ]
    vec: List[float] = []
    for k in keys:
        v = features.get(k)
        if v is None:
            vec.extend([0.0, 0.0])
            continue
        if isinstance(v, (list, np.ndarray)):
            arr = np.asarray(v)
            vec.extend([float(np.mean(arr)), float(np.std(arr))])
        else:
            vec.append(float(v))
            vec.append(0.0)
    return vec


def build_dataset(pairs: List[Tuple[str,str]], processor: TorchAudioProcessor) -> Tuple[np.ndarray, np.ndarray]:
    X: List[List[float]] = []
    y: List[int] = []
    for i,(path,label) in enumerate(pairs):
        try:
            res = processor.process_audio(path)
            feats = res.get('features')
            if not feats:
                continue
            X.append(extract_feature_vector(feats))
            y.append(EMOTION_TO_IDX[label])
            if (i+1) % 50 == 0:
                logger.info(f"Extracted features: {i+1}/{len(pairs)}")
        except Exception as e:
            logger.warning(f"Feature extraction failed for {path}: {e}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train_and_save(X: np.ndarray, y: np.ndarray, save_dir: Path, name: str) -> Dict:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=EMOTIONS, zero_division=0)

    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_dir / f"emotion_model_{name}.pkl")
    joblib.dump(scaler, save_dir / f"emotion_scaler_{name}.pkl")

    with open(save_dir / f"training_report_{name}.txt", 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    logger.info(f"Saved model and scaler: {name} (acc={acc:.3f})")
    return {"accuracy": acc, "report": report}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit-per-class', type=int, default=0, help='Optional cap per class per dataset to speed up')
    parser.add_argument('--save-dir', type=str, default='models/real', help='Directory to save models')
    parser.add_argument('--order', type=str, default='ravdess,crema,iemocap,tess', help='Dataset training order')
    args = parser.parse_args()

    base = Path('datasets')
    order = [o.strip() for o in args.order.split(',') if o.strip()]
    present = { 'ravdess': (base/'ravdess').exists(),
                'crema': (base/'crema').exists(),
                'iemocap': (base/'IEMOCAP').exists(),
                'tess': (base/'tess').exists() }

    logger.info(f"Datasets present: {present}")

    processor = TorchAudioProcessor()

    cumulative_pairs: List[Tuple[str,str]] = []
    results: Dict[str,Dict] = {}
    save_dir = Path(args.save_dir)

    for ds in order:
        if ds == 'ravdess' and present['ravdess']:
            pairs = list_ravdess(base/'ravdess', args.limit_per_class)
        elif ds == 'crema' and present['crema']:
            pairs = list_crema(base/'crema', args.limit_per_class)
        elif ds == 'iemocap' and present['iemocap']:
            pairs = list_iemocap(base/'IEMOCAP', args.limit_per_class)
        elif ds == 'tess' and present['tess']:
            pairs = list_tess(base/'tess', args.limit_per_class)
        else:
            logger.info(f"Skipping {ds}: not present")
            continue

        logger.info(f"{ds.upper()}: found {len(pairs)} files")
        cumulative_pairs.extend(pairs)

        # Build dataset for current cumulative set
        X, y = build_dataset(cumulative_pairs, processor)
        logger.info(f"Cumulative samples: {len(X)}")
        if len(X) < 50:
            logger.warning("Too few samples to train reliably; continuing")
            continue
        res = train_and_save(X, y, save_dir, name=f"upto_{ds}")
        results[ds] = {"num_samples": len(X), **res}

    # Save summary
    with open(save_dir / 'incremental_training_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("Incremental training complete. Summary saved.")

if __name__ == '__main__':
    main()
