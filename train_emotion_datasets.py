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
from sklearn.svm import SVC
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
        'mfcc_delta_mean','mfcc_delta_std','mfcc_delta2_mean','mfcc_delta2_std',
        'pitch_mean','pitch_std','pitch_max','pitch_min',
        'rms_mean','rms_std','rms_max','rms_min',
        'spectral_centroid_mean','spectral_centroid_std',
        'spectral_rolloff_mean','spectral_rolloff_std',
        'spectral_bw_mean','spectral_bw_std',
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


def build_dataset(pairs: List[Tuple[str,str]], processor: TorchAudioProcessor, augment: bool = False, augment_per_sample: int = 1) -> Tuple[np.ndarray, np.ndarray]:
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

            # Optional augmentation path
            if augment and augment_per_sample > 0:
                try:
                    waveform = processor.load_audio(path)
                    aug_waves = processor.augment_waveforms(waveform)
                    # Limit number of augments per sample
                    for aug_w in aug_waves[:augment_per_sample]:
                        aug_feats = processor.extract_features_from_waveform(aug_w)
                        X.append(extract_feature_vector(aug_feats))
                        y.append(EMOTION_TO_IDX[label])
                except Exception as ae:
                    logger.warning(f"Augmentation failed for {path}: {ae}")
            if (i+1) % 50 == 0:
                logger.info(f"Extracted features: {i+1}/{len(pairs)}")
        except Exception as e:
            logger.warning(f"Feature extraction failed for {path}: {e}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train_and_save(X: np.ndarray, y: np.ndarray, save_dir: Path, name: str) -> Dict:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)

    # Try multiple classifiers and pick the best on validation split
    candidates = {
        'rf': RandomForestClassifier(n_estimators=400, max_depth=None, random_state=42, n_jobs=-1),
        'svm_rbf': SVC(kernel='rbf', C=5.0, gamma='scale', class_weight='balanced', probability=False, random_state=42)
    }
    best_name = None
    best_model = None
    best_acc = -1.0
    best_report = ""
    for mname, m in candidates.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        acc_m = accuracy_score(y_test, y_pred)
        if acc_m > best_acc:
            best_acc = acc_m
            best_report = classification_report(y_test, y_pred, target_names=EMOTIONS, zero_division=0)
            best_model = m
            best_name = mname

    model = best_model
    acc = best_acc
    report = best_report

    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_dir / f"emotion_model_{name}.pkl")
    joblib.dump(scaler, save_dir / f"emotion_scaler_{name}.pkl")

    with open(save_dir / f"training_report_{name}.txt", 'w') as f:
        f.write(f"Selected model: {best_name}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    logger.info(f"Saved model and scaler: {name} (model={best_name}, acc={acc:.3f})")
    return {"accuracy": acc, "report": report, "model": model, "scaler": scaler, "model_name": best_name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit-per-class', type=int, default=0, help='Optional cap per class per dataset to speed up')
    parser.add_argument('--save-dir', type=str, default='models/real', help='Directory to save models')
    parser.add_argument('--order', type=str, default='ravdess,crema,iemocap,tess', help='Dataset training order')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation during feature building')
    parser.add_argument('--augment-per-sample', type=int, default=1, help='Number of augmented variants per original sample')
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

    ravdess_only_pairs: List[Tuple[str,str]] = []

    for ds in order:
        if ds == 'ravdess' and present['ravdess']:
            pairs = list_ravdess(base/'ravdess', args.limit_per_class)
            ravdess_only_pairs = list(pairs)
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
        X, y = build_dataset(cumulative_pairs, processor, augment=args.augment, augment_per_sample=args.augment_per_sample)
        logger.info(f"Cumulative samples: {len(X)}")
        if len(X) < 50:
            logger.warning("Too few samples to train reliably; continuing")
            continue
        # Patch: clean feature matrix
        from sklearn.utils import check_array
        import numpy as np
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.warning(f"Detected NaN or inf in X; cleaning with zero-fill")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        res = train_and_save(X, y, save_dir, name=f"upto_{ds}")
        results[ds] = {"num_samples": len(X), "accuracy": res["accuracy"], "report": res["report"], "model_name": res.get("model_name", "")}

        # If we just trained upto RAVDESS, also evaluate strictly on RAVDESS
        if ds == 'ravdess' and ravdess_only_pairs:
            try:
                Xr, yr = build_dataset(ravdess_only_pairs, processor, augment=args.augment, augment_per_sample=args.augment_per_sample)
                if len(Xr) >= 20:
                    Xr_s = res["scaler"].transform(Xr)
                    y_pred_r = res["model"].predict(Xr_s)
                    acc_r = accuracy_score(yr, y_pred_r)
                    report_r = classification_report(yr, y_pred_r, target_names=EMOTIONS, zero_division=0)
                    with open(save_dir / 'training_report_ravdess_eval.txt', 'w') as f:
                        f.write(f"Model: {res.get('model_name','')}\n")
                        f.write(f"Accuracy (RAVDESS only): {acc_r:.4f}\n\n")
                        f.write(report_r)
                    results[ds]["ravdess_eval"] = {"accuracy": acc_r, "report": report_r}
                    logger.info(f"RAVDESS-only evaluation acc={acc_r:.3f} saved")
                else:
                    logger.warning("Too few RAVDESS samples for per-dataset eval")
            except Exception as e:
                logger.warning(f"RAVDESS-only eval failed: {e}")

    # Save summary
    with open(save_dir / 'incremental_training_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("Incremental training complete. Summary saved.")

if __name__ == '__main__':
    main()
