#!/usr/bin/env python3
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import joblib

from train_emotion_datasets import list_ravdess, extract_feature_vector
from src.audio_processor import TorchAudioProcessor

EMOTIONS = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']

def main():
    model_path = Path('models/real_ravdess/emotion_model_upto_ravdess.pkl')
    scaler_path = Path('models/real_ravdess/emotion_scaler_upto_ravdess.pkl')
    ds_root = Path('datasets/ravdess')
    assert model_path.exists() and scaler_path.exists(), 'Model/scaler not found in models/real_ravdess'
    assert ds_root.exists(), 'datasets/ravdess not found'

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    processor = TorchAudioProcessor()

    pairs: List[Tuple[str,str]] = list_ravdess(ds_root, limit_per_class=0)
    X: List[List[float]] = []
    y: List[int] = []
    total = 0
    for (path, label) in pairs:
        try:
            res = processor.process_audio(path)
            feats = res.get('features')
            if not feats:
                continue
            vec = extract_feature_vector(feats)
            X.append(vec)
            y.append(EMOTIONS.index(label))
            total += 1
        except Exception:
            continue

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    Xs = scaler.transform(X)
    y_pred = model.predict(Xs)
    acc = float(np.mean(y_pred == y)) if len(y) else 0.0
    report = {
        'num_samples': int(len(y)),
        'overall_accuracy': acc
    }
    print(json.dumps(report, indent=2))
    with open('current_model_ravdess_eval.json', 'w') as f:
        json.dump(report, f, indent=2)

if __name__ == '__main__':
    main()


