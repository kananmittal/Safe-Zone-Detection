#!/usr/bin/env python3
import os
from pathlib import Path
from typing import List, Tuple
import json
from collections import defaultdict

from advanced_emotion_recognition import AdvancedEmotionClassifier

EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def list_ravdess_samples(root: Path, limit_per_class: int = 50) -> List[Tuple[str, str]]:
    samples: List[Tuple[str, str]] = []
    per_class = defaultdict(int)
    for actor in sorted(root.glob('Actor_*')):
        for wav in actor.glob('*.wav'):
            parts = wav.stem.split('-')
            if len(parts) >= 3:
                code = parts[2]
                label = EMOTION_MAP.get(code)
                if not label:
                    continue
                if limit_per_class and per_class[label] >= limit_per_class:
                    continue
                samples.append((str(wav), label))
                per_class[label] += 1
    return samples

def main():
    ds = Path('datasets/ravdess')
    if not ds.exists():
        print('datasets/ravdess not found')
        return
    samples = list_ravdess_samples(ds, limit_per_class=50)
    if not samples:
        print('No samples found')
        return
    clf = AdvancedEmotionClassifier(models_dir='models')
    # ensure models are loaded
    try:
        import joblib
        clf.ensemble_model = joblib.load(clf.models_dir / 'advanced_ensemble_model.pkl')
        clf.scaler = joblib.load(clf.models_dir / 'advanced_scaler.pkl')
        clf.pca = joblib.load(clf.models_dir / 'advanced_pca.pkl')
    except Exception as e:
        print(f'Failed to load advanced models: {e}')
        return
    total = 0
    correct = 0
    per_class = defaultdict(lambda: {'correct': 0, 'total': 0})
    for path, label in samples:
        res = clf.predict_emotion(path)
        pred = res.get('emotion', 'neutral')
        total += 1
        per_class[label]['total'] += 1
        if pred == label:
            correct += 1
            per_class[label]['correct'] += 1
    overall = correct / total if total else 0.0
    per_class_acc = {k: (v['correct'] / v['total'] if v['total'] else 0.0) for k, v in per_class.items()}
    out = {
        'dataset': 'ravdess',
        'num_samples': total,
        'overall_accuracy': overall,
        'per_class_accuracy': per_class_acc
    }
    print(json.dumps(out, indent=2))
    with open('advanced_evaluation_ravdess.json', 'w') as f:
        json.dump(out, f, indent=2)

if __name__ == '__main__':
    main()


