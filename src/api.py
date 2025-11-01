import os
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
import numpy as np
import joblib
import soundfile as sf
import torch
import torchaudio.transforms as T

from src.audio_processor import TorchAudioProcessor

# Paths to trained model & scaler
MODEL_PATH = os.getenv("RAVDESS_MODEL_PATH", "models/real_ravdess/emotion_model_upto_ravdess.pkl")
SCALER_PATH = os.getenv("RAVDESS_SCALER_PATH", "models/real_ravdess/emotion_scaler_upto_ravdess.pkl")

# Set up app
app = FastAPI(title="Speech Emotion Recognition API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For full access; restrict as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model & scaler once on startup
EMOTIONS = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']
model = None
scaler = None
processor = None

@app.on_event("startup")
def startup():
    global model, scaler, processor
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    processor = TorchAudioProcessor()

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/")
def root():
    # Friendly landing page: send to interactive docs
    return RedirectResponse(url="/docs")

@app.get("/model-info")
def model_info():
    summary_path = "models/real_ravdess/incremental_training_summary.json"
    info = {
        "model_path": MODEL_PATH,
        "scaler_path": SCALER_PATH,
        "summary_path": summary_path
    }
    try:
        if os.path.exists(summary_path):
            import json
            with open(summary_path, 'r') as f:
                info["training_summary"] = json.load(f)
    except Exception:
        pass
    return info

class EmotionResult(BaseModel):
    emotion: str
    emotion_scores: dict
    features: dict
    device_used: str

@app.post("/predict-emotion", response_model=EmotionResult)
def predict_emotion(file: UploadFile = File(...)):
    # Save to buffer and decode into numpy array (supports wav, flac)
    try:
        audio_bytes = io.BytesIO(file.file.read())
        wav, sr = sf.read(audio_bytes)
        # Convert to torch tensor, float32, mono
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        waveform = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        # Resample if needed
        if sr != 16000:
            resampler = T.Resample(sr, 16000)
            waveform = resampler(waveform)
        if waveform.shape[1] < 16000:
            raise ValueError("Audio too short (<1s)")
        waveform = waveform / torch.max(torch.abs(waveform)) if torch.max(torch.abs(waveform)) > 0 else waveform
        # Move to correct device
        waveform = waveform.to(processor.device)
        feats = processor.extract_features_from_waveform(waveform)
        # Compose feature vector
        from train_emotion_datasets import extract_feature_vector
        vec = extract_feature_vector(feats)
        shape_warning = None
        # Adjust feature vector for model/scaler robustness
        expected_dim = scaler.mean_.shape[0]
        orig_dim = len(vec)
        adj_vec = list(vec)
        if orig_dim > expected_dim:
            shape_warning = f"Truncating features from {orig_dim} to {expected_dim} to match model."
            adj_vec = adj_vec[:expected_dim]
        elif orig_dim < expected_dim:
            shape_warning = f"Padding features from {orig_dim} to {expected_dim} to match model."
            adj_vec += [0.0] * (expected_dim - orig_dim)
        Xs = scaler.transform(np.array([adj_vec], dtype=np.float32))
        y_pred = model.predict(Xs)
        emotion = EMOTIONS[y_pred[0]]
        # Optionally: get emotion probabilities if supported
        emotion_scores = {}
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(Xs)[0]
            emotion_scores = {EMOTIONS[i]: float(p) for i, p in enumerate(probas)}
        else:
            # No predict_proba for SVC; return blank for now
            emotion_scores = {emotion: 1.0}
        # Confidence (use model's .decision_function or processor method)
        try:
            from src.audio_processor import TorchAudioProcessor as TAP
            confidence = TAP.calculate_confidence(processor, feats, emotion_scores)
        except Exception:
            confidence = float(np.max(list(emotion_scores.values())) if emotion_scores else 0.5)
        resp = {
            "emotion": emotion,
            "emotion_scores": emotion_scores,
            "features": feats,
            "device_used": processor.device,
        }
        if shape_warning:
            resp["feature_shape_warning"] = shape_warning
        return resp
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
