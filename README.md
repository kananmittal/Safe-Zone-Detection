# Voice Distress Detection System

A real-time voice distress detection system that uses Whisper for speech-to-text, TorchAudio for rich voice features, and a fine-tuned LLM for multi-modal distress detection.

## 🚀 Features

- 🎤 **Voice Upload**: Upload audio files through a beautiful web interface
- 🗣️ **Speech-to-Text**: Convert speech to text using OpenAI Whisper
- 🔊 **TorchAudio Features**: MFCCs, Mel-spectrograms, pitch, RMS, ZCR, spectral features
- 🧠 **Multi-Modal AI Analysis**: LLM combines transcript + voice features for distress level and safety action
- 🌐 **Web Interface**: Modern, responsive web UI with drag-and-drop functionality
- 📱 **Mobile Friendly**: Works on all devices
- 🔧 **Easy Setup**: Simple installation and configuration

## 📁 Project Structure

```
llm_distress_project_v2/
├── src/
│   ├── __init__.py
│   └── api.py              # FastAPI application for emotion inference (RAVDESS model)
│   ├── audio_processor.py  # TorchAudio feature extraction + rule-based emotion
│   ├── llama_processor.py  # LLM multi-modal analysis (auto-loads latest fine-tuned checkpoint)
│   ├── fine_tuner.py       # Baseline fine-tuning script
│   ├── resume_fine_tuning.py # Resume training from checkpoint
│   └── fast_fine_tuner.py  # Faster fine-tuning configuration
├── tests/
│   ├── __init__.py
│   └── test_app.py         # Test suite
├── docs/                   # Documentation
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── datasets/               # Raw and processed datasets
│   └── processed/
│       ├── combined_dataset.csv
│       └── fine_tuning_data.json
├── download_datasets.py    # TESS/IEMOCAP instructions + status
├── src/data_processor.py   # Dataset processing to create fine_tuning_data.json
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- FFmpeg (for audio processing)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd llm_distress_project_v2
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Start the Application

```bash
# Using uvicorn directly
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

### Health Check

Open: `http://127.0.0.1:8000/healthz`

### API Usage

```bash
# Upload audio file for emotion prediction via curl
curl -X POST -F 'file=@your_audio_file.wav' http://127.0.0.1:8000/predict-emotion
```

**Response:**
```json
{
  "emotion": "happy",
  "emotion_scores": {"happy": 0.73, "neutral": 0.12, ...},
  "features": {"mfcc_mean": [...], ...},
  "device_used": "mps"
}
```

## 🧪 Testing

Run the test suite:

```bash
# Install pytest if not already installed
pip install pytest

# Run tests
pytest tests/
```

## 🔧 Configuration

### Distress Keywords

You can customize the distress detection keywords in `src/app.py`:

```python
distress_keywords = [
    "help", "emergency", "danger", "scared", "fear", "threat", "unsafe",
    "panic", "terrified", "afraid", "worried", "anxious", "distress",
    "crying", "screaming", "pain", "hurt", "attack", "robbery"
]
```

### Whisper Model

The system uses Whisper's "base" model by default. You can change this in `src/app.py`:

```python
model = whisper.load_model("base")  # Options: "tiny", "base", "small", "medium", "large"
```

## 📋 Dependencies

- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **OpenAI Whisper**: Speech-to-text
- **Pydub**: Audio processing
- **Python-multipart**: File upload handling

## 🐛 Troubleshooting

### Common Issues

1. **FFmpeg not found**: Install FFmpeg on your system
2. **Port 8000 in use**: Change the port in main.py or uvicorn command
3. **Audio file not supported**: Ensure the audio format is supported by Whisper

### Logs

Check the terminal where the server is running for error messages and logs.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `pytest tests/`
5. Commit your changes: `git commit -am 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions, please open an issue on the repository.