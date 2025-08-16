# Voice Distress Detection System

A real-time voice distress detection system that uses Whisper for speech-to-text conversion and AI for distress detection.

## 🚀 Features

- 🎤 **Voice Upload**: Upload audio files through a beautiful web interface
- 🗣️ **Speech-to-Text**: Convert speech to text using OpenAI Whisper
- 🧠 **AI Analysis**: Detect distress using keyword-based analysis
- 🌐 **Web Interface**: Modern, responsive web UI with drag-and-drop functionality
- 📱 **Mobile Friendly**: Works on all devices
- 🔧 **Easy Setup**: Simple installation and configuration

## 📁 Project Structure

```
llm_distress_project_v2/
├── src/
│   ├── __init__.py
│   └── app.py              # Main FastAPI application
├── tests/
│   ├── __init__.py
│   └── test_app.py         # Test suite
├── docs/                   # Documentation
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
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
# Method 1: Using main.py
python main.py

# Method 2: Using uvicorn directly
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```

### Access the Web Interface

Open your browser and go to: `http://127.0.0.1:8000`

### Upload Audio Files

1. **Drag and drop** any audio file onto the upload area
2. **Or click** "Choose Audio File" to browse and select a file
3. **Wait for analysis** - the system will process the audio
4. **View results** showing transcript and distress detection

### API Usage

```bash
# Upload audio file via curl
curl -X POST -F 'file=@your_audio_file.mp3' http://127.0.0.1:8000/voice-check
```

**Response:**
```json
{
  "transcript": "I'm feeling really scared right now",
  "distress": true,
  "label": "Distress"
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