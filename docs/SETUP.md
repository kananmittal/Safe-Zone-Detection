# Setup Guide

## Quick Start

1. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Run the application:**
   ```bash
   python run.py
   ```

3. **Open in browser:**
   ```
   http://127.0.0.1:8000
   ```

## Alternative Run Methods

### Method 1: Using run.py (Recommended)
```bash
python run.py
```

### Method 2: Using main.py
```bash
python main.py
```

### Method 3: Using uvicorn directly
```bash
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```

## Testing

```bash
# Install pytest
pip install pytest

# Run tests
pytest tests/
```

## Project Structure

```
llm_distress_project_v2/
├── src/
│   ├── __init__.py
│   └── app.py              # Main application
├── tests/
│   ├── __init__.py
│   └── test_app.py         # Tests
├── docs/                   # Documentation
├── run.py                  # Easy run script
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── .gitignore             # Git ignore
└── README.md              # Main documentation
```

## Features

- ✅ Clean, organized code structure
- ✅ Easy-to-use run script
- ✅ Comprehensive testing
- ✅ Git-ready with proper .gitignore
- ✅ Beautiful web interface
- ✅ Audio upload and processing
- ✅ Distress detection
- ✅ Speech-to-text conversion 