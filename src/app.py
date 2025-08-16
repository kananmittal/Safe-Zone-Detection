from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import subprocess
from pydub import AudioSegment
import tempfile

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def simple_speech_to_text(audio_path):
    """Simple speech-to-text using available tools"""
    try:
        # Try using ffmpeg and whisper if available
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"]
    except ImportError:
        # Fallback: just return a placeholder
        return "Speech-to-text not available. Please install whisper: pip install openai-whisper"
    except Exception as e:
        return f"Error in speech recognition: {str(e)}"

def detect_distress_simple(text):
    """Simple distress detection without Ollama"""
    distress_keywords = [
        "help", "emergency", "danger", "scared", "fear", "threat", "unsafe",
        "panic", "terrified", "afraid", "worried", "anxious", "distress",
        "crying", "screaming", "pain", "hurt", "attack", "robbery"
    ]
    
    text_lower = text.lower()
    for keyword in distress_keywords:
        if keyword in text_lower:
            return "Distress"
    return "Safe"

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voice Distress Detection</title>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                margin: 0;
                padding: 20px;
                color: #333;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }
            .header {
                text-align: center;
                margin-bottom: 40px;
            }
            .header h1 {
                color: #667eea;
                margin-bottom: 10px;
            }
            .upload-area { 
                border: 3px dashed #ddd; 
                padding: 40px; 
                text-align: center; 
                margin: 20px 0;
                border-radius: 15px;
                transition: all 0.3s ease;
                cursor: pointer;
            }
            .upload-area:hover {
                border-color: #667eea;
                background-color: #f8f9ff;
            }
            .result { 
                background: #f8f9fa; 
                padding: 20px; 
                margin: 20px 0; 
                border-radius: 10px;
                border-left: 4px solid #667eea;
            }
            .distress { 
                background: #ffebee; 
                border-left: 4px solid #f44336; 
            }
            .safe { 
                background: #e8f5e8; 
                border-left: 4px solid #4caf50; 
            }
            .upload-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 25px;
                font-size: 1rem;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            .upload-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
            }
            .loading {
                text-align: center;
                color: #667eea;
                font-style: italic;
            }
            .debug-info {
                background: #f0f0f0;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
                font-size: 12px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎤 Voice Distress Detection</h1>
                <p>Upload an audio file to detect distress in speech</p>
            </div>
            
            <div class="upload-area">
                <form id="uploadForm">
                    <input type="file" id="audioFile" accept="audio/*" required style="display: none;">
                    <button type="button" class="upload-btn" onclick="document.getElementById('audioFile').click()">
                        Choose Audio File
                    </button>
                    <p style="margin-top: 20px; color: #666;">Click the button above or drag and drop an audio file</p>
                </form>
            </div>
            
            <div id="result" class="result" style="display: none;"></div>
            <div id="debug" class="debug-info" style="display: none;"></div>
            
            <script>
                // Debug function
                function log(message) {
                    console.log(message);
                    const debugDiv = document.getElementById('debug');
                    debugDiv.style.display = 'block';
                    debugDiv.innerHTML += '<div>' + message + '</div>';
                }
                
                // Handle file selection
                document.getElementById('audioFile').addEventListener('change', function(e) {
                    log('File selected: ' + (e.target.files[0] ? e.target.files[0].name : 'none'));
                    if (e.target.files.length > 0) {
                        handleFileUpload(e.target.files[0]);
                    }
                });
                
                // Handle drag and drop
                const uploadArea = document.querySelector('.upload-area');
                uploadArea.addEventListener('dragover', function(e) {
                    e.preventDefault();
                    uploadArea.style.borderColor = '#667eea';
                    uploadArea.style.backgroundColor = '#f0f4ff';
                });
                
                uploadArea.addEventListener('dragleave', function(e) {
                    e.preventDefault();
                    uploadArea.style.borderColor = '#ddd';
                    uploadArea.style.backgroundColor = 'transparent';
                });
                
                uploadArea.addEventListener('drop', function(e) {
                    e.preventDefault();
                    uploadArea.style.borderColor = '#ddd';
                    uploadArea.style.backgroundColor = 'transparent';
                    const files = e.dataTransfer.files;
                    log('Files dropped: ' + files.length);
                    if (files.length > 0) {
                        handleFileUpload(files[0]);
                    }
                });
                
                function handleFileUpload(file) {
                    log('Starting upload for: ' + file.name + ' (size: ' + file.size + ' bytes)');
                    
                    if (!file.type.startsWith('audio/')) {
                        alert('Please select an audio file.');
                        return;
                    }
                    
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    const resultDiv = document.getElementById('result');
                    resultDiv.innerHTML = '<div class="loading">🔄 Analyzing audio... Please wait...</div>';
                    resultDiv.className = 'result';
                    resultDiv.style.display = 'block';
                    
                    log('Sending request to /voice-check');
                    
                    fetch('/voice-check', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => {
                        log('Response received: ' + response.status + ' ' + response.statusText);
                        if (!response.ok) {
                            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                        }
                        return response.json();
                    })
                    .then(data => {
                        log('Data received: ' + JSON.stringify(data));
                        const isDistress = data.distress;
                        resultDiv.className = 'result ' + (isDistress ? 'distress' : 'safe');
                        resultDiv.innerHTML = `
                            <h3>📊 Analysis Results</h3>
                            <p><strong>Transcript:</strong> ${data.transcript || 'No transcript available'}</p>
                            <p><strong>Status:</strong> ${isDistress ? '🚨 DISTRESS DETECTED' : '✅ SAFE'}</p>
                            <p><strong>Analysis:</strong> ${data.label || 'No analysis available'}</p>
                        `;
                    })
                    .catch(error => {
                        log('Error: ' + error.message);
                        resultDiv.className = 'result distress';
                        resultDiv.innerHTML = `
                            <h3>❌ Error</h3>
                            <p><strong>Error:</strong> ${error.message}</p>
                            <p>Please try again or check if the audio file is valid.</p>
                        `;
                    });
                }
            </script>
        </div>
    </body>
    </html>
    """

@app.post("/voice-check")
async def voice_check(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        data = await file.read()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            temp_file.write(data)
            temp_path = temp_file.name
        
        # Convert speech to text
        transcript = simple_speech_to_text(temp_path)
        
        # Detect distress
        label = detect_distress_simple(transcript)
        distress = label.lower().startswith("distress")
        
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass
        
        return {
            "transcript": transcript,
            "distress": distress,
            "label": label
        }
        
    except Exception as e:
        return {
            "transcript": "Error processing audio",
            "distress": False,
            "label": f"Error: {str(e)}"
        } 