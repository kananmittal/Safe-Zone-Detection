from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import subprocess
from pydub import AudioSegment
import tempfile
import numpy as np
import traceback

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def analyze_voice_emotion(audio_path):
    """Analyze voice characteristics using TorchAudio processor with fallback"""
    try:
        # Use the new TorchAudio processor
        from src.audio_processor import TorchAudioProcessor
        
        processor = TorchAudioProcessor()
        result = processor.process_audio(audio_path)
        
        # Return only essential information for API
        return {
            'emotion': result['emotion'],
            'confidence': float(result['confidence']),
            'emotion_scores': result.get('emotion_scores', {}),
            'device_used': result.get('device_used', 'unknown')
        }
        
    except ImportError:
        # Fallback if TorchAudio is not available
        return {
            'emotion': 'neutral',
            'confidence': 0.0,
            'error': 'TorchAudio not available for voice analysis'
        }
    except Exception as e:
        # Fallback for any other error
        return {
            'emotion': 'neutral',
            'confidence': 0.0,
            'error': f'Voice analysis error: {str(e)}'
        }

def analyze_emotion_from_features(features):
    """Analyze emotion based on audio features"""
    emotion_scores = {
        'fear': 0,
        'anger': 0,
        'sadness': 0,
        'happiness': 0,
        'neutral': 0
    }
    
    # Fear indicators
    if features.get('pitch_std', 0) > 50:  # High pitch variation
        emotion_scores['fear'] += 2
    if features.get('volume_std', 0) > 0.1:  # Volume variation
        emotion_scores['fear'] += 1
    if features.get('speech_rate', 100) > 120:  # Fast speech
        emotion_scores['fear'] += 1
    if features.get('pitch_mean', 0) > 200:  # High pitch
        emotion_scores['fear'] += 1
    
    # Anger indicators
    if features.get('volume_mean', 0) > 0.15:  # Loud voice
        emotion_scores['anger'] += 2
    if features.get('pitch_mean', 0) > 180:  # High pitch
        emotion_scores['anger'] += 1
    if features.get('speech_rate', 100) > 110:  # Fast speech
        emotion_scores['anger'] += 1
    
    # Sadness indicators
    if features.get('pitch_mean', 0) < 120:  # Low pitch
        emotion_scores['sadness'] += 2
    if features.get('volume_mean', 0) < 0.05:  # Quiet voice
        emotion_scores['sadness'] += 1
    if features.get('speech_rate', 100) < 80:  # Slow speech
        emotion_scores['sadness'] += 1
    
    # Happiness indicators
    if 120 < features.get('pitch_mean', 0) < 180:  # Moderate pitch
        emotion_scores['happiness'] += 1
    if 0.05 < features.get('volume_mean', 0) < 0.15:  # Moderate volume
        emotion_scores['happiness'] += 1
    if 80 < features.get('speech_rate', 100) < 110:  # Moderate speech rate
        emotion_scores['happiness'] += 1
    
    # Neutral (baseline)
    emotion_scores['neutral'] = 1
    
    # Return the emotion with highest score
    return max(emotion_scores, key=emotion_scores.get)

def calculate_confidence(features):
    """Calculate confidence in emotion detection"""
    # Simple confidence based on feature quality
    confidence = 0.5  # Base confidence
    
    if features.get('pitch_std', 0) > 0:
        confidence += 0.2
    if features.get('volume_std', 0) > 0:
        confidence += 0.2
    if features.get('speech_rate', 0) > 0:
        confidence += 0.1
    
    return min(confidence, 1.0)

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

            .emotion-badge {
                display: inline-block;
                padding: 5px 15px;
                border-radius: 15px;
                font-weight: bold;
                margin: 5px;
            }
            .emotion-fear { background: #ff9800; color: white; }
            .emotion-anger { background: #f44336; color: white; }
            .emotion-sadness { background: #2196f3; color: white; }
            .emotion-happiness { background: #4caf50; color: white; }
            .emotion-neutral { background: #9e9e9e; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎤 Voice Distress Detection</h1>
                <p>Upload an audio file to detect distress in speech and emotional tone</p>
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
            
            <script>
                // Debug function (console only)
                function log(message) {
                    console.log(message);
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
                        
                        // Create emotion badge
                        const emotionBadge = data.voice_emotion ? 
                            `<span class="emotion-badge emotion-${data.voice_emotion.emotion}">${data.voice_emotion.emotion.toUpperCase()}</span>` : '';
                        
                        resultDiv.innerHTML = `
                            <h3>📊 Analysis Results</h3>
                            <p><strong>Transcript:</strong> ${data.transcript || 'No transcript available'}</p>
                            <p><strong>Content Analysis:</strong> ${isDistress ? '🚨 DISTRESS DETECTED' : '✅ SAFE'}</p>
                            <p><strong>Voice Emotion:</strong> ${emotionBadge} (Confidence: ${data.voice_emotion ? Math.round(data.voice_emotion.confidence * 100) : 0}%)</p>
                            <p><strong>Processing Device:</strong> ${data.voice_emotion && data.voice_emotion.device_used ? data.voice_emotion.device_used.toUpperCase() : 'Unknown'}</p>
                            <p><strong>Overall Assessment:</strong> ${data.label || 'No analysis available'}</p>
                            ${data.voice_emotion && data.voice_emotion.emotion_scores ? `
                            <details>
                                <summary>🎭 Detailed Emotion Scores</summary>
                                <div style="margin-top: 10px; padding: 10px; background: #f5f5f5; border-radius: 5px;">
                                    ${Object.entries(data.voice_emotion.emotion_scores).map(([emotion, score]) => 
                                        `<div><strong>${emotion}:</strong> ${Math.round(score * 100)}%</div>`
                                    ).join('')}
                                </div>
                            </details>
                            ` : ''}
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
        
        try:
            # Convert speech to text
            transcript = simple_speech_to_text(temp_path)
            
            # Analyze voice emotion
            voice_emotion = analyze_voice_emotion(temp_path)
            
            # Detect distress from text content
            text_distress = detect_distress_simple(transcript)
            
            # Combine text and voice analysis
            final_distress = text_distress == "Distress" or voice_emotion['emotion'] in ['fear', 'angry', 'disgust']
            
            # Create comprehensive label
            if final_distress:
                if text_distress == "Distress" and voice_emotion['emotion'] in ['fear', 'angry', 'disgust']:
                    label = f"Distress detected in both content and voice tone ({voice_emotion['emotion']})"
                elif text_distress == "Distress":
                    label = f"Distress detected in content, voice tone: {voice_emotion['emotion']}"
                else:
                    label = f"Distress detected in voice tone ({voice_emotion['emotion']}), content appears safe"
            else:
                label = f"Safe - Content: {text_distress}, Voice: {voice_emotion['emotion']}"
            
        except Exception as e:
            # Fallback if analysis fails
            transcript = "Error in analysis"
            voice_emotion = {"emotion": "neutral", "confidence": 0.0, "error": str(e)}
            text_distress = "Safe"
            final_distress = False
            label = f"Analysis error: {str(e)}"
        
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass
        
        return {
            "transcript": transcript,
            "distress": final_distress,
            "label": label,
            "voice_emotion": voice_emotion,
            "text_analysis": text_distress
        }
        
    except Exception as e:
        # Comprehensive error handling
        return {
            "transcript": "Error processing audio",
            "distress": False,
            "label": f"Error: {str(e)}",
            "voice_emotion": {"emotion": "unknown", "confidence": 0.0, "error": str(e)},
            "text_analysis": "Error"
        } 