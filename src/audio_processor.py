#!/usr/bin/env python3
"""
TorchAudio-based Audio Processor for Voice Distress Detection
Optimized for hardware acceleration (MPS/CUDA/CPU)
"""

import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TorchAudioProcessor:
    """
    Advanced audio processor using TorchAudio for voice emotion analysis
    Optimized for Mac M3 with MPS acceleration
    """
    
    def __init__(self, sample_rate: int = 16000, device: str = "auto"):
        """
        Initialize the TorchAudio processor
        
        Args:
            sample_rate: Target sample rate for audio processing
            device: Device to use ('auto', 'cpu', 'mps', 'cuda')
        """
        self.sample_rate = sample_rate
        
        # Auto-detect device
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("Using MPS (Metal Performance Shaders) for acceleration")
            elif torch.cuda.is_available():
                self.device = "cuda"
                logger.info("Using CUDA for acceleration")
            else:
                self.device = "cpu"
                logger.info("Using CPU for processing")
        else:
            self.device = device
            
        logger.info(f"TorchAudio processor initialized on {self.device}")
        
        # Initialize transforms
        self._setup_transforms()
        
    def _setup_transforms(self):
        """Setup TorchAudio transforms for feature extraction"""
        
        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=8000
        ).to(self.device)
        
        # MFCC transform
        self.mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=13,
            melkwargs={
                'n_fft': 2048,
                'hop_length': 512,
                'n_mels': 128,
                'f_min': 0,
                'f_max': 8000
            }
        ).to(self.device)
        
        # Spectral centroid transform
        self.spectral_centroid = T.SpectralCentroid(
            sample_rate=self.sample_rate,
            n_fft=2048,
            hop_length=512
        ).to(self.device)
        
        # Zero crossing rate transform (using functional instead)
        # Note: ZeroCrossingRate transform doesn't exist in this version
        # We'll implement it manually in the feature extraction
        
        # RMS energy transform (using functional instead)
        # Note: RMS transform doesn't exist in this version
        # We'll implement it manually in the feature extraction
        
        logger.info("TorchAudio transforms initialized")
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load and preprocess audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio tensor
        """
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Move to device first
            waveform = waveform.to(self.device)
            
            # Resample if needed (after moving to device)
            if sample_rate != self.sample_rate:
                resampler = T.Resample(sample_rate, self.sample_rate).to(self.device)
                waveform = resampler(waveform)
            
            # Normalize audio (skip VAD for now as it might cause issues)
            # waveform = F.vad(waveform, sample_rate=self.sample_rate)
            
            # Simple normalization instead
            if torch.max(torch.abs(waveform)) > 0:
                waveform = waveform / torch.max(torch.abs(waveform))
            
            return waveform
            
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            raise
    
    def extract_features(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive audio features using TorchAudio
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of extracted features
        """
        try:
            # Load audio
            waveform = self.load_audio(audio_path)
            
            # Check if waveform is valid
            if waveform.numel() == 0 or torch.all(waveform == 0):
                logger.warning(f"Empty or zero waveform detected for {audio_path}")
                return {
                    'emotion': 'neutral',
                    'emotion_scores': {'neutral': 1.0},
                    'confidence': 0.0,
                    'error': 'Empty or zero waveform',
                    'device_used': self.device
                }
            
            features = {}
            
            # 1. Mel spectrogram
            mel_spec = self.mel_transform(waveform)
            features['mel_spectrogram'] = mel_spec.cpu().numpy()
            features['mel_mean'] = float(torch.mean(mel_spec).cpu().numpy())
            features['mel_std'] = float(torch.std(mel_spec).cpu().numpy())
            
            # 2. MFCC coefficients
            mfcc = self.mfcc_transform(waveform)
            features['mfcc'] = mfcc.cpu().numpy()
            features['mfcc_mean'] = float(torch.mean(mfcc).cpu().numpy())
            features['mfcc_std'] = float(torch.std(mfcc).cpu().numpy())
            
            # 3. Spectral centroid
            spectral_cent = self.spectral_centroid(waveform)
            features['spectral_centroid'] = spectral_cent.cpu().numpy()
            features['spectral_centroid_mean'] = float(torch.mean(spectral_cent).cpu().numpy())
            features['spectral_centroid_std'] = float(torch.std(spectral_cent).cpu().numpy())
            
            # 4. Zero crossing rate (manual implementation)
            zcr = self._calculate_zero_crossing_rate(waveform)
            features['zero_crossing_rate'] = zcr
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            
            # 5. RMS energy (manual implementation)
            rms = self._calculate_rms_energy(waveform)
            features['rms_energy'] = rms
            features['rms_mean'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))
            features['rms_max'] = float(np.max(rms))
            
            # 6. Pitch analysis (using autocorrelation)
            pitch = self._extract_pitch(waveform)
            features['pitch'] = pitch
            features['pitch_mean'] = float(np.mean(pitch)) if len(pitch) > 0 else 0.0
            features['pitch_std'] = float(np.std(pitch)) if len(pitch) > 0 else 0.0
            features['pitch_range'] = float(np.max(pitch) - np.min(pitch)) if len(pitch) > 0 else 0.0
            
            # 7. Spectral bandwidth
            spectral_bw = self._extract_spectral_bandwidth(waveform)
            features['spectral_bandwidth'] = spectral_bw
            features['spectral_bw_mean'] = float(np.mean(spectral_bw))
            features['spectral_bw_std'] = float(np.std(spectral_bw))
            
            # 8. Spectral rolloff
            spectral_rolloff = self._extract_spectral_rolloff(waveform)
            features['spectral_rolloff'] = spectral_rolloff
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
            
            # 9. Tempo estimation
            tempo = self._estimate_tempo(waveform)
            features['tempo'] = tempo
            
            # 10. Audio duration
            features['duration'] = float(waveform.shape[1] / self.sample_rate)
            
            logger.info(f"Extracted {len(features)} feature categories from {audio_path}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {e}")
            raise
    
    def _calculate_rms_energy(self, waveform: torch.Tensor) -> np.ndarray:
        """Calculate RMS energy manually"""
        try:
            # Convert to numpy
            audio_np = waveform.cpu().numpy().flatten()
            
            # Calculate RMS per frame
            frame_length = 2048
            hop_length = 512
            num_frames = 1 + (len(audio_np) - frame_length) // hop_length
            
            rms_values = []
            for i in range(num_frames):
                start = i * hop_length
                end = start + frame_length
                if end <= len(audio_np):
                    frame = audio_np[start:end]
                    rms = np.sqrt(np.mean(frame ** 2))
                    rms_values.append(rms)
            
            return np.array(rms_values) if rms_values else np.array([0.0])
            
        except Exception as e:
            logger.warning(f"RMS energy calculation failed: {e}")
            return np.array([0.0])
    
    def _calculate_zero_crossing_rate(self, waveform: torch.Tensor) -> np.ndarray:
        """Calculate zero crossing rate manually"""
        try:
            # Convert to numpy
            audio_np = waveform.cpu().numpy().flatten()
            
            # Calculate zero crossings
            zero_crossings = np.sum(np.diff(np.signbit(audio_np)))
            
            # Calculate rate per frame
            frame_length = 2048
            hop_length = 512
            num_frames = 1 + (len(audio_np) - frame_length) // hop_length
            
            zcr_values = []
            for i in range(num_frames):
                start = i * hop_length
                end = start + frame_length
                if end <= len(audio_np):
                    frame = audio_np[start:end]
                    crossings = np.sum(np.diff(np.signbit(frame)))
                    zcr_values.append(crossings / frame_length)
            
            return np.array(zcr_values) if zcr_values else np.array([0.0])
            
        except Exception as e:
            logger.warning(f"Zero crossing rate calculation failed: {e}")
            return np.array([0.0])
    
    def _extract_pitch(self, waveform: torch.Tensor) -> np.ndarray:
        """Extract fundamental frequency using autocorrelation"""
        try:
            # Convert to numpy for pitch extraction
            audio_np = waveform.cpu().numpy().flatten()
            
            # Simple autocorrelation-based pitch detection
            def autocorr_pitch(audio, sample_rate, min_freq=80, max_freq=400):
                # Normalize audio
                audio = audio - np.mean(audio)
                
                # Calculate autocorrelation
                corr = np.correlate(audio, audio, mode='full')
                corr = corr[len(corr)//2:]
                
                # Find peaks
                peaks = []
                for i in range(1, len(corr)-1):
                    if corr[i] > corr[i-1] and corr[i] > corr[i+1]:
                        peaks.append(i)
                
                if not peaks:
                    return []
                
                # Convert to frequencies
                freqs = [sample_rate / peak for peak in peaks if peak > 0]
                freqs = [f for f in freqs if min_freq <= f <= max_freq]
                
                return freqs
            
            pitch_freqs = autocorr_pitch(audio_np, self.sample_rate)
            return np.array(pitch_freqs) if pitch_freqs else np.array([])
            
        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}")
            return np.array([])
    
    def _extract_spectral_bandwidth(self, waveform: torch.Tensor) -> np.ndarray:
        """Extract spectral bandwidth"""
        try:
            # Calculate spectrogram
            spec = torch.stft(
                waveform.squeeze(),
                n_fft=2048,
                hop_length=512,
                return_complex=True
            )
            
            # Calculate spectral bandwidth
            freqs = torch.fft.fftfreq(2048, 1/self.sample_rate, device=self.device)
            freqs = freqs[:1025]  # Positive frequencies only
            
            # Calculate bandwidth
            spec_mag = torch.abs(spec)
            spec_power = spec_mag ** 2
            
            # Weighted average frequency
            weighted_freq = torch.sum(freqs.unsqueeze(1) * spec_power, dim=0)
            total_power = torch.sum(spec_power, dim=0)
            
            # Avoid division by zero
            total_power = torch.where(total_power == 0, 1e-10, total_power)
            centroid = weighted_freq / total_power
            
            # Calculate bandwidth
            bandwidth = torch.sqrt(
                torch.sum(((freqs.unsqueeze(1) - centroid.unsqueeze(0)) ** 2) * spec_power, dim=0) / total_power
            )
            
            return bandwidth.cpu().numpy()
            
        except Exception as e:
            logger.warning(f"Spectral bandwidth extraction failed: {e}")
            return np.array([0.0])
    
    def _extract_spectral_rolloff(self, waveform: torch.Tensor) -> np.ndarray:
        """Extract spectral rolloff"""
        try:
            # Calculate spectrogram
            spec = torch.stft(
                waveform.squeeze(),
                n_fft=2048,
                hop_length=512,
                return_complex=True
            )
            
            spec_mag = torch.abs(spec)
            spec_power = spec_mag ** 2
            
            # Calculate cumulative sum
            cumsum = torch.cumsum(spec_power, dim=0)
            total_power = torch.sum(spec_power, dim=0)
            
            # Find rolloff point (85% of total power)
            threshold = 0.85 * total_power
            
            # Find index where cumsum exceeds threshold
            rolloff_indices = torch.argmax(cumsum >= threshold.unsqueeze(0), dim=0)
            
            # Convert to frequencies
            freqs = torch.fft.fftfreq(2048, 1/self.sample_rate, device=self.device)
            freqs = freqs[:1025]  # Positive frequencies only
            
            rolloff_freqs = freqs[rolloff_indices]
            
            return rolloff_freqs.cpu().numpy()
            
        except Exception as e:
            logger.warning(f"Spectral rolloff extraction failed: {e}")
            return np.array([0.0])
    
    def _estimate_tempo(self, waveform: torch.Tensor) -> float:
        """Estimate tempo using onset detection"""
        try:
            # Simple tempo estimation using autocorrelation
            # Convert to numpy for processing
            audio_np = waveform.cpu().numpy().flatten()
            
            # Calculate autocorrelation
            corr = np.correlate(audio_np, audio_np, mode='full')
            corr = corr[len(corr)//2:]
            
            # Find peaks in autocorrelation
            peaks = []
            for i in range(1, len(corr)-1):
                if corr[i] > corr[i-1] and corr[i] > corr[i+1] and corr[i] > 0.1 * np.max(corr):
                    peaks.append(i)
            
            if len(peaks) > 1:
                # Calculate average interval between peaks
                intervals = np.diff(peaks)
                avg_interval = np.mean(intervals)
                
                # Convert to tempo (beats per minute)
                # Assuming 512 samples per frame at 16kHz
                samples_per_beat = avg_interval * 512
                tempo = (60.0 * self.sample_rate) / samples_per_beat
                
                # Clamp to reasonable range
                tempo = max(60, min(200, tempo))
                return float(tempo)
            else:
                return 120.0  # Default tempo
                
        except Exception as e:
            logger.warning(f"Tempo estimation failed: {e}")
            return 120.0  # Default tempo
    
    def analyze_emotion_from_features(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze emotion from extracted features using rule-based approach
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Dictionary with emotion probabilities
        """
        emotion_scores = {
            'neutral': 0.0,
            'calm': 0.0,
            'happy': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'fear': 0.0,
            'disgust': 0.0,
            'surprised': 0.0
        }
        
        try:
            # MFCC analysis (most discriminative feature)
            mfcc_std = features.get('mfcc_std', 0.0)
            
            if mfcc_std > 80:  # High MFCC variance - likely neutral/calm
                emotion_scores['neutral'] += 0.3
                emotion_scores['calm'] += 0.2
            elif mfcc_std > 70:  # Medium variance - likely happy/surprised
                emotion_scores['happy'] += 0.3
                emotion_scores['surprised'] += 0.2
            elif mfcc_std > 60:  # Lower variance - likely sad/fearful
                emotion_scores['sad'] += 0.3
                emotion_scores['fear'] += 0.2
            else:  # Very low variance - likely angry/disgust
                emotion_scores['angry'] += 0.3
                emotion_scores['disgust'] += 0.2
            
            # Pitch analysis (adjusted for actual ranges)
            pitch_mean = features.get('pitch_mean', 0.0)
            
            if pitch_mean > 170:  # Higher pitch
                emotion_scores['happy'] += 0.15
                emotion_scores['surprised'] += 0.1
            elif pitch_mean < 150:  # Lower pitch
                emotion_scores['sad'] += 0.15
                emotion_scores['angry'] += 0.1
            
            # Energy analysis (adjusted for actual ranges)
            rms_mean = features.get('rms_mean', 0.0)
            
            if rms_mean > 0.075:  # Higher energy
                emotion_scores['angry'] += 0.15
                emotion_scores['happy'] += 0.1
            elif rms_mean < 0.055:  # Lower energy
                emotion_scores['sad'] += 0.15
                emotion_scores['fear'] += 0.1
            
            # Spectral centroid (adjusted for actual ranges)
            spec_cent_mean = features.get('spectral_centroid_mean', 0.0)
            
            if spec_cent_mean > 2500:  # Brighter sound
                emotion_scores['happy'] += 0.1
                emotion_scores['surprised'] += 0.1
            elif spec_cent_mean < 2200:  # Darker sound
                emotion_scores['sad'] += 0.1
                emotion_scores['fear'] += 0.1
            
            # Zero crossing rate (adjusted for actual ranges)
            zcr_mean = features.get('zcr_mean', 0.0)
            
            if zcr_mean > 0.20:  # More noisy
                emotion_scores['angry'] += 0.1
                emotion_scores['fear'] += 0.1
            elif zcr_mean < 0.15:  # Smoother
                emotion_scores['sad'] += 0.1
                emotion_scores['neutral'] += 0.1
            
            # Tempo analysis (if working properly)
            tempo = features.get('tempo', 120.0)
            
            if tempo > 70:  # Faster tempo
                emotion_scores['happy'] += 0.1
                emotion_scores['angry'] += 0.1
            elif tempo < 65:  # Slower tempo
                emotion_scores['sad'] += 0.1
                emotion_scores['fear'] += 0.1
            
            # Normalize scores
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
            
            # Add neutral bias if no strong emotions detected
            max_score = max(emotion_scores.values())
            if max_score < 0.3:
                emotion_scores['neutral'] += 0.3
                # Renormalize
                total_score = sum(emotion_scores.values())
                emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
            
            return emotion_scores
            
        except Exception as e:
            logger.error(f"Error in emotion analysis: {e}")
            return {'neutral': 1.0, 'calm': 0.0, 'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'fear': 0.0, 'disgust': 0.0, 'surprised': 0.0}
    
    def calculate_confidence(self, features: Dict[str, np.ndarray], emotion_scores: Dict[str, float] = None) -> float:
        """
        Calculate confidence score for the analysis
        
        Args:
            features: Dictionary of extracted features
            emotion_scores: Emotion scores from analysis
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            confidence_factors = []
            
            # Signal quality
            rms_mean = features.get('rms_mean', 0.0)
            if rms_mean > 0.01:  # Good signal strength
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.3)
            
            # Feature completeness
            required_features = ['mel_mean', 'mfcc_mean', 'spectral_centroid_mean', 'zcr_mean', 'rms_mean']
            available_features = sum(1 for f in required_features if f in features)
            confidence_factors.append(available_features / len(required_features))
            
            # Pitch detection quality
            pitch_mean = features.get('pitch_mean', 0.0)
            if 80 <= pitch_mean <= 400:  # Human voice range
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.5)
            
            # Duration quality
            duration = features.get('duration', 0.0)
            if 1.0 <= duration <= 30.0:  # Reasonable duration
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.4)
            
            # Emotion detection clarity (if emotion_scores provided)
            if emotion_scores:
                max_score = max(emotion_scores.values())
                second_max = sorted(emotion_scores.values(), reverse=True)[1] if len(emotion_scores) > 1 else 0
                score_difference = max_score - second_max
                
                # Higher confidence if one emotion is clearly dominant
                if score_difference > 0.3:
                    confidence_factors.append(0.9)
                elif score_difference > 0.2:
                    confidence_factors.append(0.7)
                elif score_difference > 0.1:
                    confidence_factors.append(0.5)
                else:
                    confidence_factors.append(0.3)
                
                # Confidence based on maximum emotion score
                if max_score > 0.5:
                    confidence_factors.append(0.8)
                elif max_score > 0.3:
                    confidence_factors.append(0.6)
                else:
                    confidence_factors.append(0.4)
            
            # Calculate average confidence
            confidence = np.mean(confidence_factors)
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def process_audio(self, audio_path: str) -> Dict:
        """
        Complete audio processing pipeline
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Complete analysis results
        """
        try:
            # Extract features
            features = self.extract_features(audio_path)
            
            # Analyze emotion
            emotion_scores = self.analyze_emotion_from_features(features)
            
            # Get dominant emotion
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            
            # Calculate confidence
            confidence = self.calculate_confidence(features, emotion_scores)
            
            return {
                'emotion': dominant_emotion,
                'emotion_scores': emotion_scores,
                'confidence': confidence,
                'features': features,
                'device_used': self.device
            }
            
        except Exception as e:
            logger.error(f"Error in audio processing pipeline: {e}")
            return {
                'emotion': 'neutral',
                'emotion_scores': {'neutral': 1.0},
                'confidence': 0.0,
                'error': str(e),
                'device_used': self.device
            }


# Convenience function for backward compatibility
def analyze_voice_emotion_torchaudio(audio_path: str) -> Dict:
    """
    Analyze voice emotion using TorchAudio (replacement for librosa version)
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Analysis results
    """
    processor = TorchAudioProcessor()
    return processor.process_audio(audio_path)


if __name__ == "__main__":
    # Test the processor
    import tempfile
    import os
    
    # Create a simple test audio file
    sample_rate = 16000
    duration = 3.0
    frequency = 440.0
    
    t = torch.linspace(0, duration, int(sample_rate * duration))
    test_audio = torch.sin(2 * torch.pi * frequency * t)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        torchaudio.save(f.name, test_audio.unsqueeze(0), sample_rate)
        test_file = f.name
    
    try:
        # Test the processor
        processor = TorchAudioProcessor()
        result = processor.process_audio(test_file)
        
        print("🎵 TorchAudio Processor Test Results:")
        print(f"Device: {result['device_used']}")
        print(f"Emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Emotion Scores: {result['emotion_scores']}")
        
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file) 