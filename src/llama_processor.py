#!/usr/bin/env python3
"""
Llama 3 Multi-Modal Processor for Voice Distress Detection
Combines text transcripts with voice features for enhanced analysis
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Llama3Processor:
    def __init__(self, model_name: Optional[str] = None, device: str = "auto"):
        """
        Initialize Llama 3 processor for multi-modal analysis
        
        Args:
            model_name: Hugging Face model name or local checkpoint path. If None, will auto-resolve.
            device: Device to use (auto, mps, cuda, cpu)
        """
        self.model_name = model_name
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("Using MPS (Metal Performance Shaders) for Llama 3")
            elif torch.cuda.is_available():
                self.device = "cuda"
                logger.info("Using CUDA for Llama 3")
            else:
                self.device = "cpu"
                logger.info("Using CPU for Llama 3")
        else:
            self.device = device
            
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _resolve_model_path(self) -> str:
        """
        Resolve which model path/name to load:
        1) LLAMA_MODEL_PATH env var if set and exists
        2) Latest checkpoint under models/fine_tuned_llama_*/checkpoint-*
        3) Fallback to public model 'microsoft/DialoGPT-medium'
        """
        # 1) Env override
        env_path = os.getenv("LLAMA_MODEL_PATH")
        if env_path and os.path.exists(env_path):
            logger.info(f"Using model from LLAMA_MODEL_PATH: {env_path}")
            return env_path

        # 2) Latest local checkpoint
        try:
            models_dir = os.path.join(os.getcwd(), "models")
            if os.path.isdir(models_dir):
                # Find fine-tuned dirs
                candidates = []
                for root, dirs, _ in os.walk(models_dir):
                    for d in dirs:
                        if d.startswith("checkpoint-"):
                            try:
                                step = int(d.split("-", 1)[1])
                            except Exception:
                                step = -1
                            candidates.append((step, os.path.join(root, d)))
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    latest_path = candidates[-1][1]
                    logger.info(f"Using latest local checkpoint: {latest_path}")
                    return latest_path
        except Exception as e:
            logger.warning(f"Could not auto-detect local checkpoint: {e}")

        # 3) Fallback public model
        fallback = "microsoft/DialoGPT-medium"
        logger.info(f"Falling back to public model: {fallback}")
        return fallback
        
    def _load_model(self):
        """Load Llama 3 model with memory optimization"""
        try:
            # Resolve model path/name if not provided
            if not self.model_name:
                self.model_name = self._resolve_model_path()

            logger.info(f"Loading Llama 3 model: {self.model_name}")
            
            # Use simple loading without quantization for compatibility
            # Quantization disabled for compatibility
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load model without quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32
            ).to(self.device)
            
            logger.info("✅ Llama 3 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Llama 3 model: {e}")
            logger.info("Falling back to rule-based analysis")
            self.model = None
            self.tokenizer = None
    
    def _create_prompt(self, transcript: str, voice_features: Dict, emotion_scores: Dict) -> str:
        """
        Create a structured prompt for Llama 3 analysis
        
        Args:
            transcript: Speech-to-text transcript
            voice_features: Extracted voice features
            emotion_scores: Emotion detection scores
            
        Returns:
            Formatted prompt for Llama 3
        """
        # Format voice features for analysis
        feature_summary = []
        if 'pitch_mean' in voice_features:
            feature_summary.append(f"Average pitch: {voice_features['pitch_mean']:.1f} Hz")
        if 'rms_mean' in voice_features:
            feature_summary.append(f"Voice volume: {voice_features['rms_mean']:.3f}")
        if 'tempo' in voice_features:
            feature_summary.append(f"Speech tempo: {voice_features['tempo']:.1f} BPM")
        if 'zcr_mean' in voice_features:
            feature_summary.append(f"Voice clarity: {voice_features['zcr_mean']:.3f}")
            
        # Format emotion scores
        emotion_summary = []
        for emotion, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True):
            if score > 0.1:  # Only include significant emotions
                emotion_summary.append(f"{emotion}: {score:.1%}")
        
        # Create the prompt
        prompt = f"""Analyze this voice distress detection case:

TRANSCRIPT: "{transcript}"

VOICE FEATURES: {', '.join(feature_summary)}

EMOTION ANALYSIS: {', '.join(emotion_summary)}

Based on the transcript, voice characteristics, and emotional indicators, determine if this person is in distress or danger.

Respond with:
DISTRESS_LEVEL: (LOW/MEDIUM/HIGH/CRITICAL)
CONFIDENCE: (0-100%)
REASONING: Brief explanation
SAFETY_ACTION: (NONE/MONITOR/ALERT/EMERGENCY)

Analysis:"""

        return prompt
    
    def _parse_llama_response(self, response: str) -> Dict:
        """
        Parse Llama 3 response into structured format
        
        Args:
            response: Raw response from Llama 3
            
        Returns:
            Parsed analysis results
        """
        try:
            # Extract key information from response
            distress_level = "LOW"
            confidence = 50
            reasoning = "Analysis completed"
            safety_action = "NONE"
            
            # Parse response for structured information
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("DISTRESS_LEVEL:"):
                    level = line.split(":", 1)[1].strip()
                    if level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
                        distress_level = level
                elif line.startswith("CONFIDENCE:"):
                    try:
                        conf_str = line.split(":", 1)[1].strip().replace("%", "")
                        confidence = int(conf_str)
                    except:
                        pass
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
                elif line.startswith("SAFETY_ACTION:"):
                    action = line.split(":", 1)[1].strip()
                    if action in ["NONE", "MONITOR", "ALERT", "EMERGENCY"]:
                        safety_action = action
            
            # If parsing failed, try to extract from the full response
            if distress_level == "LOW" and "DISTRESS_LEVEL:" not in response:
                # Analyze the response content for distress indicators
                response_lower = response.lower()
                if any(word in response_lower for word in ["high", "critical", "emergency", "danger"]):
                    distress_level = "HIGH"
                    confidence = 75
                    reasoning = "High distress indicators detected in analysis"
                    safety_action = "ALERT"
                elif any(word in response_lower for word in ["medium", "moderate", "concern"]):
                    distress_level = "MEDIUM"
                    confidence = 60
                    reasoning = "Moderate distress indicators detected"
                    safety_action = "MONITOR"
                else:
                    distress_level = "LOW"
                    confidence = 80
                    reasoning = "No significant distress indicators"
                    safety_action = "NONE"
            
            return {
                'distress_level': distress_level,
                'confidence': confidence / 100.0,  # Convert to 0-1 scale
                'reasoning': reasoning,
                'safety_action': safety_action,
                'llama_analysis': True
            }
            
        except Exception as e:
            logger.error(f"Error parsing Llama response: {e}")
            return {
                'distress_level': 'LOW',
                'confidence': 0.5,
                'reasoning': 'Error in analysis',
                'safety_action': 'NONE',
                'llama_analysis': False
            }
    
    def analyze_multi_modal(self, transcript: str, voice_features: Dict, emotion_scores: Dict) -> Dict:
        """
        Perform multi-modal analysis using Llama 3
        
        Args:
            transcript: Speech-to-text transcript
            voice_features: Extracted voice features
            emotion_scores: Emotion detection scores
            
        Returns:
            Multi-modal analysis results
        """
        try:
            if self.model is None or self.tokenizer is None:
                logger.warning("Llama 3 model not available, using fallback analysis")
                return self._fallback_analysis(transcript, voice_features, emotion_scores)
            
            # Create prompt
            prompt = self._create_prompt(transcript, voice_features, emotion_scores)
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated response
            assistant_response = response
            
            # Parse response
            result = self._parse_llama_response(assistant_response)
            
            logger.info(f"Llama 3 analysis: {result['distress_level']} (confidence: {result['confidence']:.1%})")
            return result
            
        except Exception as e:
            logger.error(f"Error in Llama 3 analysis: {e}")
            return self._fallback_analysis(transcript, voice_features, emotion_scores)
    
    def _fallback_analysis(self, transcript: str, voice_features: Dict, emotion_scores: Dict) -> Dict:
        """
        Fallback analysis when Llama 3 is not available
        
        Args:
            transcript: Speech-to-text transcript
            voice_features: Extracted voice features
            emotion_scores: Emotion detection scores
            
        Returns:
            Fallback analysis results
        """
        # Simple rule-based analysis combining text and voice
        distress_keywords = [
            "help", "emergency", "danger", "scared", "fear", "threat", "unsafe",
            "panic", "terrified", "afraid", "worried", "anxious", "distress",
            "crying", "screaming", "pain", "hurt", "attack", "robbery"
        ]
        
        # Check text content
        text_distress = any(keyword in transcript.lower() for keyword in distress_keywords)
        
        # Check voice emotions
        high_distress_emotions = ['fear', 'angry', 'disgust']
        voice_distress = any(emotion in high_distress_emotions and score > 0.3 
                           for emotion, score in emotion_scores.items())
        
        # Determine overall distress level
        if text_distress and voice_distress:
            distress_level = "HIGH"
            confidence = 0.85
            reasoning = "Distress detected in both content and voice tone"
            safety_action = "ALERT"
        elif text_distress or voice_distress:
            distress_level = "MEDIUM"
            confidence = 0.70
            reasoning = "Distress detected in content or voice tone"
            safety_action = "MONITOR"
        else:
            distress_level = "LOW"
            confidence = 0.90
            reasoning = "No distress indicators detected"
            safety_action = "NONE"
        
        return {
            'distress_level': distress_level,
            'confidence': confidence,
            'reasoning': reasoning,
            'safety_action': safety_action,
            'llama_analysis': False
        }

# Convenience function for easy integration
def analyze_multi_modal_distress(transcript: str, voice_features: Dict, emotion_scores: Dict) -> Dict:
    """
    Convenience function for multi-modal distress analysis
    
    Args:
        transcript: Speech-to-text transcript
        voice_features: Extracted voice features
        emotion_scores: Emotion detection scores
        
    Returns:
        Multi-modal analysis results
    """
    processor = Llama3Processor()
    return processor.analyze_multi_modal(transcript, voice_features, emotion_scores)

if __name__ == "__main__":
    # Test the Llama 3 processor
    test_transcript = "I'm feeling really scared right now"
    test_features = {
        'pitch_mean': 180.0,
        'rms_mean': 0.08,
        'tempo': 120.0,
        'zcr_mean': 0.25
    }
    test_emotions = {
        'neutral': 0.1,
        'calm': 0.0,
        'happy': 0.0,
        'sad': 0.2,
        'angry': 0.1,
        'fear': 0.6,
        'disgust': 0.0,
        'surprised': 0.0
    }
    
    result = analyze_multi_modal_distress(test_transcript, test_features, test_emotions)
    print("🎯 Multi-Modal Analysis Test:")
    print(f"Distress Level: {result['distress_level']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Reasoning: {result['reasoning']}")
    print(f"Safety Action: {result['safety_action']}")
    print(f"Llama Analysis: {result['llama_analysis']}") 