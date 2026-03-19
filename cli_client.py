#!/usr/bin/env python3
import argparse
import requests
import json
import sys
from pathlib import Path


def predict_emotion(audio_file, api_url="http://localhost:8000"):
    url = f"{api_url}/predict-emotion"
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': (Path(audio_file).name, f, 'audio/wav')}
            response = requests.post(url, files=files)
        if response.status_code == 200:
            return response.json()
        return {"error": f"HTTP {response.status_code}: {response.text}"}
    except FileNotFoundError:
        return {"error": f"File not found: {audio_file}"}
    except Exception as e:
        return {"error": str(e)}


def voice_check(audio_file, api_url="http://localhost:8000"):
    url = f"{api_url}/voice-check"
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': (Path(audio_file).name, f, 'audio/wav')}
            response = requests.post(url, files=files)
        if response.status_code == 200:
            return response.json()
        return {"error": f"HTTP {response.status_code}: {response.text}"}
    except FileNotFoundError:
        return {"error": f"File not found: {audio_file}"}
    except Exception as e:
        return {"error": str(e)}


def check_health(api_url="http://localhost:8000"):
    try:
        response = requests.get(f"{api_url}/healthz", timeout=5)
        return response.status_code == 200
    except:
        return False


def print_result(result, is_voice_check=False, json_output=False):
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    if json_output:
        print(json.dumps(result, indent=2))
        return
    
    if is_voice_check:
        print("\n--- Voice Check Results ---")
        print(f"Transcript: {result.get('transcript', 'N/A')}")
        distress = result.get('distress', False)
        print(f"Status: {'DISTRESS DETECTED' if distress else 'SAFE'}")
        
        voice_emotion = result.get('voice_emotion', {})
        if voice_emotion:
            emotion = voice_emotion.get('emotion', 'unknown')
            confidence = voice_emotion.get('confidence', 0.0)
            print(f"Emotion: {emotion} ({confidence*100:.1f}% confidence)")
        
        multi_modal = result.get('multi_modal', {})
        if multi_modal:
            print(f"Analysis: {'Llama 3' if multi_modal.get('llama_used') else 'Rule-based'}")
            print(f"Safety Action: {multi_modal.get('safety_action', 'NONE')}")
        
        print(f"Assessment: {result.get('label', 'N/A')}")
    else:
        print("\n--- Emotion Prediction ---")
        print(f"Emotion: {result.get('emotion', 'unknown')}")
        if 'emotion_scores' in result:
            print("Scores:")
            for emotion, score in sorted(result['emotion_scores'].items(), 
                                       key=lambda x: x[1], reverse=True):
                print(f"  {emotion}: {score*100:.1f}%")
        if 'device_used' in result:
            print(f"Device: {result['device_used']}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call the emotion detection API")
    parser.add_argument('audio_file', nargs='?', help='Audio file to analyze')
    parser.add_argument('--emotion', action='store_true', help='Use emotion endpoint')
    parser.add_argument('--voice-check', action='store_true', help='Use voice-check endpoint')
    parser.add_argument('--health', action='store_true', help='Check if API is running')
    parser.add_argument('--url', default='http://localhost:8000', help='API URL')
    parser.add_argument('--json', action='store_true', help='Output JSON')
    args = parser.parse_args()
    
    if args.health:
        if check_health(args.url):
            print("API is running!")
            sys.exit(0)
        else:
            print("API is not responding")
            sys.exit(1)
    
    if not args.audio_file:
        parser.error("Need audio file")
    
    if not Path(args.audio_file).exists():
        print(f"File not found: {args.audio_file}")
        sys.exit(1)
    
    if not args.emotion and not args.voice_check:
        args.voice_check = True
    
    if args.emotion:
        result = predict_emotion(args.audio_file, args.url)
        print_result(result, False, args.json)
    else:
        result = voice_check(args.audio_file, args.url)
        print_result(result, True, args.json)
    
    if "error" in result:
        sys.exit(1)

