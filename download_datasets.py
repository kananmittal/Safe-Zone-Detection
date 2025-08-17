#!/usr/bin/env python3
"""
Dataset Download Script for Voice Distress Detection
Downloads TESS and provides instructions for IEMOCAP
"""

import os
import requests
import zipfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_tess_dataset():
    """Download TESS dataset"""
    logger.info("Downloading TESS dataset...")
    
    tess_dir = Path("datasets/tess")
    tess_dir.mkdir(exist_ok=True)
    
    # TESS dataset URLs (multiple sources)
    tess_urls = [
        "https://tspace.library.utoronto.ca/bitstream/1807/24487/1/TESS%20Toronto%20emotional%20speech%20set%20data.zip",
        "https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess/download?datasetVersionNumber=1"
    ]
    
    for i, url in enumerate(tess_urls):
        try:
            logger.info(f"Trying TESS download from source {i+1}...")
            
            # Download the file
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Check if it's actually a zip file
            content_type = response.headers.get('content-type', '')
            if 'zip' not in content_type and 'application/octet-stream' not in content_type:
                logger.warning(f"Source {i+1} returned {content_type}, not a zip file")
                continue
            
            # Save the file
            zip_path = tess_dir / "tess_data.zip"
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify it's a valid zip file
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tess_dir)
                os.remove(zip_path)  # Remove zip after extraction
                logger.info("✅ TESS dataset downloaded and extracted successfully!")
                return True
            except zipfile.BadZipFile:
                logger.warning(f"Source {i+1} file is not a valid zip file")
                os.remove(zip_path)
                continue
                
        except Exception as e:
            logger.warning(f"Failed to download from source {i+1}: {e}")
            continue
    
    logger.error("❌ Failed to download TESS dataset from all sources")
    return False

def create_iemocap_instructions():
    """Create instructions for IEMOCAP download"""
    logger.info("Creating IEMOCAP download instructions...")
    
    instructions = """
🎯 IEMOCAP Dataset Download Instructions:

📥 Manual Download Required:
1. Visit: https://sail.usc.edu/iemocap/
2. Click "Download IEMOCAP"
3. Fill out the form with your information
4. You'll receive download links via email
5. Download the dataset files

📁 Expected Files:
- IEMOCAP_full_release.zip (or similar)
- Contains 5 sessions with audio and annotations

📂 Extraction Instructions:
1. Extract the downloaded zip file
2. Copy contents to: datasets/iemocap/
3. Expected structure:
   datasets/iemocap/
   ├── Session1/
   ├── Session2/
   ├── Session3/
   ├── Session4/
   └── Session5/

🔧 After Download:
1. Run: python src/data_processor.py
2. This will process IEMOCAP automatically
3. Expected: ~12,000 additional samples

⏰ Time Required: 10-15 minutes for download and setup
"""
    
    # Save instructions to file
    with open("datasets/iemocap/README.md", "w") as f:
        f.write(instructions)
    
    print(instructions)
    return True

def create_synthetic_data_plan():
    """Create plan for synthetic data generation"""
    logger.info("Creating synthetic data generation plan...")
    
    plan = """
🎵 Synthetic Data Generation Plan:

📊 Current Dataset Status:
- RAVDESS: 1,440 samples ✅
- CREMA-D: 7,442 samples ✅
- TESS: Downloading... ⏳
- IEMOCAP: Manual download required ⏳

🎯 Synthetic Data Goals:
- Generate 5,000+ additional samples
- Balance safe vs distress scenarios
- Add cultural and linguistic diversity
- Create edge cases and ambiguous situations

🛠️ Implementation Options:

Option 1: Text-to-Speech Generation
- Use gTTS or pyttsx3
- Generate speech with different emotions
- Mix with background noise
- Expected: 2,000-3,000 samples

Option 2: Audio Augmentation
- Pitch shifting
- Speed variation
- Noise addition
- Reverb effects
- Expected: 1,000-2,000 samples

Option 3: Hybrid Approach
- Combine TTS + augmentation
- Create realistic scenarios
- Expected: 3,000-5,000 samples

📁 Output Structure:
datasets/synthetic/
├── tts_generated/
├── augmented/
└── hybrid/

🚀 Next Steps:
1. Complete TESS and IEMOCAP downloads
2. Run data processor to get baseline
3. Implement synthetic data generation
4. Achieve 15,000+ total samples
"""
    
    with open("datasets/synthetic/README.md", "w") as f:
        f.write(plan)
    
    print(plan)
    return True

def check_current_status():
    """Check current dataset status"""
    logger.info("Checking current dataset status...")
    
    datasets = {
        'ravdess': Path("datasets/ravdess"),
        'crema': Path("datasets/crema"),
        'tess': Path("datasets/tess"),
        'iemocap': Path("datasets/iemocap"),
        'synthetic': Path("datasets/synthetic")
    }
    
    status = {}
    
    for name, path in datasets.items():
        if path.exists():
            # Count audio files
            audio_files = list(path.rglob("*.wav")) + list(path.rglob("*.mp3"))
            status[name] = len(audio_files)
        else:
            status[name] = 0
    
    print("\n📊 Current Dataset Status:")
    print("=" * 40)
    for name, count in status.items():
        status_icon = "✅" if count > 0 else "❌"
        print(f"{status_icon} {name.upper()}: {count} audio files")
    
    total_samples = sum(status.values())
    print(f"\n📈 Total Samples: {total_samples}")
    
    if total_samples >= 8000:
        print("🎉 Sufficient data for fine-tuning!")
    elif total_samples >= 5000:
        print("🟡 Good data, consider adding more for better results")
    else:
        print("🔴 Need more data for effective fine-tuning")
    
    return status

def main():
    """Main function"""
    print("🎯 Dataset Download and Setup")
    print("=" * 40)
    
    # Check current status
    status = check_current_status()
    
    # Download TESS
    if status['tess'] == 0:
        print("\n📥 Downloading TESS dataset...")
        download_tess_dataset()
    
    # Create IEMOCAP instructions
    if status['iemocap'] == 0:
        print("\n📋 Creating IEMOCAP instructions...")
        create_iemocap_instructions()
    
    # Create synthetic data plan
    if status['synthetic'] == 0:
        print("\n🎵 Creating synthetic data plan...")
        create_synthetic_data_plan()
    
    # Final status check
    print("\n📊 Final Status:")
    check_current_status()
    
    print("\n🚀 Next Steps:")
    print("1. Complete TESS download (if needed)")
    print("2. Download IEMOCAP manually")
    print("3. Run: python src/data_processor.py")
    print("4. Start fine-tuning with available data")

if __name__ == "__main__":
    main() 