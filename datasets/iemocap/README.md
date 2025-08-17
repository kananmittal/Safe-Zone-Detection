
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
