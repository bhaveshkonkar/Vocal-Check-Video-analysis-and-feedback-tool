# Vocal-Check-Video-analysis-and-feedback-tool
Vocal Check is an intelligent, real-time voice and video analysis tool designed to help users improve their presentation skills. It provides instant feedback on speech pace, pauses, filler words, body language, eye contact, and more.



🚀 Features

🎤 Voice Analysis:
feedback on pace, pauses, pronunciation, and filler word usage.

🎥 Video Analysis:
Posture, eye contact, and gesture tracking using advanced computer vision.

🧠 AI-Powered Insights:
Smart suggestions for improving presentation delivery based on multi-modal data.

📊 Detailed Feedback Reports:
Summarized visual feedback with actionable tips for each session.




🔧 Tech Stack
JavaScript (Frontend)

Python (Backend)

Gemini API - for analyzing

Speech-to-Text API - Whisper




🌟 Use Cases
Students preparing for presentations or viva.

Professionals refining public speaking skills.

Trainers providing structured feedback.


## Setup

1. Install Python 3.8 or higher
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to `http://localhost:5000`
3. Upload a video file and wait for the analysis to complete

## Directory Structure

- `app.py` - Main Flask application
- `templates/` - HTML templates
- `uploads/` - Temporary storage for uploaded videos
- `audio/` - Extracted audio files
- `transcriptions/` - Transcription and analysis results
- `outputs/` - Body language analysis results

## Notes

- The application supports video files up to 16MB in size
- Processing time depends on the video length and complexity
- Make sure you have a stable internet connection for the AI analysis features 
