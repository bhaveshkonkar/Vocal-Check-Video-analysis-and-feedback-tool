# Video Analysis Tool

This application provides a web interface for analyzing videos, including transcription, content analysis, and body language analysis.

## Features

- Video upload and processing
- Automatic transcription using Whisper
- Content analysis using Google's Gemini AI
- Body language analysis
- Modern web interface

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