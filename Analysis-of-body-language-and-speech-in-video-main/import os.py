import os
import sys
import ctypes
import time
import json
import uuid

from flask import Flask, render_template, request, redirect, url_for, flash

# --- Monkey-patch for Windows if needed ---
if sys.platform == "win32":
    import ctypes.util
    original_find_library = ctypes.util.find_library

    def find_library(libname):
        if libname == "c":
            return "msvcrt.dll"  # For Windows, return the MSVCRT library
        return original_find_library(libname)

    ctypes.util.find_library = find_library

# --- Import heavy libraries ---
import whisper
from moviepy.editor import VideoFileClip

# Import Gemini generative AI (make sure the library is installed)
import google.generativeai as genai

# Optional: remove rich/colorama usage here if output goes to webpage
#from rich.console import Console
#from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
#from rich import print as rprint
#from colorama import init, Fore, Style

# Configure your Gemini API key (replace with your actual key)
GOOGLE_API_KEY = 'AIzaSyB5FFNNqzfBeupJOQx3MnXkyUUBr7tAXeI'  # REPLACE WITH YOUR ACTUAL KEY
genai.configure(api_key=GOOGLE_API_KEY)

# --- Flask app setup ---
app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Define upload folders (all paths are relative to this file's directory)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
VIDEO_FOLDER = os.path.join(UPLOAD_FOLDER, 'videos')
AUDIO_FOLDER = os.path.join(UPLOAD_FOLDER, 'audio')
TRANSCRIPT_FOLDER = os.path.join(UPLOAD_FOLDER, 'transcriptions')
OUTPUT_FOLDER = os.path.join(UPLOAD_FOLDER, 'outputs')

# Ensure all directories exist
for folder in [UPLOAD_FOLDER, VIDEO_FOLDER, AUDIO_FOLDER, TRANSCRIPT_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Pre-load the Whisper model to avoid reloading with every request.
whisper_model = whisper.load_model("base")
# Instantiate the Gemini model (you can reuse this object if needed)
gen_model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

# --- Core Processing Functions ---

def extract_audio(video_path, audio_path):
    """Extracts audio from the video file and saves it as an MP3."""
    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path, logger=None)
    video_clip.close()

def transcribe_audio(audio_path):
    """Transcribes the audio file using Whisper and returns the transcription text."""
    result = whisper_model.transcribe(audio_path)
    return result.get('text', '')

def analyze_transcription(transcription_text):
    """Analyzes the transcription text with Gemini and returns the analysis result."""
    prompt = (
        "Analyze the following text for topic understanding and provide a comprehensive analysis:\n\n"
        f"{transcription_text}"
    )
    response = gen_model.generate_content([prompt], request_options={"timeout": 600})
    return response.text

def analyze_body_language(video_path):
    """Uploads the video for body language analysis and returns the analysis response."""
    prompt_body_language = (
        "Analyze the body language in the provided video, focusing on the following aspects:\n"
        "1. Facial expressions and emotional cues.\n"
        "2. Eye gaze direction and eye contact patterns.\n"
        "3. Head movements and tilts.\n"
        "4. Body posture, e.g., open or closed stance.\n"
        "5. Hand and arm movements.\n"
        "6. Leg positioning and movement.\n"
        "7. Subtle movements such as touching the face or adjusting clothing.\n"
        "8. Micro-expressions and subtle shifts in demeanor.\n"
        "Finally, provide a complete analysis highlighting strengths, weaknesses, and suggestions."
    )
    # Upload the video file for processing
    video_file_obj = genai.upload_file(path=video_path)
    
    # Wait for the file to finish processing
    while video_file_obj.state.name == "PROCESSING":
        time.sleep(10)
        video_file_obj = genai.get_file(video_file_obj.name)
    
    if video_file_obj.state.name == "FAILED":
        raise Exception("Video processing failed for body language analysis.")
    
    response_body = gen_model.generate_content([video_file_obj, prompt_body_language], request_options={"timeout": 600})
    return response_body.text

def process_video(video_filepath):
    """
    Runs the entire video analysis workflow:
    1. Extracts audio.
    2. Transcribes audio.
    3. Analyzes the transcription.
    4. Performs body language analysis.
    
    Returns a dictionary with all the results.
    """
    # Create unique file names based on a uuid
    unique_id = uuid.uuid4().hex
    audio_filename = f"audio_{unique_id}.mp3"
    audio_filepath = os.path.join(AUDIO_FOLDER, audio_filename)
    
    # (Optionally) Use the same base name for transcript and analysis files if you wish to save on disk.
    transcript_txt_filename = f"transcript_{unique_id}.txt"
    transcript_txt_path = os.path.join(TRANSCRIPT_FOLDER, transcript_txt_filename)
    
    analysis_txt_filename = f"analysis_{unique_id}.txt"
    analysis_txt_path = os.path.join(TRANSCRIPT_FOLDER, analysis_txt_filename)
    
    body_analysis_filename = f"body_analysis_{unique_id}.txt"
    body_analysis_path = os.path.join(OUTPUT_FOLDER, body_analysis_filename)
    
    # -----------------------------
    # Step 1: Extract audio from video.
    extract_audio(video_filepath, audio_filepath)
    
    # -----------------------------
    # Step 2: Transcribe extracted audio.
    transcription_text = transcribe_audio(audio_filepath)
    
    # (Optional) Save transcription to disk.
    with open(transcript_txt_path, 'w', encoding='utf-8') as f:
        f.write(transcription_text)
    
    # -----------------------------
    # Step 3: Analyze the transcription text.
    analysis_text = analyze_transcription(transcription_text)
    with open(analysis_txt_path, 'w', encoding='utf-8') as f:
        f.write(analysis_text)
    
    # -----------------------------
    # Step 4: Perform body language analysis.
    body_analysis_text = analyze_body_language(video_filepath)
    with open(body_analysis_path, 'w', encoding='utf-8') as f:
        f.write(body_analysis_text)
    
    # Return a dictionary with all the results.
    return {
        "transcription": transcription_text,
        "analysis": analysis_text,
        "body_analysis": body_analysis_text,
        "audio_file": audio_filepath,
        "transcript_txt": transcript_txt_path,
        "analysis_txt": analysis_txt_path,
        "body_analysis_txt": body_analysis_path
    }

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    if 'video_file' not in request.files:
        flash("No file part in the form.")
        return redirect(request.url)

    file = request.files['video_file']
    if file.filename == '':
        flash("No file selected.")
        return redirect(request.url)

    # Save the uploaded video file in the VIDEO_FOLDER
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    video_filepath = os.path.join(VIDEO_FOLDER, unique_filename)
    file.save(video_filepath)

    try:
        # Run the video processing (this may take a while!)
        results = process_video(video_filepath)
    except Exception as e:
        flash(f"An error occurred during processing: {str(e)}")
        return redirect(url_for('index'))

    # Render the results on a new page
    return render_template('result.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)