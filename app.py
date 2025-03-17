from fastapi import FastAPI, File, UploadFile, Form
import shutil
import os
import torch
import whisper
import Levenshtein
import librosa
from moviepy.editor import VideoFileClip
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from fastapi.middleware.cors import CORSMiddleware
import nltk
import syllapy
from nltk.corpus import words
from metaphone import doublemetaphone
import dlib
from model import LipCoordNet
from inference import load_video, generate_lip_coordinates, ctc_decode

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
whisper_model = whisper.load_model("base")
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

# Load the lip-reading model
device = "cuda" if torch.cuda.is_available() else "cpu"
lip_model = LipCoordNet()
lip_model.load_state_dict(torch.load("pretrain/LipCoordNet_coords_loss_0.025581153109669685_wer_0.01746208431890914_cer_0.006488426950253695.pt", map_location=device, weights_only=False))
lip_model = lip_model.to(device)
lip_model.eval()

# Load dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("lip_coordinate_extraction/shape_predictor_68_face_landmarks_GTX.dat")

# Download word corpus if not already downloaded
nltk.download("words")

# Dolch sight words for preschoolers and kindergarteners (child-friendly words)
dolch_words = set([
    # Common Nouns
    "cat", "dog", "ball", "car", "sun", "tree", "run", "jump", "happy", "big", 
    "red", "blue", "apple", "bird", "milk", "baby", "mom", "dad", "book", "fish",
    "bed", "chair", "door", "house", "mouse", "water", "flower", "toy", "rain", "star", 
    "hat", "bat", "cup", "boat", "train", "bus", "egg", "moon", "shoes", "sock", "leaf",

    # Basic Verbs (Actions)
    "go", "come", "play", "see", "run", "jump", "eat", "sleep", "drink", "read",
    "sit", "stand", "walk", "sing", "dance", "clap", "cry", "laugh", "look", "write",

    # Adjectives (Descriptive Words)
    "big", "small", "fast", "slow", "happy", "sad", "red", "blue", "yellow", "green",
    "hot", "cold", "soft", "hard", "clean", "dirty", "loud", "quiet", "light", "dark",

    # Other Useful Words
    "yes", "no", "up", "down", "in", "out", "here", "there", "this", "that",
    "where", "who", "what", "why", "when", "how", "thank", "please", "good", "bad"
])

word_list = set(words.words())

# Define paths
TEMP_DIR = "temp_videos"
os.makedirs(TEMP_DIR, exist_ok=True)

def clean_temp_files():
    """Ensures all files in the temporary directory are deleted."""
    for file in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, file)
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def clean_temp_dir():
    """Deletes and recreates the temporary directory to ensure all files are removed."""
    try:
        shutil.rmtree(TEMP_DIR)  # Remove the entire folder
        os.makedirs(TEMP_DIR, exist_ok=True)  # Recreate it
        print("Temporary directory cleared and recreated.")
    except Exception as e:
        print(f"Error clearing temp directory: {e}")

# Extract audio from video
def extract_audio(video_path):
    audio_path = os.path.join(TEMP_DIR, "current_audio.wav")
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
    return audio_path

# Whisper transcription
def whisper_transcription(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result['text']

# Wav2Vec2 phoneme extraction
def extract_wav2vec_text(audio_path):
    speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
    inputs = wav2vec_processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = wav2vec_model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    wav2vec_text = wav2vec_processor.batch_decode(predicted_ids)
    return wav2vec_text[0]


# Function to calculate accuracy using Levenshtein distance
def calculate_accuracy(actual_text, predicted_text):
    return Levenshtein.ratio(actual_text.lower().strip(), predicted_text.lower().strip()) * 100

def suggest_similar_words(target_word, num_suggestions=5):
    """Suggests child-friendly words with similar pronunciation using Metaphone encoding."""
    target_encoding = doublemetaphone(target_word)[0]  # Get primary Metaphone encoding
    
    # Find words with the same Metaphone encoding
    similar_words = [
        word for word in dolch_words  # Use predefined Dolch words list
        if doublemetaphone(word)[0] == target_encoding and word != target_word
        and len(word) <= 6  # Limit word length for young children
        and syllapy.count(word) <= 2  # Ensure easy pronunciation
    ]
    
    # Return up to `num_suggestions` words
    return similar_words[:num_suggestions]


@app.post("/predict")
async def predict(
    video: UploadFile = File(...), 
    expected_text: str = Form(...), 
    mode: str = Form(...)
):
    """
    mode: "word" or "sentence"
    - "word": Uses audio models only (Whisper + Wav2Vec2)
    - "sentence": Uses audio + lip-reading model
    """

    # Clean previous files
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Save new uploaded video
    video_path = os.path.join(TEMP_DIR, "current_video.mp4")
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Process audio and text predictions
    audio_path = extract_audio(video_path)
    whisper_text = whisper_transcription(audio_path)
    wav2vec_text = extract_wav2vec_text(audio_path)

    lip_reading_text = None
    lip_reading_accuracy = None

    # If mode is "sentence", use lip-reading model
    if mode.lower() == "sentence":
        video_tensor = load_video(video_path, device)
        coords_tensor = generate_lip_coordinates("samples", detector, predictor)

        with torch.no_grad():
            pred = lip_model(video_tensor[None, ...].to(device), coords_tensor[None, ...].to(device))

        lip_reading_text = ctc_decode(pred[0])[-1]  # Get final output text
        lip_reading_accuracy = calculate_accuracy(expected_text, lip_reading_text)

    # Calculate accuracy for audio models
    whisper_accuracy = calculate_accuracy(expected_text, whisper_text)
    wav2vec_accuracy = calculate_accuracy(expected_text, wav2vec_text)

    # Calculate overall accuracy
    accuracies = [whisper_accuracy, wav2vec_accuracy]
    if lip_reading_text:  # Include lip-reading accuracy if used
        accuracies.append(lip_reading_accuracy)

    overall_accuracy = sum(accuracies) / len(accuracies)

    feedback = "Excellent performance!" if overall_accuracy > 85 else "Good job! Keep improving!" if overall_accuracy > 70 else "Needs more practice!"

    return {
        "expected_text": expected_text,
        "mode": mode,
        "whisper_text": whisper_text,
        "whisper_accuracy": whisper_accuracy,
        "wav2vec_text": wav2vec_text,
        "wav2vec_accuracy": wav2vec_accuracy,
        "lip_reading_text": lip_reading_text if lip_reading_text else wav2vec_text,
        "lip_reading_accuracy": lip_reading_accuracy if lip_reading_text else wav2vec_accuracy,
        "overall_accuracy": overall_accuracy,
        "feedback": feedback
    }
   