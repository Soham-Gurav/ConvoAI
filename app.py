#Voice Agent Imports
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO
import google.generativeai as genai
import os
from datetime import datetime
from gtts import gTTS
from system_prompts import SYSTEM_PROMPTS
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, load_index_from_storage, StorageContext
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings

#Call Analyser Imports
from pydub import AudioSegment
import speech_recognition as sr
from textblob import TextBlob
import nltk
from collections import defaultdict
import wave
import tempfile
import os
from collections import defaultdict
import subprocess

app = Flask(__name__)
socketio = SocketIO(app)

#Download NLTK data (only needed once)
#nltk.download('punkt')

# Configure Gemini
API_KEY = ""        # Replace with your actual API key
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

os.environ["GOOGLE_API_KEY"] = API_KEY
Settings.llm = None # Disable OpenAI model usage

# Ensure the static audio folder exists
AUDIO_DIR = "static/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Global variables
current_session_folder = None
current_kb_folder = None
query_engine = None

END_CALL_PHRASES = [
    "thanks for helping", "no more questions", "goodbye", "that's all", "i'm done", "end call"
]

def get_next_session_number():
    """Get the next available session number."""
    existing_sessions = [
        int(folder.replace("session", "")) 
        for folder in os.listdir(AUDIO_DIR) 
        if folder.startswith("session") and folder.replace("session", "").isdigit()
    ]
    return max(existing_sessions, default=0) + 1

def create_session_folder():
    """Ensure a single session folder is used throughout a call."""
    global current_session_folder
    if not current_session_folder:
        session_number = get_next_session_number()
        session_folder = os.path.join(AUDIO_DIR, f"session{session_number}")
        os.makedirs(session_folder, exist_ok=True)
        current_session_folder = session_folder
        print(f"🔵 New session folder created: {current_session_folder}")
    return current_session_folder

@app.route('/static/docs/<path:filename>')
def serve_docs(filename):
    return send_from_directory(os.path.join("static", "docs"), filename)

@app.route('/static/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)

@app.route("/")
def home():
    return render_template("index.html")

@socketio.on("start_call")
def handle_start_call():
    """Start a new call session."""
    global current_session_folder
    current_session_folder = create_session_folder()
    socketio.emit("session_started", {"session": current_session_folder})
    print(f"✅ Call started in session: {current_session_folder}")

def create_or_load_index(kb_folder):
    """Creates a LlamaIndex vector database from PDFs or loads an existing one."""
    global query_engine
    base_path = os.path.join("static", "knowledge_bases", kb_folder)

    # Ensure embedding model is set up correctly
    embed_model = GoogleGenAIEmbedding(model_name="models/embedding-001")
    Settings.embed_model = embed_model

    # Check if the storage directory exists
    if os.path.exists(base_path) and os.listdir(base_path):  # Ensures folder exists and is not empty
        print(f"💽 Loading existing knowledge base: {kb_folder}")
        storage_context = StorageContext.from_defaults(persist_dir=base_path)
        index = load_index_from_storage(storage_context)
    else:
        print(f"📂 Creating new knowledge base: {kb_folder}")
        os.makedirs(base_path, exist_ok=True)

        # Ensure document path exists
        doc_path = os.path.join("static", "docs", kb_folder)
        if not os.path.exists(doc_path) or not os.listdir(doc_path):  # Check if docs folder is empty
            print("⚠️ No valid documents found. Cannot create knowledge base.")
            return

        # Load documents and create a new index
        documents = SimpleDirectoryReader(doc_path).load_data()
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        index.storage_context.persist(persist_dir=base_path)

    # Set the query engine globally
    query_engine = index.as_query_engine()
    print("✅ Knowledge base loaded successfully!")


@app.route("/set_knowledge_base", methods=["POST"])
def set_knowledge_base():
    """Set the active knowledge base."""
    global current_kb_folder
    data = request.json
    kb_folder = data.get("kb_folder")
    if not kb_folder:
        return jsonify({"error": "No knowledge base folder provided"}), 400
    
    current_kb_folder = kb_folder
    create_or_load_index(kb_folder)
    return jsonify({"message": f"Knowledge base set to {kb_folder}"})

def handle_general_query(query):
    """Handle general queries like greetings."""
    general_keywords = ["hello", "hi", "how are you", "what is your name"]
    query_lower = query.lower()
    
    for keyword in general_keywords:
        if keyword in query_lower:
            prompt = f"You are a friendly AI assistant. Respond to the user in a conversational manner. User: {query}"
            response = model.generate_content(prompt)
            return response.text
    return None

@socketio.on("user_message")
def handle_message(data):
    global current_session_folder, query_engine, current_kb_folder
    user_input = data["message"].strip().lower()
    mode = data["mode"]  # "text" or "call"
    system_prompt = data.get("systemPrompt", "default")

    if not user_input:
        print("❌ Empty user input received. Skipping AI response.")
        return

    if any(phrase in user_input for phrase in END_CALL_PHRASES):
        socketio.emit("call_ended", {"message": "Call has been ended by the user."})
        print("✅ Call ended by user request.")
        return

    if not current_session_folder:
        current_session_folder = create_session_folder()

    # Default mode does not require a knowledge base
    if system_prompt == "default":
        query_engine = None
    elif current_kb_folder:
        if query_engine is None:
            create_or_load_index(current_kb_folder)
    else:
        ai_response = "Please select a knowledge base before proceeding."
        socketio.emit("ai_response", {"message": ai_response, "mode": mode})
        return

    prompt = f"{SYSTEM_PROMPTS[system_prompt]}\n\nUser Query: {user_input}\n\nProvide a response:"
    response = model.generate_content(prompt)
    ai_response = response.text

    save_conversation(user_input, ai_response, mode)
    
    if mode == "call":
        audio_file = text_to_speech(ai_response)
        if audio_file:
            socketio.emit("ai_response", {
                "message": ai_response, 
                "audio_file": f"/static/{audio_file}",
                "mode": mode
            })
    else:
        socketio.emit("ai_response", {"message": ai_response, "mode": mode})

def save_conversation(user_input, ai_response, mode):
    """Save conversation history."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("conversation.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] User: {user_input}\n")
        f.write(f"[{timestamp}] AI: {ai_response}\n\n")

@app.route("/save_user_audio", methods=["POST"])
def save_user_audio():
    global current_session_folder
    if not current_session_folder:
        print("❌ No active session folder.")
        return "No active session folder", 400

    if "audio" not in request.files:
        print("❌ No audio file found in request.")
        return "No audio file provided", 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        print("❌ No selected file.")
        return "No selected file", 400

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as WAV first for reliability
        temp_wav_path = os.path.join(current_session_folder, f"temp_{timestamp}.wav")
        final_mp3_path = os.path.join(current_session_folder, f"user_audio_{timestamp}.mp3")
        
        # 1. First save as WAV
        audio_file.save(temp_wav_path)
        
        # 2. Convert to MP3 with proper settings
        subprocess.run([
            "ffmpeg",
            "-y",  # Overwrite if exists
            "-i", temp_wav_path,
            "-acodec", "libmp3lame",  # Use standard MP3 codec
            "-q:a", "2",  # Quality level (0-9, 2 is good)
            "-ar", "44100",  # Standard sample rate
            "-ac", "1",  # Mono channel
            final_mp3_path
        ], check=True)
        
        # 3. Clean up temporary WAV
        os.remove(temp_wav_path)
        
        print(f"✅ User audio saved: {final_mp3_path}")
        return "User audio saved successfully", 200
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg conversion failed: {e}")
        # Try to save the original as fallback
        try:
            audio_file.save(final_mp3_path)
            print(f"⚠️ Saved original audio (not converted): {final_mp3_path}")
            return "Audio saved (not converted)", 200
        except Exception as fallback_e:
            print(f"❌ Fallback save failed: {fallback_e}")
            return "Error saving audio", 500
            
    except Exception as e:
        print(f"❌ Error saving user audio: {e}")
        # Clean up if anything was partially created
        if 'temp_wav_path' in locals() and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except:
                pass
        return "Error saving audio", 500
    
def text_to_speech(text):
    """Convert text to speech and save audio file."""
    global current_session_folder
    if not current_session_folder:
        print("❌ No active session folder. Cannot save AI audio.")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file = os.path.join(current_session_folder, f"ai_audio_{timestamp}.mp3")
    try:
        tts = gTTS(text=text, lang="en")
        tts.save(audio_file)
        print(f"✅ AI audio saved: {audio_file}")
        return audio_file.replace("static/", "")
    except Exception as e:
        print(f"❌ Error generating AI audio: {e}")
        return None
    
@app.route("/analyze_call", methods=["POST"])
def analyze_call():
    """Analyze the call recordings in the current session folder."""
    global current_session_folder
    
    if not current_session_folder:
        return jsonify({"error": "No active session to analyze"}), 400
    
    try:
        # Get all audio files in the session folder
        try:
            audio_files = sorted(
                [f for f in os.listdir(current_session_folder) if f.endswith('.mp3')],
                key=lambda x: os.path.getctime(os.path.join(current_session_folder, x))
            )
        except Exception as e:
            print(f"Error listing audio files: {e}")
            return jsonify({"error": "Could not access audio files"}), 500
        
        if not audio_files:
            return jsonify({"error": "No audio files found in session"}), 400
        
        # Initialize analysis results
        analysis_results = {
            "transcript": [],
            "sentiment": {
                "positive": 0,
                "neutral": 0,
                "negative": 0,
                "overall_sentiment": 0
            },
            "speaker_times": {
                "user": 0,
                "ai": 0
            },
            "word_counts": {
                "user": 0,
                "ai": 0
            },
            "topics": defaultdict(int),
            "errors": [],
            "warnings": []
        }
        
        # Initialize speech recognizer
        recognizer = sr.Recognizer()
        
        for audio_file in audio_files:
            file_path = os.path.join(current_session_folder, audio_file)
            speaker = "user" if "user_audio" in audio_file else "ai"
            
            try:
                # First try direct pydub processing
                try:
                    audio = AudioSegment.from_mp3(file_path)
                    
                    # Validate audio properties
                    if audio.channels == 0 or audio.frame_rate == 0:
                        raise Exception("Invalid audio properties (0 channels or 0 frame rate)")
                    
                    # Create temporary WAV file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                        temp_wav_path = temp_wav.name
                    
                    # Export with explicit parameters
                    audio.set_frame_rate(16000).set_channels(1).export(
                        temp_wav_path, 
                        format="wav",
                        parameters=["-ar", "16000", "-ac", "1"]
                    )
                    
                    # Get duration
                    duration = len(audio) / 1000  # Convert to seconds
                    analysis_results["speaker_times"][speaker] += duration
                    
                    # Perform speech recognition
                    with sr.AudioFile(temp_wav_path) as source:
                        audio_data = recognizer.record(source)
                        try:
                            text = recognizer.recognize_google(audio_data)
                            
                            # Add to transcript
                            analysis_results["transcript"].append({
                                "speaker": speaker,
                                "text": text,
                                "time": duration
                            })
                            
                            # Sentiment analysis
                            blob = TextBlob(text)
                            sentiment = blob.sentiment.polarity
                            
                            if sentiment > 0.1:
                                analysis_results["sentiment"]["positive"] += 1
                            elif sentiment < -0.1:
                                analysis_results["sentiment"]["negative"] += 1
                            else:
                                analysis_results["sentiment"]["neutral"] += 1
                            analysis_results["sentiment"]["overall_sentiment"] += sentiment
                            
                            # Word count
                            analysis_results["word_counts"][speaker] += len(text.split())
                            
                            # Topic detection
                            words = [word.lower() for word in text.split() if word.isalpha()]
                            for word in words:
                                if len(word) > 4:
                                    analysis_results["topics"][word] += 1
                            
                        except sr.UnknownValueError:
                            analysis_results["warnings"].append(f"Could not understand audio in {audio_file}")
                        except sr.RequestError as e:
                            analysis_results["warnings"].append(f"Speech recognition error for {audio_file}: {e}")
                
                except Exception as e:
                    # Fallback to FFmpeg with explicit parameters
                    analysis_results["warnings"].append(f"Pydub failed for {audio_file}, trying FFmpeg directly: {str(e)}")
                    
                    try:
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                            temp_wav_path = temp_wav.name
                        
                        # Use FFmpeg with explicit parameters
                        result = subprocess.run([
                            "ffmpeg",
                            "-y",  # Overwrite output file without asking
                            "-i", file_path,
                            "-ac", "1",  # Force mono channel
                            "-ar", "16000",  # Force sample rate
                            "-acodec", "pcm_s16le",  # Force PCM encoding
                            "-hide_banner",  # Suppress FFmpeg banner
                            "-loglevel", "error",  # Only show errors
                            temp_wav_path
                        ], capture_output=True, text=True)
                        
                        if result.returncode != 0:
                            raise Exception(f"FFmpeg failed: {result.stderr}")
                        
                        # Get duration using wave module
                        with wave.open(temp_wav_path, 'rb') as wav_file:
                            frames = wav_file.getnframes()
                            rate = wav_file.getframerate()
                            duration = frames / float(rate)
                            analysis_results["speaker_times"][speaker] += duration
                        
                        # Perform speech recognition
                        with sr.AudioFile(temp_wav_path) as source:
                            audio_data = recognizer.record(source)
                            try:
                                text = recognizer.recognize_google(audio_data)
                                
                                # Add to transcript
                                analysis_results["transcript"].append({
                                    "speaker": speaker,
                                    "text": text,
                                    "time": duration
                                })
                                
                                # Sentiment analysis
                                blob = TextBlob(text)
                                sentiment = blob.sentiment.polarity
                                
                                if sentiment > 0.1:
                                    analysis_results["sentiment"]["positive"] += 1
                                elif sentiment < -0.1:
                                    analysis_results["sentiment"]["negative"] += 1
                                else:
                                    analysis_results["sentiment"]["neutral"] += 1
                                analysis_results["sentiment"]["overall_sentiment"] += sentiment
                                
                                # Word count
                                analysis_results["word_counts"][speaker] += len(text.split())
                                
                                # Topic detection
                                words = [word.lower() for word in text.split() if word.isalpha()]
                                for word in words:
                                    if len(word) > 4:
                                        analysis_results["topics"][word] += 1
                                
                            except sr.UnknownValueError:
                                analysis_results["warnings"].append(f"Could not understand audio in {audio_file}")
                            except sr.RequestError as e:
                                analysis_results["warnings"].append(f"Speech recognition error for {audio_file}: {e}")
                    
                    except Exception as ffmpeg_e:
                        analysis_results["errors"].append(f"Failed to process {audio_file} with FFmpeg: {str(ffmpeg_e)}")
                        continue
                
                finally:
                    # Clean up temporary file if it exists
                    try:
                        if 'temp_wav_path' in locals() and os.path.exists(temp_wav_path):
                            os.unlink(temp_wav_path)
                    except Exception as clean_e:
                        analysis_results["warnings"].append(f"Could not clean up temp file: {str(clean_e)}")
            
            except Exception as outer_e:
                analysis_results["errors"].append(f"Unexpected error processing {audio_file}: {str(outer_e)}")
                continue
        
        # Calculate averages
        total_turns = len(analysis_results["transcript"])
        if total_turns > 0:
            analysis_results["sentiment"]["overall_sentiment"] /= total_turns
        
        # Get top 5 topics
        try:
            analysis_results["topics"] = dict(
                sorted(analysis_results["topics"].items(), 
                key=lambda item: item[1], 
                reverse=True
            ))[:5]
        except Exception as e:
            analysis_results["warnings"].append(f"Error sorting topics: {e}")
            analysis_results["topics"] = {}
        
        return jsonify(analysis_results)
    
    except Exception as e:
        error_msg = f"Error analyzing call: {str(e)}"
        print(error_msg)
        return jsonify({"error": error_msg}), 500
    
if __name__ == "__main__":
    socketio.run(app, debug=True, port=5001)
    
