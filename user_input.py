import google.generativeai as genai
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr
import io
import threading
import time
from flask import Flask, jsonify, request
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from llama_api_client import LlamaAPIClient

app = Flask(__name__)

# Global variables
conversation_active = False
conversation_thread = None
current_status = "idle"
last_response = ""

# Vector database variables
chunks_data = None
embeddings_model = None
chunk_embeddings = None

# Configure Llama API Client
client = LlamaAPIClient(
    api_key="API_KEY",
    base_url="https://api.llama.com/v1/",
)

def load_vector_database():
    """Load the vector database and initialize the embedding model"""
    global chunks_data, embeddings_model, chunk_embeddings
    
    try:
        print("🔄 Loading vector database...")
        
        # Load chunks data
        chunks_data = np.load('chunks.npy', allow_pickle=True)
        print(f"✅ Loaded {len(chunks_data)} chunks from database")
        
        # Initialize sentence transformer model for embeddings
        print("🔄 Loading embedding model...")
        embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Pre-compute embeddings for all chunks (this might take a moment)
        print("🔄 Computing chunk embeddings...")
        chunk_texts = [str(chunk) for chunk in chunks_data]
        chunk_embeddings = embeddings_model.encode(chunk_texts)
        
        print("✅ Vector database loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading vector database: {e}")
        return False

def find_relevant_context(query, top_k=3):
    """Find the most relevant chunks for a given query"""
    global chunks_data, embeddings_model, chunk_embeddings
    
    if chunks_data is None or embeddings_model is None or chunk_embeddings is None:
        return []
    
    try:
        # Encode the query
        query_embedding = embeddings_model.encode([query])
        
        # Calculate cosine similarity between query and all chunks
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        
        # Get top_k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_chunks = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                relevant_chunks.append({
                    'content': str(chunks_data[idx]),
                    'similarity': float(similarities[idx])
                })
        
        return relevant_chunks
        
    except Exception as e:
        print(f"❌ Error finding relevant context: {e}")
        return []

def get_enhanced_response(user_input):
    """Get response from Llama with relevant context from vector database"""
    try:
        # Find relevant context
        relevant_chunks = find_relevant_context(user_input)
        
        # Prepare the prompt with context
        context_text = ""
        if relevant_chunks:
            context_text = "\n\nRelevant context from knowledge base:\n"
            for i, chunk in enumerate(relevant_chunks, 1):
                context_text += f"{i}. {chunk['content'][:200]}...\n"
        
        # Create enhanced prompt
        enhanced_prompt = f"""User question: {user_input}
        
{context_text}

Instructions: Use the context above to provide a helpful, accurate response. Keep it short and simple - 1 sentence max for each important point. Don't be wordy! You are an expert helper who makes users feel better, stays calm, and provides assistance. If the context is relevant, use it; if not, provide general helpful guidance."""

        # Get response from Llama
        response = client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {"role": "user", "content": enhanced_prompt},
            ],
        )
        
        return response.completion_message.content.text.strip()
        
    except Exception as e:
        print(f"❌ Error getting enhanced response: {e}")
        # Fallback to original method
        response = client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {"role": "user", "content": f"{user_input} Dont make it wordy have it short and simple and to the key points like 1 sentence max for each of the important stuff DONT MAKE IT WORDY! AND you are an expert basically someone who is there to help, make the user feel better talk with them make sure are they ok, be calming and HELP!"},
            ],
        )
        return response.completion_message.content.text.strip()

def voice_assistant():
    global conversation_active, current_status, last_response
    
    # Set up speech recognizer
    recognizer = sr.Recognizer()
    print("🧠 Llama voice assistant started with vector database. Say 'thank you' to stop.")
    
    while conversation_active:
        try:
            current_status = "listening"
            with sr.Microphone() as source:
                print("\n🎤 Calibrating mic...")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("🎤 Listening...")
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=25)

            current_status = "processing"
            user_input = recognizer.recognize_google(audio)
            print("🗣️ You:", user_input)
            
            # Exit condition
            if user_input.lower().strip() == "thank you":
                print("👋 Ending conversation. Goodbye!")
                goodbye_tts = gTTS(text="You're welcome! Goodbye!", lang='en')
                fp = io.BytesIO()
                goodbye_tts.write_to_fp(fp)
                fp.seek(0)
                audio = AudioSegment.from_file(fp, format="mp3")
                play(audio)
                conversation_active = False
                current_status = "idle"
                break
            
            # Get enhanced response with vector database context
            llama_text = get_enhanced_response(user_input)
            print("🤖 Llama:", llama_text)
            last_response = llama_text
            
            # Convert Llama's text to speech and play
            current_status = "speaking"
            tts = gTTS(text=llama_text, lang='en')
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            audio = AudioSegment.from_file(fp, format="mp3")
            play(audio)
            
        except sr.WaitTimeoutError:
            # Continue listening if no speech detected
            continue
        except sr.UnknownValueError:
            print("❌ Could not understand audio")
            error_tts = gTTS(text="Sorry, I didn't catch that. Try again.", lang='en')
            fp = io.BytesIO()
            error_tts.write_to_fp(fp)
            fp.seek(0)
            audio = AudioSegment.from_file(fp, format="mp3")
            play(audio)
        except Exception as e:
            print("❌ Error:", e)
            error_tts = gTTS(text="Sorry, something went wrong. Try again.", lang='en')
            fp = io.BytesIO()
            error_tts.write_to_fp(fp)
            fp.seek(0)
            audio = AudioSegment.from_file(fp, format="mp3")
            play(audio)
    
    current_status = "idle"

@app.route("/")
def home():
    return jsonify({
        "message": "Llama Voice Assistant API with Vector Database",
        "status": current_status,
        "vector_db_loaded": chunks_data is not None,
        "endpoints": {
            "start": "/start",
            "stop": "/stop",
            "status": "/status",
            "chat": "/chat",
            "search": "/search"
        }
    })

@app.route("/start", methods=["POST"])
def start_conversation():
    global conversation_active, conversation_thread
    
    if conversation_active:
        return jsonify({"error": "Conversation already active"}), 400
    
    conversation_active = True
    conversation_thread = threading.Thread(target=voice_assistant)
    conversation_thread.start()
    
    return jsonify({"message": "Voice assistant started", "status": "active"})

@app.route("/stop", methods=["POST"])
def stop_conversation():
    global conversation_active
    
    if not conversation_active:
        return jsonify({"error": "No active conversation"}), 400
    
    conversation_active = False
    return jsonify({"message": "Voice assistant stopped", "status": "idle"})

@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({
        "status": current_status,
        "active": conversation_active,
        "last_response": last_response,
        "vector_db_loaded": chunks_data is not None,
        "total_chunks": len(chunks_data) if chunks_data is not None else 0
    })

@app.route("/chat", methods=["POST"])
def text_chat():
    """Text-based chat endpoint with vector database integration"""
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({"error": "No message provided"}), 400
        
        user_message = data['message']
        
        # Get enhanced response with vector database
        llama_text = get_enhanced_response(user_message)
        
        # Also get relevant chunks for debugging/transparency
        relevant_chunks = find_relevant_context(user_message)
        
        return jsonify({
            "user_message": user_message,
            "llama_response": llama_text,
            "relevant_chunks_count": len(relevant_chunks),
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/search", methods=["POST"])
def search_chunks():
    """Search the vector database directly"""
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
        
        query = data['query']
        top_k = data.get('top_k', 5)
        
        relevant_chunks = find_relevant_context(query, top_k)
        
        return jsonify({
            "query": query,
            "results": relevant_chunks,
            "count": len(relevant_chunks)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced 911-Style Assistant with Vector Database...")
    
    # Load vector database on startup
    if load_vector_database():
        print("✅ Vector database integration ready!")
    else:
        print("⚠️ Running without vector database - responses will use general knowledge only")
    
    app.run(debug=True, host='0.0.0.0', port=5004)



