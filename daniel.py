# ===============================================
# Daniel AI Assistant - Structured Version
# ===============================================

# --- Imports ---
import warnings
warnings.filterwarnings("ignore", message="GPT2InferenceModel has generative capabilities")
import os
import subprocess
import tempfile
import time
import atexit
import signal
import keyboard
import sounddevice as sd
import soundfile as sf
import numpy as np
import pyttsx3
import wave
import requests
import yaml
import re


from langchain.schema import HumanMessage
from memory import HybridMemory
memory = HybridMemory()
from textblob import TextBlob
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from yourtts_tts import speak_with_xtts
from personality import load_personality
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain, LLMChain






# ===============================================
# Whisper Transcription
# ===============================================
def transcribe_audio(path):
    url = "http://localhost:5005/transcribe"
    response = None  # <-- define upfront so it always exists
    with open(path, "rb") as f:
        try:
            response = requests.post(url, files={"audio": f}, timeout=60)
            response.raise_for_status()
            data = response.json()
            print("📝 Whisper response:", data)
            return data.get("text", "")
        except Exception as e:
            print("❌ Whisper request failed:", e)
            if response is not None:
                print("🔍 Raw response:", response.text)
            return ""


# ===============================================
# TTS Engine (pyttsx3 fallback, XTTS main)
# ===============================================
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 165)


# ===============================================
# Check if Dolphin-Mistral is running
# ===============================================
def is_dolphin_mistral_running():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        return "dolphin-mistral" in result.stdout.lower()
    except Exception as e:
        print("❌ Error checking Ollama:", e)
        return False
    
# ===============================================
# Load Personality from YAML
# =============================================== 
def build_personality_prompt():
    personality = load_personality()
    traits = personality.get("traits", [])
    core_values = personality.get("core_values", [])
    identity = personality.get("identity", [])
    goals = personality.get("goals", [])
    mission = personality.get("mission", "")
    style = personality.get("style", "")

    system_message = f"""
    You are {personality['name']} — {personality['role']}.
    Traits: {", ".join(traits)}.
    Core values: {", ".join(core_values)}.
    Identity and mindset: {", ".join(identity)}.
    Mission: {mission}.
    Goals: {"; ".join(goals)}.
    Speaking style: {style}.

    Your responses should always reflect these traits, goals, mission, and style.
    Keep consistency across sessions, adapting to memory and user preferences.
        """

    return ChatPromptTemplate.from_messages([
        ("ai", system_message),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])



# ===============================================
# LLM Setup
# ===============================================
def setup_llm_chain():
    llm = OllamaLLM(model="dolphin-mistral")
    prompt = build_personality_prompt()
    return LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False
    )




# ===============================================
# Microphone Check
# ===============================================
def mic_check(duration=2):
    print("🔊 Checking mic...")
    recording = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='int16')
    sd.wait()
    volume = np.linalg.norm(recording)
    return volume > 100


# ===============================================
# Record Audio
# ===============================================
def record_audio(filename="input.wav", duration=5):
    print("🎙️ Recording... Speak now.")
    recording = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='int16')
    sd.wait()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(recording.tobytes())

    return filename


# ===============================================
# Detect Silence
# ===============================================
def is_silent(filename, threshold=500):
    with wave.open(filename, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        return np.abs(audio).mean() < threshold


# ===============================================
# Detect Emotion from Text
# ===============================================
def detect_emotion(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return "happy"
    elif polarity < -0.2:
        return "sad"
    return "neutral"


# ===============================================
# Speak with XTTS
# ===============================================
def speak(text):
    speak_with_xtts(text, "output.wav")
    data, samplerate = sf.read("output.wav")
    sd.play(data, samplerate)
    sd.wait()


# ===============================================
# Main Loop
# ===============================================
def main():
    print("🧠 Setting up Daniel...")

    # === Setup ===
    llm_chain = setup_llm_chain()
    memory = HybridMemory()

    print("🎯 Checking if Dolphin-Mistral is running...")
    if not is_dolphin_mistral_running():
        print("🟡 Attempting to launch Dolphin-Mistral...")
        subprocess.Popen(["ollama", "run", "dolphin-mistral"], stdout=subprocess.DEVNULL)
        time.sleep(10)

    if not is_dolphin_mistral_running():
        print("❌ Dolphin-Mistral is not running. Please start it first.")
        return

    if not mic_check():
        print("⚠️ Mic input is low. Using SPACEBAR for text activation.")

    print("\n🤖 Daniel is ready! Say 'hey daniel' or press SPACEBAR to start.\n")

    while True:
        try:
            # === 1. Wake word or spacebar ===
            print("🛎️ Waiting for wake word or SPACEBAR...")
            wake_triggered = False

            while not wake_triggered:
                if keyboard.is_pressed("space"):
                    wake_triggered = True
                    print("⎇ SPACEBAR pressed — entering text mode.")
                    break

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                    record_audio(tmp_path, duration=2)
                    spoken = transcribe_audio(tmp_path).lower().strip()

                if "hey daniel" in spoken:
                    print(f"🔊 Wake word detected: {spoken}")
                    wake_triggered = True
                    break

            # === 2. Capture user input ===
            audio_file = record_audio()
            if is_silent(audio_file):
                print("😶 Silence detected. Type your message.")
                user_input = input("💬 You: ").strip()
                if not user_input:
                    continue
            else:
                user_input = transcribe_audio(audio_file).strip()
                print(f"🗣️ You said: {user_input}")
                if not user_input:
                    user_input = input("💬 You (type): ").strip()
                    if not user_input:
                        continue

            # === 3. Emotion detection ===
            emotion = detect_emotion(user_input)
            print(f"🧠 Emotion detected: {emotion}")
            if emotion and emotion != "neutral":
                user_input = f"[User seems {emotion}] {user_input}"

            # === 4. Save turn to memory ===
            memory.save_turn("user", user_input)

            # === 5. Recall facts ===
            facts = memory.recall(user_input)
            facts_text = "\n".join([f"- {f}" for f in facts]) if facts else "None"

            # === 6. Build input for chain ===
            history = memory.load_history()
            history_msgs = [
                HumanMessage(content=f"{m['role'].capitalize()}: {m['content']}")
                for m in history
            ]

            # final input: user + facts
            final_input = f"""
Relevant knowledge:
{facts_text}

{user_input}
"""

            print("💡 Thinking...")
            result = llm_chain.predict(input=final_input, history=history_msgs)

            # === 7. Extract text safely ===
            if isinstance(result, dict):
                response = result.get("response") or result.get("text") or str(result)
            else:
                response = str(result)

            # Clean unwanted prefixes/junk
            response = re.sub(r"(response:|input:).*", "", response, flags=re.IGNORECASE)
            response = response.strip(" {}\"\n")
            if not response:
                response = "Sorry, I didn’t quite get that."

            # === 8. Output ===
            print(f"🤖 Daniel: {response}")
            speak(response)

            # === 9. Save Daniel’s response to memory ===
            memory.save_turn("daniel", response)

        except KeyboardInterrupt:
            print("\n👋 Shutting down Daniel. Goodbye!")
            break
        
# ===============================================
# Run Main
# ===============================================
if __name__ == "__main__":
    main()
