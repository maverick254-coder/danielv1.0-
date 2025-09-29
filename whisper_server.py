# whisper_server.py
from faster_whisper import WhisperModel
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# Auto-detect device
if torch.cuda.is_available():
    device = "cuda"
    compute_type = "float16"
    print("⚡ Using GPU (CUDA) with float16")
else:
    device = "cpu"
    compute_type = "int8"  # smaller & faster on CPU
    print("🖥️ Using CPU with int8")

# Load Whisper model
print("Loading Whisper...")
model = WhisperModel("small", device=device, compute_type=compute_type)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_file = request.files["audio"]
    segments, _ = model.transcribe(audio_file, beam_size=5)

    text = " ".join([segment.text for segment in segments])
    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(port=5005)
