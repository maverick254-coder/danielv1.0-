import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs   # <-- add XttsArgs here
from TTS.config.shared_configs import BaseDatasetConfig

# Allowlist the XTTS + dataset configs for torch.load
torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    BaseDatasetConfig,
    XttsArgs   # <-- new allowlisted class
])

# Check if CUDA is available and print what device will be used
use_gpu = torch.cuda.is_available()
device = "cuda" if use_gpu else "cpu"
print(f"[XTTS] Using device: {device.upper()}")

# Load XTTS-v2 with device-aware setting
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=use_gpu)

# Reference voice sample
speaker_wav = "jarvis.wav"
language = "en"

def speak_with_xtts(text, output_path="output.wav"):
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
        file_path=output_path
    )
