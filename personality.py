import yaml
import os

def load_personality(path="personality.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Personality file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:   # <-- force utf-8
        return yaml.safe_load(f)
