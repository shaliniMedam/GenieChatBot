import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from backend.model_manager import model_manager

model_manager._load_blocking()
if not model_manager.model_loaded:
    print("Model not loaded:", model_manager.load_error)
    sys.exit(1)

messages = [
    {"role": "system", "content": "Your name is Genie. You are a helpful AI. You must NEVER say your name is Gemma. You are NOT Gemma."},
    {"role": "user", "content": "heyy"}
]

print("Generating sync...")
res = model_manager.generate_sync(messages=messages, temperature=1.0, max_tokens=100)
print(res)
