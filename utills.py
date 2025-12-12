import json
from datetime import datetime

def chunk_text(text, max_chars=1000):
    """Split text into smaller chunks for TTS"""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+max_chars])
        start += max_chars
    return chunks



def log_interaction(role, message):
    """Save chat interactions to a file for analysis."""
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "message": message
    }

    # Save into a log file
    with open("chat_logs.jsonl", "a") as f:
        f.write(json.dumps(log_data) + "\n")
