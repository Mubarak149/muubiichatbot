def chunk_text(text, max_chars=1000):
    """Split text into smaller chunks for TTS"""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+max_chars])
        start += max_chars
    return chunks
