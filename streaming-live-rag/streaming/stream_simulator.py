from typing import Iterator
from telemetry.schema import StreamChunk

def simulate_stream(utterance: str, words_per_chunk: int = 3, ms_per_chunk: int = 300) -> Iterator[StreamChunk]:
    """
    Simulates ASR output by yielding partial text chunks from a full utterance.
    """
    words = utterance.split()
    current_text = ""
    
    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i:i+words_per_chunk]
        if current_text:
            current_text += " "
        current_text += " ".join(chunk_words)
        
        # Simulate time passing
        t_offset = (i // words_per_chunk + 1) * (ms_per_chunk / 1000.0)
        
        yield StreamChunk(
            t_offset_s=round(t_offset, 2),
            partial_text=current_text
        )
