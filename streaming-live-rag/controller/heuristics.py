import re

def is_stable_enough(text: str) -> bool:
    """
    Returns True if the text is 'stable enough' to even bother checking with the LLM classifier.
    This acts as a cheap filter to prevent thrashing the LLM on every single word.
    """
    text = text.strip()
    words = text.split()
    
    # 1. Minimum token count: don't check if less than 3 words (unless there's a strong punctuation mark)
    has_terminal_punctuation = bool(re.search(r'[.?!]$', text))
    
    if len(words) < 3 and not has_terminal_punctuation:
        return False
        
    return True
