import re
import unicodedata

def clean_text(text: str) -> str:
    """
    Cleans raw extracted text:
    - Normalizes unicode characters
    - Fixes hyphenated line-breaks (e.g., 'cli-\nmate' -> 'climate')
    - Collapses abnormal multiple whitespaces while preserving paragraph breaks
    - Removes non-printable control characters
    """
    if not text:
        return ""
    
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    
    # Fix hyphenated words broken across lines
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Replace carriage returns
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove control characters except standard whitespace
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    
    # Normalize multiple blank lines to double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Collapse multiple spaces or tabs into a single space
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
    cleaned = '\n'.join(lines).strip()
    
    return cleaned

