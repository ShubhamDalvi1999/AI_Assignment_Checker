import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count the number of tokens in a text string."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")  # Fallback encoding
    return len(enc.encode(text))

def truncate_by_token_limit(texts: list[str], max_tokens: int, model: str = "gpt-4o") -> str:
    """Truncate a list of text chunks to stay within token limit."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")  # Fallback encoding
        
    current_tokens = 0
    output_chunks = []

    for text in texts:
        tokens = enc.encode(text)
        if current_tokens + len(tokens) > max_tokens:
            # If we can't fit the entire chunk, see if we can fit part of it
            remaining_tokens = max_tokens - current_tokens
            if remaining_tokens > 50:  # If we have space for at least 50 tokens
                partial_text = enc.decode(tokens[:remaining_tokens])
                output_chunks.append(partial_text + "... [truncated]")
            break
        output_chunks.append(text)
        current_tokens += len(tokens)

    return "\n\n".join(output_chunks)

def safe_truncate_code(code: str, max_tokens: int, model: str = "gpt-4o") -> str:
    """Safely truncate code to a maximum token count."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")  # Fallback encoding
        
    tokens = enc.encode(code)
    if len(tokens) <= max_tokens:
        return code
    
    # When truncating, add a comment indicating truncation
    truncated = enc.decode(tokens[:max_tokens])
    return truncated + "\n# ... [code truncated due to length]" 