"""Token counting and text chunking utilities."""

import tiktoken


def count_tokens(text: str, model_name: str = "gpt-5-mini-2025-08-07") -> int:
    """Return token count for `text` with tiktoken for given model."""
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def split_into_chunks(text: str, max_tokens: int = 250000) -> list:
    """Split text into chunks that fit within max_tokens."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Rough fallback: 4 chars per token
        chunk_size = max_tokens * 4
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    tokens = enc.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(enc.decode(chunk_tokens))
    return chunks


def split_texts_into_token_batches(texts: list, max_tokens_per_batch: int = 100000,
                                   char_trunc: int = 30000) -> list:
    """Split a list of texts into batches that fit within a token budget.

    Each text is optionally truncated to `char_trunc` characters if very long.
    Returns list of lists of texts.
    """
    batches = []
    current_batch = []
    current_tokens = 0

    for t in texts:
        if len(t) > char_trunc:
            t = t[:char_trunc]
        tok = count_tokens(t)
        if current_tokens + tok > max_tokens_per_batch and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(t)
        current_tokens += tok

    if current_batch:
        batches.append(current_batch)
    return batches
