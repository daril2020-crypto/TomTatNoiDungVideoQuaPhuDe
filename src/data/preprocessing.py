import re
import nltk
from typing import Optional


def normalize_speakers(text: str) -> str:
    """Replace #PersonN# tags with [SPN] to avoid tokenizer issues with '#'."""
    return re.sub(r"#Person(\d+)#", r"[SP\1]", text)


def clean_dialogue(text: str) -> str:
    """Strip whitespace noise and collapse multiple spaces."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\x00", "")
    return text


def preprocess_dialogue(text: str) -> str:
    """Full preprocessing: clean + normalize speakers."""
    return normalize_speakers(clean_dialogue(text))


def chunk_long_dialogue(text: str, max_words: int = 900) -> list[str]:
    """Split a very long dialogue into overlapping chunks at sentence boundaries.

    Used for MediaSum transcripts that can exceed 5,000 words.
    Returns a list of chunks, each under max_words.
    """
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    sentences = nltk.sent_tokenize(text)
    chunks, current, current_len = [], [], 0

    for sent in sentences:
        word_count = len(sent.split())
        if current_len + word_count > max_words and current:
            chunks.append(" ".join(current))
            # overlap: keep last 2 sentences for context
            current = current[-2:]
            current_len = sum(len(s.split()) for s in current)
        current.append(sent)
        current_len += word_count

    if current:
        chunks.append(" ".join(current))

    return chunks


def tokenize_for_model(
    examples: dict,
    tokenizer,
    max_source_length: int = 1024,
    max_target_length: int = 128,
    dialogue_col: str = "dialogue",
    summary_col: str = "summary",
) -> dict:
    """HuggingFace .map()-compatible tokenization function for Seq2Seq training.

    Preprocesses dialogues and tokenizes inputs + labels.
    Note: only clean_dialogue() is applied (no speaker normalization) so that
    the original #PersonN# tags are preserved — consistent with published baselines.
    """
    dialogues = [clean_dialogue(d) for d in examples[dialogue_col]]
    summaries = list(examples[summary_col])

    model_inputs = tokenizer(
        dialogues,
        max_length=max_source_length,
        truncation=True,
        padding=False,
    )
    # Tokenize targets separately to allow a different max_length
    labels = tokenizer(
        text_target=summaries,
        max_length=max_target_length,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
