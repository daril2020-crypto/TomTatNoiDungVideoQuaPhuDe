from datasets import load_dataset, Dataset
from typing import Optional


def load_dialogsum(cache_dir: Optional[str] = None) -> dict:
    """Load DialogSum dataset from HuggingFace Hub.

    Returns a DatasetDict with keys: train, validation, test.
    Each sample has fields: id, dialogue, summary, topic.
    """
    dataset = load_dataset("knkarthick/dialogsum", cache_dir=cache_dir)
    return dataset


def load_mediasum_sample(n: int = 2000, cache_dir: Optional[str] = None) -> Dataset:
    """Load a sample of MediaSum for cross-domain evaluation.

    Falls back to CNN/DailyMail if MediaSum loader fails.
    Maps fields to (dialogue, summary) for compatibility.
    """
    try:
        dataset = load_dataset(
            "ccdv/mediasum",
            split=f"test[:{n}]",
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        # MediaSum fields: document (transcript), summary
        dataset = dataset.rename_column("document", "dialogue")
        return dataset
    except Exception:
        print("[loader] MediaSum failed, falling back to CNN/DailyMail test set.")
        dataset = load_dataset(
            "cnn_dailymail",
            "3.0.0",
            split=f"test[:{n}]",
            cache_dir=cache_dir,
        )
        # CNN/DM fields: article, highlights
        dataset = dataset.rename_column("article", "dialogue")
        dataset = dataset.rename_column("highlights", "summary")
        return dataset
