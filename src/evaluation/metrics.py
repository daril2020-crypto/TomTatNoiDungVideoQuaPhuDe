from evaluate import load as hf_load


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores.

    Returns a dict with keys: rouge1, rouge2, rougeL (all F1, 0–1 scale).
    """
    rouge = hf_load("rouge")
    result = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    return {
        "rouge1": round(result["rouge1"], 4),
        "rouge2": round(result["rouge2"], 4),
        "rougeL": round(result["rougeL"], 4),
    }


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    lang: str = "en",
    model_type: str = "roberta-large",
) -> dict:
    """Compute BERTScore Precision, Recall, F1.

    Falls back to distilbert-base-uncased if roberta-large download fails.
    Returns a dict with keys: precision, recall, f1 (mean over samples).
    """
    bertscore = hf_load("bertscore")
    try:
        result = bertscore.compute(
            predictions=predictions,
            references=references,
            lang=lang,
            model_type=model_type,
        )
    except Exception:
        print("[metrics] roberta-large failed, falling back to distilbert-base-uncased")
        result = bertscore.compute(
            predictions=predictions,
            references=references,
            lang=lang,
            model_type="distilbert-base-uncased",
        )

    return {
        "precision": round(sum(result["precision"]) / len(result["precision"]), 4),
        "recall": round(sum(result["recall"]) / len(result["recall"]), 4),
        "f1": round(sum(result["f1"]) / len(result["f1"]), 4),
    }


def compute_bleu(predictions: list[str], references: list[str]) -> dict:
    """Compute corpus-level BLEU-4 score.

    Returns a dict with key: bleu (0–1 scale).
    """
    sacrebleu = hf_load("sacrebleu")
    # sacrebleu expects references as list of lists
    refs = [[r] for r in references]
    result = sacrebleu.compute(predictions=predictions, references=refs)
    return {"bleu": round(result["score"] / 100, 4)}  # normalize 0–100 → 0–1


def compute_all_metrics(
    predictions: list[str],
    references: list[str],
    include_bertscore: bool = True,
    include_bleu: bool = True,
) -> dict:
    """Compute all metrics in one call. Returns unified dict."""
    scores = compute_rouge(predictions, references)
    if include_bertscore:
        bs = compute_bertscore(predictions, references)
        scores["bertscore_f1"] = bs["f1"]
    if include_bleu:
        bleu = compute_bleu(predictions, references)
        scores["bleu"] = bleu["bleu"]
    return scores
