import json
import torch
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

from src.data.preprocessing import preprocess_dialogue
from src.evaluation.metrics import compute_rouge


DEFAULT_MODEL = "google/flan-t5-base"

PROMPT_TEMPLATE = (
    "Summarize the following conversation in 2-3 sentences:\n\n"
    "{dialogue}\n\nSummary:"
)


class FlanT5Summarizer:
    """Zero-shot summarizer using Flan-T5-base via instruction prompting.

    No fine-tuning — demonstrates instruction-tuning without domain adaptation.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = 128,
        num_beams: int = 4,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "text2text-generation",
            model=model_name,
            device=device,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        )

    def _format_prompt(self, dialogue: str) -> str:
        cleaned = preprocess_dialogue(dialogue)
        return PROMPT_TEMPLATE.format(dialogue=cleaned)

    def summarize(self, dialogue: str) -> str:
        prompt = self._format_prompt(dialogue)
        output = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            early_stopping=True,
        )
        return output[0]["generated_text"].strip()

    def summarize_batch(self, dialogues: list[str], batch_size: int = 16) -> list[str]:
        prompts = [self._format_prompt(d) for d in dialogues]
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            outputs = self.pipe(
                batch,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                early_stopping=True,
            )
            results.extend([o["generated_text"].strip() for o in outputs])
        return results


def save_results(predictions: list[str], references: list[str], path: str) -> dict:
    scores = compute_rouge(predictions, references)
    payload = {"scores": scores, "n_samples": len(predictions)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Flan-T5] Results saved to {path}")
    return scores
