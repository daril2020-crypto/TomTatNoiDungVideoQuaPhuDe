import json
import yaml
import torch
from pathlib import Path
from functools import partial
from typing import Optional

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import DatasetDict

from src.data.preprocessing import tokenize_for_model
from src.evaluation.metrics import compute_rouge


DEFAULT_MODEL = "facebook/bart-large-cnn"
DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "configs" / "bart_config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer


def preprocess_dataset(
    dataset: DatasetDict,
    tokenizer,
    max_source_length: int = 1024,
    max_target_length: int = 128,
) -> DatasetDict:
    fn = partial(
        tokenize_for_model,
        tokenizer=tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    return dataset.map(fn, batched=True, remove_columns=dataset["train"].column_names)


def _make_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = [
            [(l if l != -100 else tokenizer.pad_token_id) for l in label]
            for label in labels
        ]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        scores = compute_rouge(decoded_preds, decoded_labels)
        return {
            "rouge1": scores["rouge1"],
            "rouge2": scores["rouge2"],
            "rougeL": scores["rougeL"],
        }

    return compute_metrics


def train(
    dataset: DatasetDict,
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str = "models/bart_finetuned",
):
    config = load_config(config_path)
    model_name = config.get("model_name", DEFAULT_MODEL)

    model, tokenizer = load_model_and_tokenizer(model_name)
    tokenized = preprocess_dataset(
        dataset,
        tokenizer,
        max_source_length=config.get("max_source_length", 1024),
        max_target_length=config.get("max_target_length", 128),
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 3e-5),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_ratio=config.get("warmup_ratio", 0.06),
        bf16=config.get("bf16", True) and torch.cuda.is_bf16_supported(),
        predict_with_generate=True,
        generation_max_length=config.get("max_target_length", 128),
        eval_strategy=config.get("evaluation_strategy", "epoch"),
        save_strategy=config.get("save_strategy", "epoch"),
        load_best_model_at_end=True,
        metric_for_best_model=config.get("metric_for_best_model", "rougeL"),
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_make_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[BART] Model saved to {output_dir}")
    return trainer


def predict(
    dialogues: list[str],
    model_dir: str = "models/bart_finetuned",
    batch_size: int = 8,
    max_new_tokens: int = 128,
    num_beams: int = 4,
) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    summaries = []
    for i in range(0, len(dialogues), batch_size):
        batch = dialogues[i : i + batch_size]
        inputs = tokenizer(
            batch,
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        summaries.extend(decoded)

    return summaries


def save_results(predictions: list[str], references: list[str], path: str) -> dict:
    scores = compute_rouge(predictions, references)
    payload = {"scores": scores, "n_samples": len(predictions)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[BART] Results saved to {path}")
    return scores
