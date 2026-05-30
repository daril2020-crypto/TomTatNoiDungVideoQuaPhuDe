"""
Retrain BART and PEGASUS with fixed preprocessing (clean_dialogue only, no normalize_speakers).
Run: python3 scripts/retrain_all.py
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, '/workspace')

from src.data.loader import load_dialogsum
from src.methods.bart_finetune import train as bart_train, predict as bart_predict, save_results as bart_save
from src.methods.pegasus_finetune import train as pegasus_train, predict as pegasus_predict, save_results as pegasus_save
from src.evaluation.metrics import compute_rouge, compute_bertscore, compute_bleu


def run_inference_and_eval(dialogues, references, predict_fn, model_dir, result_path, pred_path, label):
    print(f"\n[{label}] Running inference on {len(dialogues)} samples...", flush=True)
    preds = predict_fn(dialogues, model_dir=model_dir)
    print(f"[{label}] Computing metrics...", flush=True)
    scores = compute_rouge(preds, references)
    bs = compute_bertscore(preds, references)
    bl = compute_bleu(preds, references)
    scores["bertscore_f1"] = bs["f1"]
    scores["bleu"] = bl["bleu"]

    Path(result_path).parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump({"scores": scores, "n_samples": len(preds)}, f, indent=2)
    with open(pred_path, "w") as f:
        json.dump({"predictions": preds[:50], "references": references[:50]}, f, indent=2)

    print(f"[{label}] ROUGE-1={scores['rouge1']:.4f} ROUGE-2={scores['rouge2']:.4f} ROUGE-L={scores['rougeL']:.4f} BERTScore={scores['bertscore_f1']:.4f} BLEU={scores['bleu']:.4f}", flush=True)
    return scores


print("="*60)
print("Loading DialogSum dataset...")
dataset = load_dialogsum('/workspace/data/raw/dialogsum')
test_dialogues = list(dataset['test']['dialogue'])
test_references = list(dataset['test']['summary'])
print(f"Train: {len(dataset['train'])}  Val: {len(dataset['validation'])}  Test: {len(dataset['test'])}")

# ── BART ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 1: BART FINE-TUNING")
print("="*60)
t0 = time.time()
trainer = bart_train(
    dataset,
    config_path='/workspace/configs/bart_config.yaml',
    output_dir='/workspace/models/bart_finetuned',
)
bart_time = time.time() - t0
print(f"[BART] Training done in {bart_time/60:.1f} minutes")

# Save training log
log = trainer.state.log_history
Path('/workspace/results/scores').mkdir(parents=True, exist_ok=True)
with open('/workspace/results/scores/bart_train_log.json', 'w') as f:
    json.dump(log, f, indent=2)

bart_scores = run_inference_and_eval(
    test_dialogues, test_references,
    bart_predict,
    model_dir='/workspace/models/bart_finetuned',
    result_path='/workspace/results/scores/bart_results.json',
    pred_path='/workspace/results/scores/bart_predictions.json',
    label='BART',
)

# ── PEGASUS ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 2: PEGASUS FINE-TUNING")
print("="*60)
t0 = time.time()
trainer = pegasus_train(
    dataset,
    config_path='/workspace/configs/pegasus_config.yaml',
    output_dir='/workspace/models/pegasus_finetuned',
)
peg_time = time.time() - t0
print(f"[PEGASUS] Training done in {peg_time/60:.1f} minutes")

log = trainer.state.log_history
with open('/workspace/results/scores/pegasus_train_log.json', 'w') as f:
    json.dump(log, f, indent=2)

peg_scores = run_inference_and_eval(
    test_dialogues, test_references,
    pegasus_predict,
    model_dir='/workspace/models/pegasus_finetuned',
    result_path='/workspace/results/scores/pegasus_results.json',
    pred_path='/workspace/results/scores/pegasus_predictions.json',
    label='PEGASUS',
)

print("\n" + "="*60)
print("ALL TRAINING COMPLETE")
print(f"  BART   ROUGE-L={bart_scores['rougeL']:.4f}  BERTScore={bart_scores['bertscore_f1']:.4f}")
print(f"  PEGASUS ROUGE-L={peg_scores['rougeL']:.4f}  BERTScore={peg_scores['bertscore_f1']:.4f}")
