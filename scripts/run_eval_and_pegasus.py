"""
1. Run BART inference + eval (model already trained)
2. Train PEGASUS
3. Run PEGASUS inference + eval
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, '/workspace')

from src.data.loader import load_dialogsum
from src.methods.bart_finetune import predict as bart_predict
from src.methods.pegasus_finetune import train as pegasus_train, predict as pegasus_predict
from src.evaluation.metrics import compute_rouge, compute_bertscore, compute_bleu


def infer_and_save(dialogues, references, predict_fn, model_dir, result_path, pred_path, label):
    print(f"\n[{label}] Running inference on {len(dialogues)} samples...", flush=True)
    preds = predict_fn(dialogues, model_dir=model_dir)
    print(f"[{label}] Computing ROUGE...", flush=True)
    scores = compute_rouge(preds, references)
    print(f"[{label}] Computing BERTScore...", flush=True)
    bs = compute_bertscore(preds, references)
    print(f"[{label}] Computing BLEU...", flush=True)
    bl = compute_bleu(preds, references)
    scores["bertscore_f1"] = bs["f1"]
    scores["bleu"] = bl["bleu"]

    Path(result_path).parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump({"scores": scores, "n_samples": len(preds)}, f, indent=2)
    with open(pred_path, "w") as f:
        json.dump({"predictions": preds[:50], "references": references[:50]}, f, indent=2)

    print(f"[{label}] ROUGE-1={scores['rouge1']:.4f} ROUGE-2={scores['rouge2']:.4f} "
          f"ROUGE-L={scores['rougeL']:.4f} BERTScore={scores['bertscore_f1']:.4f} "
          f"BLEU={scores['bleu']:.4f}", flush=True)
    return scores


print("Loading DialogSum dataset...", flush=True)
dataset = load_dialogsum('/workspace/data/raw/dialogsum')
test_dialogues = list(dataset['test']['dialogue'])
test_references = list(dataset['test']['summary'])
print(f"Test: {len(test_dialogues)} samples", flush=True)

# ── BART inference ────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE: BART INFERENCE (model already trained)")
print("="*60)
bart_scores = infer_and_save(
    test_dialogues, test_references,
    bart_predict,
    model_dir='/workspace/models/bart_finetuned',
    result_path='/workspace/results/scores/bart_results.json',
    pred_path='/workspace/results/scores/bart_predictions.json',
    label='BART',
)

# ── PEGASUS training ──────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE: PEGASUS FINE-TUNING")
print("="*60)
t0 = time.time()
trainer = pegasus_train(
    dataset,
    config_path='/workspace/configs/pegasus_config.yaml',
    output_dir='/workspace/models/pegasus_finetuned',
)
peg_time = time.time() - t0
print(f"[PEGASUS] Training done in {peg_time/60:.1f} minutes", flush=True)

log = trainer.state.log_history
with open('/workspace/results/scores/pegasus_train_log.json', 'w') as f:
    json.dump(log, f, indent=2)

peg_scores = infer_and_save(
    test_dialogues, test_references,
    pegasus_predict,
    model_dir='/workspace/models/pegasus_finetuned',
    result_path='/workspace/results/scores/pegasus_results.json',
    pred_path='/workspace/results/scores/pegasus_predictions.json',
    label='PEGASUS',
)

print("\n" + "="*60)
print("ALL DONE")
print(f"  BART    ROUGE-L={bart_scores['rougeL']:.4f}  BERTScore={bart_scores['bertscore_f1']:.4f}")
print(f"  PEGASUS ROUGE-L={peg_scores['rougeL']:.4f}  BERTScore={peg_scores['bertscore_f1']:.4f}")
