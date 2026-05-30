"""
Regenerate all figures and update report.docx with latest results.
Run after retrain_all.py completes.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, '/workspace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.evaluation.visualize import plot_comparison_bar, plot_training_curves

RESULTS_DIR = Path('/workspace/results/scores')
FIGURES_DIR = Path('/workspace/results/figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_scores(name):
    path = RESULTS_DIR / f'{name}_results.json'
    with open(path) as f:
        return json.load(f)['scores']


def load_log(name):
    path = RESULTS_DIR / f'{name}_train_log.json'
    with open(path) as f:
        return json.load(f)


print("Loading results...", flush=True)
results = {
    'TextRank': load_scores('textrank'),
    'BART':     load_scores('bart'),
    'PEGASUS':  load_scores('pegasus'),
    'Flan-T5':  load_scores('flant5'),
}

for method, scores in results.items():
    print(f"  {method:12s}: ROUGE-1={scores['rouge1']:.4f}  ROUGE-2={scores['rouge2']:.4f}  ROUGE-L={scores['rougeL']:.4f}  BERTScore={scores['bertscore_f1']:.4f}  BLEU={scores['bleu']:.4f}")

# ── Figure 1: ROUGE comparison ─────────────────────────────────────
print("\nGenerating ROUGE comparison chart...", flush=True)
plot_comparison_bar(
    results,
    metrics=['rouge1', 'rouge2', 'rougeL'],
    save_path=str(FIGURES_DIR / 'rouge_comparison.png'),
    title='So sánh điểm ROUGE giữa các phương pháp',
)

# ── Figure 2: BERTScore + BLEU ─────────────────────────────────────
print("Generating BERTScore/BLEU chart...", flush=True)
plot_comparison_bar(
    results,
    metrics=['bertscore_f1', 'bleu'],
    save_path=str(FIGURES_DIR / 'bertscore_bleu_comparison.png'),
    title='So sánh BERTScore F1 và BLEU-4',
)

# ── Figure 3: BART training curves ────────────────────────────────
print("Generating BART training curves...", flush=True)
bart_log = load_log('bart')
plot_training_curves(
    bart_log,
    metric='rougeL',
    save_path=str(FIGURES_DIR / 'training_curves_bart.png'),
    title='BART Fine-tuning: Train Loss & Val ROUGE-L',
)

# ── Figure 4: PEGASUS training curves ─────────────────────────────
print("Generating PEGASUS training curves...", flush=True)
peg_log = load_log('pegasus')
plot_training_curves(
    peg_log,
    metric='rougeL',
    save_path=str(FIGURES_DIR / 'training_curves_pegasus.png'),
    title='PEGASUS Fine-tuning: Train Loss & Val ROUGE-L',
)

plt.close('all')
print("\nAll figures saved to results/figures/", flush=True)

# ── Regenerate report ─────────────────────────────────────────────
print("\nRegenerating report.docx...", flush=True)
import subprocess, sys
result = subprocess.run(
    [sys.executable, '/workspace/report/generate_report.py'],
    cwd='/workspace',
    capture_output=True,
    text=True,
)
print(result.stdout)
if result.returncode != 0:
    print("ERROR:", result.stderr[-2000:])
else:
    print("Done. Report saved to report/report.docx", flush=True)
