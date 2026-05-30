"""
Chạy toàn bộ pipeline dự án từ đầu đến cuối:
  1. TextRank (baseline extractive)
  2. Flan-T5 zero-shot
  3. BART fine-tuned
  4. PEGASUS fine-tuned
  5. Sinh hình ảnh (7 figures)
  6. Xuất báo cáo report.docx

Chạy từ thư mục gốc:
    python scripts/run_all.py
Hoặc bỏ qua bước training (dùng model đã có):
    python scripts/run_all.py --skip-train
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_dialogsum
from src.evaluation.metrics import compute_rouge, compute_bertscore, compute_bleu

SCORES_DIR  = Path("results/scores")
FIGURES_DIR = Path("results/figures")
SCORES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _header(msg):
    print("\n" + "=" * 60, flush=True)
    print(msg, flush=True)
    print("=" * 60, flush=True)


def eval_and_save(preds, references, result_path, pred_path, label):
    print(f"[{label}] Computing ROUGE...",     flush=True)
    scores = compute_rouge(preds, references)
    print(f"[{label}] Computing BERTScore...", flush=True)
    bs = compute_bertscore(preds, references)
    print(f"[{label}] Computing BLEU...",      flush=True)
    bl = compute_bleu(preds, references)
    scores["bertscore_f1"] = bs["f1"]
    scores["bleu"]         = bl["bleu"]

    with open(result_path, "w") as f:
        json.dump({"scores": scores, "n_samples": len(preds)}, f, indent=2)
    with open(pred_path, "w") as f:
        json.dump({"predictions": preds[:50], "references": references[:50]}, f, indent=2)

    print(
        f"[{label}] ROUGE-1={scores['rouge1']:.4f}  ROUGE-2={scores['rouge2']:.4f}  "
        f"ROUGE-L={scores['rougeL']:.4f}  BERTScore={scores['bertscore_f1']:.4f}  "
        f"BLEU={scores['bleu']:.4f}",
        flush=True,
    )
    return scores


# ─────────────────────────────────────────────────────────────────
# Figure generation (7 figures)
# ─────────────────────────────────────────────────────────────────

def generate_figures(results, bart_log, peg_log):
    from src.evaluation.visualize import plot_comparison_bar, plot_training_curves

    _header("PHASE: GENERATING FIGURES")

    # ── Fig 1: Pipeline diagram ──────────────────────────────────
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    fig.patch.set_facecolor("#FAFAFA")
    COLS = {"input": "#4472C4", "preproc": "#ED7D31",
            "method": "#70AD47", "eval": "#FFC000", "output": "#7030A0"}

    def box(x, y, w, h, txt, c, fs=9.5, tc="white"):
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                     boxstyle="round,pad=0.08", facecolor=c,
                     edgecolor="white", linewidth=1.5, zorder=3))
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold", zorder=4, multialignment="center")

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#595959",
                                   lw=1.8, mutation_scale=18), zorder=2)

    box(1.2, 2.5, 2.0, 0.9, "Phụ đề Video\n(SRT/VTT)",          COLS["input"])
    box(3.3, 2.5, 2.0, 0.9, "Tiền xử lý\n(clean + tokenize)",   COLS["preproc"])
    arr(2.2, 2.5, 2.3, 2.5)
    for mx, my, lbl in [(5.55,4.15,"TextRank\n(Extractive)"),
                         (5.55,3.05,"BART\nFine-tuned"),
                         (5.55,1.95,"PEGASUS\nFine-tuned"),
                         (5.55,0.85,"Flan-T5\nZero-shot")]:
        box(mx, my, 1.85, 0.75, lbl, COLS["method"], fs=8.5)
        arr(4.3, 2.5, mx-0.93, my)
    box(8.0, 2.5, 2.1, 0.9, "Đánh giá\nROUGE / BERTScore / BLEU",
        COLS["eval"], tc="#333333")
    for mx, my, _ in [(5.55,4.15,""),(5.55,3.05,""),(5.55,1.95,""),(5.55,0.85,"")]:
        arr(mx+0.93, my, 7.5, 2.5+(my-2.5)*0.1)
    box(10.5, 2.5, 1.8, 0.9, "Kết quả\n& Báo cáo", COLS["output"])
    arr(9.05, 2.5, 9.6, 2.5)
    ax.set_title("Sơ đồ Pipeline Hệ thống Tóm tắt Phụ đề Video",
                 fontsize=13, fontweight="bold", pad=12, color="#1F3864")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/"pipeline_diagram.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("Saved pipeline_diagram.png", flush=True)

    # ── Fig 2: Length distribution ───────────────────────────────
    ds   = load_dialogsum(cache_dir="data/raw/dialogsum")
    test = ds["test"]
    dlen = [len(d.split()) for d in test["dialogue"]]
    slen = [len(s.split()) for s in test["summary"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor("white")

    def hist_stats(ax, data, color, title, xlabel):
        ax.hist(data, bins=40, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        m, md = np.mean(data), np.median(data)
        ax.axvline(m,  color="#C00000", lw=2, ls="--", label=f"Mean: {m:.0f}")
        ax.axvline(md, color="#FF8C00", lw=2, ls=":",  label=f"Median: {md:.0f}")
        ax.set_title(title, fontsize=12, fontweight="bold", color="#1F3864", pad=8)
        ax.set_xlabel(xlabel, fontsize=10); ax.set_ylabel("Số mẫu", fontsize=10)
        ax.legend(fontsize=9.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_facecolor("#F8F8F8")

    hist_stats(ax1, dlen, "#4472C4", "Phân phối độ dài Hội thoại", "Số từ")
    hist_stats(ax2, slen, "#70AD47", "Phân phối độ dài Tóm tắt",   "Số từ")
    fig.suptitle("Phân phối độ dài văn bản trong DialogSum test set (1.500 mẫu)",
                 fontsize=13, fontweight="bold", color="#1F3864", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/"length_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved length_distribution.png", flush=True)

    # ── Fig 3 & 4: ROUGE and BERTScore/BLEU bar charts ──────────
    plot_comparison_bar(results, metrics=["rouge1","rouge2","rougeL"],
                        save_path=str(FIGURES_DIR/"rouge_comparison.png"),
                        title="So sánh điểm ROUGE giữa các phương pháp")
    print("Saved rouge_comparison.png", flush=True)

    plot_comparison_bar(results, metrics=["bertscore_f1","bleu"],
                        save_path=str(FIGURES_DIR/"bertscore_bleu_comparison.png"),
                        title="So sánh BERTScore F1 và BLEU-4")
    print("Saved bertscore_bleu_comparison.png", flush=True)

    # ── Fig 5: Radar chart ───────────────────────────────────────
    cats = ["ROUGE-1","ROUGE-2","ROUGE-L","BERTScore\nF1","BLEU-4"]
    morder  = ["TextRank","Flan-T5","BART","PEGASUS"]
    metrics = ["rouge1","rouge2","rougeL","bertscore_f1","bleu"]
    raw  = {m: [results[met][m] for met in morder] for m in metrics}
    norm = {}
    for m in metrics:
        mn, mx = min(raw[m]), max(raw[m])
        norm[m] = [(v-mn)/(mx-mn+1e-9) for v in raw[m]]
    N = len(cats)
    angles = [n/float(N)*2*np.pi for n in range(N)] + [0]
    cmap = ["#9DC3E6","#F4B183","#2E609A","#C00000"]
    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white"); ax.set_facecolor("#F8F8F8")
    for idx, method in enumerate(morder):
        vals = [norm[m][idx] for m in metrics] + [norm[metrics[0]][idx]]
        ax.plot(angles, vals, "o-", lw=2.2, color=cmap[idx], label=method, ms=6)
        ax.fill(angles, vals, alpha=0.12, color=cmap[idx])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, fontsize=11, fontweight="bold")
    ax.set_yticks([0.25,0.5,0.75,1.0]); ax.set_yticklabels(["25%","50%","75%","100%"],
                                                              fontsize=8, color="gray")
    ax.set_ylim(0,1); ax.grid(color="#DDDDDD", ls="--", lw=0.8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32,1.12), fontsize=11,
              framealpha=0.9, edgecolor="#CCCCCC")
    ax.set_title("So sánh tổng hợp 5 độ đo (giá trị chuẩn hóa)\n",
                 fontsize=13, fontweight="bold", color="#1F3864", pad=18)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/"radar_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved radar_chart.png", flush=True)

    # ── Fig 6 & 7: Training curves ───────────────────────────────
    plot_training_curves(bart_log, metric="rougeL",
                         save_path=str(FIGURES_DIR/"training_curves_bart.png"),
                         title="BART Fine-tuning: Train Loss & Val ROUGE-L")
    print("Saved training_curves_bart.png", flush=True)

    plot_training_curves(peg_log, metric="rougeL",
                         save_path=str(FIGURES_DIR/"training_curves_pegasus.png"),
                         title="PEGASUS Fine-tuning: Train Loss & Val ROUGE-L")
    print("Saved training_curves_pegasus.png", flush=True)

    plt.close("all")
    print("\n7 figures saved to results/figures/", flush=True)


# ─────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run full NLP pipeline")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip BART/PEGASUS training (use existing checkpoints)")
    args = parser.parse_args()

    total_t0 = time.time()

    # ── Load dataset ─────────────────────────────────────────────
    _header("PHASE 0: LOADING DATASET")
    ds = load_dialogsum(cache_dir="data/raw/dialogsum")
    test_dialogues  = list(ds["test"]["dialogue"])
    test_references = list(ds["test"]["summary"])
    print(f"Train: {len(ds['train'])}  Val: {len(ds['validation'])}  "
          f"Test: {len(ds['test'])}", flush=True)

    scores = {}

    # ── Phase 1: TextRank ─────────────────────────────────────────
    _header("PHASE 1: TEXTRANK (BASELINE)")
    from src.methods.textrank import TextRankSummarizer
    tr = TextRankSummarizer(top_n=3)
    t0 = time.time()
    preds_tr = [tr.summarize(d) for d in test_dialogues]
    print(f"TextRank inference done in {time.time()-t0:.0f}s", flush=True)
    scores["TextRank"] = eval_and_save(
        preds_tr, test_references,
        SCORES_DIR/"textrank_results.json",
        SCORES_DIR/"textrank_predictions.json",
        "TextRank",
    )

    # ── Phase 2: Flan-T5 zero-shot ───────────────────────────────
    _header("PHASE 2: FLAN-T5 ZERO-SHOT")
    from src.methods.flant5_zeroshot import FlanT5Summarizer
    ft5 = FlanT5Summarizer()
    t0 = time.time()
    preds_ft5 = ft5.summarize_batch(test_dialogues)
    print(f"Flan-T5 inference done in {time.time()-t0:.0f}s", flush=True)
    scores["Flan-T5"] = eval_and_save(
        preds_ft5, test_references,
        SCORES_DIR/"flant5_results.json",
        SCORES_DIR/"flant5_predictions.json",
        "Flan-T5",
    )
    del ft5

    # ── Phase 3: BART ────────────────────────────────────────────
    _header("PHASE 3: BART FINE-TUNING" if not args.skip_train
            else "PHASE 3: BART INFERENCE (skip training)")
    from src.methods.bart_finetune import train as bart_train, predict as bart_predict

    bart_log = None
    if not args.skip_train:
        t0 = time.time()
        trainer = bart_train(ds,
                             config_path="configs/bart_config.yaml",
                             output_dir="models/bart_finetuned")
        print(f"BART training done in {(time.time()-t0)/60:.1f} min", flush=True)
        bart_log = trainer.state.log_history
        with open(SCORES_DIR/"bart_train_log.json", "w") as f:
            json.dump(bart_log, f, indent=2)
    else:
        print("Skipping BART training, using existing checkpoint.", flush=True)
        try:
            with open(SCORES_DIR/"bart_train_log.json") as f:
                bart_log = json.load(f)
        except FileNotFoundError:
            bart_log = []

    preds_bart = bart_predict(test_dialogues, model_dir="models/bart_finetuned")
    scores["BART"] = eval_and_save(
        preds_bart, test_references,
        SCORES_DIR/"bart_results.json",
        SCORES_DIR/"bart_predictions.json",
        "BART",
    )

    # ── Phase 4: PEGASUS ─────────────────────────────────────────
    _header("PHASE 4: PEGASUS FINE-TUNING" if not args.skip_train
            else "PHASE 4: PEGASUS INFERENCE (skip training)")
    from src.methods.pegasus_finetune import train as peg_train, predict as peg_predict

    peg_log = None
    if not args.skip_train:
        t0 = time.time()
        trainer = peg_train(ds,
                            config_path="configs/pegasus_config.yaml",
                            output_dir="models/pegasus_finetuned")
        print(f"PEGASUS training done in {(time.time()-t0)/60:.1f} min", flush=True)
        peg_log = trainer.state.log_history
        with open(SCORES_DIR/"pegasus_train_log.json", "w") as f:
            json.dump(peg_log, f, indent=2)
    else:
        print("Skipping PEGASUS training, using existing checkpoint.", flush=True)
        try:
            with open(SCORES_DIR/"pegasus_train_log.json") as f:
                peg_log = json.load(f)
        except FileNotFoundError:
            peg_log = []

    preds_peg = peg_predict(test_dialogues, model_dir="models/pegasus_finetuned")
    scores["PEGASUS"] = eval_and_save(
        preds_peg, test_references,
        SCORES_DIR/"pegasus_results.json",
        SCORES_DIR/"pegasus_predictions.json",
        "PEGASUS",
    )

    # ── Phase 5: Figures ─────────────────────────────────────────
    generate_figures(scores, bart_log or [], peg_log or [])

    # ── Phase 6: Report ──────────────────────────────────────────
    _header("PHASE 6: GENERATING REPORT")
    import subprocess
    res = subprocess.run(
        [sys.executable, "report/generate_report.py"],
        cwd=str(Path(__file__).parent.parent),
        capture_output=True, text=True,
    )
    print(res.stdout, flush=True)
    if res.returncode != 0:
        print("ERROR generating report:\n", res.stderr[-2000:], flush=True)

    # ── Summary ──────────────────────────────────────────────────
    elapsed = time.time() - total_t0
    _header("ALL DONE")
    print(f"Total time: {elapsed/60:.1f} minutes\n", flush=True)
    header = f"{'Method':<12} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'BERTScore':>10} {'BLEU':>8}"
    print(header)
    print("-" * len(header))
    for method in ["TextRank", "Flan-T5", "BART", "PEGASUS"]:
        s = scores[method]
        print(f"{method:<12} {s['rouge1']:>8.4f} {s['rouge2']:>8.4f} "
              f"{s['rougeL']:>8.4f} {s['bertscore_f1']:>10.4f} {s['bleu']:>8.4f}")
    print(f"\nReport: report/report.docx", flush=True)
    print(f"Figures: results/figures/ (7 files)", flush=True)


if __name__ == "__main__":
    main()
