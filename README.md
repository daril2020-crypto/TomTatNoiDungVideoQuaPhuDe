# Đề tài #17 — Tóm tắt Nội dung Video dựa trên Phụ đề

**Môn:** Xử lý Ngôn ngữ Tự nhiên | **Đơn vị:** ĐH Công nghiệp TP.HCM

Hệ thống so sánh **4 phương pháp tóm tắt văn bản** trên tập dữ liệu hội thoại DialogSum, đánh giá bằng ROUGE-1/2/L, BERTScore F1 và BLEU-4.

---

## Kết quả nhanh

| Phương pháp    | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | BLEU-4 |
|----------------|---------|---------|---------|-------------|--------|
| TextRank       | 0.2139  | 0.0679  | 0.1696  | 0.8524      | 0.0393 |
| Flan-T5 (0-shot)| 0.1912 | 0.0770  | 0.1660  | 0.8531      | 0.0364 |
| BART (fine-tuned)| 0.3711| 0.1553  | 0.2876  | 0.9015      | 0.1189 |
| **PEGASUS (fine-tuned)**| **0.4321**|**0.1792**|**0.3535**|**0.9120**|**0.1904**|

---

## Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu | Khuyến nghị |
|---|---|---|
| OS | Ubuntu 20.04 / Windows 10 | Ubuntu 22.04 / Windows 11 |
| Python | 3.10+ | 3.12 |
| GPU | NVIDIA 8 GB VRAM | RTX A5000 24 GB |
| CUDA | 11.8+ | 12.4 |
| RAM | 16 GB | 32 GB |
| Dung lượng | 20 GB (models + data) | 30 GB |

> **Không có GPU:** Pipeline vẫn chạy được trên CPU, nhưng bước training BART/PEGASUS
> mỗi model sẽ mất 12–24 giờ thay vì ~75 phút.

---

## Cấu trúc thư mục

```
.
├── configs/
│   ├── bart_config.yaml        # Siêu tham số BART
│   └── pegasus_config.yaml     # Siêu tham số PEGASUS
├── data/
│   └── raw/dialogsum/          # Cache dataset HuggingFace (tự tải)
├── models/
│   ├── bart_finetuned/         # Checkpoint sau training
│   └── pegasus_finetuned/
├── notebooks/                  # Jupyter notebooks từng bước
│   ├── 00_setup_and_eda.ipynb
│   ├── 02_textrank_baseline.ipynb
│   ├── 03_bart_finetune.ipynb
│   ├── 04_pegasus_finetune.ipynb
│   ├── 05_flant5_zeroshot.ipynb
│   └── 06_evaluation_comparison.ipynb
├── report/
│   ├── generate_report.py      # Script xuất báo cáo DOCX
│   └── report.docx             # Báo cáo Word (output)
├── results/
│   ├── figures/                # 7 hình ảnh biểu đồ (PNG)
│   └── scores/                 # Điểm đánh giá + predictions (JSON)
├── scripts/
│   ├── run_all.py              # Orchestrator chính (Python)
│   ├── retrain_all.py          # Chỉ train BART + PEGASUS
│   └── update_figures_and_report.py
├── src/
│   ├── data/
│   │   ├── loader.py           # load_dialogsum()
│   │   └── preprocessing.py    # clean_dialogue(), tokenize_for_model()
│   ├── evaluation/
│   │   ├── metrics.py          # compute_rouge(), compute_bertscore(), compute_bleu()
│   │   └── visualize.py        # plot_comparison_bar(), plot_training_curves()
│   └── methods/
│       ├── textrank.py         # TextRankSummarizer
│       ├── bart_finetune.py    # train(), predict()
│       ├── pegasus_finetune.py # train(), predict()
│       └── flant5_zeroshot.py  # FlanT5Summarizer
├── requirements.txt
├── run_all.sh                  # Script chạy toàn bộ (Linux/macOS)
└── run_all.ps1                 # Script chạy toàn bộ (Windows PowerShell)
```

---

## Cách chạy

### Linux / macOS

```bash
# Cấp quyền thực thi lần đầu
chmod +x run_all.sh

# Chạy toàn bộ (train + eval + báo cáo) — ~2.5 giờ với GPU A5000
./run_all.sh

# Bỏ qua bước training nếu model đã có sẵn trong models/
./run_all.sh --skip-train
```

### Windows (PowerShell)

```powershell
# Cho phép chạy script lần đầu (chỉ cần làm 1 lần)
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# Chạy toàn bộ
.\run_all.ps1

# Bỏ qua bước training
.\run_all.ps1 -SkipTrain
```

Cả hai script sẽ tự động:
1. Tạo môi trường ảo Python `.venv`
2. Cài PyTorch (CUDA 12.4) và toàn bộ dependencies
3. Tải dataset DialogSum từ HuggingFace Hub
4. Chạy 4 phương pháp và tính điểm
5. Xuất 7 hình ảnh biểu đồ
6. Tạo file `report/report.docx`

---

## Chạy từng bước thủ công

Nếu muốn kiểm soát từng giai đoạn, kích hoạt venv trước:

```bash
# Linux
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Sau đó chạy các lệnh sau từ thư mục gốc (`/workspace`):

### 1. Kiểm tra môi trường

```python
# Kiểm tra GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Kiểm tra dataset
python -c "from src.data.loader import load_dialogsum; d = load_dialogsum(); print(d)"

# Smoke test TextRank
python -c "from src.methods.textrank import TextRankSummarizer; s = TextRankSummarizer(); print(s.summarize('Hello. How are you? I am fine. What about you?'))"
```

### 2. Chạy từng phương pháp độc lập

```python
# TextRank (không cần GPU, ~2 phút)
python -c "
from src.data.loader import load_dialogsum
from src.methods.textrank import TextRankSummarizer
ds = load_dialogsum(cache_dir='data/raw/dialogsum')
summarizer = TextRankSummarizer(top_n=3)
pred = summarizer.summarize(ds['test'][0]['dialogue'])
print(pred)
"

# Training BART + PEGASUS + Eval (~2.5 giờ tổng)
python scripts/retrain_all.py

# Chỉ tạo lại hình ảnh + báo cáo (sau khi đã có results/scores/)
python scripts/update_figures_and_report.py
```

### 3. Tạo báo cáo Word riêng lẻ

```bash
python report/generate_report.py
```

---

## Cài đặt thủ công (không dùng script)

```bash
# 1. Tạo và kích hoạt venv
python3 -m venv .venv
source .venv/bin/activate          # Linux
# .\.venv\Scripts\Activate.ps1    # Windows

# 2. Cài PyTorch CUDA 12.4
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 3. Cài các dependencies còn lại
pip install -r requirements.txt
pip install python-docx lxml

# 4. Tải NLTK resources
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
```

---

## Siêu tham số chính

### BART (`configs/bart_config.yaml`)

| Tham số | Giá trị |
|---|---|
| Model gốc | `facebook/bart-large-cnn` |
| Max source length | 1024 tokens |
| Learning rate | 3×10⁻⁵ |
| Batch size (hiệu dụng) | 16 (4 per device × 4 grad accum) |
| Epochs | 3 |
| Mixed precision | bf16 |

### PEGASUS (`configs/pegasus_config.yaml`)

| Tham số | Giá trị |
|---|---|
| Model gốc | `google/pegasus-xsum` |
| Max source length | 512 tokens |
| Learning rate | 3×10⁻⁵ |
| Batch size (hiệu dụng) | 16 |
| Epochs | 3 |
| Mixed precision | bf16 |

---

## Output sau khi chạy xong

```
results/
├── scores/
│   ├── textrank_results.json       # Điểm TextRank
│   ├── flant5_results.json         # Điểm Flan-T5
│   ├── bart_results.json           # Điểm BART
│   ├── pegasus_results.json        # Điểm PEGASUS
│   ├── bart_train_log.json         # Training log BART
│   ├── pegasus_train_log.json      # Training log PEGASUS
│   ├── *_predictions.json          # 50 ví dụ predictions
│   └── ...
└── figures/
    ├── pipeline_diagram.png        # Sơ đồ kiến trúc hệ thống
    ├── length_distribution.png     # Phân phối độ dài dataset
    ├── rouge_comparison.png        # So sánh ROUGE-1/2/L
    ├── bertscore_bleu_comparison.png
    ├── radar_chart.png             # So sánh 5 độ đo (chuẩn hóa)
    ├── training_curves_bart.png    # Đường cong huấn luyện BART
    └── training_curves_pegasus.png

report/
└── report.docx                     # Báo cáo hoàn chỉnh (Times New Roman 13pt)
```

---

## Ước tính thời gian chạy

| Giai đoạn | GPU A5000 | CPU (i9) |
|---|---|---|
| TextRank (1500 mẫu) | ~2 phút | ~2 phút |
| Flan-T5 zero-shot | ~10 phút | ~60 phút |
| BART training (3 epochs) | ~75 phút | ~15 giờ |
| PEGASUS training (3 epochs) | ~60 phút | ~12 giờ |
| Evaluation metrics | ~15 phút | ~30 phút |
| Figures + Report | ~2 phút | ~2 phút |
| **Tổng** | **~2.5 giờ** | **~28 giờ** |

---

## Tài liệu tham khảo

- **DialogSum** — Chen et al. (2021), ACL Findings
- **BART** — Lewis et al. (2020), ACL
- **PEGASUS** — Zhang et al. (2020), ICML
- **Flan-T5** — Chung et al. (2022), arXiv:2210.11416
- **ROUGE** — Lin (2004), ACL Workshop
- **BERTScore** — Zhang et al. (2020), ICLR
