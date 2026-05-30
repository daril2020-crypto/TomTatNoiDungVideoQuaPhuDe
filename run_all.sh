#!/usr/bin/env bash
# ============================================================
# run_all.sh — Cài đặt và chạy toàn bộ pipeline dự án
# Hệ điều hành: Linux / macOS
# Yêu cầu: Python 3.10+, GPU NVIDIA (CUDA 12.x) khuyến nghị
#
# Cách dùng:
#   chmod +x run_all.sh
#   ./run_all.sh                  # full pipeline (train + eval + report)
#   ./run_all.sh --skip-train     # bỏ qua training, dùng model đã có
# ============================================================

set -euo pipefail

SKIP_TRAIN=""
if [[ "${1:-}" == "--skip-train" ]]; then
    SKIP_TRAIN="--skip-train"
    echo "[INFO] Chế độ skip-train: bỏ qua huấn luyện BART/PEGASUS."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Kiểm tra Python ──────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Bước 1: Kiểm tra phiên bản Python"
echo "============================================================"
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" -c "import sys; print(sys.version_info[:2])")
        if "$candidate" -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
            PYTHON_BIN="$candidate"
            echo "[OK] Dùng $PYTHON_BIN (version: $($PYTHON_BIN --version))"
            break
        fi
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    echo "[LỖI] Không tìm thấy Python >= 3.10. Vui lòng cài đặt Python trước."
    exit 1
fi

# ── Tạo môi trường ảo ────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Bước 2: Tạo môi trường ảo Python (.venv)"
echo "============================================================"
if [[ ! -d ".venv" ]]; then
    "$PYTHON_BIN" -m venv .venv
    echo "[OK] Đã tạo .venv"
else
    echo "[OK] .venv đã tồn tại, bỏ qua."
fi

source .venv/bin/activate
echo "[OK] Đã kích hoạt .venv"

# ── Cài đặt PyTorch ──────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Bước 3: Cài đặt PyTorch (CUDA 12.4)"
echo "============================================================"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[OK] PyTorch + CUDA đã sẵn sàng. Bỏ qua."
else
    echo "[INFO] Đang cài PyTorch CUDA 12.4..."
    pip install --upgrade pip --quiet
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --quiet
    # Kiểm tra lại; nếu không có GPU thì cài CPU-only
    if ! python -c "import torch" 2>/dev/null; then
        echo "[CẢNH BÁO] Cài CUDA wheel thất bại, thử CPU-only..."
        pip install torch==2.5.1 --quiet
    fi
    echo "[OK] PyTorch đã cài xong."
fi

# ── Cài đặt dependencies ─────────────────────────────────────────
echo ""
echo "============================================================"
echo " Bước 4: Cài đặt dependencies (requirements.txt)"
echo "============================================================"
pip install -r requirements.txt --quiet
pip install python-docx lxml --quiet
echo "[OK] Tất cả packages đã cài xong."

# ── Tải dữ liệu NLTK ─────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Bước 5: Tải dữ liệu NLTK"
echo "============================================================"
python -c "
import nltk
for r in ['punkt_tab', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{r}' if r == 'punkt_tab' else f'corpora/{r}')
        print(f'[OK] {r} đã có')
    except LookupError:
        nltk.download(r, quiet=True)
        print(f'[OK] {r} đã tải')
"

# ── Kiểm tra GPU ─────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Bước 6: Kiểm tra GPU"
echo "============================================================"
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'[GPU {i}] {name} ({mem:.1f} GB VRAM)')
else:
    print('[CẢNH BÁO] Không phát hiện GPU. Pipeline sẽ chạy trên CPU (rất chậm).')
    echo ''
    echo 'Bạn có muốn tiếp tục trên CPU không? [y/N]'
    read -r reply
    if [[ ! \"$reply\" =~ ^[Yy]$ ]]; then
        echo 'Đã hủy.'
        exit 1
    fi
"

# ── Tạo thư mục output ───────────────────────────────────────────
mkdir -p results/scores results/figures models/bart_finetuned models/pegasus_finetuned

# ── Chạy pipeline chính ──────────────────────────────────────────
echo ""
echo "============================================================"
echo " Bước 7: Chạy toàn bộ pipeline"
echo "============================================================"
echo "[INFO] Bắt đầu lúc: $(date)"
echo "[INFO] Ước tính thời gian: ~2.5 giờ (có GPU A5000)"
echo ""

python scripts/run_all.py $SKIP_TRAIN

echo ""
echo "============================================================"
echo " HOÀN THÀNH!"
echo "============================================================"
echo ""
echo "Kết quả:"
echo "  Điểm đánh giá : results/scores/"
echo "  Hình ảnh (7)  : results/figures/"
echo "  Báo cáo Word  : report/report.docx"
echo "  Models        : models/bart_finetuned/  models/pegasus_finetuned/"
echo ""
echo "Xem báo cáo: report/report.docx"
