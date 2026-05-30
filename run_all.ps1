# ============================================================
# run_all.ps1 — Cài đặt và chạy toàn bộ pipeline dự án
# Hệ điều hành: Windows 10/11 (PowerShell 5.1+)
# Yêu cầu: Python 3.10+, GPU NVIDIA (CUDA 12.x) khuyến nghị
#
# Cách dùng (mở PowerShell với quyền thường):
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # lần đầu
#   .\run_all.ps1                   # full pipeline
#   .\run_all.ps1 --skip-train      # bỏ qua training
# ============================================================

param(
    [switch]$SkipTrain
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Write-Header($msg) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " $msg" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Write-OK($msg)   { Write-Host "[OK] $msg"      -ForegroundColor Green  }
function Write-Info($msg) { Write-Host "[INFO] $msg"    -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[LOI] $msg"     -ForegroundColor Red    }

# ── Bước 1: Kiểm tra Python ──────────────────────────────────────
Write-Header "Bước 1: Kiểm tra phiên bản Python"

$PythonBin = $null
foreach ($candidate in @("python3.12","python3.11","python3.10","python3","python")) {
    try {
        $ver = & $candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($ver) {
            $parts = $ver.Split(".")
            if ([int]$parts[0] -ge 3 -and [int]$parts[1] -ge 10) {
                $PythonBin = $candidate
                Write-OK "Dùng $PythonBin (version: $ver)"
                break
            }
        }
    } catch {}
}

if (-not $PythonBin) {
    Write-Err "Không tìm thấy Python >= 3.10."
    Write-Info "Tải tại: https://www.python.org/downloads/"
    exit 1
}

# ── Bước 2: Tạo môi trường ảo ────────────────────────────────────
Write-Header "Bước 2: Tạo môi trường ảo Python (.venv)"

if (-not (Test-Path ".venv")) {
    & $PythonBin -m venv .venv
    Write-OK "Đã tạo .venv"
} else {
    Write-OK ".venv đã tồn tại, bỏ qua."
}

$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
$VenvPip    = Join-Path $ScriptDir ".venv\Scripts\pip.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Err ".venv không hợp lệ. Xóa .venv và thử lại."
    exit 1
}
Write-OK "Đã xác định: $VenvPython"

# ── Bước 3: Cài đặt PyTorch ──────────────────────────────────────
Write-Header "Bước 3: Cài đặt PyTorch (CUDA 12.4)"

$hasCuda = & $VenvPython -c "import torch; print(torch.cuda.is_available())" 2>$null
if ($hasCuda -eq "True") {
    Write-OK "PyTorch + CUDA đã sẵn sàng."
} else {
    Write-Info "Đang cài PyTorch CUDA 12.4..."
    & $VenvPip install --upgrade pip --quiet
    try {
        & $VenvPip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --quiet
        Write-OK "PyTorch CUDA đã cài xong."
    } catch {
        Write-Info "Cài CUDA wheel thất bại, thử CPU-only..."
        & $VenvPip install torch==2.5.1 --quiet
        Write-OK "PyTorch CPU đã cài xong (training sẽ rất chậm)."
    }
}

# ── Bước 4: Cài đặt dependencies ─────────────────────────────────
Write-Header "Bước 4: Cài đặt dependencies (requirements.txt)"

& $VenvPip install -r requirements.txt --quiet
& $VenvPip install python-docx lxml --quiet
Write-OK "Tất cả packages đã cài xong."

# ── Bước 5: Tải dữ liệu NLTK ─────────────────────────────────────
Write-Header "Bước 5: Tải dữ liệu NLTK"

& $VenvPython -c @"
import nltk
for r in ['punkt_tab', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{r}' if r == 'punkt_tab' else f'corpora/{r}')
        print(f'[OK] {r} da co')
    except LookupError:
        nltk.download(r, quiet=True)
        print(f'[OK] {r} da tai')
"@

# ── Bước 6: Kiểm tra GPU ─────────────────────────────────────────
Write-Header "Bước 6: Kiểm tra GPU"

& $VenvPython -c @"
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'[GPU {i}] {name} ({mem:.1f} GB VRAM)')
else:
    print('[CANH BAO] Khong phat hien GPU. Pipeline se chay tren CPU (rat cham).')
"@

# ── Tạo thư mục output ───────────────────────────────────────────
$dirs = @("results\scores","results\figures","models\bart_finetuned","models\pegasus_finetuned","data\raw\dialogsum")
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

# ── Bước 7: Chạy pipeline chính ──────────────────────────────────
Write-Header "Bước 7: Chạy toàn bộ pipeline"

$startTime = Get-Date
Write-Info "Bắt đầu lúc: $startTime"
Write-Info "Ước tính thời gian: ~2.5 giờ (có GPU A5000)"
Write-Host ""

$extraArgs = if ($SkipTrain) { "--skip-train" } else { "" }

if ($extraArgs) {
    & $VenvPython scripts\run_all.py $extraArgs
} else {
    & $VenvPython scripts\run_all.py
}

if ($LASTEXITCODE -ne 0) {
    Write-Err "Pipeline thất bại với exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

$endTime  = Get-Date
$elapsed  = ($endTime - $startTime).TotalMinutes

Write-Header "HOAN THANH!"
Write-Host ""
Write-OK "Tổng thời gian chạy: $([Math]::Round($elapsed, 1)) phút"
Write-Host ""
Write-Host "Kết quả:" -ForegroundColor White
Write-Host "  Điểm đánh giá : results\scores\"
Write-Host "  Hình ảnh (7)  : results\figures\"
Write-Host "  Báo cáo Word  : report\report.docx"
Write-Host "  Models        : models\bart_finetuned\   models\pegasus_finetuned\"
Write-Host ""
Write-Info "Mở báo cáo: Start-Process report\report.docx"
