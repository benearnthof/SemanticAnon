#!/usr/bin/env bash
set -Eeuo pipefail

# Run from the project root (normally /workspace):
#   bash setup.sh
#
# Optional end-to-end transcription test:
#   TEST_AUDIO=/workspace/data/audio/example.mp3 bash setup.sh

WORKDIR="${WORKDIR:-/workspace/SemanticAnon}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV="${WORKDIR}/.venv"

echo "=============================================="
echo " SemanticAnon GPU environment setup"
echo "=============================================="
echo "Project directory: ${WORKDIR}"
echo "Python version:    ${PYTHON_VERSION}"
echo "Virtualenv:        ${VENV}"

if [[ ! -f "${WORKDIR}/pyproject.toml" ]]; then
    echo "ERROR: pyproject.toml not found in ${WORKDIR}"
    echo "Place setup.sh and pyproject.toml in the project root, then rerun."
    exit 1
fi
cd "${WORKDIR}"

echo
echo "[1/6] Installing system dependencies..."
if [[ "${EUID}" -eq 0 ]]; then
    SUDO=""
else
    command -v sudo >/dev/null 2>&1 || {
        echo "ERROR: apt installation requires root or sudo."
        exit 1
    }
    SUDO="sudo"
fi
${SUDO} apt-get update
${SUDO} apt-get install -y ca-certificates curl ffmpeg build-essential

echo
echo "[2/6] Checking NVIDIA driver..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "WARNING: nvidia-smi not found. Continuing, but GPU inference will not work."
fi

echo
echo "[3/6] Installing uv and Python ${PYTHON_VERSION}..."
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is not on PATH after installation."
    exit 1
}
uv python install "${PYTHON_VERSION}"

echo
echo "[4/6] Creating/updating the project environment..."
export UV_PROJECT_ENVIRONMENT="${VENV}"

# First run resolves dependencies and creates uv.lock. Commit uv.lock to Git;
# subsequent installs use the exact locked versions.
if [[ ! -f "${WORKDIR}/uv.lock" ]]; then
    echo "No uv.lock found; resolving dependencies for the first time..."
    uv lock
fi

uv sync --locked --python "${PYTHON_VERSION}" \
    --all-extras

PYTHON="${VENV}/bin/python"

echo
echo "[5/6] Configuring CUDA shared-library paths..."
CUDA_LIB_DIRS="$("${PYTHON}" - <<'PY'
import importlib.util
from pathlib import Path

directories = []
for package in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime"):
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    if spec and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            libdir = Path(location) / "lib"
            if libdir.is_dir():
                directories.append(str(libdir))
print(":".join(dict.fromkeys(directories)))
PY
)"

if [[ -z "${CUDA_LIB_DIRS}" ]]; then
    echo "ERROR: Could not locate NVIDIA CUDA libraries in the virtual environment."
    exit 1
fi
export LD_LIBRARY_PATH="${CUDA_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cat > "${WORKDIR}/activate.sh" <<EOF
#!/usr/bin/env bash
source "${VENV}/bin/activate"
export LD_LIBRARY_PATH="${CUDA_LIB_DIRS}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
EOF
chmod +x "${WORKDIR}/activate.sh"

echo "CUDA library directories:"
printf '%s\n' "${CUDA_LIB_DIRS}" | tr ':' '\n'

echo
echo "[6/6] Verifying installed packages and GPU access..."
"${PYTHON}" - <<'PY'
import importlib
import sys

print("Python:", sys.version.split()[0])
for name in (
    "requests", "yt_dlp", "gdown", "faster_whisper", "whisperx",
    "torch", "torchaudio", "sentence_transformers", "transformers",
    "onnxruntime", "spacy", "ctranslate2",
):
    module = importlib.import_module(name)
    print(f"  OK  {name}: {getattr(module, '__version__', 'installed')}")

import torch
import ctranslate2

print()
print("PyTorch CUDA available:", torch.cuda.is_available())
print("CTranslate2 CUDA devices:", ctranslate2.get_cuda_device_count())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot access CUDA; check the NVIDIA driver and PyTorch wheel.")
if ctranslate2.get_cuda_device_count() < 1:
    raise SystemExit("ERROR: CTranslate2 cannot access a CUDA device; check CUDA libraries.")
print("GPU environment checks passed.")
PY

if [[ -n "${TEST_AUDIO:-}" ]]; then
    [[ -f "${TEST_AUDIO}" ]] || {
        echo "ERROR: TEST_AUDIO does not exist: ${TEST_AUDIO}"
        exit 1
    }
    echo
    echo "Running optional Faster-Whisper transcription test..."
    "${PYTHON}" - "${TEST_AUDIO}" <<'PY'
import sys
from faster_whisper import WhisperModel

model = WhisperModel("tiny", device="cuda", device_index=0, compute_type="float16")
segments, info = model.transcribe(
    sys.argv[1], beam_size=1, clip_timestamps="0,30"
)
segments = list(segments)
print("Language:", info.language)
print("Language probability:", info.language_probability)
for segment in segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
print("Faster-Whisper GPU test passed.")
PY
fi

echo
echo "=============================================="
echo " Setup complete"
echo "=============================================="
echo "Activate with: source ${WORKDIR}/activate.sh"
echo "Environment:   ${VENV}"
echo "Credentials:   ${WORKDIR}/secrets.env (never commit this file)"
