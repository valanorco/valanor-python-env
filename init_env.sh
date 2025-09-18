#!/usr/bin/env bash
# File: init_env.sh
# Purpose: Ultra-fast Python env setup using uv, with selectable tiers and optional CUDA-aware ML frameworks.
# Platform: Ubuntu 24.04/25.04 (Python 3.12+ / 3.13)
#
# Examples:
#   ./init_env.sh                            # standard tier
#   ./init_env.sh -t minimal                 # minimal tier
#   ./init_env.sh -t full --gpu all          # full tier + Torch/TF/JAX (CUDA auto-detect)
#   ./init_env.sh --gpu torch --cuda cu126 --torch-version 2.5.1

set -euo pipefail

# -------------------------
# Defaults / CLI
# -------------------------
VENV_DIR=".venv"
PYTHON_BIN="python3"
TIER="standard"            # minimal | standard | full
INSTALL_GPU="none"         # none | torch | tf | jax | all
CUDA_MODE="auto"           # auto | cpu | cu121 | cu124 | cu126 | cu128
TORCH_VERSION=""
TF_PACKAGE="tensorflow"    # e.g. tensorflow==2.17.*
JAX_VERSION=""
TORCH_EXTRAS="none"        # DEFAULT: none | vision | audio | all

print_help() {
  cat <<'EOF'
init_env.sh - Create and initialize a Python virtual environment using uv.

Options:
  -n, --name <dir>         Venv directory (default: .venv)
  -p, --python <bin>       Python interpreter (default: python3)
  -t, --tier <tier>        minimal | standard | full   (default: standard)
  --gpu <which>            none | torch | tf | jax | all  (default: none)
  --cuda <mode>            auto | cpu | cu121 | cu124 | cu126 | cu128 (default: auto)
  --torch-version <ver>    Pin torch version (e.g., 2.5.1)
  --torch-extras <opt>     none | vision | audio | all  (default: none)
  --tf-pkg <pkg>           TensorFlow package (default: tensorflow)
  --jax-version <ver>      Pin jax/jaxlib version
  -h, --help               Show help
EOF
}

while [[ "${1-}" != "" ]]; do
  case "$1" in
    -n|--name)            VENV_DIR="$2"; shift 2 ;;
    -p|--python)          PYTHON_BIN="$2"; shift 2 ;;
    -t|--tier)            TIER="$2"; shift 2 ;;
    --gpu)                INSTALL_GPU="$2"; shift 2 ;;
    --cuda)               CUDA_MODE="$2"; shift 2 ;;
    --torch-version)      TORCH_VERSION="$2"; shift 2 ;;
    --tf-pkg)             TF_PACKAGE="$2"; shift 2 ;;
    --jax-version)        JAX_VERSION="$2"; shift 2 ;;
    --torch-extras)       TORCH_EXTRAS="$2"; shift 2 ;;
    -h|--help)            print_help; exit 0 ;;
    *) echo "Unknown option: $1"; print_help; exit 1 ;;
  esac
done

need() { command -v "$1" >/dev/null 2>&1; }

get_pkg_mgr_hint() {
  if   need apt;    then echo "sudo apt update && sudo apt install -y"
  elif need dnf;    then echo "sudo dnf update && sudo dnf install -y"
  elif need yum;    then echo "sudo yum update && sudo yum install -y"
  elif need zypper; then echo "sudo zypper update && sudo zypper install -y"
  elif need pacman; then echo "sudo pacman -Syu --noconfirm"
  else echo "sudo <YOUR_PACKAGE_MANAGER> update && sudo <YOUR_PACKAGE_MANAGER> install -y"
  fi
}

# -------------------------
# Pre-flight checks
# -------------------------
if ! need uv; then
  echo "ERROR: 'uv' not found on PATH."
  echo "Install:  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
if ! need "$PYTHON_BIN"; then
  echo "ERROR: '$PYTHON_BIN' not found."
  echo "Hint: $(get_pkg_mgr_hint) python3 python3-venv python3-pip"
  exit 1
fi

# -------------------------
# Create & activate venv (uv)
# -------------------------
if [[ ! -d "$VENV_DIR" ]]; then
  echo "📦 Creating uv venv in '$VENV_DIR' ..."
  uv venv -p "$PYTHON_BIN" "$VENV_DIR"
else
  echo "♻️  Reusing existing venv at '$VENV_DIR'."
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
echo "✅ Activated: $(python -c 'import sys; print(sys.prefix)')"

# Seed pip inside the venv so tools that call `python -m pip` don’t break
uv pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true

# Enforce venv usage (PEP 668 guard) and prefer venv bin
export PATH="$VENV_DIR/bin:$PATH"
if python - <<'PY' | grep -q '^FALSE$'; then
import sys
print("TRUE" if (hasattr(sys, "base_prefix") and sys.prefix != sys.base_prefix) else "FALSE")
PY
  echo "ERROR: Not using the virtual environment Python. Aborting to avoid PEP 668."
  echo "Hint: source \"$VENV_DIR/bin/activate\" and retry."
  exit 1
fi

# Wrapper: use uv pip first (fast); on error fall back to venv pip
pip_install() {
  set +e
  uv pip install "$@"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "⚠️  uv pip failed or refused. Falling back to venv pip..."
    "$VENV_DIR/bin/pip" install "$@"
  fi
}

# -------------------------
# Generate requirements.txt for selected tier
# -------------------------
REQ_FILE="requirements.txt"

if [[ ! -f "$REQ_FILE" ]]; then
  echo "📝 Generating $REQ_FILE for tier: ${TIER} ..."
  case "$TIER" in
    minimal)
      cat > "$REQ_FILE" <<'EOF'
ipython
ipykernel
numpy>=2.3
pandas>=2.3,<3
pyarrow
fastapi
uvicorn[standard]
pydantic
SQLAlchemy
psycopg2-binary
EOF
      ;;
    standard)
      cat > "$REQ_FILE" <<'EOF'
ipython
ipykernel
jupyterlab
numpy>=2.3
pandas>=2.3,<3
pyarrow
duckdb
polars
dask[distributed]
pandera
great-expectations
scikit-learn
matplotlib
plotly
mlflow
fastapi
uvicorn[standard]
pydantic
SQLAlchemy
psycopg2-binary
EOF
      ;;
    full)
      cat > "$REQ_FILE" <<'EOF'
ipython
ipykernel
jupyterlab
numpy>=2.3
pandas>=2.3,<3
pyarrow
duckdb
polars
dask[distributed]
pyspark
pandera
great-expectations
scikit-learn
xgboost
lightgbm
matplotlib
plotly
mlflow
fastapi
uvicorn[standard]
pydantic
SQLAlchemy
psycopg2-binary
opencv-python-headless
# Optional cloud connectors (uncomment as needed)
# boto3
# google-cloud-storage
# google-cloud-bigquery
# pandas-gbq
# db-dtypes
# snowflake-connector-python
# snowflake-sqlalchemy
EOF
      ;;
    *)
      echo "ERROR: Unknown tier '$TIER'"; exit 1 ;;
  esac
fi

# -------------------------
# Install (venv-safe) with PySpark two-phase
# -------------------------
echo "📥 Installing ${TIER} stack (venv-safe) ..."

if [[ "$TIER" == "full" ]]; then
  # 1) Wheels-only for everything EXCEPT pyspark
  awk 'tolower($0) !~ /^pyspark/' "$REQ_FILE" > .req_nospark.txt
  pip_install --only-binary=:all: -r .req_nospark.txt
  rm -f .req_nospark.txt

  # 2) pyspark from sdist (pure Python; wheels may be missing on Py 3.13)
  echo "📥 Installing pyspark from sdist ..."
  pip_install --no-binary=pyspark "pyspark==4.0.0"
else
  pip_install --only-binary=:all: -r "$REQ_FILE"
fi

# -------------------------
# JAVA check (for PySpark)
# -------------------------
if [[ "$TIER" == "full" ]] && ! command -v java >/dev/null 2>&1; then
  echo "⚠️  Java not found; PySpark may fail at runtime. Try: sudo apt install -y openjdk-17-jre"
fi

# -------------------------
# CUDA detection (optional GPU frameworks)
# -------------------------
GPU_PRESENT=0
CUDA_TAG="cpu"

detect_cuda() {
  local forced="$1"
  if [[ "$forced" != "auto" ]]; then
    CUDA_TAG="$forced"
    [[ "$forced" != "cpu" ]] && GPU_PRESENT=1
    return 0
  fi
  if need nvidia-smi; then GPU_PRESENT=1; fi
  if [[ "$GPU_PRESENT" -eq 1 ]]; then
    if need nvcc; then
      # fixed regex: single escaped dot
      local ver; ver="$(nvcc --version 2>/dev/null | grep -Eo 'release[[:space:]]+[0-9]+\.[0-9]+' | awk '{print $2}' || true)"
      case "$ver" in
        12.8*) CUDA_TAG="cu128" ;;
        12.6*) CUDA_TAG="cu126" ;;
        12.4*) CUDA_TAG="cu124" ;;
        12.*)  CUDA_TAG="cu121" ;;
        *)     CUDA_TAG="cu121" ;;
      esac
    else
      CUDA_TAG="cu121"
    fi
  else
    CUDA_TAG="cpu"
  fi
}

install_torch() {
  local ver_suf=""
  [[ -n "$TORCH_VERSION" ]] && ver_suf="==${TORCH_VERSION}"

  case "$CUDA_TAG" in
    cpu)                        pip_install "torch${ver_suf}" --index-url https://download.pytorch.org/whl/cpu ;;
    cu121|cu124|cu126|cu128)    pip_install "torch${ver_suf}" --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" ;;
    *)                          pip_install "torch${ver_suf}" --index-url https://download.pytorch.org/whl/cpu ;;
  esac

  if [[ "$TORCH_EXTRAS" == "vision" || "$TORCH_EXTRAS" == "all" ]]; then
    set +e
    echo "🧩 Trying to install torchvision ..."
    case "$CUDA_TAG" in
      cpu)                     pip_install torchvision --index-url https://download.pytorch.org/whl/cpu ;;
      cu121|cu124|cu126|cu128) pip_install torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" ;;
      *)                       pip_install torchvision --index-url https://download.pytorch.org/whl/cpu ;;
    esac
    set -e
  fi

  if [[ "$TORCH_EXTRAS" == "audio" || "$TORCH_EXTRAS" == "all" ]]; then
    set +e
    echo "🧩 Trying to install torchaudio ..."
    case "$CUDA_TAG" in
      cpu)                     pip_install torchaudio --index-url https://download.pytorch.org/whl/cpu ;;
      cu121|cu124|cu126|cu128) pip_install torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" ;;
      *)                       pip_install torchaudio --index-url https://download.pytorch.org/whl/cpu ;;
    esac
    set -e
  fi
}

install_tf() { pip_install "${TF_PACKAGE}"; }
install_jax() {
  if [[ "$CUDA_TAG" == "cpu" ]]; then
    pip_install ${JAX_VERSION:+jax==$JAX_VERSION} || pip_install jax
  else
    pip_install "jax[cuda12]" ${JAX_VERSION:+jaxlib==$JAX_VERSION}
    pip_install --upgrade jax-cuda12-plugin || true
  fi
}

if [[ "$INSTALL_GPU" != "none" ]]; then
  detect_cuda "$CUDA_MODE"
  echo "🧮 CUDA decision: GPU=${GPU_PRESENT}  TAG=${CUDA_TAG}  |  Torch extras: ${TORCH_EXTRAS}"
  [[ "$INSTALL_GPU" == "torch" || "$INSTALL_GPU" == "all" ]] && install_torch
  [[ "$INSTALL_GPU" == "tf"    || "$INSTALL_GPU" == "all" ]] && install_tf
  [[ "$INSTALL_GPU" == "jax"   || "$INSTALL_GPU" == "all" ]] && install_jax
fi

# -------------------------
# Jupyter kernel
# -------------------------
KERNEL_NAME="$(basename "$(pwd)")-venv"
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_NAME" >/dev/null 2>&1 || true
echo "🧠 Registered Jupyter kernel: $KERNEL_NAME"

# -------------------------
# ENVIRONMENT.md (neutral report)
# -------------------------
ENV_MD="ENVIRONMENT.md"
{
  echo "# Environment Report"
  echo
  echo "**Generated:** $(date)"
  echo
  echo "- Tier: $TIER"
  echo "- GPU Frameworks: $INSTALL_GPU"
  echo "- CUDA: $CUDA_TAG"
  echo "- Torch extras: $TORCH_EXTRAS"
  echo
  echo "## Installed Packages"
  uv pip list | awk 'NR>2 {print "- " $1 "==" $2}'
} > "$ENV_MD"
echo "🧾 Generated $ENV_MD"

# -------------------------
# Demo FastAPI app
# -------------------------
if [[ ! -f "main.py" ]]; then
  cat > main.py <<'EOF'
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Valanor Demo API")

class Item(BaseModel):
    name: str
    value: float

@app.get("/")
def read_root():
    return {"message": "Hello from Valanor standard environment 🚀"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/items/")
def create_item(item: Item):
    return {"received": item.dict()}
EOF
  echo "🧪 Wrote FastAPI demo: main.py"
fi

echo
echo "🎉 Done."
echo "Activate later: source \"$VENV_DIR/bin/activate\""
echo "Tier: $TIER | GPU: $INSTALL_GPU | CUDA: $CUDA_TAG | Torch extras: $TORCH_EXTRAS"
