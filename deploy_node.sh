#!/bin/bash
# Prepara un nodo después de `git pull`: actualiza .conda_env, crea los
# directorios de trabajo y verifica el entorno con un job de SLURM.
# Uso (en la raíz del repo, en cada nodo): bash deploy_node.sh
#
# Las versiones deben coincidir con environment.yml.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CONDA="$(command -v conda || echo /opt/miniconda3/condabin/conda)"
PY=./.conda_env/bin/python

echo "=== 1/4 Entorno conda (.conda_env) ==="
if [ ! -x "$PY" ]; then
    echo ".conda_env no existe: creándolo desde environment.yml"
    "$CONDA" env create -p ./.conda_env -f environment.yml
else
    # forgeffects exige tensorflow==2.13, que exige numpy<=1.24.3.
    "$CONDA" install -y -p ./.conda_env -c https://repo.anaconda.com/pkgs/main --override-channels "numpy=1.24.3"
    "$PY" -m pip install \
        "numpy==1.24.3" \
        "forgeffects==0.2.5" \
        "tensorflow==2.13.0" \
        "tensorflow-probability==0.20.0" \
        "nvidia-cudnn-cu11==8.6.0.163" \
        "pandas==2.3.3" \
        "pytest==9.1.1"
fi

echo "=== 2/4 Versiones instaladas ==="
"$PY" -c "import numpy, tensorflow, forgeffects, pandas; print('numpy', numpy.__version__, '| tensorflow', tensorflow.__version__, '| pandas', pandas.__version__)"

echo "=== 3/4 Directorios de trabajo ==="
mkdir -p logs jobs_results jobs_inputs jobs_failed_notify

echo "=== 4/4 Verificación vía SLURM (tests/check_env.sh) ==="
JOB_ID=$(sbatch --wait --parsable tests/check_env.sh) || true
LOG="logs/check_env.${JOB_ID}"
if [ -n "$JOB_ID" ] && grep -q "CHECK_ENV OK" "$LOG" 2>/dev/null; then
    grep -E "numpy|GPUs|matmul" "$LOG"
    echo "DEPLOY OK (job $JOB_ID)"
else
    echo "DEPLOY FALLIDO: revisar $LOG" >&2
    [ -f "$LOG" ] && tail -20 "$LOG" >&2
    exit 1
fi
