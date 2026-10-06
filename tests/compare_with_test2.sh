#!/bin/bash -l
#SBATCH -n 1
#SBATCH --job-name=compare_test2
#SBATCH --cpus-per-task=8
#SBATCH --no-requeue
#SBATCH --output=logs/%x.%j

# Criterio de aceptación 1: compara sweep_job con test2.py.
# Uso: sbatch tests/compare_with_test2.sh [ruta/a/test2.py]

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

exec "$SCRIPT_DIR/.conda_env/bin/python" "$SCRIPT_DIR/tests/compare_with_test2.py" "$@"
