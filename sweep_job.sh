#!/bin/bash -l
#SBATCH -n 1
#SBATCH --job-name=sweep_job
#SBATCH --cpus-per-task=8
#SBATCH --no-requeue
#SBATCH --output=logs/%x.%j

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# exec: python reemplaza al shell para recibir directamente el SIGTERM de SLURM (--time).
exec "$SCRIPT_DIR/.conda_env/bin/python" "$SCRIPT_DIR/sweep_job.py" "$@"
