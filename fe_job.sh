#!/bin/bash -l
#SBATCH -n 1
#SBATCH --job-name=fe_job
#SBATCH --cpus-per-task=8
#SBATCH --no-requeue
#SBATCH --output=logs/%x.%j

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# CUDA 11.8 + cuDNN 8.6 para TF 2.13, desde .conda_env (ver tf_env.sh).
source "$SCRIPT_DIR/tf_env.sh"

# exec: python reemplaza al shell para recibir directamente las señales de SLURM.
exec "$SCRIPT_DIR/.conda_env/bin/python" "$SCRIPT_DIR/fe_job.py" "$@"
