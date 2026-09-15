#!/bin/bash -l
#SBATCH -n 1
#SBATCH --job-name=forgethreads-notifier
#SBATCH --cpus-per-task=1
#SBATCH --no-requeue
#SBATCH --output=logs/%x.%j

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

"$SCRIPT_DIR/.conda_env/bin/python" "$SCRIPT_DIR/notifier.py" "$@"
