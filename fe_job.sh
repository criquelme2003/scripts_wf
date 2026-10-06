#!/bin/bash -l
#SBATCH -n 1
#SBATCH --job-name=fe_job
#SBATCH --cpus-per-task=8
#SBATCH --no-requeue
#SBATCH --output=logs/%x.%j

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# TF 2.13 (forgeffects) necesita cuDNN 8 (pip) y CUDA 11.8 (sistema); el sistema trae cuDNN 9.
CUDNN_LIB="$SCRIPT_DIR/.conda_env/lib/python3.10/site-packages/nvidia/cudnn/lib"
CUBLAS_LIB="$SCRIPT_DIR/.conda_env/lib/python3.10/site-packages/nvidia/cublas/lib"
export LD_LIBRARY_PATH="$CUDNN_LIB:$CUBLAS_LIB:/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# exec: python reemplaza al shell para recibir directamente las señales de SLURM.
exec "$SCRIPT_DIR/.conda_env/bin/python" "$SCRIPT_DIR/fe_job.py" "$@"
