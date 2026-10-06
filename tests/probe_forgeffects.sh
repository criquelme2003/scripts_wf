#!/bin/bash -l
#SBATCH -n 1
#SBATCH --job-name=probe_forgeffects
#SBATCH --cpus-per-task=8
#SBATCH --no-requeue
#SBATCH --output=logs/%x.%j

# Sondeo empírico de forgeffects.FE (T2). Uso: sbatch tests/probe_forgeffects.sh

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# TF 2.13 necesita cuDNN 8 (pip) y CUDA 11.8 (sistema).
CUDNN_LIB="$SCRIPT_DIR/.conda_env/lib/python3.10/site-packages/nvidia/cudnn/lib"
CUBLAS_LIB="$SCRIPT_DIR/.conda_env/lib/python3.10/site-packages/nvidia/cublas/lib"
export LD_LIBRARY_PATH="$CUDNN_LIB:$CUBLAS_LIB:/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$SCRIPT_DIR/.conda_env/bin/python" "$SCRIPT_DIR/tests/probe_forgeffects.py"
