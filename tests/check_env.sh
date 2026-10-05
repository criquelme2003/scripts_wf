#!/bin/bash -l
#SBATCH -n 1
#SBATCH --job-name=check_env
#SBATCH --cpus-per-task=8
#SBATCH --no-requeue
#SBATCH --output=logs/%x.%j

# Verifica el entorno del nodo: imports de new_job/sweep_job/fe_job y que
# TensorFlow 2.13 vea la GPU. Uso: sbatch tests/check_env.sh

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="$SCRIPT_DIR/.conda_env/bin/python"

# TF 2.13 necesita cuDNN 8 (pip) y CUDA 11.8 (sistema).
CUDNN_LIB="$SCRIPT_DIR/.conda_env/lib/python3.10/site-packages/nvidia/cudnn/lib"
CUBLAS_LIB="$SCRIPT_DIR/.conda_env/lib/python3.10/site-packages/nvidia/cublas/lib"
export LD_LIBRARY_PATH="$CUDNN_LIB:$CUBLAS_LIB:/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$SCRIPT_DIR" || exit 1
"$PY" - <<'EOF'
import sys
import numpy, pandas
import forgethreads
import forgeffects
import tensorflow as tf

print("python", sys.version.split()[0])
print("numpy", numpy.__version__, "| pandas", pandas.__version__, "| tensorflow", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs visibles para TF:", len(gpus), gpus)
if len(gpus) != 1:
    sys.exit("ERROR: TensorFlow no ve exactamente 1 GPU")
with tf.device("/GPU:0"):
    x = tf.random.uniform((512, 512))
    y = tf.linalg.matmul(x, x)
print("matmul en", y.device, "suma", float(tf.reduce_sum(y)))
if "GPU" not in y.device:
    sys.exit("ERROR: el cómputo de TF no corrió en GPU")
print("CHECK_ENV OK")
EOF
