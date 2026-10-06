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

# CUDA 11.8 + cuDNN 8.6 para TF 2.13, desde .conda_env (ver tf_env.sh).
source "$SCRIPT_DIR/tf_env.sh"

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
    img = tf.random.uniform((1, 32, 32, 3))
    conv = tf.nn.conv2d(img, tf.random.uniform((3, 3, 3, 8)), strides=1, padding="SAME")  # usa cuDNN
print("matmul en", y.device, "suma", float(tf.reduce_sum(y)))
print("conv2d en", conv.device, "suma", float(tf.reduce_sum(conv)))
if "GPU" not in y.device or "GPU" not in conv.device:
    sys.exit("ERROR: el cómputo de TF no corrió en GPU")

# Las librerías CUDA de TF deben venir de .conda_env (tf_env.sh), no del toolkit del nodo.
# Excepciones: el driver (libcuda.so) y libnvrtc, que cuDNN 8.6 carga como "libnvrtc.so" sin
# versión y resuelve ldconfig (decisión del usuario: se usa la del sistema).
import os, re
conda_env = os.path.realpath(os.path.join(os.path.dirname(sys.executable), ".."))
cuda_lib = re.compile(r"/lib(cudart|cublas|cublasLt|cudnn\w*|cufft|curand|cusolver|cusparse|nvrtc\w*|cupti|nvJitLink)\.so")
system_allowed = re.compile(r"/libnvrtc[\w-]*\.so")
loaded = sorted({line.split()[-1] for line in open("/proc/self/maps") if cuda_lib.search(line)})
inside = [path for path in loaded if os.path.realpath(path).startswith(conda_env)]
allowed = [path for path in loaded if path not in inside and system_allowed.search(path)]
outside = [path for path in loaded if path not in inside and path not in allowed]
print("librerías CUDA cargadas:")
for path in loaded:
    tag = "ok     " if path in inside else ("sistema" if path in allowed else "FUERA  ")
    print("  ", tag, path)
if not inside or outside:
    sys.exit("ERROR: TF cargó librerías CUDA fuera de .conda_env (o ninguna)")
print("CHECK_ENV OK")
EOF
