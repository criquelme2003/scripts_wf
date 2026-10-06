# Se carga con `source` desde los scripts que usan TensorFlow (fe_job.sh, tests/check_env.sh,
# tests/probe_forgeffects.sh). Requiere SCRIPT_DIR = raíz del repo.
#
# TF 2.13 (exigido por forgeffects) está compilado contra CUDA 11.8 + cuDNN 8.6. Todas esas
# librerías vienen por pip dentro de .conda_env (paquetes nvidia-*-cu11), así que el nodo solo
# necesita un driver NVIDIA >= 520, sin importar qué toolkit CUDA tenga instalado (11.8, 12.x...).
# Van primero en LD_LIBRARY_PATH para ganarle a las versiones registradas en ldconfig.

NVIDIA_PIP_DIR="$SCRIPT_DIR/.conda_env/lib/python3.10/site-packages/nvidia"
TF_CUDA_LIBS=""
for lib_dir in "$NVIDIA_PIP_DIR"/*/lib; do
    [ -d "$lib_dir" ] && TF_CUDA_LIBS="${TF_CUDA_LIBS:+$TF_CUDA_LIBS:}$lib_dir"
done
export LD_LIBRARY_PATH="$TF_CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset lib_dir TF_CUDA_LIBS
