# scripts_wf: requerimientos para la nueva versión

Documento de trabajo para un agente que desarrolla en el repositorio [`scripts_wf`](https://github.com/criquelme2003/scripts_wf).
Es autocontenido: no requiere acceso al backend `web_forgethreads`, salvo para consultar el contrato descrito en la sección 3.

## 1. Contexto

`scripts_wf` contiene los scripts que el backend `web_forgethreads` (FastAPI) lanza por SSH en un cluster SLURM con GPU. El patrón actual es:

```
backend ──ssh──▶ sbatch new_job.sh --nodos N --thr T --conectividad C --seed S
                   └─ devuelve "Submitted batch job <JOBID>"
         ──ssh──▶ sbatch --dependency=afterok:<JOBID> notifier.sh \
                     --job-id <JOBID> --auth-token <TOKEN> --callback-url <URL>
                   └─ notifier.py lee jobs_results/<JOBID>.json y logs/forgethreads-new-job.<JOBID>
                      y hace POST al backend con Authorization: Bearer <TOKEN>
```

El backend agrega dos funcionalidades nuevas que necesitan scripts nuevos en el cluster:

1. **Cálculo de efectos olvidados con `forgeffects`** sobre las matrices CC, CE y EE que ingresan expertos (grafo bipartito encadenado). Se calcula por experto y también de forma agregada (varios expertos apilados).
2. **Barrido de simulación**: un solo job que recorre listas de `N` y `c` con `R` repeticiones y reporta el orden efectivo de cada ejecución. Debe replicar [`feCUDA/comprobacion_empirica/test2.py`](https://github.com/criquelme2003/feCUDA/blob/gpu_paths/comprobacion_empirica/test2.py) (rama `gpu_paths`).

### Estado actual del repo

| Archivo | Rol |
|---|---|
| `new_job.sh` / `new_job.py` | Cómputo MaxMin individual con `forgethreads.maxmin` sobre una matriz de `matrix_construction.sparse_supercritical_block_matrix`. Escribe `jobs_results/<JOBID>.json`. |
| `notifier.sh` / `notifier.py` | Lee el resultado y el log del job padre y hace POST al backend con 3 intentos; si fallan, guarda `jobs_failed_notify/<id>.json`. |
| `matrix_construction.py` | Matriz por bloques `[NN NM; 0 MM]`, reflexiva. Lanza `ValueError` si `c / (N/2 − 1) > 1`. |
| `parser.py` | Parsers argparse de `new_job` y `notifier`. |
| `forgethreads.cpython-310-x86_64-linux-gnu.so` | Extensión nativa (requiere Python 3.10). |
| `environment.yml` | Entorno conda local `./.conda_env` (Python 3.10, `numpy`, `requests`). |
| `tests/` | `test_job_dependency.sh` (integración con SLURM real) y `mock_callback_server.py`. |

### Entorno de ejecución (verificado en `cuda3`, 2026-10-05)

| Aspecto | Valor | Consecuencia |
|---|---|---|
| Topología (igual en los 3 nodos) | Cada nodo es su **propio cluster SLURM de un solo nodo** (`ClusterName=cuda3`, partición `RunCuda`, 16 CPU). Disco local, no compartido. | El repo está desplegado por separado en los 3 nodos. Los `JOBID` son locales a cada nodo: el backend debe guardar el par (nodo, JOBID). El usuario despliega cada cambio con `git pull` en cada nodo. |
| GPU | 1× RTX 4090 (24 GB), driver 595.71.05, CUDA 13.2. Toolkits en `/usr/local/cuda-{11.8,12.x,13.2}`. | Ver T4. |
| GRES | `GresTypes=(null)`. SLURM **no** gestiona la GPU. | `#SBATCH --gres=gpu:1` sería rechazado. No usarlo. Dos jobs pueden compartir la GPU a la vez; esto se acepta (sección 6). |
| Memoria SLURM | `RealMemory=1` (MB) con `CR_CORE_MEMORY`. | No usar `--mem`/`--mem-per-cpu`: cualquier valor > 1 MB hace que el job no se pueda planificar. |
| `forgethreads.so` | CUDA runtime enlazado estáticamente; no usa CuPy. | `cp.get_default_memory_pool().free_all_blocks()` no libera memoria de `forgethreads`. CuPy es opcional en T3. |
| `.conda_env` | Python 3.10.21. Antes de T1: `numpy 2.2.5`, `requests`. **Después de T1:** `numpy 1.24.3` (conda), `tensorflow 2.13.0`, `tensorflow-probability 0.20.0`, `forgeffects 0.2.5`, `nvidia-cudnn-cu11 8.6.0.163`, `pandas 2.3.3`, `pytest 9.1.1`. | Hay que actualizar cada nodo a mano (ver T4). |
| `forgeffects` | Todas las versiones (0.1.9–0.2.5) exigen `tensorflow==2.13` y `tensorflow_probability==0.20.0`, con Python ≤ 3.11. TF 2.13 exige `numpy<=1.24.3` y necesita CUDA 11.8 + cuDNN 8.6. | Un solo entorno con numpy 1.24.3 (decisión del usuario). Se verificó que `forgethreads`/`new_job` da el mismo resultado que con numpy 2.2.5. |
| `logs/` | Está en `.gitignore` y `#SBATCH --output=logs/...` lo necesita. | En un clon nuevo el job falla sin dejar log. El despliegue debe crear `logs/` (o los `.sh` deben usar una ruta que exista). |

`test2.py` está en `.gitignore`: solo existe localmente en este nodo. Importa `sparse_supercritical_block_matrix2`, que es la misma función que `sparse_supercritical_block_matrix` (sección 6).

## 2. Tareas

### T1. Notifier: encadenamiento con `afterany` y estado `partial`

**Problema:** con `--dependency=afterok`, si el job de cómputo falla o se corta por tiempo, SLURM nunca libera la dependencia: el notifier queda pendiente para siempre y el error nunca llega al backend. El backend pasará a lanzar el notifier con `--dependency=afterany:<JOBID>`. El notifier debe soportar los tres desenlaces.

Requisitos:

- **Log del job padre**: hoy la ruta está fija como `logs/forgethreads-new-job.<id>`. Debe buscarse por id con `logs/*.<id>`, para que funcione con cualquier `--job-name` (`new_job`, `fe_job`, `sweep_job`). Mantener la compatibilidad con los jobs existentes.
- **Límite de tamaño del log**: enviar solo los últimos `NOTIFIER_MAX_LOG_BYTES` bytes (por defecto 200000). Los logs de un barrido largo pueden ser grandes.
- **Determinación del estado** a partir de `jobs_results/<id>.json`:

  | Situación | `status` enviado |
  |---|---|
  | El JSON existe y no tiene `"state": "running"` | `success` |
  | El JSON existe con `"state": "running"` (el job murió antes de terminar) | `partial` |
  | El JSON no existe o no es JSON válido | `error` |

- El payload conserva los campos actuales (`status`, `logs`, `job_id`) más todo el contenido del JSON de resumen, igual que hoy.

### T2. `fe_job.sh` / `fe_job.py`: efectos olvidados con `forgeffects`

**Entrada:**

```
sbatch fe_job.sh --input-dir jobs_inputs/<request_id>
```

El backend sube el directorio por SFTP antes de lanzar el job:

```
jobs_inputs/<request_id>/
  CC.npy      float32, forma (k, m, m)
  CE.npy      float32, forma (k, m, n)
  EE.npy      float32, forma (k, n, n)
  meta.json   {"causes": [m etiquetas], "effects": [n etiquetas],
               "thr": 0.5, "maxorder": 3, "reps": 1,
               "seed": 42}        # seed: opcional (int)
```

`k` es el número de expertos: 1 para el cálculo por experto, ≥ 2 para el agregado.

**Proceso:**

1. Validar que las formas sean consistentes con `k`, `m` y `n` y con las etiquetas, y que los valores estén en [0,1]. Si algo falla, terminar con código ≠ 0 y un mensaje claro en el log.
2. Ejecutar:
   ```python
   forgeffects.FE(CC, CE, EE, causes=..., effects=..., THR=thr, maxorder=maxorder, rep=reps, device="GPU")
   ```
   El parámetro de `FE` se llama **`rep`** (no `reps`) y su valor por defecto es 1000: siempre hay que pasar `rep=meta["reps"]`. `causes`/`effects` deben ser `list` de `str`. `THR` debe estar en [0,1] y `maxorder` debe ser ≥ 2 (si no, `FE` lanza `ValueError`).
3. **Comportamiento de `FE`, verificado** (código de `forgeffects 0.2.5` + `tests/probe_forgeffects.sh`, job 1005, dataset del paquete k=10, m=16, n=4):
   - Devuelve una `list` de `pandas.DataFrame`. **`resultado[i]` es el orden `i + 2`**; también se puede deducir por las columnas (`orden = cantidad de columnas Through + 1`).
   - Columnas: `From`, `Through1` … `Through{o-1}`, `To`, `Count` (int64), `Mean` (float32), `SD` (float32). Los nodos aparecen con las etiquetas de `causes` + `effects`. Las filas vienen ordenadas por `Count` de mayor a menor.
   - **Órdenes sin caminos:** no aparecen como `None` ni como DataFrame vacío. La lista **se corta**: tiene largo ≤ `maxorder − 1` y `FE` imprime `Effects were only found up to order X` por stdout. Ejemplo: k=1 con `maxorder=4` (o 12) devolvió solo el orden 2, y k=10 con `maxorder=4` devolvió los órdenes 2 y 3. Además, según el código, un orden ≥ 3 puede venir como DataFrame con 0 filas, si hay efectos pero ningún camino encadena; `fe_job` lo trata igual que un orden vacío.
   - **Sin efectos de orden 2:** `FE` lanza `ValueError("No effects found with thr …")`. `fe_job` lo trata como un resultado válido sin caminos: escribe el resumen con `orders: []` y `rows_per_order: {}`, y termina con código 0. El notifier reporta `success` (decisión del usuario).
   - `Count` es el número de réplicas (de `rep`) en que aparece el camino. Las réplicas salen de percentiles aleatorios sobre el eje de expertos. Con k=1 todas las réplicas son la misma matriz.
   - **Determinismo:** con k=1 el resultado se repite. Con k ≥ 2 **cambia entre llamadas**, porque `FEempirical` usa `tf.random` sin semilla. **Decisión:** si `meta.json` trae `"seed"` (int), `fe_job` llama a `tf.random.set_seed(seed)` antes de `FE` y la incluye en el resumen. Verificado: con la misma semilla, dos procesos dan el mismo CSV (mismo hash), y con otra semilla el resultado cambia. Sin `"seed"`, el comportamiento es el de hoy. El campo nuevo hay que coordinarlo con el backend.
   - La diagonal de CC y EE se fuerza a 1 (`reflexive`), así que lo que venga en la diagonal de entrada no influye.
   - Rendimiento: k=10, m=16, n=4, `rep=100` tarda unos 0,1 s en GPU, después de la primera llamada (~0,6 s de inicialización).

**Salida:**

```
jobs_results/<JOBID>/paths_order_<o>.csv   una por orden; columnas tal como las devuelve FE
                                           (From, Through1..Through{o-1}, To, Count, Mean, SD)
jobs_results/<JOBID>.json                  resumen para el notifier:
  {"kind": "fe", "k": 1, "seed": 42 | null, "orders": [2, 3], "rows_per_order": {"2": 120, "3": 87},
   "computation-time(s)": 12.3}
```

El resumen se escribe al final; si no existe, el notifier reporta `error`.

### T3. `sweep_job.sh` / `sweep_job.py`: barrido de simulación

**Entrada:**

```
sbatch --time=<HH:MM:SS> sweep_job.sh --ns 100,250,500 --cs 0.125,0.25,1,2.5 \
       --reps 20 --thr 0.5 --seed-base 0
```

**Proceso (replica `test2.py`):**

1. **Validar cada combinación con una única regla, la de `matrix_construction.py`:** `N` par, `c > 0` y `c ≤ N/2 − 1` (equivale a `p_N ≤ 1` y `p_M ≤ 1` con `n_N = n_M = N/2`). Las combinaciones inválidas se omiten y se listan en el resumen.

   - **`c` es `float` en todo el flujo:** `--cs` se parsea con `float` (lista separada por comas, cada valor > 0), y la comparación se hace en punto flotante con `N/2` como `float` (`c <= n / 2 - 1`), nunca con división entera. `n` es `int` (`--ns`, enteros positivos y pares).
   - **Fuente única:** extraer la condición a una función en `matrix_construction.py`, p. ej. `check_admissible(n_N, n_M, c) -> str | None` (devuelve el motivo del rechazo o `None`). `sparse_supercritical_block_matrix` la usa para lanzar su `ValueError` y `sweep_job.py` la usa para validar. La semántica de `matrix_construction.py` no cambia: rechaza exactamente los mismos casos que hoy. El usuario aprobó editar ese archivo solo para esto.
   - `new_job` no se modifica: ya aplica la misma regla, porque el `ValueError` sale de la misma función.
   - Diferencia con `test2.py` (`c < N/2`): para `c` entero las dos reglas dan lo mismo. Solo difieren con `c` fraccionario en `(N/2 − 1, N/2)`, por ejemplo `N=100, c=49.5`. En ese caso `test2.py` termina con `ValueError` y aquí la combinación se omite.
2. **Calentamiento de GPU**: `sparse_supercritical_block_matrix(100, 100, 4, seed=0)`, `reshape(1, 200, 200)`, `float16` y una llamada a `ft.maxmin` descartada.
3. **Bucle en secuencia**: `for c in cs: for n in ns: for rep in range(reps):`
   - `seed = seed_base * 10**9 + rep * 1000 + n`. Con `seed_base = 0` se obtiene exactamente la semilla de `test2.py`.
   - `E, _, _ = sparse_supercritical_block_matrix(n // 2, n // 2, c, seed=seed)`
   - `m1 = E.reshape(1, n, n).astype(np.float16)`; `m2 = m1.copy()`
   - `_, _, eff_order = ft.maxmin(m1, m2, thr, 100)`
   - Liberar el pool de memoria de CuPy en cada repetición (`cp.get_default_memory_pool().free_all_blocks()`), como en `test2.py`. Importar `cupy` de forma opcional: si no está instalado, omitir este paso (no afecta los resultados, porque `forgethreads` no usa CuPy).
   - Si una repetición lanza una excepción, registrarla en el resumen (`c`, `n`, `rep`, mensaje) y **continuar**.
4. Después de cada combinación (`c`, `n`), escribir las filas en el CSV y hacer `flush`. Si el job se corta, todo lo ya escrito queda disponible.

**Salida:**

```
jobs_results/<JOBID>.csv    encabezado: c,n,repeticion,orden_efectivo   (mismas columnas que test2.py)
                            c se escribe como float (p. ej. 0.125, 2.5, 4.0); n, repeticion y orden_efectivo como int
                            solo repeticiones exitosas; eff_order como int
jobs_results/<JOBID>.json   resumen; se crea al inicio y se actualiza después de cada combinación:
  {"kind": "sweep", "state": "running" | "completed",
   "params": {"ns": [...], "cs": [...], "reps": 20, "thr": 0.5, "seed_base": 0},
   "skipped_combinations": [{"n": 100, "c": 50.0, "reason": "Inadmissible Bernoulli parameter: ..."}],
   "completed_combinations": 37, "total_combinations": 40,
   "failures": [{"c": 2.5, "n": 100, "rep": 3, "error": "..."}],
   "computation-time(s)": 812.4}
```

`reason` es el mensaje de `check_admissible`, o `"N debe ser par"` para un N impar. El resumen se escribe de forma atómica (archivo temporal + `os.replace`), así que si el job muere a mitad de una escritura el notifier sigue leyendo un JSON válido.

`state` pasa a `completed` solo al terminar todo el barrido. Así el notifier (T1) distingue `success` de `partial`.

**Hecho (T7/T8):** `sweep_job.sh` lanza python con `exec`. Sin `exec`, SLURM manda el SIGTERM al `bash` del batch, que muere en el acto, y python muere por SIGKILL sin ejecutar el handler (verificado: job 1018). Es la única diferencia con `new_job.sh`. Antes era **opcional:** capturar `SIGTERM` (SLURM lo envía antes de matar el job por tiempo) para escribir el CSV y el resumen pendientes antes de salir.

### T4. Entorno

- **Hecho en T1:** `environment.yml` fija `numpy=1.24.3` (conda, junto con `mkl_fft`/`mkl_random`) y, por pip, `forgeffects==0.2.5`, `tensorflow==2.13.0`, `tensorflow-probability==0.20.0`, `nvidia-cudnn-cu11==8.6.0.163`, `pandas==2.3.3` y `pytest==9.1.1`. No se instala `cupy`: su soporte en T3 es opcional.
- **GPU para TF 2.13 (actualizado 2026-10-06):** todas las librerías CUDA 11.8 + cuDNN 8.6 se instalan por pip (`nvidia-{cudnn,cuda-runtime,cublas,cufft,curand,cusolver,cusparse}-cu11`) y `tf_env.sh` las pone primero en `LD_LIBRARY_PATH`. Lo usan `fe_job.sh`, `tests/check_env.sh` y `tests/probe_forgeffects.sh`. El nodo solo necesita **driver ≥ 520**, con cualquier toolkit (por ejemplo, un nodo que solo tenga CUDA 12.4). Excepción: cuDNN carga `libnvrtc.so` sin versión desde el sistema (hoy CUDA 13.2); el usuario decidió dejarlo así. `check_env` verifica en `/proc/self/maps` que el resto venga de `.conda_env` (job 1051). Antes de este cambio, TF tomaba en silencio cudart/cufft/cusolver/etc. del CUDA 11.8 del sistema vía ldconfig.
- No hacen falta `numba` ni `matplotlib`: `test2.py` los importa, pero aquí no se grafica.
- **GRES resuelto:** los nodos no tienen GRES configurado (`GresTypes=(null)`), así que **no** usar `--gres=gpu:1` en `fe_job.sh` ni `sweep_job.sh`. Mantener las mismas directivas `#SBATCH` que `new_job.sh` (sin `--mem`).
- Comprobar con `sbatch tests/check_env.sh` (no directamente en el nodo) que TF ve 1 GPU y que un `matmul` corre en `GPU:0`. **Verificado en `cuda3` (job 1004).**
- SLURM no tiene accounting (`sacct` no funciona). Los tests usan `scontrol`.
- Documentar el procedimiento para actualizar `.conda_env` en cada uno de los 3 nodos. La instalación la hace el usuario o el agente con su aprobación.

### T5. README

- Documentar `fe_job`, `sweep_job` y el notifier con `afterany` y el estado `partial`.
- Corregir lo desactualizado: `new_job.sh` **no** recibe `--auth-token` (el parser no lo tiene); el token es **por job**, generado por el backend, no por sesión; y el resultado se escribe en `jobs_results/<JOBID>.json`, con extensión.

### T6. Tests

- **Tests unitarios con pytest** que corran sin SLURM ni GPU. Usar `SLURM_JOB_ID` falso en el entorno, y mocks de `forgethreads`, `forgeffects` y `cupy`. Deben cubrir:
  - la fórmula de semillas (con `seed_base = 0` coincide con `rep*1000 + n`);
  - la validación de combinaciones: `check_admissible` coincide con que `sparse_supercritical_block_matrix` lance o no `ValueError` en una grilla de `N` y `c` que incluya `c` fraccionarios en el borde (`N/2 − 1`, `49.5` con `N=100`, `N/2`);
  - el formato del CSV y del resumen;
  - que el barrido continúe tras una repetición fallida;
  - la validación de entrada de `fe_job`;
  - la búsqueda de logs y la tabla de estados del notifier (`success`, `partial`, `error`).
- **Test de integración:** extender `tests/test_job_dependency.sh` para cubrir `afterany` con un job que falla, y verificar que el callback llega con `status: error`.

## 3. Contrato con el backend

Esto es lo que el backend va a ejecutar y esperar. Cualquier cambio aquí debe coordinarse con `web_forgethreads`.

| Paso | Comando o archivo |
|---|---|
| Subir input de FE | SFTP a `$HOME/<scripts_wf>/jobs_inputs/<request_id>/` (CC.npy, CE.npy, EE.npy, meta.json) |
| Lanzar FE | `cd <scripts_wf> && sbatch fe_job.sh --input-dir jobs_inputs/<request_id>` |
| Lanzar barrido | `cd <scripts_wf> && sbatch --time=<HH:MM:SS> sweep_job.sh --ns ... --cs ... --reps R --thr T --seed-base B` |
| Lanzar MaxMin individual | sin cambios: `sbatch new_job.sh --nodos N --thr T --conectividad C --seed S` |
| Notifier (todos) | `sbatch --dependency=afterany:<JOBID> notifier.sh --job-id <JOBID> --auth-token <TOKEN> --callback-url <URL>` |
| JOBID | El backend lo parsea con la regex `Submitted batch job (\d+)` sobre el stdout de `sbatch`. No cambiar esa salida. |
| Callback | `POST <URL>`, `Authorization: Bearer <TOKEN>`, JSON con `status` ∈ {`success`, `error`, `partial`}, `job_id`, `logs` y los campos del resumen. |
| Descarga de resultados | SFTP desde `jobs_results/<JOBID>/` (FE) o `jobs_results/<JOBID>.csv` (barrido). |

## 4. Restricciones

- No romper `new_job.sh`, sus argumentos ni su salida: el backend actual depende de ellos.
- No cambiar la semántica de `matrix_construction.py`.
- Todo corre con `./.conda_env/bin/python` (Python 3.10), por la extensión `forgethreads`.
- No versionar `.conda_env/`, `jobs_results/`, `jobs_inputs/`, `logs/` ni `jobs_failed_notify/`. Hoy falta `jobs_inputs/` en `.gitignore`.
- Convenciones de trabajo del usuario: commits sin línea `Co-Authored-By` de Claude, y sin abrir pull requests (el usuario los abre).

### Permisos del agente en este nodo

Este nodo ejecuta jobs reales del backend con el mismo usuario. Las reglas aplicables también están en `.claude/settings.json`; el resto depende del criterio del agente.

**Siempre (sin preguntar):**
- Leer el estado de SLURM: `sinfo`, `squeue`, `sacct`, `scontrol show`.
- Lanzar con `sbatch` jobs de prueba **pequeños** (minutos, matrices chicas) y `bash tests/test_job_dependency.sh`. Toda ejecución que use la GPU pasa por `sbatch`.
- Correr tests unitarios: `./.conda_env/bin/python -m pytest tests/`.
- Editar código, tests, `environment.yml`, `README.md` y este spec.
- Usar el scratchpad o `/tmp` para archivos temporales, y borrar los archivos que generen sus propios jobs de prueba (`jobs_results/<id>*`, `logs/*.<id>`).

**Preguntar antes:**
- `scancel`: el backend lanza jobs con el mismo usuario y no hay forma de distinguirlos con una regla. Cancelar solo los IDs que el propio agente lanzó, y confirmar.
- Cualquier `pip`/`conda install` o cambio en `.conda_env/`.
- Barridos o jobs largos (más de ~10 min o que ocupen la GPU mucho tiempo).
- Modificar `matrix_construction.py`, `new_job.sh`/`new_job.py` o la salida de `sbatch` (contrato con el backend).
- `git commit` y `git push`.

**Nunca:**
- `git pull`/`git fetch`/`git merge` de remoto, `git push --force`, `gh pr create`.
- Ejecutar código en la GPU fuera de SLURM.
- Usar `ssh`/`scp`/`rsync` hacia los otros nodos: el despliegue en ellos lo hace el usuario.
- Editar `.conda_env/`, `forgethreads*.so`, `jobs_results/`, `jobs_inputs/`, `logs/` ni `jobs_failed_notify/` (salvo los archivos de sus propios jobs de prueba).
- `scontrol update/hold/release`, `rm -rf`, `sudo`.
- Agregar `Co-Authored-By` de Claude en los commits.

## 5. Criterios de aceptación

1. Un barrido con `--seed-base 0` y los mismos `ns`, `cs`, `reps` y `thr` que `test2.py` produce **el mismo CSV** (mismas filas y mismos órdenes; `c` se compara como número, porque `test2.py` puede escribir `4` donde aquí se escribe `4.0`) que `test2.py` para las combinaciones que ambos consideran válidas.
2. Un barrido cortado por `--time` deja un CSV con las combinaciones terminadas, y el callback llega con `status: partial`.
3. Un `fe_job` con `k = 1` y otro con `k ≥ 2` generan un CSV por orden y un resumen correcto. Un input con formas inconsistentes termina en `status: error` con el motivo en `logs`.
4. Un job que falla con `afterany` genera un callback con `status: error`. Hoy ese callback nunca llega.
5. Los tests unitarios pasan sin SLURM ni GPU.
6. `fe_job.sh` y `sweep_job.sh` corren en `cuda3` sin `--gres` ni `--mem`, y la instalación del entorno queda documentada para los otros 2 nodos.

## 6. Decisiones (resueltas con el usuario, 2026-10-05)

1. **`sparse_supercritical_block_matrix2`** (de `test2.py`) es la misma función que `sparse_supercritical_block_matrix`. `sweep_job.py` usa la de `matrix_construction.py` y el criterio de aceptación 1 se verifica contra ella.
2. **Lanzamiento de jobs**: `fe_job.sh` y `sweep_job.sh` siguen la misma estructura que `new_job.sh` (mismas directivas `#SBATCH`, wrapper que llama a `./.conda_env/bin/python`) y se lanzan con `sbatch`. Sin `--exclusive`, `--gres` ni `--mem`. Se acepta que dos jobs compartan la GPU.
3. **Los 3 nodos son iguales** (GPU, driver y configuración SLURM): lo verificado en `cuda3` vale para todos.
4. **Despliegue manual**: el usuario hace `git pull` en cada nodo y actualiza `.conda_env` a mano. Por lo tanto:
   - Todo lo que el nodo necesita debe estar versionado o documentado en el README: los pasos para actualizar `.conda_env` y `mkdir -p logs jobs_results jobs_inputs` (para clones nuevos).
   - Los cambios deben funcionar tras un `git pull` sin pasos ocultos. Si un cambio exige actualizar el entorno, el README lo dice explícitamente.
   - El agente nunca hace `git pull` ni toca los otros nodos.

## 7. Actualización 2026-10-26: documento «scripts_wf: requerimientos y cambios» del backend

El backend (`web_forgethreads`, desde `f621a6a`) publicó una versión consolidada de estos requerimientos. Casi todo ya está implementado y verificado en `cuda3` (T1–T11 de `tasks/todo.md`). Esta sección registra solo las diferencias y las decisiones nuevas.

### 7.1 Cambios a implementar

| Id | Cambio | Motivo |
|---|---|---|
| D1 | `sweep_job` crea `jobs_results/<JOBID>.csv` con el encabezado **antes** del calentamiento (junto con el resumen inicial). | Si el job muere durante el calentamiento, el resumen dice `running`, el notifier reporta `partial` y el backend descarga un CSV que hoy no existe. |
| D2 | `c` se escribe como lo manda el backend: entero si es entero (`2`), decimal si no (`2.5`), en el CSV, en `params.cs`, en `skipped_combinations` y en `failures`. Se normaliza al parsear `--cs` (`2` y `2.0` dan `2`). La regla de admisibilidad y las matrices no cambian. | El backend manda `--cs 2` y su ejemplo es `"cs": [0.5, 2, 2.5]`. Además coincide con lo que escribe `test2.py`. Reemplaza el «`c` se escribe como float» de T3. |
| D3 | Campo nuevo en el resumen del barrido: `combination_times`, una lista de `{"c", "n", "seconds"}` por cada combinación terminada (incluye el calentamiento solo en `computation-time(s)`). | Calibrar la estimación de duración del backend (`1 s + 2 s × (N/1000)³`). Opcional para el backend. |
| D4 | `fe_job` valida `reps` en `[1, 10000]`. | Rango del documento del backend. |
| D5 | `tests/test_fe_integration.sh` verifica, en el caso k=10, que `Count`, `Mean` y `SD` existen y que `max(Count) ≤ reps`. | Criterio de aceptación «el de k ≥ 2 trae `Count`, `Mean` y `SD` con el `reps` pedido». |
| D6 | README: `combination_times`, formato de `c`, tope de `reps`. | T5. |

### 7.2 Decisiones (usuario, 2026-10-26)

- **cupy:** no se instala, aunque el documento lo pide en T4. `forgethreads` no usa CuPy, así que `free_all_blocks()` no cambia los resultados. Se mantiene el import opcional.
- **Benchmark de FE** (reps de 1000 a 10000, k de 5 a 10): queda pendiente, no se lanza por ahora.
- `seed` opcional en `meta.json`: se mantiene. El documento del backend no lo menciona; sin él, k ≥ 2 no es reproducible.

### 7.3 Puntos abiertos del documento, estado

| Punto | Estado |
|---|---|
| Semillas ≥ 2³² | **Cerrado.** `matrix_construction.py` usa `np.random.default_rng`, que acepta enteros de cualquier tamaño (probado con `seed ≈ 10¹⁰`). No hace falta limitar `seed-base`. |
| Semillas repetidas con N ≥ 1000 | Aceptado por el backend. Sin cambios. |
| GPU en SLURM | **Cerrado.** `GresTypes=(null)`: sin `--gres` ni `--mem` (ver §1). |
| Resultado de FE | **Cerrado** salvo el benchmark: `resultado[i]` es el orden `i+2`; la lista se corta en el último orden con caminos; k=1 con `reps=1` probado. Memoria y tiempo con `reps` alto: pendiente (7.2). |
| Calibración de la duración | D3. |

### 7.4 Para coordinar con el backend (sin cambios de código)

- Las columnas son `Through1`, `Through2`, … (sin `_`). El backend acepta cualquier prefijo `Through`, así que funciona.
- `reason` en `skipped_combinations` es el mensaje de `check_admissible` (o `"N debe ser par"`), no `"c > N/2 - 1"`. El backend las recalcula.
- El resumen de FE trae además `"seed"` (int o `null`).
- Un barrido que falla con una excepción fuera de una repetición (por ejemplo, en el calentamiento) queda con `state: running`, así que se reporta `partial`, con el CSV vacío (solo el encabezado). Se cumple la tabla del notifier.

### 7.5 Criterios de aceptación de esta actualización

1. `pytest tests/` en verde, con tests nuevos para D1 (el CSV existe con el encabezado si el calentamiento falla), D2 (`--cs 2,2.5,0.125` → CSV y JSON con `2`, `2.5`, `0.125`), D3 (una entrada por combinación terminada) y D4 (`reps` = 0, 10000 y 10001).
2. `tests/compare_with_test2.sh` sigue dando `IGUALES`.
3. `tests/test_sweep_partial.sh` y `tests/test_fe_integration.sh` (con la verificación de D5) en PASS en `cuda3`.
4. `new_job.sh` sin cambios (`git diff` vacío en `new_job.*` y en la salida de `sbatch`).
