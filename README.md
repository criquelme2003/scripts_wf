# scripts_wf — jobs SLURM para web_forgethreads

Scripts que el backend `web_forgethreads` lanza por SSH en los nodos de cómputo (cluster SLURM
con GPU). Hay tres jobs de cómputo y un job de notificación que reporta el resultado al backend
por HTTP:

| Job | Librería | Qué hace | Resultado |
|---|---|---|---|
| `new_job.sh` | `forgethreads` | Un `maxmin` sobre una matriz sintética de N nodos | `jobs_results/<JOBID>.json` |
| `sweep_job.sh` | `forgethreads` | Barrido de N × c × repeticiones (replica `test2.py`) | `jobs_results/<JOBID>.csv` + `.json` |
| `fe_job.sh` | `forgeffects` | Efectos olvidados sobre las matrices CC/CE/EE de expertos | `jobs_results/<JOBID>/paths_order_<o>.csv` + `<JOBID>.json` |
| `notifier.sh` | — | Lee el resultado y el log del job padre y hace POST al backend | callback HTTP |

`new_job` y `sweep_job` **generan** el digrafo en el nodo con
`matrix_construction.sparse_supercritical_block_matrix` (bloques `[NN NM; 0 MM]` binarios). `fe_job`
**recibe** los bloques de los expertos, y `forgeffects` arma el grafo `[CC CE; 0 EE]`.

## Flujo de llamada

```
backend (SSH al nodo, cd <scripts_wf>)
   │
   ├─ 1. sbatch <job>.sh <args>                      → "Submitted batch job <JOBID>"
   │
   └─ 2. sbatch --dependency=afterany:<JOBID> notifier.sh \
              --job-id <JOBID> --auth-token <TOKEN> --callback-url <URL>
            └─ queda PENDING (Dependency) hasta que el job termine, con éxito o no
            └─ hace POST a <URL> con status success | partial | error
```

- Se usa **`afterany`**, no `afterok`. Con `afterok`, si el job falla o se corta por tiempo, la
  dependencia nunca se cumple y el notifier queda pendiente para siempre.
- El backend obtiene el `JOBID` con la regex `Submitted batch job (\d+)` sobre el stdout de `sbatch`.
- **Cada nodo es un cluster SLURM independiente** con disco local, así que los `JOBID` solo valen
  dentro de su nodo: el backend debe guardar el par (nodo, JOBID).

## `new_job.sh` — MaxMin individual

```
sbatch new_job.sh --nodos <int> --thr <float> --conectividad <float> --seed <int>
```

| Argumento | Descripción |
|---|---|
| `--nodos` | N. La matriz se divide en dos bloques iguales de N/2 |
| `--thr` | Threshold del `maxmin` |
| `--conectividad` | c: aristas promedio por nodo. Debe cumplir `c ≤ N/2 − 1`; si no, el job falla con `ValueError` |
| `--seed` | Semilla de la matriz |

`new_job` **no** recibe `--auth-token`: el token solo lo usa el notifier.

**Salida:** `jobs_results/<JOBID>.json`

```json
{"effective-order": 3, "computation-time(s)": 0.13}
```

**Log:** `logs/forgethreads-new-job.<JOBID>`

## `sweep_job.sh` — barrido de simulación

```
sbatch --time=<HH:MM:SS> sweep_job.sh --ns 100,250,500 --cs 0.125,0.25,1,2.5 \
       --reps 20 --thr 0.5 --seed-base 0
```

| Argumento | Descripción |
|---|---|
| `--ns` | Lista de N separada por comas (enteros positivos; los impares se omiten) |
| `--cs` | Lista de c separada por comas (float > 0) |
| `--reps` | Repeticiones por combinación |
| `--thr` | Threshold del `maxmin` |
| `--seed-base` | Opcional, por defecto 0. `seed = seed_base·10⁹ + rep·1000 + N`; con 0 reproduce `test2.py` |

- Recorre `for c in cs: for n in ns: for rep in range(reps)`, en ese orden.
- Las combinaciones con `N` impar o `c > N/2 − 1` se omiten (misma regla que `matrix_construction`,
  función `check_admissible`) y se listan en `skipped_combinations`.
- Si una repetición falla, se registra en `failures` y el barrido continúa.
- Al terminar cada combinación se escriben sus filas en el CSV (con `flush`) y se actualiza el resumen.
  Si el job se corta, lo ya escrito queda disponible.
- **Corte por `--time`:** SLURM manda SIGTERM. El job deja el resumen con `state: running` y sale
  con código 143; el notifier lo reporta como **`partial`**. Por eso `sweep_job.sh` lanza python con
  `exec`: sin él, el SIGTERM le llegaría al `bash` y no a python.

**Salida:**

```
jobs_results/<JOBID>.csv     c,n,repeticion,orden_efectivo   (c float; el resto int)
jobs_results/<JOBID>.json
```

```json
{"kind": "sweep", "state": "running | completed",
 "params": {"ns": [100, 250], "cs": [0.125, 1.0], "reps": 20, "thr": 0.5, "seed_base": 0},
 "skipped_combinations": [{"n": 100, "c": 50.0, "reason": "Inadmissible Bernoulli parameter: ..."}],
 "completed_combinations": 37, "total_combinations": 40,
 "failures": [{"c": 2.5, "n": 100, "rep": 3, "error": "RuntimeError: ..."}],
 "computation-time(s)": 812.4}
```

**Log:** `logs/sweep_job.<JOBID>`

## `fe_job.sh` — efectos olvidados con `forgeffects`

```
sbatch fe_job.sh --input-dir jobs_inputs/<request_id>
```

El backend sube antes el directorio por SFTP:

```
jobs_inputs/<request_id>/
  CC.npy      forma (k, m, m), valores en [0,1]
  CE.npy      forma (k, m, n)
  EE.npy      forma (k, n, n)
  meta.json   {"causes": [m etiquetas], "effects": [n etiquetas],
               "thr": 0.5, "maxorder": 3, "reps": 100, "seed": 42}
```

| Campo de `meta.json` | Regla |
|---|---|
| `causes`, `effects` | Listas de strings de largo m y n, sin etiquetas repetidas entre ambas |
| `thr` | Número en [0,1] |
| `maxorder` | Entero ≥ 2 |
| `reps` | Entero ≥ 1: réplicas de `forgeffects` (su parámetro `rep`) |
| `seed` | Opcional (entero). Con k ≥ 2, `FE` es aleatorio; con `seed` el resultado es reproducible |

- `k` es el número de expertos: 1 para el cálculo por experto, ≥ 2 para el agregado.
- Si la entrada es inválida, el job termina con código 1 y el log lista **todos** los problemas
  (el notifier reporta `error` con ese texto en `logs`).
- Si no hay efectos de orden 2 con ese `thr`, el resultado es válido: `orders: []`.
- `fe_job.sh` agrega a `LD_LIBRARY_PATH` el cuDNN 8.6 que instala pip y `/usr/local/cuda-11.8/lib64`,
  porque TF 2.13 no funciona con el cuDNN 9 del sistema.

**Salida:**

```
jobs_results/<JOBID>/paths_order_<o>.csv   uno por orden con caminos:
                                           From, Through1..Through{o-1}, To, Count, Mean, SD
jobs_results/<JOBID>.json                  se escribe al final (si no existe -> error)
```

```json
{"kind": "fe", "k": 10, "seed": 42, "orders": [2, 3], "rows_per_order": {"2": 2098, "3": 1},
 "computation-time(s)": 0.4}
```

**Log:** `logs/fe_job.<JOBID>`

## `notifier.sh` — callback al backend

```
sbatch --dependency=afterany:<JOBID> notifier.sh \
  --job-id <JOBID>        JOBID del job padre (NO el del propio notifier)
  --auth-token <TOKEN>    se envía como Authorization: Bearer <TOKEN>
  --callback-url <URL>    endpoint http(s) del backend
```

Lee `jobs_results/<JOBID>.json` y el log `logs/*.<JOBID>` (sirve para cualquier `--job-name`), y
determina el estado:

| Situación | `status` |
|---|---|
| El JSON existe y no tiene `"state": "running"` | `success` |
| El JSON existe con `"state": "running"` (el job murió antes de terminar) | `partial` |
| El JSON no existe o no es un objeto JSON válido | `error` |

**Payload:** el contenido del JSON de resumen más `status`, `job_id` y `logs`:

```json
{"effective-order": 3, "computation-time(s)": 0.13,
 "status": "success", "logs": "<últimos bytes del log>", "job_id": "937"}
```

En caso de error: `{"status": "error", "logs": "<log o null>", "job_id": "938"}`.

### Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `NOTIFIER_HTTP_TIMEOUT` | `30` | Timeout en segundos del POST |
| `NOTIFIER_MAX_LOG_BYTES` | `200000` | Solo se envían los últimos N bytes del log |

### Comportamiento de red

- Hasta 3 intentos (1 inicial + 2 reintentos) ante un error de red o un status HTTP de error, con un
  delay fijo entre intentos. El endpoint debe responder 2xx.
- Si los 3 fallan, el payload se guarda en `jobs_failed_notify/<JOBID>.json`.

### Credenciales

- El token es **por job**: lo genera el backend al lanzar el job y lo pasa **solo** al notifier.
- El backend valida el `Authorization: Bearer <TOKEN>` del callback y correlaciona por `job_id`.

## Entorno

Todo corre con `./.conda_env/bin/python` (Python 3.10, que exige la extensión nativa
`forgethreads.cpython-310-x86_64-linux-gnu.so`). Las versiones están fijadas en `environment.yml`:

- `forgeffects 0.2.5` exige `tensorflow==2.13`, y TF 2.13 exige `numpy<=1.24.3`. Por eso el entorno
  usa **numpy 1.24.3**; se verificó que `forgethreads` da los mismos resultados que con numpy 2.2.5.
- `tensorflow-probability 0.20.0`, `nvidia-cudnn-cu11 8.6.0.163`, `pandas`, `pytest`.
- Los nodos tienen 1 GPU RTX 4090 y SLURM **no** gestiona la GPU (no hay GRES): los `.sh` no usan
  `--gres`. Tampoco usan `--mem` (el nodo declara `RealMemory=1`).

## Despliegue en un nodo

En cada nodo, desde la raíz del repo:

```bash
git pull
bash deploy_node.sh
```

`deploy_node.sh`:

1. Actualiza `.conda_env` con las versiones fijadas (o lo crea desde `environment.yml` si no existe).
2. Imprime las versiones instaladas.
3. Crea `logs/`, `jobs_results/`, `jobs_inputs/` y `jobs_failed_notify/`. Sin `logs/`, los jobs
   fallan sin dejar log.
4. Lanza `sbatch --wait tests/check_env.sh` y termina en `DEPLOY OK` si TensorFlow hace un `matmul`
   en la GPU.

Las versiones están escritas en `environment.yml` **y** en `deploy_node.sh`: si cambia una, hay que
cambiar la otra.

## Pruebas

```bash
# Unitarias (sin SLURM ni GPU; usan mocks de forgethreads, forgeffects y cupy):
./.conda_env/bin/python -m pytest tests/ -q

# Integración (SLURM real y GPU del nodo):
bash tests/test_job_dependency.sh     # notifier afterany: success y error (~10 s)
bash tests/test_sweep_partial.sh      # sweep cortado por --time -> partial (~80 s)
bash tests/test_fe_integration.sh     # fe_job: k=1, k=10, entrada inválida, sin efectos (~25 s)

# Verificaciones puntuales (jobs de sbatch, resultado en logs/<nombre>.<JOBID>):
sbatch tests/check_env.sh             # entorno y GPU de TensorFlow
sbatch tests/probe_forgeffects.sh     # formato de salida de forgeffects.FE
sbatch tests/compare_with_test2.sh    # sweep_job == test2.py (solo donde exista test2.py)
```

Los tests de integración usan un servidor de callback local (`tests/mock_callback_server.py`) y
borran al terminar los archivos de sus propios jobs.
