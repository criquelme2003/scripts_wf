# scripts_wf — HPC Job Pipeline

Pipeline de dos jobs SLURM para simulación numérica, pensado para ser lanzado remotamente
por una API externa vía SSH. Un job de cómputo (`new_job`) genera un resultado; un job de
notificación (`notifier`), encadenado como dependencia SLURM, reporta ese resultado a un
endpoint HTTP de la API llamante.

## Flujo de llamada esperado

```
API externa (SSH al cluster)
   │
   ├─ 1. sbatch new_job.sh --nodos N --thr T --conectividad C --seed S --auth-token TOKEN
   │        └─ devuelve JOBID (p. ej. "937")
   │        └─ new_job.py corre el cómputo y escribe jobs_results/<JOBID>
   │
   └─ 2. sbatch --dependency=afterok:<JOBID> notifier.sh \
              --job-id <JOBID> --auth-token TOKEN --callback-url <URL>
            └─ SLURM retiene este job en PENDING (Dependency) hasta que new_job termine
            └─ al completar new_job, notifier.py corre, lee el resultado
              y hace POST del resultado a <URL>
```

La API es responsable de:
- Lanzar ambos `sbatch` (no hay un único script que encadene todo internamente).
- Capturar el `JOBID` devuelto por el primer `sbatch` y pasarlo tanto en `--dependency=afterok:<JOBID>` como en `--job-id <JOBID>` del segundo.
- Exponer el endpoint HTTP que recibirá el callback de `notifier.py`.

## `new_job.sh` / `new_job.py` — job de cómputo

Construye una matriz de bloques dispersa (tipo Bernoulli, estructura reflexiva N/M) y ejecuta
el algoritmo `maxmin` (módulo nativo `forgethreads`) sobre ella para calcular un "orden
efectivo".

```
sbatch new_job.sh \
  --nodos <int>          Número de nodos de la matriz (se divide en dos bloques N/M iguales)
  --thr <float>          Threshold para filtrado de efectos olvidados
  --conectividad <float> Aristas promedio por nodo
  --seed <int>           Semilla para la construcción de la matriz
  --auth-token <str>     Token de usuario/sesión (no es un token por-job), se propaga al resultado para que la API identifique al usuario que lanzó el job
```

**Salida:** `jobs_results/<SLURM_JOB_ID>` (JSON, sin extensión), con:
```json
{
  "authToken": "<el token recibido>",
  "effective-order": 3,
  "computation-time(s)": 0.42
}
```

**Log:** `logs/forgethreads-new-job.<SLURM_JOB_ID>` (stdout/stderr del job, incluye el log
del algoritmo `maxmin`).

## `notifier.sh` / `notifier.py` — job de notificación

Lee el resultado y el log de un `new_job` ya finalizado, y reporta el estado a la API vía
HTTP POST.

```
sbatch --dependency=afterok:<JOBID> notifier.sh \
  --job-id <str>        SLURM_JOB_ID del new_job cuyo resultado se reporta
                         (NO es el propio SLURM_JOB_ID de este job de notificación)
  --auth-token <str>    Token a enviar como credencial Bearer en el callback
  --callback-url <str>  URL del endpoint de la API que recibe el resultado
```

**Importante:** `--job-id` es obligatorio y distinto del `SLURM_JOB_ID` que SLURM asigna al
propio proceso de `notifier` — cada `sbatch` recibe su propio id, así que la API debe pasar
explícitamente el id del `new_job` padre.

### Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `NOTIFIER_HTTP_TIMEOUT` | `30` | Timeout en segundos para el POST del callback |

### Comportamiento de red

- Hasta 3 intentos totales (1 inicial + 2 reintentos) ante fallo de red o status HTTP de error, con un pequeño delay fijo entre intentos.
- Si los 3 intentos fallan, el resultado se guarda en `jobs_failed_notify/<job-id>.json` como fallback, para poder identificar qué jobs no llegaron a reportarse a la API.

## Credenciales esperadas

- **`--auth-token`**: token de **usuario/sesión** emitido por la API (no un token generado por-job) — el mismo token puede usarse para lanzar múltiples jobs. Se pasa a `new_job.sh` al lanzar el cómputo, y debe volver a pasarse a `notifier.sh` (mismo valor) para que el callback pueda autenticarse. Correlacionar el resultado con el job específico es responsabilidad del `--job-id` (ver sección anterior), no del token.
- **Autenticación del callback:** el POST de `notifier.py` hacia `<callback-url>` incluye el header:
  ```
  Authorization: Bearer <auth-token>
  ```
  La API debe validar este Bearer token en su endpoint para confirmar que el callback proviene de un job legítimo lanzado por ese usuario/sesión.

## Estructura esperada del endpoint de callback

El endpoint de la API (`--callback-url`) debe aceptar `POST` con body JSON y responder con un
status HTTP 2xx en caso de éxito (cualquier otro status o error de red dispara un reintento).

**Caso éxito** (`new_job` completó y generó resultado):
```json
{
  "effective-order": 3,
  "computation-time(s)": 0.42,
  "status": "success",
  "logs": "<contenido completo del log de new_job>",
  "job_id": "<SLURM_JOB_ID del new_job, el mismo pasado como --job-id a notifier.sh>"
}
```

El token se entrega exclusivamente como credencial Bearer en el
header `Authorization` (ver sección Credenciales). La API debe identificar el job por el token
recibido en el header.

**Caso error** (`new_job` falló o no generó resultado — por ejemplo el archivo
`jobs_results/<job-id>` no existe o no es JSON válido):
```json
{
  "status": "error",
  "logs": "<contenido del log de new_job, o null si tampoco existe>",
  "job_id": "<SLURM_JOB_ID del new_job>"
}
```

Los campos `status` y `job_id` están siempre presentes: `status` (`"success"` | `"error"`) es el
que la API debe usar para distinguir ambos casos, y `job_id` es el que permite identificar a qué
solicitud pertenece la respuesta (coincide con el `--job-id` pasado a `notifier.sh` y con el
`JOBID` devuelto por el `sbatch` de `new_job.sh`). Los demás campos (`effective-order`,
`computation-time(s)`) solo aparecen en el caso de éxito.

> El formato de este payload es provisional: se definió a partir del comportamiento actual del
> pipeline, ya que el endpoint real de la API aún no existía al momento de escribir este
> documento. Ajustar aquí y en `notifier.py` si la API requiere otro formato.

## Requisitos de entorno

Todo el pipeline corre sobre un entorno conda **local al workspace** (`.conda_env/`, Python
3.10 + `numpy` + `requests`), no sobre `.venv/` (Python 3.12, incompatible con la extensión
nativa `forgethreads.so`) ni sobre entornos conda globales del usuario.

```bash
# Crear/recrear el entorno en cualquier nodo del cluster:
conda env create -p ./.conda_env -f environment.yml
```

Los wrappers `new_job.sh` y `notifier.sh` ya apuntan a `./.conda_env/bin/python` de forma
relativa al directorio de envío (`$SLURM_SUBMIT_DIR`), así que no requieren activación manual
del entorno.

## Pruebas

```bash
# Test de integración end-to-end (requiere sbatch/squeue/scontrol reales):
bash tests/test_job_dependency.sh
```

Verifica que `notifier.sh` permanece `PENDING` mientras `new_job.sh` corre, que el orden
temporal de ejecución respeta la dependencia `afterok`, y que el callback recibe el resultado
esperado.
