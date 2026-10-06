#!/bin/bash
# Test de integración: notifier.sh encadenado con --dependency=afterany.
#
# Escenario 1 (success): new_job termina bien; notifier queda PENDING (Dependency)
#   mientras new_job corre, arranca después de que termina y el callback recibe
#   status success con el Bearer correcto.
# Escenario 2 (error): new_job falla (conectividad inadmisible -> ValueError);
#   con afterany el notifier igual corre y el callback recibe status error con el
#   traceback en logs. Con afterok este callback nunca llegaba.
#
# Requiere: sbatch/squeue/scontrol reales disponibles (SLURM local o cluster).

# Uso: bash tests/test_job_dependency.sh

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

FAILED=0
CALLBACK_PORT=8791
CALLBACK_LOG="tests/mock_callback_received.jsonl"
AUTH_TOKEN="dep-test-token-$$"
JOB_IDS=()
SERVER_PID=""

fail() { echo "FAIL: $1"; FAILED=1; }
pass() { echo "PASS: $1"; }

job_state() {
    scontrol show job "$1" 2>/dev/null | grep -oP 'JobState=\K[A-Z]+'
}

job_reason() {
    scontrol show job "$1" 2>/dev/null | grep -oP 'Reason=\K\S+'
}

job_field() {
    # job_field <jobid> <FieldName>
    scontrol show job "$1" 2>/dev/null | grep -oP "$2=\K\S+"
}

job_done() {
    # Ausencia de estado (job ya purgado de la cola) cuenta como terminado.
    local st
    st=$(job_state "$1")
    [ -z "$st" ] || [[ "$st" =~ ^(COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)$ ]]
}

cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    # Cancelar cualquier job propio de este test que siga vivo (PENDING/RUNNING),
    # para no dejar basura en la cola si el test aborta a mitad de camino.
    for jid in "${JOB_IDS[@]}"; do
        st=$(job_state "$jid")
        case "$st" in
            PENDING|RUNNING|CONFIGURING|SUSPENDED)
                echo "cleanup: cancelando job $jid (estado $st)"
                scancel "$jid" 2>/dev/null
                ;;
        esac
        rm -f "jobs_results/${jid}.json" "jobs_failed_notify/${jid}.json" logs/*."${jid}"
    done
    rm -f "$CALLBACK_LOG"
}
trap cleanup EXIT

callback_for() {
    # callback_for <job_id>: imprime el registro del callback de ese job (o nada).
    ./.conda_env/bin/python - "$CALLBACK_LOG" "$1" <<'EOF'
import json, sys
path, job_id = sys.argv[1], sys.argv[2]
try:
    lines = open(path, encoding="utf-8").read().splitlines()
except FileNotFoundError:
    lines = []
for line in lines:
    record = json.loads(line)
    if (record.get("body") or {}).get("job_id") == job_id:
        print(json.dumps(record))
EOF
}

# run_scenario <nombre> <status esperado> <texto esperado en logs> <args de new_job...>
run_scenario() {
    local name="$1" expected_status="$2" expected_log="$3"
    shift 3

    echo ""
    echo "######## Escenario $name ########"
    echo "=== Paso 1: lanzar new_job.sh vía sbatch ($*) ==="
    local new_job_id notifier_id
    new_job_id=$(sbatch --parsable new_job.sh "$@" 2>&1)
    if ! [[ "$new_job_id" =~ ^[0-9]+$ ]]; then
        fail "[$name] sbatch de new_job.sh no devolvió un job id válido: $new_job_id"
        return
    fi
    JOB_IDS+=("$new_job_id")
    echo "new_job job id = $new_job_id"

    echo "=== Paso 2: lanzar notifier.sh con --dependency=afterany:$new_job_id ==="
    notifier_id=$(sbatch --parsable --dependency="afterany:${new_job_id}" notifier.sh \
        --job-id "$new_job_id" \
        --auth-token "$AUTH_TOKEN" \
        --callback-url "http://127.0.0.1:${CALLBACK_PORT}/callback" 2>&1)
    if ! [[ "$notifier_id" =~ ^[0-9]+$ ]]; then
        fail "[$name] sbatch de notifier.sh (con dependencia) no devolvió un job id válido: $notifier_id"
        return
    fi
    JOB_IDS+=("$notifier_id")
    echo "notifier job id = $notifier_id"

    echo "=== Paso 3: mientras new_job corre, notifier debe estar PENDING con razón Dependency ==="
    local observed=0 nt_state nt_reason
    for i in $(seq 1 20); do
        nt_state=$(job_state "$notifier_id")
        nt_reason=$(job_reason "$notifier_id")
        [ -z "$nt_state" ] && break
        if [ "$nt_state" = "PENDING" ] && [[ "$nt_reason" == Dependency* ]]; then
            observed=1
            echo "  observado: $nt_state $nt_reason"
            break
        fi
        sleep 0.5
    done
    if [ "$observed" = "1" ]; then
        pass "[$name] notifier permaneció PENDING con razón Dependency mientras new_job no había terminado"
    else
        echo "  (no se alcanzó a observar el estado Dependency antes de que el job avanzara — se valida por orden temporal en el Paso 5)"
    fi

    echo "=== Paso 4: esperar a que ambos jobs terminen (timeout 120s) ==="
    for i in $(seq 1 120); do
        job_done "$new_job_id" && job_done "$notifier_id" && break
        sleep 1
    done
    local nj_final nt_final
    nj_final=$(job_state "$new_job_id")
    nt_final=$(job_state "$notifier_id")
    echo "new_job final state: $nj_final"
    echo "notifier final state: $nt_final"

    if [ "$expected_status" = "success" ]; then
        [ "$nj_final" = "COMPLETED" ] && pass "[$name] new_job terminó COMPLETED" \
            || fail "[$name] new_job no terminó COMPLETED: $nj_final"
    else
        [ "$nj_final" = "FAILED" ] && pass "[$name] new_job terminó FAILED (fallo esperado)" \
            || fail "[$name] new_job debía terminar FAILED: $nj_final"
    fi
    [ "$nt_final" = "COMPLETED" ] && pass "[$name] notifier terminó COMPLETED" \
        || fail "[$name] notifier no terminó COMPLETED: $nt_final"

    echo "=== Paso 5: verificar orden temporal real (new_job debe terminar antes de que notifier arranque) ==="
    local nj_end nt_start
    nj_end=$(job_field "$new_job_id" EndTime)
    nt_start=$(job_field "$notifier_id" StartTime)
    echo "new_job EndTime=$nj_end  notifier StartTime=$nt_start"
    if [ -n "$nj_end" ] && [ -n "$nt_start" ] && [[ ! ( "$nj_end" > "$nt_start" ) ]]; then
        pass "[$name] notifier arrancó en o después de que new_job terminara (orden de dependencia respetado)"
    else
        fail "[$name] no se pudo confirmar el orden temporal esperado: End=$nj_end Start=$nt_start"
    fi

    echo "=== Paso 6: verificar el callback ==="
    local received
    received=$(callback_for "$new_job_id")
    if [ -z "$received" ]; then
        fail "[$name] el mock callback server no recibió el POST del job $new_job_id"
        return
    fi
    echo "  recibido: ${received:0:300}..."
    echo "$received" | grep -q "\"status\": \"${expected_status}\"" \
        && pass "[$name] callback recibió status ${expected_status}" \
        || fail "[$name] callback no recibió status ${expected_status}"
    echo "$received" | grep -q "Bearer ${AUTH_TOKEN}" \
        && pass "[$name] callback recibió el Authorization Bearer correcto" \
        || fail "[$name] callback no recibió el Bearer esperado"
    echo "$received" | grep -q "$expected_log" \
        && pass "[$name] logs del callback contienen '$expected_log'" \
        || fail "[$name] logs del callback no contienen '$expected_log'"
}

echo "=== Setup: iniciando mock callback server en puerto $CALLBACK_PORT ==="
rm -f "$CALLBACK_LOG"
./.conda_env/bin/python tests/mock_callback_server.py "$CALLBACK_PORT" "$CALLBACK_LOG" &
SERVER_PID=$!
sleep 1

run_scenario "success" "success" "Convergencia" --nodos 20 --thr 0.5 --conectividad 4 --seed 0
# c=100 con N=20 es inadmisible: matrix_construction lanza ValueError y new_job sale con código != 0.
run_scenario "error" "error" "Inadmissible Bernoulli parameter" --nodos 20 --thr 0.5 --conectividad 100 --seed 0

echo ""
if [ "$FAILED" = "0" ]; then
    echo "=== RESULTADO: TODOS LOS CHECKS PASARON ==="
    exit 0
else
    echo "=== RESULTADO: HAY CHECKS FALLIDOS ==="
    exit 1
fi
