#!/bin/bash
# Test de integración: verifica que notifier.sh solo se ejecuta después de que
# new_job.sh termine (dependencia SLURM afterok), y que el resultado de
# new_job está disponible y es reportado correctamente por notifier.
#
# Requiere: sbatch/squeue/scontrol reales disponibles (SLURM local o cluster).

# Uso: bash tests/test_job_dependency.sh

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

FAILED=0
CALLBACK_PORT=8791
CALLBACK_LOG="tests/mock_callback_received.jsonl"
AUTH_TOKEN="dep-test-token-$$"
NEW_JOB_ID=""
NOTIFIER_JOB_ID=""
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

cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    # Cancelar cualquier job propio de este test que siga vivo (PENDING/RUNNING),
    # para no dejar basura en la cola si el test aborta a mitad de camino.
    for jid in "$NEW_JOB_ID" "$NOTIFIER_JOB_ID"; do
        [ -z "$jid" ] && continue
        st=$(job_state "$jid")
        case "$st" in
            PENDING|RUNNING|CONFIGURING|SUSPENDED)
                echo "cleanup: cancelando job $jid (estado $st)"
                scancel "$jid" 2>/dev/null
                ;;
        esac
    done
    rm -f "jobs_results/${NEW_JOB_ID}"
    rm -f "jobs_failed_notify/${NEW_JOB_ID}.json"
    rm -f logs/forgethreads-new-job."${NEW_JOB_ID}"
    rm -f logs/forgethreads-notifier."${NOTIFIER_JOB_ID}"
    rm -f "$CALLBACK_LOG"
}
trap cleanup EXIT

echo "=== Setup: iniciando mock callback server en puerto $CALLBACK_PORT ==="
rm -f "$CALLBACK_LOG"
./.conda_env/bin/python tests/mock_callback_server.py "$CALLBACK_PORT" "$CALLBACK_LOG" &
SERVER_PID=$!
sleep 1

echo "=== Paso 1: lanzar new_job.sh vía sbatch ==="
NEW_JOB_SUBMIT=$(sbatch --parsable new_job.sh --nodos 20 --thr 0.5 --conectividad 4 --seed 0 --auth-token "$AUTH_TOKEN" 2>&1)
if ! [[ "$NEW_JOB_SUBMIT" =~ ^[0-9]+$ ]]; then
    fail "sbatch de new_job.sh no devolvió un job id válido: $NEW_JOB_SUBMIT"
    exit 1
fi
NEW_JOB_ID="$NEW_JOB_SUBMIT"
echo "new_job job id = $NEW_JOB_ID"

echo "=== Paso 2: lanzar notifier.sh con --dependency=afterok:$NEW_JOB_ID ==="
NOTIFIER_SUBMIT=$(sbatch --parsable --dependency="afterok:${NEW_JOB_ID}" notifier.sh \
    --job-id "$NEW_JOB_ID" \
    --auth-token "$AUTH_TOKEN" \
    --callback-url "http://127.0.0.1:${CALLBACK_PORT}/callback" 2>&1)
if ! [[ "$NOTIFIER_SUBMIT" =~ ^[0-9]+$ ]]; then
    fail "sbatch de notifier.sh (con dependencia) no devolvió un job id válido: $NOTIFIER_SUBMIT"
    exit 1
fi
NOTIFIER_JOB_ID="$NOTIFIER_SUBMIT"
echo "notifier job id = $NOTIFIER_JOB_ID"

echo "=== Paso 3: mientras new_job corre, notifier debe estar PENDING con razón Dependency ==="
DEPENDENCY_OBSERVED=0
for i in $(seq 1 20); do
    NT_STATE=$(job_state "$NOTIFIER_JOB_ID")
    NT_REASON=$(job_reason "$NOTIFIER_JOB_ID")
    [ -z "$NT_STATE" ] && break
    if [ "$NT_STATE" = "PENDING" ] && [[ "$NT_REASON" == Dependency* ]]; then
        DEPENDENCY_OBSERVED=1
        echo "  observado: $NT_STATE $NT_REASON"
        break
    fi
    sleep 0.5
done

if [ "$DEPENDENCY_OBSERVED" = "1" ]; then
    pass "notifier permaneció PENDING con razón Dependency mientras new_job no había terminado"
else
    echo "  (no se alcanzó a observar el estado Dependency antes de que el job avanzara — se valida por orden temporal en el Paso 5)"
fi

echo "=== Paso 4: esperar a que ambos jobs terminen (timeout 120s) ==="
for i in $(seq 1 120); do
    NJ_STATE=$(job_state "$NEW_JOB_ID")
    NT_STATE=$(job_state "$NOTIFIER_JOB_ID")
    # Ausencia de estado (job ya purgado de la cola) cuenta como terminado.
    NJ_DONE=0; NT_DONE=0
    [ -z "$NJ_STATE" ] || [[ "$NJ_STATE" =~ ^(COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL)$ ]] && NJ_DONE=1
    [ -z "$NT_STATE" ] || [[ "$NT_STATE" =~ ^(COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL)$ ]] && NT_DONE=1
    [ "$NJ_DONE" = "1" ] && [ "$NT_DONE" = "1" ] && break
    sleep 1
done

NJ_FINAL=$(job_state "$NEW_JOB_ID")
NT_FINAL=$(job_state "$NOTIFIER_JOB_ID")
echo "new_job final state: $NJ_FINAL"
echo "notifier final state: $NT_FINAL"

if [ "$NJ_FINAL" = "COMPLETED" ]; then
    pass "new_job terminó con estado COMPLETED"
else
    fail "new_job no terminó COMPLETED: $NJ_FINAL"
fi

if [ "$NT_FINAL" = "COMPLETED" ]; then
    pass "notifier terminó con estado COMPLETED"
else
    fail "notifier no terminó COMPLETED: $NT_FINAL"
fi

echo "=== Paso 5: verificar orden temporal real (new_job debe terminar antes de que notifier arranque) ==="
NJ_END=$(job_field "$NEW_JOB_ID" EndTime)
NT_START=$(job_field "$NOTIFIER_JOB_ID" StartTime)
echo "new_job EndTime=$NJ_END  notifier StartTime=$NT_START"
if [ -n "$NJ_END" ] && [ -n "$NT_START" ] && [[ ! ( "$NJ_END" > "$NT_START" ) ]]; then
    pass "notifier arrancó en o después de que new_job terminara (orden de dependencia respetado)"
else
    fail "no se pudo confirmar el orden temporal esperado: End=$NJ_END Start=$NT_START"
fi

echo "=== Paso 6: verificar que el callback recibió el resultado correcto ==="
if [ -s "$CALLBACK_LOG" ]; then
    RECEIVED=$(cat "$CALLBACK_LOG")
    echo "  recibido: $RECEIVED"
    if echo "$RECEIVED" | grep -q '"status": "success"'; then
        pass "callback recibió status success"
    else
        fail "callback no recibió status success: $RECEIVED"
    fi
    if echo "$RECEIVED" | grep -q "Bearer ${AUTH_TOKEN}"; then
        pass "callback recibió el Authorization Bearer correcto"
    else
        fail "callback no recibió el Bearer esperado: $RECEIVED"
    fi
else
    fail "el mock callback server no recibió ningún POST"
fi

echo ""
if [ "$FAILED" = "0" ]; then
    echo "=== RESULTADO: TODOS LOS CHECKS PASARON ==="
    exit 0
else
    echo "=== RESULTADO: HAY CHECKS FALLIDOS ==="
    exit 1
fi
