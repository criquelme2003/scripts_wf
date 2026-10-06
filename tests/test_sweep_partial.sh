#!/bin/bash
# Test de integración: un sweep_job cortado por --time deja un CSV con las
# combinaciones terminadas y el callback llega con status partial.
#
# El barrido pedido (N=4000, ~0.5 s/rep en una RTX 4090 -> ~20 s por combinación,
# 8 combinaciones ~160 s) no alcanza a terminar: con --time=00:01:00 SLURM le manda
# SIGTERM al minuto (KillWait=30s antes del SIGKILL). Duración total ~2 min.
#
# Requiere: sbatch/scontrol reales. Uso: bash tests/test_sweep_partial.sh

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

FAILED=0
CALLBACK_PORT=8792
CALLBACK_LOG="tests/mock_callback_partial.jsonl"
AUTH_TOKEN="partial-test-token-$$"
REPS=40
SWEEP_ID=""
NOTIFIER_ID=""
SERVER_PID=""

fail() { echo "FAIL: $1"; FAILED=1; }
pass() { echo "PASS: $1"; }

job_state() {
    scontrol show job "$1" 2>/dev/null | grep -oP 'JobState=\K[A-Z]+'
}

job_done() {
    local st
    st=$(job_state "$1")
    [ -z "$st" ] || [[ "$st" =~ ^(COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)$ ]]
}

cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    for jid in "$SWEEP_ID" "$NOTIFIER_ID"; do
        [ -z "$jid" ] && continue
        case "$(job_state "$jid")" in
            PENDING|RUNNING|CONFIGURING|SUSPENDED)
                echo "cleanup: cancelando job $jid"
                scancel "$jid" 2>/dev/null
                ;;
        esac
        rm -f "jobs_results/${jid}.json" "jobs_results/${jid}.json.tmp" "jobs_results/${jid}.csv" \
              "jobs_failed_notify/${jid}.json" logs/*."${jid}"
    done
    rm -f "$CALLBACK_LOG"
}
trap cleanup EXIT

echo "=== Setup: mock callback server en puerto $CALLBACK_PORT ==="
rm -f "$CALLBACK_LOG"
./.conda_env/bin/python tests/mock_callback_server.py "$CALLBACK_PORT" "$CALLBACK_LOG" &
SERVER_PID=$!
sleep 1

echo "=== Paso 1: lanzar sweep_job.sh con --time=00:01:00 y un barrido largo ==="
SWEEP_ID=$(sbatch --parsable --time=00:01:00 sweep_job.sh \
    --ns 4000 --cs 1,2,3,4,5,6,7,8 --reps "$REPS" --thr 0.5 --seed-base 0 2>&1)
if ! [[ "$SWEEP_ID" =~ ^[0-9]+$ ]]; then
    fail "sbatch de sweep_job.sh no devolvió un job id válido: $SWEEP_ID"
    exit 1
fi
echo "sweep job id = $SWEEP_ID"

echo "=== Paso 2: lanzar notifier.sh con --dependency=afterany:$SWEEP_ID ==="
NOTIFIER_ID=$(sbatch --parsable --dependency="afterany:${SWEEP_ID}" notifier.sh \
    --job-id "$SWEEP_ID" --auth-token "$AUTH_TOKEN" \
    --callback-url "http://127.0.0.1:${CALLBACK_PORT}/callback" 2>&1)
if ! [[ "$NOTIFIER_ID" =~ ^[0-9]+$ ]]; then
    fail "sbatch de notifier.sh no devolvió un job id válido: $NOTIFIER_ID"
    exit 1
fi
echo "notifier job id = $NOTIFIER_ID"

echo "=== Paso 3: esperar a que ambos jobs terminen (timeout 240s) ==="
for i in $(seq 1 240); do
    job_done "$SWEEP_ID" && job_done "$NOTIFIER_ID" && break
    sleep 1
done
SW_FINAL=$(job_state "$SWEEP_ID")
NT_FINAL=$(job_state "$NOTIFIER_ID")
echo "sweep final state: $SW_FINAL"
echo "notifier final state: $NT_FINAL"
[ "$SW_FINAL" = "TIMEOUT" ] && pass "sweep_job cortado por --time (TIMEOUT)" \
    || fail "sweep_job debía terminar TIMEOUT: $SW_FINAL"
[ "$NT_FINAL" = "COMPLETED" ] && pass "notifier terminó COMPLETED" \
    || fail "notifier no terminó COMPLETED: $NT_FINAL"

echo "=== Paso 4: verificar el log, el CSV y el resumen ==="
grep -q "SIGTERM: barrido cortado" logs/sweep_job."$SWEEP_ID" 2>/dev/null \
    && pass "el handler de SIGTERM corrió: $(grep 'SIGTERM:' logs/sweep_job."$SWEEP_ID")" \
    || fail "el log no muestra el corte por SIGTERM"

./.conda_env/bin/python - "$SWEEP_ID" "$REPS" <<'EOF' || FAILED=1
import csv, json, sys
job_id, reps = sys.argv[1], int(sys.argv[2])
summary = json.load(open(f"jobs_results/{job_id}.json", encoding="utf-8"))
rows = list(csv.reader(open(f"jobs_results/{job_id}.csv", newline="")))
done, total = summary["completed_combinations"], summary["total_combinations"]
checks = [
    (summary["state"] == "running", f"resumen en state running ({summary['state']})"),
    (0 < done < total, f"combinaciones terminadas parciales: {done}/{total}"),
    (rows[0] == ["c", "n", "repeticion", "orden_efectivo"], "encabezado del CSV"),
    (len(rows) - 1 == done * reps - len(summary["failures"]),
     f"CSV con solo combinaciones cerradas: {len(rows) - 1} filas = {done} x {reps} - {len(summary['failures'])} fallos"),
]
for ok, msg in checks:
    print(("PASS: " if ok else "FAIL: ") + msg)
sys.exit(0 if all(ok for ok, _ in checks) else 1)
EOF

echo "=== Paso 5: verificar el callback ==="
if [ -s "$CALLBACK_LOG" ]; then
    RECEIVED=$(cat "$CALLBACK_LOG")
    echo "  recibido: ${RECEIVED:0:300}..."
    echo "$RECEIVED" | grep -q '"status": "partial"' && pass "callback recibió status partial" \
        || fail "callback no recibió status partial"
    echo "$RECEIVED" | grep -q "\"job_id\": \"${SWEEP_ID}\"" && pass "callback con el job_id del sweep" \
        || fail "callback sin el job_id esperado"
    echo "$RECEIVED" | grep -q "Bearer ${AUTH_TOKEN}" && pass "callback con el Authorization Bearer correcto" \
        || fail "callback sin el Bearer esperado"
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
