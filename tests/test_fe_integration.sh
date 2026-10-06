#!/bin/bash
# Test de integración de fe_job (criterio de aceptación 3) con notifier afterany:
#   k1       : k=1                         -> success, CSV por orden
#   k10      : k=10 agregado, seed=42      -> success, CSV por orden
#   broken   : EE con k distinto           -> fe_job FAILED, callback error con el motivo
#   noeffects: thr=1.0 (sin efectos)       -> success con orders: []
#
# Requiere: sbatch/scontrol reales. Uso: bash tests/test_fe_integration.sh

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

FAILED=0
CALLBACK_PORT=8793
CALLBACK_LOG="tests/mock_callback_fe.jsonl"
AUTH_TOKEN="fe-test-token-$$"
INPUT_PREFIX="jobs_inputs/_test_$$"
JOB_IDS=()
SERVER_PID=""
PY=./.conda_env/bin/python

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
    for jid in "${JOB_IDS[@]}"; do
        case "$(job_state "$jid")" in
            PENDING|RUNNING|CONFIGURING|SUSPENDED) scancel "$jid" 2>/dev/null ;;
        esac
        rm -f "jobs_results/${jid}.json" "jobs_results/${jid}"/paths_order_*.csv \
              "jobs_failed_notify/${jid}.json" logs/*."${jid}"
        rmdir "jobs_results/${jid}" 2>/dev/null
    done
    for d in "${INPUT_PREFIX}"_*; do
        [ -d "$d" ] || continue
        rm -f "$d"/CC.npy "$d"/CE.npy "$d"/EE.npy "$d"/meta.json
        rmdir "$d"
    done
    rm -f "$CALLBACK_LOG"
}
trap cleanup EXIT

callback_for() {
    "$PY" - "$CALLBACK_LOG" "$1" <<'EOF'
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

# submit <nombre> <args de make_fe_input...>: crea el input y lanza fe_job + notifier.
declare -A FE_ID NT_ID
submit() {
    local name="$1"; shift
    local input="${INPUT_PREFIX}_${name}"
    "$PY" tests/fixtures/make_fe_input.py "$input" "$@" >/dev/null || { fail "[$name] no se pudo crear el input"; return; }
    FE_ID[$name]=$(sbatch --parsable fe_job.sh --input-dir "$input" 2>&1)
    NT_ID[$name]=$(sbatch --parsable --dependency="afterany:${FE_ID[$name]}" notifier.sh \
        --job-id "${FE_ID[$name]}" --auth-token "$AUTH_TOKEN" \
        --callback-url "http://127.0.0.1:${CALLBACK_PORT}/callback" 2>&1)
    JOB_IDS+=("${FE_ID[$name]}" "${NT_ID[$name]}")
    echo "[$name] fe_job=${FE_ID[$name]} notifier=${NT_ID[$name]}"
}

# check <nombre> <estado fe_job> <status callback> <texto en logs> <órdenes esperados ("" = ninguno)>
check() {
    local name="$1" fe_state="$2" status="$3" log_text="$4" orders="$5"
    local fid="${FE_ID[$name]}"
    echo "=== Escenario $name (fe_job $fid) ==="
    [ "$(job_state "$fid")" = "$fe_state" ] && pass "[$name] fe_job terminó $fe_state" \
        || fail "[$name] fe_job debía terminar $fe_state: $(job_state "$fid")"
    [ "$(job_state "${NT_ID[$name]}")" = "COMPLETED" ] && pass "[$name] notifier COMPLETED" \
        || fail "[$name] notifier no terminó COMPLETED: $(job_state "${NT_ID[$name]}")"

    local received
    received=$(callback_for "$fid")
    [ -n "$received" ] || { fail "[$name] no llegó el callback"; return; }
    "$PY" - "$received" "$status" "$log_text" "$orders" "$fid" <<'EOF' || FAILED=1
import json, sys
from pathlib import Path
record, status, log_text, orders, fid = json.loads(sys.argv[1]), *sys.argv[2:]
body = record["body"]
expected = [int(o) for o in orders.split(",")] if orders else []
checks = [(body["status"] == status, f"status {body['status']} (esperado {status})"),
          (log_text in (body["logs"] or ""), f"logs contienen '{log_text}'")]
if status == "success":
    checks.append((body["orders"] == expected, f"orders {body['orders']} (esperado {expected})"))
    for o in expected:
        csv = Path(f"jobs_results/{fid}/paths_order_{o}.csv")
        lines = len(csv.read_text().splitlines()) - 1 if csv.exists() else -1
        checks.append((lines == body["rows_per_order"][str(o)],
                       f"{csv} con {lines} filas = rows_per_order[{o}]"))
for ok, msg in checks:
    print(("PASS: " if ok else "FAIL: ") + f"[{sys.argv[5]}] " + msg)
sys.exit(0 if all(ok for ok, _ in checks) else 1)
EOF
}

echo "=== Setup: mock callback server en puerto $CALLBACK_PORT ==="
rm -f "$CALLBACK_LOG"
"$PY" tests/mock_callback_server.py "$CALLBACK_PORT" "$CALLBACK_LOG" &
SERVER_PID=$!
sleep 1

submit k1 --k 1 --thr 0.5 --maxorder 3 --reps 1
submit k10 --k 10 --thr 0.5 --maxorder 4 --reps 100 --seed 42
submit broken --k 2 --broken
submit noeffects --k 1 --thr 1.0 --maxorder 3 --reps 1

echo "=== Esperando a que terminen los jobs (timeout 300s) ==="
for i in $(seq 1 300); do
    all_done=1
    for jid in "${JOB_IDS[@]}"; do job_done "$jid" || { all_done=0; break; }; done
    [ "$all_done" = "1" ] && break
    sleep 1
done

check k1 COMPLETED success "orden 2" "2"
check k10 COMPLETED success "orden 3" "2,3"
check broken FAILED error "k (expertos) inconsistente" ""
check noeffects COMPLETED success "sin caminos" ""

echo ""
if [ "$FAILED" = "0" ]; then
    echo "=== RESULTADO: TODOS LOS CHECKS PASARON ==="
    exit 0
else
    echo "=== RESULTADO: HAY CHECKS FALLIDOS ==="
    exit 1
fi
