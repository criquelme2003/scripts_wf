import csv
import json
import os
import sys
import time as ti

import numpy as np

from matrix_construction import check_admissible, sparse_supercritical_block_matrix
from parser import get_sweep_parser

ORDER = 100
CSV_HEADER = ["c", "n", "repeticion", "orden_efectivo"]


def seed_for(seed_base: int, rep: int, n: int) -> int:
    # Con seed_base = 0 coincide con la semilla de test2.py (rep * 1000 + n).
    return seed_base * 10**9 + rep * 1000 + n


def split_combinations(ns: list, cs: list) -> tuple[list, list]:
    """Separa las combinaciones (c, n) válidas de las omitidas, en el orden del barrido."""
    valid, skipped = [], []
    for c in cs:
        for n in ns:
            if n % 2 != 0:
                reason = "N debe ser par"
            else:
                reason = check_admissible(n // 2, n // 2, c)
            if reason is None:
                valid.append((c, n))
            else:
                skipped.append({"n": n, "c": c, "reason": reason})
    return valid, skipped


def get_free_memory():
    """Libera el pool de CuPy como test2.py; si CuPy no está instalado no hace nada."""
    try:
        import cupy as cp
    except ImportError:
        return lambda: None
    return lambda: cp.get_default_memory_pool().free_all_blocks()


def write_summary(path: str, summary: dict) -> None:
    # Escritura atómica: si el job muere a mitad, el resumen anterior sigue siendo JSON válido.
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)
    os.replace(tmp_path, path)


def warmup(ft, thr: float) -> None:
    m_w, _, _ = sparse_supercritical_block_matrix(100, 100, 4, seed=0)
    m_w = m_w.reshape(1, 200, 200).astype(np.float16)
    ft.maxmin(m_w.copy(), m_w.copy(), thr, ORDER)


def run_sweep(params: dict, job_id: str, ft, free_memory) -> dict:
    valid, skipped = split_combinations(params["ns"], params["cs"])
    csv_path = f"jobs_results/{job_id}.csv"
    summary_path = f"jobs_results/{job_id}.json"
    start = ti.time()

    summary = {
        "kind": "sweep",
        "state": "running",
        "params": params,
        "skipped_combinations": skipped,
        "completed_combinations": 0,
        "total_combinations": len(valid),
        "failures": [],
        "computation-time(s)": 0.0,
    }
    os.makedirs("jobs_results", exist_ok=True)
    write_summary(summary_path, summary)

    warmup(ft, params["thr"])

    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_HEADER)
        file.flush()

        for c, n in valid:
            orders = []
            for rep in range(params["reps"]):
                free_memory()
                try:
                    E, _, _ = sparse_supercritical_block_matrix(n // 2, n // 2, c,
                                                                seed=seed_for(params["seed_base"], rep, n))
                    m1 = E.reshape(1, n, n).astype(np.float16)
                    m2 = m1.copy()
                    _, _, eff_order = ft.maxmin(m1, m2, params["thr"], ORDER)
                    orders.append((rep, int(eff_order)))
                except Exception as e:
                    summary["failures"].append({"c": c, "n": n, "rep": rep, "error": f"{type(e).__name__}: {e}"})

            writer.writerows([c, n, rep, o] for rep, o in orders)
            file.flush()
            if orders:
                values = [o for _, o in orders]
                print(f"c={c}  n={n}  mean={np.mean(values):.2f}  min={min(values)}  max={max(values)}", flush=True)

            summary["completed_combinations"] += 1
            summary["computation-time(s)"] = ti.time() - start
            write_summary(summary_path, summary)

    summary["state"] = "completed"
    summary["computation-time(s)"] = ti.time() - start
    write_summary(summary_path, summary)
    return summary


if __name__ == "__main__":
    args = get_sweep_parser().parse_args(sys.argv[1:])

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id is None:
        raise KeyError("SLURM_JOB_ID IS REQUIRED")

    import forgethreads as ft

    run_sweep(
        {"ns": args.ns, "cs": args.cs, "reps": args.reps, "thr": args.thr, "seed_base": args.seed_base},
        slurm_job_id,
        ft,
        get_free_memory(),
    )
