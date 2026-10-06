"""Criterio de aceptación 1: sweep_job reproduce el CSV de test2.py.

Ejecuta run_sweep() de test2.py SIN modificar el archivo (se lee y se ejecuta en
memoria) y sweep_job.run_sweep con los mismos ns, cs, reps y thr (seed_base=0), y
compara las filas por valor. Se lanza con: sbatch tests/compare_with_test2.sh

Para que test2.py corra aquí:
- numba, matplotlib y cupy se reemplazan por módulos falsos (no se grafica; cupy
  solo libera un pool que forgethreads no usa);
- sparse_supercritical_block_matrix2 es alias de sparse_supercritical_block_matrix
  (misma función, decisión del usuario);
- no se ejecutan las dos últimas líneas del script (run_sweep() y plot_results()).
"""
import csv
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

NS = [20, 100, 250]
CS = [0.125, 1, 2.5, 4, 9]  # todos admisibles para ambas reglas en estos N
REPEATS = 5
THR = 0.5


def install_fakes():
    numba = types.ModuleType("numba")
    numba.njit = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    matplotlib.pyplot = pyplot
    cupy = types.ModuleType("cupy")
    cupy.get_default_memory_pool = lambda: types.SimpleNamespace(free_all_blocks=lambda: None)
    sys.modules.update({"numba": numba, "matplotlib": matplotlib, "matplotlib.pyplot": pyplot, "cupy": cupy})

    import matrix_construction
    matrix_construction.sparse_supercritical_block_matrix2 = matrix_construction.sparse_supercritical_block_matrix


def run_test2(test2_path: Path, csv_out: Path):
    source = test2_path.read_text(encoding="utf-8")
    tail = "results = run_sweep()"
    if tail not in source:
        sys.exit(f"ERROR: no se encontró '{tail}' en {test2_path}")
    namespace = {"__name__": "test2"}
    exec(compile(source[:source.index(tail)], str(test2_path), "exec"), namespace)
    namespace.update(NS=NS, CS=CS, REPEATS=REPEATS, THR=THR, CSV_OUT=str(csv_out))
    namespace["run_sweep"]()


def read_rows(path: Path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["c", "n", "repeticion", "orden_efectivo"], rows[0]
    return [(float(c), int(n), int(rep), int(o)) for c, n, rep, o in rows[1:]]


if __name__ == "__main__":
    test2_path = Path(sys.argv[1] if len(sys.argv) > 1 else REPO / "test2.py").resolve()
    if not test2_path.exists():
        sys.exit(f"ERROR: no existe {test2_path} (test2.py está en .gitignore; solo existe en el nodo original)")

    install_fakes()
    import forgethreads as ft
    import sweep_job

    workdir = Path(tempfile.mkdtemp(prefix="compare_test2_"))
    os.chdir(workdir)

    test2_csv = workdir / "test2.csv"
    run_test2(test2_path, test2_csv)

    params = {"ns": NS, "cs": [float(c) for c in CS], "reps": REPEATS, "thr": THR, "seed_base": 0}
    sweep_job.run_sweep(params, "compare", ft, free_memory=lambda: None)

    expected, got = read_rows(test2_csv), read_rows(workdir / "jobs_results" / "compare.csv")
    os.chdir(REPO)
    shutil.rmtree(workdir)
    print(f"\nfilas test2.py={len(expected)} sweep_job={len(got)}")
    diffs = [(a, b) for a, b in zip(expected, got) if a != b]
    for a, b in diffs[:10]:
        print(f"  DIFERENCIA test2={a} sweep_job={b}")
    if len(expected) == len(got) and not diffs:
        print(f"IGUALES ({len(got)} filas, {len(NS) * len(CS)} combinaciones x {REPEATS} reps)")
    else:
        sys.exit("DISTINTOS")
