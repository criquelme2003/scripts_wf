import json
import os
import sys
import time as ti
from pathlib import Path

import numpy as np

from parser import get_fe_parser

MATRICES = ("CC", "CE", "EE")
# FE lanza ValueError con este prefijo cuando no hay efectos de orden 2: es un resultado válido.
NO_EFFECTS_PREFIX = "No effects found"


class InputError(Exception):
    pass


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _load_array(path: Path, name: str, errors: list):
    file = path / f"{name}.npy"
    if not file.is_file():
        errors.append(f"falta {name}.npy")
        return None
    try:
        array = np.load(file, allow_pickle=False)
    except (ValueError, OSError) as e:
        errors.append(f"{name}.npy no se puede leer: {e}")
        return None
    if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)):
        errors.append(f"{name} debe ser numérico (dtype {array.dtype})")
        return None
    if array.ndim != 3:
        errors.append(f"{name} debe ser 3D (k, filas, columnas), forma recibida {array.shape}")
        return None
    if not np.all(np.isfinite(array)) or array.min(initial=0) < 0 or array.max(initial=0) > 1:
        errors.append(f"{name} tiene valores fuera de [0,1] o no finitos")
    return array


def _check_shapes(arrays: dict, errors: list):
    CC, CE, EE = (arrays[name] for name in MATRICES)
    if CC is None or CE is None or EE is None:
        return None, None
    k, m, n = CE.shape
    if CC.shape[0] != k or EE.shape[0] != k:
        errors.append(f"k (expertos) inconsistente: CC={CC.shape[0]}, CE={k}, EE={EE.shape[0]}")
    if CC.shape[1:] != (m, m):
        errors.append(f"CC debe tener forma (k, {m}, {m}) según CE, recibida {CC.shape}")
    if EE.shape[1:] != (n, n):
        errors.append(f"EE debe tener forma (k, {n}, {n}) según CE, recibida {EE.shape}")
    if k < 1 or m < 1 or n < 1:
        errors.append(f"dimensiones vacías: k={k}, m={m}, n={n}")
    return m, n


def _check_meta(meta: dict, m, n, errors: list) -> None:
    for field in ("causes", "effects", "thr", "maxorder", "reps"):
        if field not in meta:
            errors.append(f"meta.json: falta el campo '{field}'")

    for field, expected in (("causes", m), ("effects", n)):
        labels = meta.get(field)
        if labels is None:
            continue
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            errors.append(f"meta.json: '{field}' debe ser una lista de strings")
        elif expected is not None and len(labels) != expected:
            errors.append(f"meta.json: '{field}' tiene {len(labels)} etiquetas y la matriz espera {expected}")

    labels = (meta.get("causes") or []) + (meta.get("effects") or [])
    if all(isinstance(label, str) for label in labels) and len(set(labels)) != len(labels):
        errors.append("meta.json: hay etiquetas repetidas entre 'causes' y 'effects'")

    if "thr" in meta and not (_is_number(meta["thr"]) and 0 <= meta["thr"] <= 1):
        errors.append(f"meta.json: 'thr' debe ser un número en [0,1], recibido {meta['thr']!r}")
    if "maxorder" in meta and not (_is_int(meta["maxorder"]) and meta["maxorder"] >= 2):
        errors.append(f"meta.json: 'maxorder' debe ser un entero >= 2, recibido {meta['maxorder']!r}")
    if "reps" in meta and not (_is_int(meta["reps"]) and meta["reps"] >= 1):
        errors.append(f"meta.json: 'reps' debe ser un entero >= 1, recibido {meta['reps']!r}")
    if meta.get("seed") is not None and not _is_int(meta["seed"]):
        errors.append(f"meta.json: 'seed' debe ser un entero, recibido {meta['seed']!r}")


def load_and_validate(input_dir) -> dict:
    """Carga el input de fe_job; lanza InputError con todos los problemas encontrados."""
    path = Path(input_dir)
    if not path.is_dir():
        raise InputError(f"el directorio de entrada no existe: {path}")

    errors = []
    arrays = {name: _load_array(path, name, errors) for name in MATRICES}
    m, n = _check_shapes(arrays, errors)

    meta = None
    meta_file = path / "meta.json"
    if not meta_file.is_file():
        errors.append("falta meta.json")
    else:
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:
            errors.append(f"meta.json no es JSON válido: {e}")
        if meta is not None and not isinstance(meta, dict):
            errors.append("meta.json debe ser un objeto JSON")
            meta = None
    if meta is not None:
        _check_meta(meta, m, n, errors)

    if errors:
        raise InputError("entrada inválida en " + str(path) + ":\n  - " + "\n  - ".join(errors))

    return {
        **{name: arrays[name].astype(np.float32) for name in MATRICES},
        "causes": meta["causes"],
        "effects": meta["effects"],
        "thr": float(meta["thr"]),
        "maxorder": meta["maxorder"],
        "reps": meta["reps"],
        "seed": meta.get("seed"),
    }


def run_fe(data: dict, job_id: str, forgeffects, set_seed) -> dict:
    """Ejecuta FE, escribe un CSV por orden con caminos y, al final, el resumen para el notifier."""
    if data["seed"] is not None:
        set_seed(data["seed"])

    start = ti.time()
    try:
        results = forgeffects.FE(
            data["CC"], data["CE"], data["EE"],
            causes=data["causes"], effects=data["effects"],
            THR=data["thr"], maxorder=data["maxorder"], rep=data["reps"], device="GPU",
        )
    except ValueError as e:
        if not str(e).startswith(NO_EFFECTS_PREFIX):
            raise
        print(f"[fe_job] {e} -> sin caminos", flush=True)
        results = []
    total_time = ti.time() - start

    out_dir = Path("jobs_results") / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_per_order = {}
    for i, df in enumerate(results):
        order = i + 2  # resultado[i] es el orden i + 2 (verificado en T2)
        if len(df) == 0:
            continue
        df.to_csv(out_dir / f"paths_order_{order}.csv", index=False)
        rows_per_order[str(order)] = len(df)
        print(f"[fe_job] orden {order}: {len(df)} caminos", flush=True)

    summary = {
        "kind": "fe",
        "k": int(data["CE"].shape[0]),
        "seed": data["seed"],
        "orders": [int(order) for order in rows_per_order],
        "rows_per_order": rows_per_order,
        "computation-time(s)": total_time,
    }
    with open(Path("jobs_results") / f"{job_id}.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)
    return summary


def main(argv=None) -> None:
    args = get_fe_parser().parse_args(sys.argv[1:] if argv is None else argv)

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id is None:
        raise KeyError("SLURM_JOB_ID IS REQUIRED")

    try:
        data = load_and_validate(args.input_dir)
    except InputError as e:
        print(f"[fe_job] ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    print(f"[fe_job] entrada válida: k={data['CE'].shape[0]} m={data['CE'].shape[1]} n={data['CE'].shape[2]}",
          flush=True)

    # Import diferido: TensorFlow tarda en cargar y no hace falta si la entrada es inválida.
    import tensorflow as tf
    import forgeffects

    run_fe(data, slurm_job_id, forgeffects, tf.random.set_seed)


if __name__ == "__main__":
    main()
