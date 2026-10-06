"""Genera un directorio de entrada para fe_job a partir del dataset de forgeffects
(k=10 expertos, m=16 causas, n=4 efectos).

Uso: python tests/fixtures/make_fe_input.py <out_dir> --k K [--thr T] [--maxorder O]
                                             [--reps R] [--seed S] [--broken]
--broken deja EE con un k distinto (entrada inválida para probar el camino de error).
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def dataset_dir() -> Path:
    # Sin importar forgeffects (cargaría TensorFlow): se ubica el paquete en disco.
    spec = importlib.util.find_spec("forgeffects")
    return Path(list(spec.submodule_search_locations)[0]) / "dataset"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--maxorder", type=int, default=3)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--broken", action="store_true")
    args = parser.parse_args()

    src = dataset_dir()
    CC, CE, EE = (np.load(src / f"{name}.npy").astype(np.float32)[:args.k] for name in ("CC", "CE", "EE"))
    if args.broken:
        EE = EE[:max(1, args.k - 1)] if args.k > 1 else np.concatenate([EE, EE])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, array in (("CC", CC), ("CE", CE), ("EE", EE)):
        np.save(out / f"{name}.npy", array)
    meta = {
        "causes": [f"a{i + 1}" for i in range(CE.shape[1])],
        "effects": [f"b{j + 1}" for j in range(CE.shape[2])],
        "thr": args.thr, "maxorder": args.maxorder, "reps": args.reps,
    }
    if args.seed is not None:
        meta["seed"] = args.seed
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{out}: CC{CC.shape} CE{CE.shape} EE{EE.shape}")


if __name__ == "__main__":
    main()
