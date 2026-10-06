"""Sondeo empírico de forgeffects.FE (T2). Se lanza con: sbatch tests/probe_forgeffects.sh

Usa el dataset incluido en forgeffects (k=10 expertos, m=16 causas, n=4 efectos) y
reporta, por caso, qué devuelve FE: largo de la lista, columnas por índice, filas,
cómo se comporta sin caminos y si el resultado es determinista.
"""
import time

import numpy as np
import pandas as pd
import forgeffects
from forgeffects import load_test_data

CC = load_test_data("CC.npy")
CE = load_test_data("CE.npy")
EE = load_test_data("EE.npy")
k, m, n = CE.shape
causes = [f"c{i}" for i in range(m)]
effects = [f"e{j}" for j in range(n)]
print(f"dataset: k={k} m={m} n={n}")


def describe(label, **kwargs):
    print(f"\n=== {label}: {kwargs.get('extra', '')}")
    kwargs.pop("extra", None)
    start = time.time()
    try:
        res = forgeffects.FE(causes=causes, effects=effects, device="GPU", **kwargs)
    except Exception as e:
        print(f"  EXCEPCION {type(e).__name__}: {e}")
        return None
    print(f"  tiempo={time.time() - start:.2f}s  type={type(res).__name__}  len={len(res)}")
    for i, df in enumerate(res):
        through = [c for c in df.columns if c.startswith("Through")]
        print(f"  [{i}] {type(df).__name__} filas={len(df)} through={len(through)} "
              f"-> orden inferido={len(through) + 1}  columnas={list(df.columns)}")
        if len(df):
            print("      dtypes:", dict(df.dtypes.astype(str)))
            print("      primera fila:", df.iloc[0].to_dict())
    return res


r1 = describe("A k=1", CC=CC[:1], CE=CE[:1], EE=EE[:1], THR=0.5, maxorder=4, rep=1, extra="experto 0")
r_agg = describe("B k=10 agregado", CC=CC, CE=CE, EE=EE, THR=0.5, maxorder=4, rep=100)
describe("C sin efectos", CC=CC[:1], CE=CE[:1], EE=EE[:1], THR=1.0, maxorder=3, rep=1, extra="THR=1.0")
describe("D maxorder alto", CC=CC[:1], CE=CE[:1], EE=EE[:1], THR=0.5, maxorder=12, rep=1)

print("\n=== E determinismo")
for label, kw in [("k=1 rep=1", dict(CC=CC[:1], CE=CE[:1], EE=EE[:1], rep=1)),
                  ("k=10 rep=100", dict(CC=CC, CE=CE, EE=EE, rep=100))]:
    a = forgeffects.FE(causes=causes, effects=effects, THR=0.5, maxorder=3, device="GPU", **kw)
    b = forgeffects.FE(causes=causes, effects=effects, THR=0.5, maxorder=3, device="GPU", **kw)
    same = len(a) == len(b) and all(
        x.sort_values(list(x.columns)).reset_index(drop=True).equals(
            y.sort_values(list(y.columns)).reset_index(drop=True)) for x, y in zip(a, b))
    print(f"  {label}: dos llamadas iguales = {same}")

print("\nPROBE OK")
