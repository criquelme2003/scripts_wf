import sys
import forgethreads as ft
from matrix_construction import sparse_supercritical_block_matrix
from parser import get_newjob_parser
import numpy as np
import time as ti
import os
import json
from pathlib import Path

ORDER = 100
if __name__ == "__main__":
    parser = get_newjob_parser()
    args = parser.parse_args(sys.argv[1:])

    N = args.nodos
    thr = args.thr
    c = args.conectividad
    seed = args.seed
    authToken = args.authToken

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id is None:
        raise KeyError("SLURM_JOB_ID IS REQUIRED")
      
    n = m = int(N / 2)
    N = n * 2

    matrix, _, _ = sparse_supercritical_block_matrix(n, m, c, seed)
    m1: np.ndarray = matrix.reshape(1, N, N).astype(np.float16)

    m2 = m1.copy()
    start = ti.time()
    _, _, eff_order = ft.maxmin(m1, m2, thr, ORDER)
    end = ti.time()

    total_time = end-start

    data = {
        "authToken": authToken,
        "effective-order": eff_order,
        "computation-time(s)": total_time,
    }

    os.makedirs("jobs_results", exist_ok=True)
    with open(f"jobs_results/{slurm_job_id}.json", "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
        
    
