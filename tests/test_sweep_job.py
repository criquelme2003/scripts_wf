import csv
import json
import sys

import numpy as np
import pytest

import sweep_job
from matrix_construction import sparse_supercritical_block_matrix
from parser import get_sweep_parser

JOB_ID = "4242"


def params(**overrides):
    base = {"ns": [20, 40], "cs": [1.0, 2.5], "reps": 2, "thr": 0.5, "seed_base": 0}
    base.update(overrides)
    return base


def read_csv():
    with open(f"jobs_results/{JOB_ID}.csv", newline="") as f:
        return list(csv.reader(f))


def read_summary():
    with open(f"jobs_results/{JOB_ID}.json", encoding="utf-8") as f:
        return json.load(f)


# --- parser -----------------------------------------------------------------

def test_parser_parses_lists_and_types():
    args = get_sweep_parser().parse_args(
        ["--ns", "100,250", "--cs", "0.125,1,2.5", "--reps", "20", "--thr", "0.5", "--seed-base", "3"])
    assert args.ns == [100, 250]
    assert args.cs == [0.125, 1.0, 2.5] and all(isinstance(c, float) for c in args.cs)
    assert (args.reps, args.thr, args.seed_base) == (20, 0.5, 3)


def test_parser_seed_base_defaults_to_zero():
    args = get_sweep_parser().parse_args(["--ns", "20", "--cs", "1", "--reps", "1", "--thr", "0.5"])
    assert args.seed_base == 0


@pytest.mark.parametrize("flag, value", [("--ns", "0,20"), ("--ns", "20,x"), ("--cs", "-1"), ("--cs", ""),
                                         ("--reps", "0")])
def test_parser_rejects_invalid_values(flag, value):
    argv = {"--ns": "20", "--cs": "1", "--reps": "1", "--thr": "0.5"}
    argv[flag] = value
    with pytest.raises(SystemExit):
        get_sweep_parser().parse_args([x for kv in argv.items() for x in kv])


# --- semillas y combinaciones ---------------------------------------------

@pytest.mark.parametrize("rep, n", [(0, 100), (3, 250), (19, 500)])
def test_seed_with_base_zero_matches_test2(rep, n):
    assert sweep_job.seed_for(0, rep, n) == rep * 1000 + n


def test_seed_base_offsets_by_1e9():
    assert sweep_job.seed_for(2, 3, 100) == 2 * 10**9 + 3100


def test_split_combinations_uses_matrix_rule_and_c_major_order():
    valid, skipped = sweep_job.split_combinations(ns=[20, 21, 100], cs=[0.125, 9.0, 9.5, 50.0])
    assert valid == [(0.125, 20), (0.125, 100), (9.0, 20), (9.0, 100), (9.5, 100)]
    assert {(s["c"], s["n"]) for s in skipped} == {(0.125, 21), (9.0, 21), (9.5, 20), (9.5, 21),
                                                    (50.0, 20), (50.0, 21), (50.0, 100)}
    odd = next(s for s in skipped if s["n"] == 21)
    assert "par" in odd["reason"]
    border = next(s for s in skipped if (s["c"], s["n"]) == (9.5, 20))
    assert "Inadmissible" in border["reason"]


# --- barrido ---------------------------------------------------------------

def test_run_sweep_writes_csv_and_completed_summary(fake_forgethreads):
    fake_forgethreads.maxmin.return_value = (None, None, np.int64(7))
    sweep_job.run_sweep(params(), JOB_ID, fake_forgethreads, free_memory=lambda: None)

    rows = read_csv()
    assert rows[0] == ["c", "n", "repeticion", "orden_efectivo"]
    assert rows[1:] == [
        ["1.0", "20", "0", "7"], ["1.0", "20", "1", "7"], ["1.0", "40", "0", "7"], ["1.0", "40", "1", "7"],
        ["2.5", "20", "0", "7"], ["2.5", "20", "1", "7"], ["2.5", "40", "0", "7"], ["2.5", "40", "1", "7"],
    ]
    summary = read_summary()
    assert summary["kind"] == "sweep" and summary["state"] == "completed"
    assert summary["params"] == params()
    assert summary["completed_combinations"] == summary["total_combinations"] == 4
    assert summary["skipped_combinations"] == [] and summary["failures"] == []
    assert summary["computation-time(s)"] >= 0


def test_run_sweep_feeds_maxmin_like_test2(fake_forgethreads):
    sweep_job.run_sweep(params(ns=[20], cs=[2.5], reps=2), JOB_ID, fake_forgethreads, free_memory=lambda: None)

    calls = fake_forgethreads.maxmin.call_args_list
    warmup, *reps = calls
    assert warmup.args[0].shape == (1, 200, 200)
    for rep, call in enumerate(reps):
        m1, m2, thr, order = call.args
        expected, _, _ = sparse_supercritical_block_matrix(10, 10, 2.5, seed=rep * 1000 + 20)
        assert m1.dtype == np.float16 and m1.shape == (1, 20, 20)
        assert np.array_equal(m1[0], expected.astype(np.float16))
        assert np.array_equal(m1, m2) and m1 is not m2
        assert (thr, order) == (0.5, 100)


def test_run_sweep_continues_after_failed_repetition(fake_forgethreads):
    results = iter([(None, None, 1), (None, None, 2), RuntimeError("boom"), (None, None, 4), (None, None, 5)])

    def maxmin(*_):
        r = next(results)
        if isinstance(r, Exception):
            raise r
        return r

    fake_forgethreads.maxmin.side_effect = maxmin
    sweep_job.run_sweep(params(ns=[20, 40], cs=[1.0], reps=2), JOB_ID, fake_forgethreads, free_memory=lambda: None)

    assert [r[3] for r in read_csv()[1:]] == ["2", "4", "5"]
    summary = read_summary()
    assert summary["state"] == "completed"
    assert summary["failures"] == [{"c": 1.0, "n": 20, "rep": 1, "error": "RuntimeError: boom"}]


def test_summary_is_running_and_csv_flushed_during_sweep(fake_forgethreads):
    seen = []

    def maxmin(m1, *_):
        if m1.shape == (1, 40, 40):
            seen.append((read_summary()["state"], read_summary()["completed_combinations"], len(read_csv())))
        return None, None, 3

    fake_forgethreads.maxmin.side_effect = maxmin
    sweep_job.run_sweep(params(ns=[20, 40], cs=[1.0], reps=1), JOB_ID, fake_forgethreads, free_memory=lambda: None)
    assert seen == [("running", 1, 2)]  # header + 1 fila de la combinación (1.0, 20) ya escritas


def test_skipped_combinations_go_to_summary(fake_forgethreads):
    sweep_job.run_sweep(params(ns=[20], cs=[1.0, 50.0], reps=1), JOB_ID, fake_forgethreads,
                        free_memory=lambda: None)
    summary = read_summary()
    assert summary["total_combinations"] == 1
    assert [(s["c"], s["n"]) for s in summary["skipped_combinations"]] == [(50.0, 20)]


def test_sigterm_keeps_finished_combinations_and_running_state(fake_forgethreads):
    import os
    import signal

    def maxmin(m1, *_):
        if m1.shape == (1, 40, 40):  # SLURM avisa del corte durante la segunda combinación
            os.kill(os.getpid(), signal.SIGTERM)
        return None, None, 3

    fake_forgethreads.maxmin.side_effect = maxmin
    previous = signal.getsignal(signal.SIGTERM)
    try:
        sweep_job.install_sigterm_handler()
        with pytest.raises(SystemExit) as exc:
            sweep_job.run_sweep(params(ns=[20, 40], cs=[1.0], reps=2), JOB_ID, fake_forgethreads,
                                free_memory=lambda: None)
    finally:
        signal.signal(signal.SIGTERM, previous)

    assert exc.value.code == 143
    assert read_csv()[1:] == [["1.0", "20", "0", "3"], ["1.0", "20", "1", "3"]]
    summary = read_summary()
    assert summary["state"] == "running"
    assert summary["completed_combinations"] == 1 and summary["failures"] == []


def test_free_memory_called_once_per_repetition(fake_forgethreads, fake_cupy):
    free = sweep_job.get_free_memory()
    sweep_job.run_sweep(params(), JOB_ID, fake_forgethreads, free_memory=free)
    assert fake_cupy.pool.free_all_blocks.call_count == 8


def test_free_memory_is_noop_without_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)  # import cupy -> ImportError
    sweep_job.get_free_memory()()  # no lanza
