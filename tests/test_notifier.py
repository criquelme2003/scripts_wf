import json

import pytest

import notifier

JOB_ID = "1234"


def write_result(content):
    with open(f"jobs_results/{JOB_ID}.json", "w", encoding="utf-8") as f:
        f.write(content if isinstance(content, str) else json.dumps(content))


def write_log(name, text):
    with open(f"logs/{name}", "w", encoding="utf-8") as f:
        f.write(text)


@pytest.mark.parametrize("job_name", ["forgethreads-new-job", "fe_job", "sweep_job"])
def test_find_log_by_job_id_for_any_job_name(job_name):
    write_log(f"{job_name}.{JOB_ID}", "hola")
    write_log(f"forgethreads-notifier.99{JOB_ID}", "otro job")
    write_log(f"{job_name}.{JOB_ID}9", "otro job")
    assert notifier.find_log(JOB_ID) == f"logs/{job_name}.{JOB_ID}"


def test_find_log_missing_returns_none():
    assert notifier.find_log(JOB_ID) is None


def test_read_log_tail_limits_to_last_bytes(monkeypatch):
    write_log(f"sweep_job.{JOB_ID}", "a" * 100 + "FIN")
    monkeypatch.setenv("NOTIFIER_MAX_LOG_BYTES", "10")
    assert notifier.read_log(JOB_ID) == "a" * 7 + "FIN"


def test_read_log_default_limit_is_200000(monkeypatch):
    monkeypatch.delenv("NOTIFIER_MAX_LOG_BYTES", raising=False)
    write_log(f"sweep_job.{JOB_ID}", "x" * 250000)
    assert len(notifier.read_log(JOB_ID)) == 200000


def test_read_log_tolerates_cut_multibyte_char(monkeypatch):
    write_log(f"fe_job.{JOB_ID}", "ñ" * 10)
    monkeypatch.setenv("NOTIFIER_MAX_LOG_BYTES", "5")
    assert notifier.read_log(JOB_ID).endswith("ññ")


def test_success_when_summary_has_no_running_state():
    write_log(f"forgethreads-new-job.{JOB_ID}", "log ok")
    write_result({"effective-order": 3, "computation-time(s)": 0.1})
    payload = notifier.build_payload(JOB_ID)
    assert payload == {
        "effective-order": 3, "computation-time(s)": 0.1,
        "status": "success", "logs": "log ok", "job_id": JOB_ID,
    }


def test_success_when_state_completed():
    write_result({"kind": "sweep", "state": "completed"})
    assert notifier.build_payload(JOB_ID)["status"] == "success"


def test_partial_when_state_running_keeps_summary():
    write_log(f"sweep_job.{JOB_ID}", "cortado")
    write_result({"kind": "sweep", "state": "running", "completed_combinations": 3})
    payload = notifier.build_payload(JOB_ID)
    assert payload["status"] == "partial"
    assert payload["completed_combinations"] == 3
    assert payload["logs"] == "cortado"


@pytest.mark.parametrize("content", [None, "{no es json", "[1, 2]"])
def test_error_when_summary_missing_or_invalid(content):
    write_log(f"fe_job.{JOB_ID}", "Traceback: ValueError")
    if content is not None:
        write_result(content)
    assert notifier.build_payload(JOB_ID) == {
        "status": "error", "logs": "Traceback: ValueError", "job_id": JOB_ID,
    }


def test_error_without_log_sends_null_logs():
    assert notifier.build_payload(JOB_ID) == {"status": "error", "logs": None, "job_id": JOB_ID}
