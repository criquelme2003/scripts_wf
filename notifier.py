import glob
import json
import os
import sys
import time
from pathlib import Path
import requests

from parser import get_notifier_parser

MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2
DEFAULT_HTTP_TIMEOUT = 30
DEFAULT_MAX_LOG_BYTES = 200000


def find_log(slurm_job_id: str) -> str | None:
    # El log es logs/<job-name>.<id> para cualquier --job-name (new_job, fe_job, sweep_job).
    matches = sorted(glob.glob(f"logs/*.{glob.escape(slurm_job_id)}"))
    return matches[0] if matches else None


def read_log(slurm_job_id: str) -> str | None:
    path = find_log(slurm_job_id)
    if path is None:
        return None
    max_bytes = int(os.environ.get("NOTIFIER_MAX_LOG_BYTES", DEFAULT_MAX_LOG_BYTES))
    try:
        with open(path, "rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            log_file.seek(max(0, log_file.tell() - max_bytes))
            return log_file.read().decode("utf-8", errors="ignore")
    except OSError:
        return None


def build_payload(slurm_job_id: str) -> dict:
    logs = read_log(slurm_job_id)

    try:
        with open(f"jobs_results/{slurm_job_id}.json", "r", encoding="utf-8") as file:
            summary = json.load(file)
    except (OSError, json.JSONDecodeError):
        summary = None

    if isinstance(summary, dict):
        payload = summary
        # "running" en el resumen indica que el job murió antes de terminar.
        payload["status"] = "partial" if summary.get("state") == "running" else "success"
        payload["logs"] = logs
    else:
        payload = {"status": "error", "logs": logs}

    payload["job_id"] = slurm_job_id
    return payload


def send_with_retries(callback_url: str, auth_token: str, payload: dict, timeout: int) -> str | None:
    headers = {"Authorization": f"Bearer {auth_token}"}

    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.post(callback_url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            return None
        except requests.RequestException as e:
            last_error = str(e)
            if e.response is not None:
                try:
                    last_error = json.dumps({
                        "error": str(e),
                        "response": e.response.json(),
                    })
                except ValueError:
                    last_error = json.dumps({
                        "error": str(e),
                        "response": e.response.text,
                    })
            print(f"[notifier] intento {attempt}/{MAX_ATTEMPTS} fallido: {e}", file=sys.stderr)
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS)

    return last_error


def save_fallback(slurm_job_id: str, payload: dict, error: str) -> None:
    os.makedirs("jobs_failed_notify", exist_ok=True)
    payload_with_error = {**payload, "notify_error": error}
    with open(f"jobs_failed_notify/{slurm_job_id}.json", "w", encoding="utf-8") as file:
        json.dump(payload_with_error, file, indent=4)


if __name__ == "__main__":
    parser = get_notifier_parser()
    args = parser.parse_args(sys.argv[1:])
    Path("jobs_failed_notify").mkdir( exist_ok=True)

    if os.environ.get("SLURM_JOB_ID") is None:
        raise KeyError("SLURM_JOB_ID IS REQUIRED")

    timeout = int(os.environ.get("NOTIFIER_HTTP_TIMEOUT", DEFAULT_HTTP_TIMEOUT))

    payload = build_payload(args.jobId)
    error = send_with_retries(args.callbackUrl, args.authToken, payload, timeout)

    if error is not None:
        save_fallback(args.jobId, payload, error)
