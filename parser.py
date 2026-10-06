import argparse
import re
from urllib.parse import urlparse

JOB_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
ALLOWED_CALLBACK_SCHEMES = ("https", "http")


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"debe ser un entero positivo, recibido: {value}")
    return ivalue


def positive_float(value: str) -> float:
    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError(f"debe ser un número positivo, recibido: {value}")
    return fvalue


def job_id_type(value: str) -> str:
    if not JOB_ID_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "job-id inválido: solo se permiten letras, números, '.', '_' y '-' "
            f"(recibido: {value!r})"
        )
    return value


def callback_url_type(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in ALLOWED_CALLBACK_SCHEMES or not parsed.netloc:
        raise argparse.ArgumentTypeError(
            f"callback-url inválida: debe ser http(s) con host (recibido: {value!r})"
        )
    return value


def auth_token_type(value: str) -> str:
    if not value or any(c in value for c in ("\r", "\n")):
        raise argparse.ArgumentTypeError("auth-token inválido: no puede estar vacío ni contener saltos de línea")
    return value


def get_newjob_parser() -> argparse.ArgumentParser :
    parser = argparse.ArgumentParser(description="forgethreads module parser")
    parser.add_argument(
        "--nodos",
        type=positive_int,
        required=True,
        dest="nodos",
        help="Definir número de nodos para la matriz (entero positivo)",
    )

    parser.add_argument(
            "--thr",
            type=positive_float,
            required=True,
            dest="thr",
            help="Definir threshold para filtrado de efectos olvidados (número positivo)",
    )

    parser.add_argument(
                "--conectividad",
                type=positive_float,
                required=True,
                dest="conectividad",
                help="Definir aristas promedio por nodo (número positivo)",
    )

    parser.add_argument(
                    "--seed",
                    type=int,
                    required=True,
                    dest="seed",
                    help="Definir semilla para la construcción de la matriz",
    )

    return parser


def comma_list(item_type):
    def parse(value: str) -> list:
        items = [item.strip() for item in value.split(",")]
        if not all(items):
            raise argparse.ArgumentTypeError(f"lista separada por comas inválida: {value!r}")
        return [item_type(item) for item in items]
    return parse


def get_sweep_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="forgethreads sweep parser")
    parser.add_argument(
        "--ns",
        type=comma_list(positive_int),
        required=True,
        dest="ns",
        help="Lista de N separada por comas (enteros positivos; los impares se omiten)",
    )

    parser.add_argument(
        "--cs",
        type=comma_list(positive_float),
        required=True,
        dest="cs",
        help="Lista de conectividades c separada por comas (números positivos)",
    )

    parser.add_argument(
        "--reps",
        type=positive_int,
        required=True,
        dest="reps",
        help="Repeticiones por combinación (c, N)",
    )

    parser.add_argument(
        "--thr",
        type=positive_float,
        required=True,
        dest="thr",
        help="Threshold para filtrado de efectos olvidados (número positivo)",
    )

    parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        dest="seed_base",
        help="Base de semillas: seed = seed_base * 10**9 + rep * 1000 + N (0 reproduce test2.py)",
    )

    return parser


def get_fe_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="forgeffects FE parser")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        dest="input_dir",
        help="Directorio con CC.npy, CE.npy, EE.npy y meta.json (p. ej. jobs_inputs/<request_id>)",
    )

    return parser


def get_notifier_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="forgethreads module notifier parser")
    parser.add_argument(
        "--job-id",
        type=job_id_type,
        required=True,
        dest="jobId",
        help="SLURM_JOB_ID del new_job cuyo resultado debe reportarse (no el del propio notifier). "
             "Solo letras, números, '.', '_' y '-'",
    )

    parser.add_argument(
        "--auth-token",
        type=auth_token_type,
        required=True,
        dest="authToken",
        help="Token para autenticar el callback ante la API (enviado como Bearer)",
    )

    parser.add_argument(
        "--callback-url",
        type=callback_url_type,
        required=True,
        dest="callbackUrl",
        help="URL del endpoint de la API que recibirá el resultado del job (http/https)",
    )

    return parser
