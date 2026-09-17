import argparse
def get_newjob_parser() -> argparse.ArgumentParser :
    parser = argparse.ArgumentParser(description="forgethreads module parser")
    parser.add_argument(
        "--nodos",
        type=int,
        required=True,
        dest="nodos",
        help="Definir número de nodos para la matriz",
    )
    
    parser.add_argument(
            "--thr",
            type=float,
            required=True,
            dest="thr",
            help="Definir threshold para filtrado de efectos olvidados",
    )
    
    parser.add_argument(
                "--conectividad",
                type=float,
                required=True,
                dest="conectividad",
                help="Definir aristas promedio por nodo",
    )
    
    parser.add_argument(
                    "--seed",
                    type=int,
                    required=True,
                    dest="seed",
                    help="Definir semilla para la construcción de la matriz",
    )
    
    return parser


def get_notifier_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="forgethreads module notifier parser")
    parser.add_argument(
        "--job-id",
        type=str,
        required=True,
        dest="jobId",
        help="SLURM_JOB_ID del new_job cuyo resultado debe reportarse (no el del propio notifier)",
    )

    parser.add_argument(
        "--auth-token",
        type=str,
        required=True,
        dest="authToken",
        help="Token para autenticar el callback ante la API (enviado como Bearer)",
    )

    parser.add_argument(
        "--callback-url",
        type=str,
        required=True,
        dest="callbackUrl",
        help="URL del endpoint de la API que recibirá el resultado del job",
    )

    return parser

