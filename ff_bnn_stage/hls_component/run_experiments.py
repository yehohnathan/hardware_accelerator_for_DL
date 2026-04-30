"""Ejecuta barridos de experimentos MNIST usando el Makefile local.

El script automatiza las combinaciones de resolucion, bits por pixel e
hiperparametros. Cada subgrupo de resultados se guarda en un archivo de texto
agrupado por configuracion y por normalizacion de bits.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "experiment_results"
MAKE_COMMAND = "make"

RESOLUTIONS = [(28, 28), (20, 20), (16, 16)]

PIXEL_BITS_LIST = [1, 4, 6]

HYPERPARAMETER_GROUPS = [
    # Smoke test rapido para validar el flujo completo sin esperar mucho tiempo
    {"name": "smoke",
     "train_samples": 256,
     "eval_samples": 64,
     "epochs": 1},
    # Debug con pocas muestras y epocas, y tuning
    {"name": "debug",
     "train_samples": 4096,
     "eval_samples": 1024,
     "epochs": 5},
    # Comparacion con configuracion de referencia y tuning
    {"name": "comparacion",
     "train_samples": 35700,
     "eval_samples": 6300,
     "epochs": 20},
    # Barrido final con configuracion de referencia y tuning, y mas epocas
    {"name": "barrido_final",
     "train_samples": 35700,
     "eval_samples": 6300,
     "epochs": 30}
]

GROUP_CHOICES = [group["name"] for group in HYPERPARAMETER_GROUPS]

METRIC_PATTERNS = {
    "input_mode": r"input_mode\s*=\s*(.+)",
    "image_resolution": r"image_resolution\s*=\s*(.+)",
    "stage_b_ok": r"stage_b_validation_ok\s*=\s*(.+)",
    "holdout_correct_count": r"holdout_correct_count\s*=\s*(.+)",
    "holdout_accuracy": r"holdout_accuracy\s*=\s*(.+)",
    "loss_pos": r"epoch\[\d+\]\s+loss_pos=([^\s]+)",
    "loss_neg": r"epoch\[\d+\].*?\sloss_neg=([^\s]+)",
    "goodness_pos": r"epoch\[\d+\].*?\sg_pos=([^\s]+)",
    "goodness_neg": r"epoch\[\d+\].*?\sg_neg=([^\s]+)",
    "goodness_gap": r"epoch\[\d+\].*?\sgap=([^\s]+)",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Representa una corrida concreta del Makefile.

    Parametros:
        group_name: Nombre del grupo de hiperparametros.
        image_width: Ancho de la imagen que se pasa al Makefile.
        image_height: Alto de la imagen que se pasa al Makefile.
        pixel_bits: Cantidad de bits por pixel empaquetado.
        train_samples: Cantidad de muestras de entrenamiento.
        eval_samples: Cantidad de muestras de evaluacion.
        epochs: Cantidad de epocas de entrenamiento.

    Retorna:
        La clase no retorna valores directamente. Sus instancias agrupan los
        parametros necesarios para ejecutar un experimento.
    """

    group_name: str
    image_width: int
    image_height: int
    pixel_bits: int
    train_samples: int
    eval_samples: int
    epochs: int


@dataclass(frozen=True)
class ExperimentResult:
    """Almacena el resultado capturado de una corrida.

    Parametros:
        config: Configuracion usada para ejecutar el experimento.
        command: Comando completo enviado a subprocess.
        return_code: Codigo de retorno del proceso.
        elapsed_seconds: Duracion de la corrida en segundos.
        stdout: Salida estandar capturada.
        stderr: Salida de error capturada.

    Retorna:
        La clase no retorna valores directamente. Sus instancias conservan los
        datos necesarios para escribir el resumen final.
    """

    config: ExperimentConfig
    command: list[str]
    return_code: int
    elapsed_seconds: float
    stdout: str
    stderr: str


def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la linea de comandos.

    Parametros:
        No recibe parametros.

    Retorna:
        Un objeto Namespace con las opciones de ejecucion seleccionadas.
    """

    parser = argparse.ArgumentParser(
        description="Ejecuta experimentos con make run y agrupa resultados.",
    )
    parser.add_argument(
        "--make",
        default=MAKE_COMMAND,
        help="Ejecutable de make que se usara para lanzar el Makefile.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Carpeta donde se escribiran los resumenes txt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra los comandos sin ejecutarlos.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Tiempo maximo por corrida en segundos.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=GROUP_CHOICES,
        default=GROUP_CHOICES,
        help="Grupos de hiperparametros que se quieren ejecutar.",
    )
    return parser.parse_args()


def build_experiments(selected_groups: set[str]) -> list[ExperimentConfig]:
    """Construye todas las combinaciones del barrido experimental.

    Parametros:
        selected_groups: Nombres de grupos que deben incluirse.

    Retorna:
        Una lista con las configuraciones de experimento seleccionadas.
    """

    experiments: list[ExperimentConfig] = []

    for group in HYPERPARAMETER_GROUPS:
        if group["name"] not in selected_groups:
            continue

        for pixel_bits in PIXEL_BITS_LIST:
            for image_width, image_height in RESOLUTIONS:
                experiments.append(
                    ExperimentConfig(
                        group_name=group["name"],
                        image_width=image_width,
                        image_height=image_height,
                        pixel_bits=pixel_bits,
                        train_samples=group["train_samples"],
                        eval_samples=group["eval_samples"],
                        epochs=group["epochs"],
                    )
                )

    return experiments


def build_make_command(
    make_command: str,
    config: ExperimentConfig,
) -> list[str]:
    """Crea el comando make run para una configuracion concreta.

    Parametros:
        make_command: Ejecutable de make que se invocara.
        config: Configuracion de experimento.

    Retorna:
        Una lista de argumentos lista para subprocess.run.
    """

    return [
        make_command,
        "run",
        f"IMG_W={config.image_width}",
        f"IMG_H={config.image_height}",
        f"PIX_BITS={config.pixel_bits}",
        f"TRAIN_SAMPLES={config.train_samples}",
        f"EVAL_SAMPLES={config.eval_samples}",
        f"EPOCHS={config.epochs}",
    ]


def run_experiment(
    make_command: str,
    config: ExperimentConfig,
    timeout: int | None,
    dry_run: bool,
) -> ExperimentResult:
    """Ejecuta una corrida individual y captura su salida.

    Parametros:
        make_command: Ejecutable de make que se invocara.
        config: Configuracion de experimento.
        timeout: Tiempo maximo permitido en segundos.
        dry_run: Indica si el comando solo debe simularse.

    Retorna:
        Un ExperimentResult con stdout, stderr, codigo de retorno y tiempo.
    """

    command = build_make_command(make_command, config)
    start_time = dt.datetime.now()

    if dry_run:
        return ExperimentResult(
            config=config,
            command=command,
            return_code=0,
            elapsed_seconds=0.0,
            stdout="DRY RUN: comando no ejecutado.",
            stderr="",
        )

    try:
        completed = subprocess.run(
            command,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        elapsed = (dt.datetime.now() - start_time).total_seconds()

        return ExperimentResult(
            config=config,
            command=command,
            return_code=completed.returncode,
            elapsed_seconds=elapsed,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = (dt.datetime.now() - start_time).total_seconds()
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stderr += "\nERROR: la corrida supero el timeout configurado."

        return ExperimentResult(
            config=config,
            command=command,
            return_code=-1,
            elapsed_seconds=elapsed,
            stdout=stdout,
            stderr=stderr,
        )
    except OSError as exc:
        elapsed = (dt.datetime.now() - start_time).total_seconds()

        return ExperimentResult(
            config=config,
            command=command,
            return_code=-2,
            elapsed_seconds=elapsed,
            stdout="",
            stderr=f"ERROR: no fue posible ejecutar el comando: {exc}",
        )


def extract_last_metric(output: str, pattern: str) -> str:
    """Extrae la ultima coincidencia de una metrica en el texto.

    Parametros:
        output: Texto completo de stdout y stderr.
        pattern: Expresion regular de la metrica buscada.

    Retorna:
        El ultimo valor encontrado o N/A si la metrica no aparece.
    """

    matches = re.findall(pattern, output)

    if not matches:
        return "N/A"

    return str(matches[-1]).strip()


def summarize_result(result: ExperimentResult) -> str:
    """Genera un resumen textual de una corrida.

    Parametros:
        result: Resultado capturado de la corrida.

    Retorna:
        Un bloque de texto con comando, estado, metricas y salida capturada.
    """

    combined_output = f"{result.stdout}\n{result.stderr}"
    metrics = {
        name: extract_last_metric(combined_output, pattern)
        for name, pattern in METRIC_PATTERNS.items()
    }
    status = "OK" if result.return_code == 0 else "FAIL"
    cfg = result.config

    lines = [
        "=" * 78,
        f"experimento       = {cfg.group_name}",
        f"resolucion        = {cfg.image_width}x{cfg.image_height}",
        f"bits_pixel        = {cfg.pixel_bits}",
        f"train_samples     = {cfg.train_samples}",
        f"eval_samples      = {cfg.eval_samples}",
        f"epochs            = {cfg.epochs}",
        f"status            = {status}",
        f"return_code       = {result.return_code}",
        f"elapsed_seconds   = {result.elapsed_seconds:.2f}",
        f"command           = {' '.join(result.command)}",
        "-" * 78,
        "metricas_extraidas:",
    ]

    for metric_name, metric_value in metrics.items():
        lines.append(f"{metric_name:22s}= {metric_value}")

    lines.extend(
        [
            "-" * 78,
            "stdout:",
            result.stdout.strip() or "N/A",
            "-" * 78,
            "stderr:",
            result.stderr.strip() or "N/A",
            "",
        ]
    )

    return "\n".join(lines)


def group_experiments(
    experiments: Iterable[ExperimentConfig],
) -> dict[tuple[str, int], list[ExperimentConfig]]:
    """Agrupa experimentos por configuracion y bits por pixel.

    Parametros:
        experiments: Coleccion de configuraciones de experimento.

    Retorna:
        Un diccionario cuya clave es (grupo, bits) y cuyo valor contiene las
        resoluciones pertenecientes a ese subgrupo.
    """

    grouped: dict[tuple[str, int], list[ExperimentConfig]] = {}

    for experiment in experiments:
        key = (experiment.group_name, experiment.pixel_bits)
        grouped.setdefault(key, []).append(experiment)

    return grouped


def write_group_summary(
    output_dir: Path,
    group_name: str,
    pixel_bits: int,
    results: list[ExperimentResult],
) -> Path:
    """Escribe el archivo txt de un subgrupo de experimentos.

    Parametros:
        output_dir: Carpeta de salida.
        group_name: Nombre del grupo de hiperparametros.
        pixel_bits: Bits por pixel del subgrupo.
        results: Resultados capturados para el subgrupo.

    Retorna:
        La ruta del archivo escrito.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{group_name}_bits{pixel_bits}.txt"
    ok_count = sum(1 for result in results if result.return_code == 0)
    fail_count = len(results) - ok_count
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header = [
        f"resumen_generado = {timestamp}",
        f"grupo            = {group_name}",
        f"bits_pixel       = {pixel_bits}",
        f"corridas         = {len(results)}",
        f"ok               = {ok_count}",
        f"fallos           = {fail_count}",
        "",
    ]
    body = [summarize_result(result) for result in results]

    output_path.write_text("\n".join(header + body), encoding="utf-8")

    return output_path


def main() -> int:
    """Ejecuta el flujo completo de automatizacion.

    Parametros:
        No recibe parametros directos. Lee opciones desde argparse.

    Retorna:
        Cero si todas las corridas terminan correctamente, uno si alguna falla.
    """

    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    selected_groups = set(args.groups)
    experiments = build_experiments(selected_groups)
    grouped = group_experiments(experiments)
    failed_runs = 0

    print(f"Total de corridas planificadas: {len(experiments)}")
    print(f"Grupos seleccionados: {', '.join(args.groups)}")
    print(f"Carpeta de salida: {output_dir}")

    for group_name, pixel_bits in grouped:
        print(f"\nGrupo: {group_name}, bits: {pixel_bits}")
        results: list[ExperimentResult] = []

        for experiment in grouped[(group_name, pixel_bits)]:
            label = f"{experiment.image_width}x{experiment.image_height}"
            print(f"  Ejecutando {label}...", end="", flush=True)

            result = run_experiment(
                make_command=args.make,
                config=experiment,
                timeout=args.timeout,
                dry_run=args.dry_run,
            )
            results.append(result)

            if result.return_code == 0:
                print(" OK")
            else:
                failed_runs += 1
                print(f" FAIL ({result.return_code})")

        summary_path = write_group_summary(
            output_dir=output_dir,
            group_name=group_name,
            pixel_bits=pixel_bits,
            results=results,
        )
        print(f"  Resumen escrito en: {summary_path}")

    if failed_runs:
        print(f"\nCorridas con fallo: {failed_runs}")
        return 1

    print("\nTodas las corridas finalizaron correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
