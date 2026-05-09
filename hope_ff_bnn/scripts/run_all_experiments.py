#!/usr/bin/env python3
"""Ejecuta baterías generales de entrenamiento Forward-Forward.

Etapa del flujo:
    Orquestación de simulaciones C++ desde Make.

Entrada esperada:
    Directorio con datasets `.bin` preprocesados y grupos de hiperparámetros.

Salida generada:
    Una ejecución `make run` por combinación dataset/grupo. Cada ejecución
    produce `log.txt`, `metrics.csv` y `summary.md` por corrida.

Relación con Forward-Forward:
    Permite comparar el entrenamiento FF-BNN sobre resoluciones y
    cuantizaciones MNIST distintas sin editar manualmente comandos.
"""


from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


GROUPS = {
    "smoke": {
        "TRAIN_SAMPLES": "256",
        "EVAL_SAMPLES": "64",
        "EPOCHS": "1",
        "EVAL_EVERY_EPOCH": "0",
    },
    "debug": {
        "TRAIN_SAMPLES": "4096",
        "EVAL_SAMPLES": "1024",
        "EPOCHS": "5",
        "EVAL_EVERY_EPOCH": "0",
    },
    "comparacion": {
        "TRAIN_SAMPLES": "35700",
        "EVAL_SAMPLES": "6300",
        "EPOCHS": "20",
        "EVAL_EVERY_EPOCH": "1",
    },
    "barrido_final": {
        "TRAIN_SAMPLES": "35700",
        "EVAL_SAMPLES": "6300",
        "EPOCHS": "30",
        "EVAL_EVERY_EPOCH": "1",
    },
}


def parse_dataset_shape(path: Path) -> tuple[str, str, str]:
    """Extrae la forma del dataset desde su nombre de archivo.

    Args:
        path: Ruta del `.bin`, con nombre esperado tipo
            `mnist_20x20_4b_packed.bin`.

    Returns:
        Tupla `(ancho, alto, bits)` como cadenas para pasarlas a Make.

    Raises:
        ValueError: Si el nombre no contiene resolución y cuantización MNIST.

    Impacto:
        Evita compilar el testbench con macros incompatibles con el layout del
        dataset, una fuente común de métricas FF inválidas.
    """

    match = re.search(r"mnist_(\d+)x(\d+)_(\d+)b", path.name)
    if not match:
        raise ValueError(f"Cannot parse dataset shape from {path.name}")
    return match.group(1), match.group(2), match.group(3)


def main() -> int:
    """Lanza todas las combinaciones dataset/grupo mediante Make.

    Args:
        No recibe parámetros directos; usa `argparse` sobre CLI.

    Returns:
        Código de salida de estilo POSIX. Devuelve cero si todas las corridas
        completan, o el código del primer `make run` fallido.

    Raises:
        No propaga excepciones deliberadamente; los errores esperados se
        reportan por stderr y se convierten en códigos de salida.

    Impacto:
        Automatiza la validación comparativa del entrenamiento Forward-Forward
        para todos los datasets disponibles y los grupos definidos.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--make", default="make")
    parser.add_argument("--data-dir", default="../mnist/data/processed")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--groups",
        default="smoke debug comparacion barrido_final",
        help="Space-separated group names to run.",
    )
    parser.add_argument(
        "extra_make_vars",
        nargs=argparse.REMAINDER,
        help="Extra VAR=VALUE arguments passed to each recursive make run.",
    )
    args = parser.parse_args()

    # El descubrimiento dinámico permite agregar datasets sin editar el script.
    data_dir = Path(args.data_dir)
    datasets = sorted(data_dir.glob("*.bin"))
    if not datasets:
        print(f"ERROR: no .bin datasets found in {data_dir}", file=sys.stderr)
        return 1

    # Validar los grupos antes de ejecutar evita barridos largos a medias.
    groups = args.groups.split()
    unknown_groups = [group for group in groups if group not in GROUPS]
    if unknown_groups:
        print(f"ERROR: unknown groups: {', '.join(unknown_groups)}",
              file=sys.stderr)
        return 1

    extra = [item for item in args.extra_make_vars if item != "--"]
    total = len(datasets) * len(groups)
    run_id = 1

    print(f"Running {total} experiments")
    for dataset in datasets:
        img_w, img_h, pix_bits = parse_dataset_shape(dataset)
        for group in groups:
            group_vars = GROUPS[group]
            # Make sigue siendo la fuente de verdad para compilar macros HLS,
            # por eso el script delega cada experimento como una receta `run`.
            cmd = [
                args.make,
                "run",
                f"DATASET={dataset.as_posix()}",
                f"GROUP={group}",
                f"RESULTS_DIR={args.results_dir}",
                f"IMG_W={img_w}",
                f"IMG_H={img_h}",
                f"PIX_BITS={pix_bits}",
            ]
            cmd.extend(f"{key}={value}" for key, value in group_vars.items())
            cmd.extend(extra)

            print(f"[{run_id:02d}/{total:02d}] {' '.join(cmd)}", flush=True)
            completed = subprocess.run(cmd)
            if completed.returncode != 0:
                print(
                    f"ERROR: experiment failed for {dataset.name} / {group}",
                    file=sys.stderr,
                )
                return completed.returncode
            run_id += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
