#!/usr/bin/env python3
"""Ejecuta el barrido grande de arquitecturas FF-BNN.

Etapa del flujo:
    Exploración experimental de hiperparámetros y arquitecturas.

Entrada esperada:
    Directorio de datasets `.bin` y perfiles de arquitectura definidos en
    `ARCH_PROFILES`.

Salida generada:
    Comandos `make run` para cada combinación dataset/perfil, con resultados en
    `results/sweep100` salvo que se indique otra ruta.

Relación con Forward-Forward:
    Compara accuracy y complejidad para redes BNN FF de distinto tamaño,
    paralelismo, threshold y número de épocas.
"""


from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ARCH_PROFILES = [
    {
        "layers": 1,
        "values": "16",
        "parallel": 4,
        "threshold": 48,
        "epochs": 20,
        "eval": 0,
    },
    {
        "layers": 1,
        "values": "32",
        "parallel": 4,
        "threshold": 56,
        "epochs": 20,
        "eval": 1,
    },
    {
        "layers": 1,
        "values": "64",
        "parallel": 8,
        "threshold": 72,
        "epochs": 30,
        "eval": 1,
    },
    {
        "layers": 1,
        "values": "96",
        "parallel": 8,
        "threshold": 80,
        "epochs": 30,
        "eval": 0,
    },
    {
        "layers": 1,
        "values": "128",
        "parallel": 8,
        "threshold": 96,
        "epochs": 40,
        "eval": 1,
    },
    {
        "layers": 2,
        "values": "32 8",
        "parallel": 8,
        "threshold": 64,
        "epochs": 20,
        "eval": 1,
    },
    {
        "layers": 2,
        "values": "32 16",
        "parallel": 4,
        "threshold": 72,
        "epochs": 30,
        "eval": 0,
    },
    {
        "layers": 2,
        "values": "64 16",
        "parallel": 8,
        "threshold": 80,
        "epochs": 30,
        "eval": 1,
    },
    {
        "layers": 2,
        "values": "64 32",
        "parallel": 8,
        "threshold": 96,
        "epochs": 40,
        "eval": 1,
    },
    {
        "layers": 2,
        "values": "96 32",
        "parallel": 8,
        "threshold": 112,
        "epochs": 40,
        "eval": 0,
    },
    {
        "layers": 3,
        "values": "64 32 16",
        "parallel": 8,
        "threshold": 96,
        "epochs": 30,
        "eval": 1,
    },
    {
        "layers": 3,
        "values": "96 48 16",
        "parallel": 8,
        "threshold": 128,
        "epochs": 40,
        "eval": 1,
    },
]


def parse_dataset_shape(path: Path) -> tuple[str, str, str]:
    """Obtiene resolución y bits desde el nombre del dataset.

    Args:
        path: Ruta del `.bin` preprocesado.

    Returns:
        Tupla `(ancho, alto, bits)` como cadenas para variables Make.

    Raises:
        ValueError: Si el nombre no sigue el patrón MNIST esperado.

    Impacto:
        Garantiza que cada perfil se compile con dimensiones coherentes con el
        dataset usado.
    """

    match = re.search(r"mnist_(\d+)x(\d+)_(\d+)b", path.name)
    if not match:
        raise ValueError(f"Cannot parse dataset shape from {path.name}")
    return match.group(1), match.group(2), match.group(3)


def arch_tag(profile: dict[str, object]) -> str:
    """Construye una etiqueta compacta de arquitectura.

    Args:
        profile: Diccionario con capas, neuronas y paralelismo.

    Returns:
        Texto como `L2_32_8_P8`.

    Raises:
        KeyError: Si falta una clave obligatoria en el perfil.

    Impacto:
        Separa carpetas de resultados por complejidad de red y paralelismo HLS.
    """

    values = str(profile["values"]).replace(" ", "_")
    return f"L{profile['layers']}_{values}_P{profile['parallel']}"


def run_tag(profile: dict[str, object]) -> str:
    """Construye una etiqueta para threshold y época.

    Args:
        profile: Perfil de barrido.

    Returns:
        Texto como `thr64_ep20_ev1`.

    Raises:
        KeyError: Si el perfil no contiene threshold, epochs o eval.

    Impacto:
        Permite repetir arquitecturas con reglas de entrenamiento distintas sin
        sobrescribir resultados.
    """

    return (
        f"thr{profile['threshold']}_ep{profile['epochs']}"
        f"_ev{profile['eval']}"
    )


def main() -> int:
    """Planifica y ejecuta el barrido `sweep100`.

    Args:
        No recibe parámetros directos; usa argumentos CLI.

    Returns:
        0 si todas las simulaciones completan o si `--dry-run` solo imprime
        comandos. Devuelve el error del primer `make run` fallido.

    Raises:
        ValueError: Puede propagarse si algún dataset no permite inferir forma.

    Impacto:
        Produce una batería amplia para ordenar modelos por accuracy y coste
        relativo, útil para decidir qué arquitectura cabe mejor en FPGA.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--make", default="make")
    parser.add_argument("--data-dir", default="../mnist/data/processed")
    parser.add_argument("--results-dir", default="results/sweep100")
    parser.add_argument("--limit", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra_make_vars", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Cada dataset se combina con cada perfil para cubrir resoluciones,
    # cuantizaciones y tamaños de BNN.
    data_dir = Path(args.data_dir)
    datasets = sorted(data_dir.glob("*.bin"))
    if not datasets:
        print(f"ERROR: no .bin datasets found in {data_dir}", file=sys.stderr)
        return 1

    # `--limit` permite validar comandos sin lanzar las 100+ simulaciones.
    limit = 0
    if args.limit.strip():
        limit = int(args.limit)

    extra = [item for item in args.extra_make_vars if item != "--"]
    planned = [
        (dataset, profile)
        for dataset in datasets
        for profile in ARCH_PROFILES
    ]
    if limit > 0:
        planned = planned[:limit]

    print(f"Planned sweep100 experiments: {len(planned)}")
    print("Default samples: TRAIN_SAMPLES=35700 EVAL_SAMPLES=6300")
    if len(datasets) * len(ARCH_PROFILES) < 100 and limit == 0:
        print(
            "ERROR: generated sweep has fewer than 100 experiments",
            file=sys.stderr,
        )
        return 2

    # Imprimir antes de ejecutar ayuda a reanudar un barrido largo.
    for idx, (dataset, profile) in enumerate(planned, start=1):
        img_w, img_h, pix_bits = parse_dataset_shape(dataset)
        cmd = [
            args.make,
            "run",
            f"DATASET={dataset.as_posix()}",
            "GROUP=barrido_final",
            f"RESULTS_DIR={args.results_dir}",
            f"IMG_W={img_w}",
            f"IMG_H={img_h}",
            f"PIX_BITS={pix_bits}",
            "TRAIN_SAMPLES=35700",
            "EVAL_SAMPLES=6300",
            f"EPOCHS={profile['epochs']}",
            f"EVAL_EVERY_EPOCH={profile['eval']}",
            f"HIDDEN_LAYERS={profile['layers']}",
            f"HIDDEN_VALUES={profile['values']}",
            f"PARALLEL_NEURONS={profile['parallel']}",
            f"THRESHOLD={profile['threshold']}",
            f"RUN_TAG={run_tag(profile)}",
        ]
        cmd.extend(extra)

        print(
            f"[{idx:03d}/{len(planned):03d}] "
            f"{dataset.name} {arch_tag(profile)} {run_tag(profile)}",
            flush=True,
        )
        print("  " + " ".join(cmd), flush=True)
        if args.dry_run:
            continue

        completed = subprocess.run(cmd)
        if completed.returncode != 0:
            print(
                (
                    f"ERROR: sweep failed for {dataset.name} / "
                    f"{arch_tag(profile)}"
                ),
                file=sys.stderr,
            )
            return completed.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
