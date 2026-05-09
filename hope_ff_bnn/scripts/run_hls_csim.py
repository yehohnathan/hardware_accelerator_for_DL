#!/usr/bin/env python3
"""Ejecuta C simulation de Vitis HLS para los kernels FF-BNN.

Etapa del flujo:
    Validación HLS previa a síntesis.

Entrada esperada:
    Ejecutable `vitis-run`, archivo de configuración HLS, dataset y parámetros
    de entrenamiento/evaluación.

Salida generada:
    Log reproducible con configuración, entorno y salida completa de csim.

Relación con Forward-Forward:
    Verifica que `ff_train_kernel` e `ff_infer_kernel` pueden ejecutarse en la
    ruta de Vitis/Vivado 2024.2 con la arquitectura seleccionada.
"""


from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def resolve_existing(path_text: str) -> str:
    """Normaliza una ruta si existe localmente.

    Args:
        path_text: Ruta recibida desde Make.

    Returns:
        Ruta absoluta si existe; texto original si Vitis debe resolverla.

    Raises:
        No genera excepciones esperadas; `Path.exists()` maneja rutas inválidas
        de forma segura en este contexto.

    Impacto:
        Evita que csim falle por cambios de directorio de trabajo al localizar
        el dataset MNIST.
    """

    path = Path(path_text)
    if path.exists():
        return str(path.resolve())
    return path_text


def run_command(
    vitis_run: str,
    config: str,
    work_dir: str,
    env: dict[str, str],
    log,
) -> int:
    """Ejecuta `vitis-run --mode hls --csim`.

    Args:
        vitis_run: Ruta o nombre del lanzador Vitis.
        config: Archivo HLS generado por `write_hls_config.py`.
        work_dir: Directorio de trabajo HLS.
        env: Variables de entorno para el testbench de csim.
        log: Archivo abierto donde se redirige stdout/stderr.

    Returns:
        Código de salida devuelto por Vitis.

    Raises:
        subprocess.SubprocessError: Puede propagarse si el proceso no puede
        crearse.

    Impacto:
        Centraliza la ejecución de csim y soporta tanto lanzadores `.bat/.cmd`
        de Windows como ejecutables directos.
    """

    cmd = [
        vitis_run,
        "--mode",
        "hls",
        "--csim",
        "--config",
        config,
        "--work_dir",
        work_dir,
    ]

    log.write("--- vitis-run --mode hls --csim output ---\n")
    log.flush()

    # Los wrappers `.bat/.cmd` necesitan `shell=True`; los ejecutables directos
    # no, lo que reduce problemas de quoting en Linux/MSYS2.
    suffix = Path(vitis_run).suffix.lower()
    if suffix in {".bat", ".cmd"}:
        command_text = subprocess.list2cmdline(cmd)
        completed = subprocess.run(
            command_text,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            shell=True,
        )
    else:
        completed = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    return completed.returncode


def main() -> int:
    """Prepara entorno, registra metadatos y ejecuta HLS csim.

    Args:
        No recibe parámetros directos; usa CLI.

    Returns:
        Código de salida de Vitis o 127 si el ejecutable indicado no existe.

    Raises:
        OSError: Puede propagarse si no se puede crear el log o work_dir.

    Impacto:
        Conecta el flujo Make con Vitis HLS, dejando trazabilidad de dataset,
        arquitectura, parte FPGA y parámetros Forward-Forward.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--vitis-run", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-samples", required=True)
    parser.add_argument("--eval-samples", required=True)
    parser.add_argument("--epochs", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--hls-part", required=True)
    parser.add_argument("--img", required=True)
    parser.add_argument("--pixel-bits", required=True)
    parser.add_argument("--hidden-layers", required=True)
    parser.add_argument("--hidden-values", required=True)
    parser.add_argument("--parallel-neurons", required=True)
    parser.add_argument("--threshold", required=True)
    args = parser.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    # El testbench HLS lee estos valores desde entorno para mantener el archivo
    # de configuración enfocado en macros de compilación.
    dataset = resolve_existing(args.dataset)
    env = os.environ.copy()
    env["FF_DATASET_BIN"] = dataset
    env["FF_TRAIN_SAMPLES"] = args.train_samples
    env["FF_EVAL_SAMPLES"] = args.eval_samples
    env["FF_EPOCHS"] = args.epochs
    env["FF_SEED"] = args.seed

    # La cabecera del log permite reproducir la csim sin inspeccionar Makefile.
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"timestamp={args.timestamp}\n")
        log.write(f"variant={args.variant}\n")
        log.write(f"vitis_run={args.vitis_run}\n")
        log.write(f"hls_part={args.hls_part}\n")
        log.write(f"hls_cfg={args.config}\n")
        log.write(f"hls_work_dir={args.work_dir}\n")
        log.write(f"dataset={dataset}\n")
        log.write(f"img={args.img}\n")
        log.write(f"pixel_bits={args.pixel_bits}\n")
        log.write(f"hidden_layers={args.hidden_layers}\n")
        log.write(f"hidden_values={args.hidden_values}\n")
        log.write(f"parallel_neurons={args.parallel_neurons}\n")
        log.write(f"threshold={args.threshold}\n")
        log.write(f"train_samples={args.train_samples}\n")
        log.write(f"eval_samples={args.eval_samples}\n")
        log.write(f"epochs={args.epochs}\n")
        log.write(f"seed={args.seed}\n")

        vitis_path = Path(args.vitis_run)
        if not vitis_path.exists() and vitis_path.name != args.vitis_run:
            log.write(f"ERROR: Vitis executable not found: {args.vitis_run}\n")
            code = 127
        else:
            code = run_command(
                args.vitis_run,
                args.config,
                args.work_dir,
                env,
                log,
            )

    text = log_path.read_text(encoding="utf-8", errors="replace")
    print(text)
    print(f"CSIM_LOG_SAVED={log_path.as_posix()}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
