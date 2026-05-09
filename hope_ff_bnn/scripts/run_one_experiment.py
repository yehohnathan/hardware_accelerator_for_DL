#!/usr/bin/env python3
"""Ejecuta una simulación C++ individual de la FF-BNN.

Etapa del flujo:
    Ejecución controlada del testbench `train_tb`.

Entrada esperada:
    Ruta del ejecutable compilado, dataset, hiperparámetros y arquitectura ya
    seleccionados por Make.

Salida generada:
    `log.txt` con stdout/stderr del testbench y `status.txt` con OK/FAIL.

Relación con Forward-Forward:
    Conserva la evidencia experimental de una corrida de entrenamiento:
    goodness positiva/negativa, accuracy y cambios de pesos/bias.
"""


from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Ejecuta el testbench y persiste su resultado.

    Args:
        No recibe parámetros directos; consume argumentos CLI definidos por
        `argparse`.

    Returns:
        Código de salida del ejecutable `train_tb`.

    Raises:
        OSError: Solo se captura al reimprimir el log; errores al crear carpeta
        o lanzar el proceso se dejan fallar porque indican un problema local.

    Impacto:
        Separa la ejecución del binario C++ de la lógica del Makefile y asegura
        que cada simulación deje un log auditable para el TFG.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-samples", required=True)
    parser.add_argument("--eval-samples", required=True)
    parser.add_argument("--epochs", required=True)
    parser.add_argument("--eval-every-epoch", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-layers", required=True)
    parser.add_argument("--hidden-values", required=True)
    parser.add_argument("--parallel-neurons", required=True)
    parser.add_argument("--seed", default="0x1234")
    args = parser.parse_args()

    # La carpeta se crea antes de lanzar el binario para conservar evidencia
    # incluso si falla lectura de dataset o validación de configuración.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log.txt"
    status_path = output_dir / "status.txt"

    # Se usa una lista de argumentos, no un string de shell, para evitar
    # diferencias de quoting entre PowerShell, Git Bash/MSYS2 y Linux.
    cmd = [
        args.exe,
        "--dataset",
        args.dataset,
        "--train-samples",
        args.train_samples,
        "--eval-samples",
        args.eval_samples,
        "--epochs",
        args.epochs,
        "--eval-every-epoch",
        args.eval_every_epoch,
        "--run-name",
        args.run_name,
        "--group",
        args.group,
        "--output-dir",
        str(output_dir),
        "--hidden-layers",
        args.hidden_layers,
        "--hidden-values",
        args.hidden_values,
        "--parallel-neurons",
        args.parallel_neurons,
        "--seed",
        args.seed,
    ]

    print(f"RUN {' '.join(cmd)}")
    print(f"LOG {log_path}")
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        completed = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    # El resumen global usa este archivo para clasificar corridas incompletas.
    status = "OK" if completed.returncode == 0 else "FAIL"
    status_path.write_text(status + "\n", encoding="utf-8")

    try:
        print(log_path.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        print(f"WARNING: could not echo log: {exc}", file=sys.stderr)

    print(f"RESULT_DIR={output_dir}")
    print(f"STATUS={status}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
