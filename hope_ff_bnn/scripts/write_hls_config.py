#!/usr/bin/env python3
"""Genera la configuración HLS usada por `make csim-run`.

Etapa del flujo:
    Preparación de Vitis/Vivado HLS 2024.2.

Entrada esperada:
    Parte FPGA, top function, flags C++ del modelo, testbench y fuentes.

Salida generada:
    Archivo `.cfg` con sección `[hls]` para `vitis-run --mode hls --csim`.

Relación con Forward-Forward:
    Fija qué kernel FF-BNN se simula y con qué arquitectura compilada.
"""


from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    """Escribe un archivo de configuración para Vitis HLS.

    Args:
        No recibe parámetros directos; usa argumentos CLI.

    Returns:
        0 cuando el archivo se escribe correctamente.

    Raises:
        OSError: Si la ruta de salida no puede crearse o escribirse.

    Impacto:
        Hace reproducible la validación HLS al capturar top, parte FPGA y flags
        de compilación en un archivo versionable/logueable.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--part", required=True)
    parser.add_argument("--top", required=True)
    parser.add_argument("--cflags", required=True)
    parser.add_argument("--tb", required=True)
    parser.add_argument("--files", nargs="+", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # `package.output.syn=false` mantiene csim ligero: se valida compatibilidad
    # C/HLS sin ejecutar síntesis completa en cada prueba rápida.
    lines = [
        f"part={args.part}",
        "",
        "[hls]",
        "flow_target=vivado",
        "package.output.format=ip_catalog",
        "package.output.syn=false",
        f"syn.top={args.top}",
        f"syn.cflags={args.cflags}",
        f"tb.cflags={args.cflags}",
        f"tb.file={args.tb}",
    ]

    # Cada fuente sintetizable se registra explícitamente para que Vitis no
    # dependa de globbing del shell ni del directorio de trabajo.
    lines.extend(f"syn.file={source}" for source in args.files)
    output.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"HLS_CONFIG={output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
