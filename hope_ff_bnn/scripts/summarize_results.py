#!/usr/bin/env python3
"""Resume resultados experimentales de la BNN Forward-Forward.

Etapa del flujo:
    Análisis posterior a simulaciones.

Entrada esperada:
    Directorio de resultados con subcarpetas que contienen `summary.md`,
    `metrics.csv` y opcionalmente `status.txt`.

Salida generada:
    Tabla Markdown global, normalmente `results/summary_all.md`.

Relación con Forward-Forward:
    Reúne accuracy, goodness positiva/negativa, gap y complejidad para comparar
    datasets y arquitecturas FF-BNN.
"""


from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


HEADERS = [
    "ID",
    "Dataset",
    "Resolucion",
    "Bits",
    "Grupo",
    "Arquitectura",
    "Hidden layers",
    "Hidden values",
    "Parallel neurons",
    "Threshold",
    "Train samples",
    "Eval samples",
    "Epochs",
    "Eval cada epoca",
    "Accuracy",
    "Goodness positiva",
    "Goodness negativa",
    "Goodness gap",
    "Complexity score",
    "Params",
    "Weight bits est",
    "BRAM18 est",
    "Ops/sample est",
    "Tiempo",
    "Estado",
]

GROUP_ORDER = {
    "smoke": 0,
    "debug": 1,
    "comparacion": 2,
    "barrido_final": 3,
}


def parse_dataset_name(name: str) -> tuple[str, str]:
    """Extrae resolución y bits desde el nombre lógico del dataset.

    Args:
        name: Nombre registrado en `summary.md`.

    Returns:
        Tupla `(resolucion, bits)`. Usa `unknown` cuando no puede inferirse.

    Raises:
        No lanza excepciones; los nombres no reconocidos se degradan a
        `unknown`.

    Impacto:
        Mantiene la tabla global completa incluso si una corrida produjo un
        resumen parcial.
    """

    match = re.search(r"mnist_(\d+x\d+)_(\d+)b", name)
    if not match:
        return "unknown", "unknown"
    return match.group(1), f"{match.group(2)}b"


def read_summary(path: Path) -> dict[str, str]:
    """Lee pares `clave: valor` desde un resumen de experimento.

    Args:
        path: Ruta a `summary.md`.

    Returns:
        Diccionario normalizado por claves en minúscula con guiones bajos.

    Raises:
        No lanza errores por archivo ausente; devuelve diccionario vacío.

    Impacto:
        Convierte la salida del testbench C++ en datos comparables por script.
    """

    data: dict[str, str] = {}
    if not path.exists():
        return data
    text = path.read_text(encoding="utf-8", errors="ignore")
    for raw_line in text.splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        data[key.strip().lower().replace(" ", "_")] = value.strip()
    return data


def read_last_metrics(path: Path) -> dict[str, str]:
    """Lee la última fila de métricas CSV de una corrida.

    Args:
        path: Ruta a `metrics.csv`.

    Returns:
        Última fila como diccionario, o vacío si el archivo no existe.

    Raises:
        csv.Error: Puede propagarse si el CSV está corrupto.

    Impacto:
        Recupera el estado final de entrenamiento cuando `summary.md` no trae
        todas las métricas.
    """

    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def md_escape(value: object) -> str:
    """Escapa valores para insertarlos en una tabla Markdown.

    Args:
        value: Celda a renderizar.

    Returns:
        Texto con barras verticales escapadas.

    Raises:
        No lanza excepciones esperadas.

    Impacto:
        Evita que rutas o arquitecturas con `|` rompan el formato Obsidian.
    """

    return str(value).replace("|", "\\|")


def format_accuracy(value: str) -> str:
    """Formatea accuracy como porcentaje legible.

    Args:
        value: Valor crudo, usualmente en rango 0..1.

    Returns:
        Cadena porcentual o el valor original si no es numérico.

    Raises:
        No propaga `ValueError`; conserva el texto original.

    Impacto:
        Hace comparable la métrica principal de clasificación MNIST.
    """

    if value == "":
        return ""
    try:
        parsed = float(value)
    except ValueError:
        return value
    if parsed <= 1.0:
        return f"{parsed * 100.0:.2f}%"
    return f"{parsed:.2f}%"


def format_time(value: str) -> str:
    """Formatea segundos de ejecución.

    Args:
        value: Tiempo en segundos como texto.

    Returns:
        Tiempo con sufijo `s`, o el texto original si no es numérico.

    Raises:
        No propaga `ValueError`.

    Impacto:
        Permite comparar coste temporal de arquitecturas e hiperparámetros.
    """

    if value == "":
        return ""
    try:
        return f"{float(value):.2f}s"
    except ValueError:
        return value


def parse_float(value: str) -> float:
    """Convierte texto a flotante tolerando datos incompletos.

    Args:
        value: Texto numérico.

    Returns:
        Valor flotante o 0.0 ante texto vacío/no numérico.

    Raises:
        No propaga `ValueError`.

    Impacto:
        Permite ordenar corridas aunque algunas hayan fallado parcialmente.
    """

    if value == "":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def parse_arch_dims(architecture: str) -> list[int]:
    """Extrae dimensiones numéricas desde una cadena de arquitectura.

    Args:
        architecture: Texto como `410 -> 32 -> 8`.

    Returns:
        Lista de dimensiones `[410, 32, 8]`.

    Raises:
        ValueError: Puede propagarse si un token numérico no cabe en `int`.

    Impacto:
        Alimenta la estimación de parámetros y coste relativo de FPGA.
    """

    values = []
    for item in re.findall(r"\d+", architecture):
        values.append(int(item))
    return values


def estimate_complexity(architecture: str) -> dict[str, str]:
    """Estima complejidad de una red densa BNN.

    Args:
        architecture: Cadena con dimensiones de capas.

    Returns:
        Diccionario con parámetros, bits estimados, BRAM18 y operaciones.

    Raises:
        No lanza errores esperados cuando la arquitectura está vacía.

    Impacto:
        Da una métrica aproximada para ordenar accuracy contra coste de
        almacenamiento y cómputo en FPGA.
    """

    dims = parse_arch_dims(architecture)
    if len(dims) < 2:
        return {
            "Complexity score": "",
            "Params": "",
            "Weight bits est": "",
            "BRAM18 est": "",
            "Ops/sample est": "",
        }

    weight_params = sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))
    bias_params = sum(dims[1:])
    params = weight_params + bias_params
    weight_bits = params * 16
    bram18 = int(math.ceil(weight_bits / 18432.0))
    ops = weight_params
    return {
        "Complexity score": str(weight_params),
        "Params": str(params),
        "Weight bits est": str(weight_bits),
        "BRAM18 est": str(bram18),
        "Ops/sample est": str(ops),
    }


def row_sort_key(row: dict[str, str], mode: str) -> tuple[object, ...]:
    """Calcula la clave de ordenamiento de una fila de resultados.

    Args:
        row: Fila normalizada de la tabla global.
        mode: `path` o `accuracy-complexity`.

    Returns:
        Tupla usada por `list.sort`.

    Raises:
        ValueError: Puede propagarse si el campo de complejidad contiene texto
        no convertible y no está vacío.

    Impacto:
        Permite priorizar modelos con alta accuracy y menor coste estimado.
    """

    accuracy = parse_float(row.get("_accuracy_raw", "0"))
    complexity = int(row.get("Complexity score") or "999999999")
    if mode == "accuracy-complexity":
        return (
            -accuracy,
            complexity,
            row.get("Dataset", ""),
            row.get("Arquitectura", ""),
        )
    return (row.get("Dataset", ""), GROUP_ORDER.get(row.get("Grupo", ""), 99),
            complexity, row.get("Arquitectura", ""))


def build_rows(results_dir: Path) -> list[dict[str, str]]:
    """Construye filas de resumen desde carpetas de experimentos.

    Args:
        results_dir: Directorio raíz de resultados.

    Returns:
        Lista de filas listas para renderizar en Markdown.

    Raises:
        OSError: Puede propagarse si hay archivos ilegibles.

    Impacto:
        Une salidas de testbench (`summary.md` y `metrics.csv`) en una vista
        única para análisis académico.
    """

    rows: list[dict[str, str]] = []

    if not results_dir.exists():
        return rows

    # Cada `summary.md` identifica una corrida individual de entrenamiento.
    for summary_path in sorted(results_dir.rglob("summary.md")):
        run_dir = summary_path.parent
        summary = read_summary(summary_path)
        metrics = read_last_metrics(run_dir / "metrics.csv")
        if not summary and not metrics:
            continue

        dataset = summary.get("dataset", run_dir.parent.parent.name)
        resolution, bits = parse_dataset_name(dataset)
        status = summary.get("status", "")
        if not status:
            status_path = run_dir / "status.txt"
            status = (
                status_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                ).strip()
                if status_path.exists()
                else "UNKNOWN"
            )

        # La complejidad se infiere del texto de arquitectura para no depender
        # de archivos auxiliares generados por una versión concreta del TB.
        architecture = summary.get("architecture", "")
        complexity = estimate_complexity(architecture)
        accuracy_raw = summary.get("accuracy", metrics.get("accuracy", ""))
        row = {
            "ID": "0",
            "Dataset": dataset,
            "Resolucion": summary.get("resolution", resolution),
            "Bits": summary.get("bits", bits),
            "Grupo": summary.get("group", ""),
            "Arquitectura": architecture,
            "Hidden layers": summary.get("hidden_layers", ""),
            "Hidden values": summary.get("hidden_values", ""),
            "Parallel neurons": summary.get("parallel_neurons", ""),
            "Threshold": summary.get("threshold", ""),
            "Train samples": summary.get(
                "train_samples", metrics.get("train_samples", "")
            ),
            "Eval samples": summary.get(
                "eval_samples", metrics.get("eval_samples", "")
            ),
            "Epochs": summary.get("epochs", metrics.get("epoch", "")),
            "Eval cada epoca": summary.get("eval_every_epoch", ""),
            "Accuracy": format_accuracy(accuracy_raw),
            "Goodness positiva": summary.get(
                "avg_g_pos", metrics.get("avg_g_pos", "")
            ),
            "Goodness negativa": summary.get(
                "avg_g_neg", metrics.get("avg_g_neg", "")
            ),
            "Goodness gap": summary.get("avg_gap", metrics.get("avg_gap", "")),
            "Tiempo": format_time(
                summary.get("elapsed_sec", metrics.get("elapsed_sec", ""))
            ),
            "Estado": status,
            "_accuracy_raw": str(parse_float(accuracy_raw)),
        }
        row.update(complexity)
        rows.append(row)

    return rows


def render_markdown(rows: list[dict[str, str]]) -> str:
    """Renderiza las filas como tabla Markdown compatible con Obsidian.

    Args:
        rows: Filas normalizadas.

    Returns:
        Documento Markdown completo.

    Raises:
        No lanza excepciones esperadas.

    Impacto:
        Produce la tabla final para comparar accuracy, goodness y coste.
    """

    lines = [
        "# Resumen global de experimentos",
        "",
        f"Total de experimentos encontrados: {len(rows)}",
        "",
        "| " + " | ".join(HEADERS) + " |",
        "| " + " | ".join(["---"] * len(HEADERS)) + " |",
    ]
    for idx, row in enumerate(rows, start=1):
        row["ID"] = str(idx)
        line = "| " + " | ".join(
            md_escape(row.get(h, "")) for h in HEADERS
        ) + " |"
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Punto de entrada del generador de resumen global.

    Args:
        No recibe parámetros directos; usa argumentos CLI.

    Returns:
        0 si el Markdown se escribe correctamente.

    Raises:
        OSError: Puede propagarse si no se puede leer o escribir resultados.

    Impacto:
        Completa el flujo experimental al convertir corridas FF-BNN en una
        tabla auditable.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", nargs="?", default="results")
    parser.add_argument(
        "--sort",
        choices=["path", "accuracy-complexity"],
        default="path",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = build_rows(results_dir)
    rows.sort(key=lambda row: row_sort_key(row, args.sort))
    output = render_markdown(rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output)
        if args.output
        else results_dir / "summary_all.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
