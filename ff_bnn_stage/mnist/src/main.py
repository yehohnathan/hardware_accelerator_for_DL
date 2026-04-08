# ================================ LIBRERÍAS ================================ #
from pathlib import Path

# ============================ IMPORTAR FUNCIONES =========================== #
# Importa las funciones que generan y empaquetan las variantes del dataset.
from ff_bnn_stage.mnist.src.mnist_data_packer import (
    binarize_pixels_1bit,
    build_variant_artifacts,
    labels_to_onehot,
    load_train_table,
    quantize_pixels_4bit,
    validate_dataset,
)

# Importa las funciones que vuelven a leer e inspeccionar los binarios.
from read_bin import (
    inspect_first_digits,
    load_packed_ff_binary_dataset,
    print_dataset_summary,
)

# ============================ VARIABLES GLOBALES =========================== #
# Define el umbral fijo usado durante la binarizacion a 1 bit.
BIN_THRESHOLD = 127


# =================================== MAIN ================================== #
def main() -> None:
    """Ejecuta la generacion y lectura de los binarios 1b y 4b."""
    # Obtiene el directorio real del script para construir rutas estables.
    script_dir = Path(__file__).resolve().parent

    # Define la ruta del CSV original y el directorio de salida.
    raw_csv_path = script_dir.parent / "data" / "raw" / "train.csv"
    output_dir = script_dir.parent / "data" / "processed"

    print("===== ETAPA 1: CARGA DEL CSV =====")

    # Carga las imagenes y valida que el dataset tenga el formato esperado.
    X, y = load_train_table(raw_csv_path)
    validate_dataset(X, y)

    print("\n===== ETAPA 2: ETIQUETAS ONE-HOT =====")

    # Convierte las etiquetas una sola vez para reutilizarlas en ambos modos.
    y_onehot = labels_to_onehot(y)
    print("Conversion one-hot completada.")

    print("\n===== ETAPA 3: CUANTIZACION 1b Y 4b =====")

    # Genera las dos vistas del mismo dataset: binaria y cuantizada.
    pixels_1bit = binarize_pixels_1bit(X, threshold=BIN_THRESHOLD)
    pixels_4bit = quantize_pixels_4bit(X)
    print("Binarizacion 1b completada.")
    print("Cuantizacion 4b completada.")

    print("\n===== ETAPA 4: GENERACION DEL BINARIO 1b =====")

    # Construye y guarda el binario correspondiente al modo 1b.
    artifacts_1b = build_variant_artifacts(
        output_dir=output_dir,
        y_onehot=y_onehot,
        pixel_values=pixels_1bit,
        pixel_bits=1,
    )

    print("\n===== ETAPA 5: GENERACION DEL BINARIO 4b =====")

    # Construye y guarda el binario correspondiente al modo 4b.
    artifacts_4b = build_variant_artifacts(
        output_dir=output_dir,
        y_onehot=y_onehot,
        pixel_values=pixels_4bit,
        pixel_bits=4,
    )

    print("\n===== ETAPA 6: RELECTURA E INSPECCION 1b =====")

    # Recarga el binario 1b desde disco para comprobar su lectura.
    packed_1b = load_packed_ff_binary_dataset(
        file_path=artifacts_1b["packed_path"],
        packed_words=artifacts_1b["layout"]["words_per_sample"],
    )

    # Muestra un resumen del archivo y las primeras dos imagenes.
    print_dataset_summary(
        packed_data=packed_1b,
        pixel_bits=1,
        file_path=artifacts_1b["packed_path"],
    )
    inspect_first_digits(
        packed_data=packed_1b,
        pixel_bits=1,
        num_digits=2,
        show_words=False,
    )

    print("\n===== ETAPA 7: RELECTURA E INSPECCION 4b =====")

    # Recarga el binario 4b desde disco para comprobar su lectura.
    packed_4b = load_packed_ff_binary_dataset(
        file_path=artifacts_4b["packed_path"],
        packed_words=artifacts_4b["layout"]["words_per_sample"],
    )

    # Muestra un resumen del archivo y las primeras dos imagenes.
    print_dataset_summary(
        packed_data=packed_4b,
        pixel_bits=4,
        file_path=artifacts_4b["packed_path"],
    )
    inspect_first_digits(
        packed_data=packed_4b,
        pixel_bits=4,
        num_digits=2,
        show_words=False,
    )

    print("\n===== ETAPA 8: VALIDACION FINAL =====")
    print("Proceso completado correctamente.")


if __name__ == "__main__":
    main()
