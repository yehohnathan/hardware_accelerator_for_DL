# ================================ LIBRERIAS ================================ #
from pathlib import Path


# ============================ IMPORTAR FUNCIONES =========================== #
# Intenta importar el empaquetador desde el paquete del proyecto.
try:
    from ff_bnn_stage.mnist.src.mnist_data_packer import (
        IMG_SIZE,
        build_variant_artifacts,
        labels_to_onehot,
        load_train_table,
        prepare_pixels_for_packing,
        resize_images,
        validate_dataset,
        validate_pixel_bits,
    )

# Recurre al modulo local cuando el script se ejecuta de forma directa.
except ModuleNotFoundError:
    from mnist_data_packer import (
        IMG_SIZE,
        build_variant_artifacts,
        labels_to_onehot,
        load_train_table,
        prepare_pixels_for_packing,
        resize_images,
        validate_dataset,
        validate_pixel_bits,
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

# Define la resolucion de salida usada para validar el redimensionado.
TARGET_WIDTH = 16

# Define la resolucion de salida usada para validar el redimensionado.
TARGET_HEIGHT = 16

# Define los modos de empaquetado que se desean generar y validar.
PACKING_BITS_LIST = [1, 4, 6]

# Define cuantas muestras se imprimiran durante la inspeccion final.
NUM_DIGITS_TO_SHOW = 1


# =================================== MAIN ================================== #
def main() -> None:
    """
    Ejecuta la generacion y lectura de los binarios configurados.

    Parametros
    ----------
    None
        No recibe parametros externos. Usa la configuracion global del script.

    Retorna
    -------
    None
        No retorna ningun valor. Solo ejecuta el flujo completo.
    """
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

    # Convierte las etiquetas una sola vez para reutilizarlas.
    y_onehot = labels_to_onehot(y)
    print("Conversion one-hot completada.")

    print("\n===== ETAPA 3: REDIMENSIONADO =====")

    # Reduce las imagenes a la resolucion elegida para esta validacion.
    X_resized = resize_images(
        X=X,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        source_width=IMG_SIZE,
        source_height=IMG_SIZE,
    )
    print(
        f"Redimensionado completado: {IMG_SIZE}x{IMG_SIZE} -> "
        f"{TARGET_WIDTH}x{TARGET_HEIGHT}"
    )

    # Reserva el diccionario donde se guardaran los artefactos generados.
    artifacts_by_bits: dict[int, dict[str, object]] = {}

    for stage_idx, pixel_bits in enumerate(PACKING_BITS_LIST, start=4):
        # Valida cada modo antes de lanzar el empaquetado.
        validate_pixel_bits(pixel_bits)

        print(
            f"\n===== ETAPA {stage_idx}: "
            f"GENERACION DEL BINARIO {pixel_bits}b ====="
        )

        # Prepara los pixeles con la cuantizacion elegida.
        pixel_values = prepare_pixels_for_packing(
            X=X_resized,
            pixel_bits=pixel_bits,
            threshold=BIN_THRESHOLD,
        )

        if pixel_bits == 1:
            print("Binarizacion 1b completada.")
        else:
            print(f"Cuantizacion {pixel_bits}b completada.")

        # Construye y guarda el binario correspondiente al modo actual.
        artifacts_by_bits[pixel_bits] = build_variant_artifacts(
            output_dir=output_dir,
            y_onehot=y_onehot,
            pixel_values=pixel_values,
            pixel_bits=pixel_bits,
            image_width=TARGET_WIDTH,
            image_height=TARGET_HEIGHT,
            source_width=IMG_SIZE,
            source_height=IMG_SIZE,
        )

    read_stage_start = 4 + len(PACKING_BITS_LIST)

    for offset, pixel_bits in enumerate(PACKING_BITS_LIST):
        stage_idx = read_stage_start + offset
        artifacts = artifacts_by_bits[pixel_bits]

        print(
            f"\n===== ETAPA {stage_idx}: "
            f"RELECTURA E INSPECCION {pixel_bits}b ====="
        )

        # Recarga el binario desde disco para comprobar su cabecera.
        metadata, packed_data = load_packed_ff_binary_dataset(
            file_path=artifacts["packed_path"],
        )

        # Muestra el resumen del archivo y las primeras imagenes.
        print_dataset_summary(
            metadata=metadata,
            packed_data=packed_data,
            file_path=artifacts["packed_path"],
        )
        inspect_first_digits(
            packed_data=packed_data,
            metadata=metadata,
            num_digits=NUM_DIGITS_TO_SHOW,
            show_words=False,
        )

    final_stage = read_stage_start + len(PACKING_BITS_LIST)
    print(f"\n===== ETAPA {final_stage}: VALIDACION FINAL =====")
    print("Proceso completado correctamente.")


if __name__ == "__main__":
    main()
