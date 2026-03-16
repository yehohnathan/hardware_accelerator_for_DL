# =============================== Librerías ================================= #
from pathlib import Path                     # Manejo limpio de rutas
from mnist_binarize_onehot_pack import (     # Funciones del preprocesamiento
    FF_INPUT_BITS,
    FF_PACK_WORDS,
    load_train_table,
    validate_dataset,
    binarize_pixels,
    labels_to_onehot,
    concatenate_ff_input,
    pack_bits_matrix,
    verify_bit_packing,
    save_packed_ff_dataset,
    save_numpy_debug,
)
from read_bin import (                       # Funciones de lectura y prueba
    test_binary_dataset,
    print_first_words,
    inspect_sample_structure,
)


# =========================== Variables globales ============================ #
INPUT_CSV = "data/raw/train.csv"
# Ruta del archivo CSV original de MNIST

OUTPUT_BIN = "data/processed/mnist_ff_input_packed.bin"
# Ruta del archivo binario FF corregido

DEBUG_PREFIX = "data/processed/debug_mnist_ff"
# Prefijo base para guardar archivos .npy de depuración

BIN_THRESHOLD = 127
# Umbral de binarización usado para convertir píxeles a 0/1

ONEHOT_POSITION = "prefix"
# Posición del one-hot en la entrada FF
# Para este proyecto debe ser "prefix" para obtener:
# [label_onehot || imagen_binaria]


# =============================== Función main ============================== #
def main() -> None:
    """
    Ejecuta el flujo completo de generación y validación del binario FF.
    """
    input_csv_path = Path(INPUT_CSV)
    # Convierte la ruta de entrada a objeto Path

    output_bin_path = Path(OUTPUT_BIN)
    # Convierte la ruta de salida a objeto Path

    print("===== ETAPA 1: CARGA DEL CSV =====")
    # Indica el inicio de la carga del dataset original

    X, y = load_train_table(str(input_csv_path))
    # Carga imágenes y etiquetas desde el CSV

    validate_dataset(X, y)
    # Verifica forma y rangos del dataset

    print("\n===== ETAPA 2: BINARIZACION =====")
    # Indica el inicio del proceso de binarización

    X_bin = binarize_pixels(
        X=X,
        threshold=BIN_THRESHOLD,
        show_process=False,
    )
    # Convierte todos los píxeles a binario usando el umbral definido

    print("Binarización completada.")
    # Informa que la binarización terminó

    print("\n===== ETAPA 3: ONE-HOT =====")
    # Indica el inicio de la generación de etiquetas one-hot

    y_onehot = labels_to_onehot(y)
    # Convierte las etiquetas escalares a formato one-hot

    print("Conversión one-hot completada.")
    # Informa que la codificación one-hot terminó

    print("\n===== ETAPA 4: CONCATENACION FF =====")
    # Indica el inicio de la construcción de la entrada FF

    X_ff = concatenate_ff_input(
        X_bin=X_bin,
        y_onehot=y_onehot,
        onehot_position=ONEHOT_POSITION,
    )
    # Construye la entrada FF con estructura [one-hot || imagen]

    print("Concatenación FF completada.")
    # Informa que la entrada FF fue construida

    print("Forma de X_ff:", X_ff.shape)
    # Imprime la forma de la matriz FF

    print("\n===== ETAPA 5: EMPAQUETADO =====")
    # Indica el inicio del empaquetado en uint32

    X_ff_packed = pack_bits_matrix(
        X_bin=X_ff,
        total_bits=FF_INPUT_BITS,
        pack_bits=32,
    )
    # Empaqueta cada muestra FF en 25 palabras uint32

    print("Empaquetado completado.")
    # Informa que el empaquetado terminó

    print("Forma de X_ff_packed:", X_ff_packed.shape)
    # Imprime la forma de la matriz empaquetada

    print("\n===== ETAPA 6: VERIFICACION PACK/UNPACK =====")
    # Indica el inicio de la verificación de reversibilidad

    verify_bit_packing(
        X_original=X_ff,
        X_packed=X_ff_packed,
        total_bits=FF_INPUT_BITS,
        pack_bits=32,
        num_samples=10,
    )
    # Verifica que pack y unpack conserven la información

    print("\n===== ETAPA 7: GUARDADO DEL BINARIO FF =====")
    # Indica el inicio del guardado del binario final

    save_packed_ff_dataset(
        output_path=str(output_bin_path),
        X_ff_packed=X_ff_packed,
    )
    # Guarda el nuevo binario sin byte extra de label

    print("\n===== ETAPA 8: GUARDADO DE ARCHIVOS DEBUG =====")
    # Indica el inicio del guardado de archivos auxiliares

    save_numpy_debug(
        prefix=DEBUG_PREFIX,
        X=X,
        y=y,
        X_bin=X_bin,
        y_onehot=y_onehot,
        X_ff=X_ff,
        X_ff_packed=X_ff_packed,
    )
    # Guarda matrices intermedias en formato .npy para depuración

    print("\n===== ETAPA 9: RELECTURA DEL BINARIO =====")
    # Indica el inicio de la prueba de lectura del binario generado

    packed, X_reconstructed = test_binary_dataset(
        file_path=str(output_bin_path),
        packed_words=FF_PACK_WORDS,
        total_bits=FF_INPUT_BITS,
    )
    # Vuelve a leer el binario recién creado y reconstruye sus bits

    print("\n===== ETAPA 10: INSPECCION DE LA PRIMERA MUESTRA =====")
    # Indica el inicio de la inspección estructural de la primera muestra

    print_first_words(
        packed_data=packed,
        sample_idx=0,
        num_words=FF_PACK_WORDS,
    )
    # Imprime las primeras 25 palabras de la muestra 0

    inspect_sample_structure(
        packed_data=packed,
        sample_idx=0,
    )
    # Imprime label, imagen y padding de la muestra 0

    print("\n===== ETAPA 11: VALIDACION FINAL =====")
    # Indica el cierre del flujo de validación

    print("Dimensión reconstruida:", X_reconstructed.shape)
    # Imprime la dimensión del arreglo reconstruido

    print("Proceso completado correctamente.")
    # Mensaje final de éxito


# ============================= Punto de entrada ============================ #
if __name__ == "__main__":
    # Ejecuta main solo si este archivo se lanza directamente
    main()
