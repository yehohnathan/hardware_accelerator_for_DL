"""
Este script:
1. Carga el dataset MNIST desde un archivo CSV/TXT.
2. Valida que tenga el formato esperado.
3. Binariza los píxeles.
4. Convierte las etiquetas a one-hot.
5. Concatena imagen binaria + etiqueta one-hot.
6. Empaqueta bits en palabras de 32 bits.
7. Guarda archivos binarios listos para FPGA.
8. Guarda archivos .npy para depuración.
"""
# =============================== Librerías ================================= #
import numpy as np                  # Cálculo numérico eficiente

# =========================== Importar Funciones ============================ #
import mnist_binarize_onehot_pack as mnist_utils


# =========================== Variables globales ============================ #
# Tamaño lateral de la imagen MNIST: 28x28
IMG_SIZE = 28

# Total de píxeles por imagen
NUM_PIXELS = IMG_SIZE * IMG_SIZE

# Número de clases en MNIST: dígitos del 0 al 9
NUM_CLASSES = 10

# Entrada total para FF: 784 bits de imagen + 10 bits one-hot
FF_INPUT_BITS = NUM_PIXELS + NUM_CLASSES

# Cantidad de bits por palabra de empaquetado
PACK_BITS = 32

# Número de palabras de 32 bits necesarias para 784 bits
PIX_PACK_WORDS = (NUM_PIXELS + PACK_BITS - 1) // PACK_BITS

# Número de palabras de 32 bits necesarias para 794 bits
FF_PACK_WORDS = (FF_INPUT_BITS + PACK_BITS - 1) // PACK_BITS


# ================================== Main =================================== #
if __name__ == "__main__":
    input_path = "data\\raw\\train.csv"
    # Ruta del archivo de entrada

    # Carga el dataset original desde archivo
    print("\n# 1. Cargando dataset MNIST #")
    X, y = mnist_utils.load_train_table(input_path)

    print(f"Dataset cargado: {X.shape[0]} muestras.")
    print(f"Etiquetas cargadas: {y.shape[0]} muestras.")
    print(f"{X.shape[1]} píxeles por imagen.")

    # Revisa estructura, dimensiones y rangos
    print("\n# 2. Revisando estructura, dimensiones y rangos #")
    mnist_utils.validate_dataset(X, y)

    # Convierte los píxeles de 0-255 a bits 0/1
    print("\n# 3. Binarizando píxeles #")
    threshold = 127     # Umbral de binarización
    X_bin = mnist_utils.binarize_pixels(X, threshold=threshold,
                                        show_process=False, sample_idx=0,)

    # Convierte cada etiqueta escalar en vector one-hot
    print("\n# 4. Convirtiendo etiquetas a one-hot #")
    y_onehot = mnist_utils.labels_to_onehot(y, num_classes=NUM_CLASSES)
    print(f"Ejemplo de etiqueta one-hot: {y_onehot[42]}"
          f" (para etiqueta original {y[42]})")

    # Concatena imagen binaria + etiqueta one-hot
    print("\n# 5. Concatenando imagen binaria + etiqueta one-hot #")
    X_ff = mnist_utils.concatenate_ff_input(X_bin, y_onehot,
                                            onehot_position="prefix",)
    print(f"y_onehot[42]:   {y_onehot[42]} | y[42]: {y[42]}")
    print(f"EX_ff[42][:20]: {X_ff[42][:20]}")

    # Empaqueta solo los píxeles binarios
    print("\n# 6. Empaquetando bits en palabras de 32 bits #")
    X_pix_packed = mnist_utils.pack_bits_matrix(X_bin, total_bits=NUM_PIXELS,
                                                pack_bits=PACK_BITS,)

    # Comprobar consistencia posterior al empaquetado de píxeles
    print("\n# 7. Comprobando consistencia posterior al empaquetado #")
    mnist_utils.verify_bit_packing(X_bin, X_pix_packed, total_bits=NUM_PIXELS,
                                   pack_bits=PACK_BITS,)

    print("\n# 8. Empaquetando (+ ONEHOTENCODE) bits en palabras de 32 bits #")
    # Empaqueta la entrada completa para Forward-Forward
    X_ff_packed = mnist_utils.pack_bits_matrix(X_ff, total_bits=FF_INPUT_BITS,
                                               pack_bits=PACK_BITS,)

    # Comprobar consistencia posterior al empaquetado de píxeles
    print("\n# 9. Comprobando consistencia posterior al empaquetado"
          " (+ ONEHOTENCODE) #")
    mnist_utils.verify_bit_packing(X_ff, X_ff_packed, total_bits=FF_INPUT_BITS,
                                   pack_bits=PACK_BITS,)

    # Comprobación final: contraste en bruto
    print("\n# 10. Comprobación final: contraste en bruto #")
    X_unpacked_42 = mnist_utils.unpack_bits_row(X_ff_packed[42],
                                                total_bits=FF_INPUT_BITS,
                                                pack_bits=PACK_BITS,)
    print(f"X_ff_packed[42]: {X_ff_packed[42]}")
    # print(f"X_ff[42]: {X_ff[42]}")
    # print(f"X_unpacked_42: {X_unpacked_42}")
    # X_ff[42][3] = 22    # Modificación para verificar que detecta diferencias
    print(f"X_ff == X_unpacked_42: {np.array_equal(X_ff[42], X_unpacked_42)}")

    # Guarda el archivo binario con solo píxeles empaquetados
    mnist_utils.save_packed_pixels_dataset(
        "data\\processed\\mnist_bnn_pixels_packed.bin", X_pix_packed, y,)

    # Guarda el archivo binario con entrada FF empaquetada
    mnist_utils.save_packed_ff_dataset(
        "data\\processed\\mnist_ff_input_packed.bin", X_ff_packed, y,)

    # Guarda archivos intermedios en formato .npy para depuración
    mnist_utils.save_numpy_debug(
        "data\\processed\\mnist", X=X, y=y, X_bin=X_bin, y_onehot=y_onehot,
        X_ff=X_ff, X_pix_packed=X_pix_packed, X_ff_packed=X_ff_packed,)

    # Prueba de lectura y desempaquetado para verificar integridad
    """
    labels, packed, X_rec = mnist_utils.test_binary_dataset(
        "data\\processed\\mnist_bnn_pixels_packed.bin",
        packed_words=PIX_PACK_WORDS, total_bits=NUM_PIXELS)

    labels, packed, X_rec = mnist_utils.test_binary_dataset(
        "data\\processed\\mnist_ff_input_packed.bin",
        packed_words=FF_PACK_WORDS, total_bits=FF_INPUT_BITS)
    """
