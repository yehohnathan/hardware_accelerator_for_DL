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
# import numpy as np                  # Cálculo numérico eficiente

# =========================== Importar Funciones ============================ #
# import mnist_binarize_onehot_pack as mnist_utils
import read_bin

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
    input_path = "data/processed/mnist_ff_input_packed.bin"
    sample_idx = 2

    labels, packed = read_bin.load_packed_binary_dataset(input_path,
                                                         FF_PACK_WORDS)

    read_bin.print_full_sample(packed, sample_idx=sample_idx)

    read_bin.print_first_words(packed, sample_idx=sample_idx)

    read_bin.print_first_bits(packed, sample_idx=sample_idx)

    labels, packed = read_bin.load_packed_binary_dataset(input_path,
                                                         PIX_PACK_WORDS)

    read_bin.inspect_sample_structure(packed, sample_idx=sample_idx)
