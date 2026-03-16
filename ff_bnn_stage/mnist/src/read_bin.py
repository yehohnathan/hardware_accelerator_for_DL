# =============================== Librerías ================================= #
from pathlib import Path            # Manejo limpio de rutas de archivos
import numpy as np                  # Cálculo numérico eficiente


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


# =============================== Funciones ================================= #
def load_packed_binary_dataset(file_path: str, packed_words: int,
                               ) -> tuple[np.ndarray, np.ndarray]:
    """
    Lee un archivo binario con estructura:

    [label:1 byte][packed_words uint32]

    Args:
        file_path: Ruta del archivo binario.
        packed_words: Cantidad de palabras uint32 por muestra.

    Returns:
        labels: Vector de etiquetas (N,)
        packed_data: Matriz empaquetada (N, packed_words)
    """

    file_path = Path(file_path)

    labels = []
    packed_rows = []

    with open(file_path, "rb") as f:

        while True:

            label_bytes = f.read(1)
            if not label_bytes:
                break

            label = int.from_bytes(label_bytes, "little")

            packed = np.frombuffer(
                f.read(packed_words * 4),
                dtype=np.uint32
            )

            labels.append(label)
            packed_rows.append(packed)

    labels = np.array(labels, dtype=np.uint8)
    packed_rows = np.vstack(packed_rows)

    return labels, packed_rows


def unpack_bits_matrix(packed_matrix: np.ndarray, total_bits: int,
                       pack_bits: int = 32,) -> np.ndarray:
    """
    Convierte datos empaquetados en su representación binaria original.

    Args:
        packed_matrix: Matriz (N, words) con uint32.
        total_bits: Cantidad total de bits a reconstruir.
        pack_bits: Bits por palabra.

    Returns:
        Matriz binaria (N, total_bits)
    """

    N = packed_matrix.shape[0]

    X_bits = np.zeros((N, total_bits), dtype=np.uint8)

    for n in range(N):

        for i in range(total_bits):

            word_idx = i // pack_bits
            bit_idx = i % pack_bits

            X_bits[n, i] = (
                packed_matrix[n, word_idx] >> bit_idx
            ) & 1

    return X_bits


def test_binary_dataset(file_path: str, packed_words: int, total_bits: int,):
    """
    Carga un dataset binario y reconstruye los vectores binarios.
    """

    labels, packed = load_packed_binary_dataset(
        file_path,
        packed_words
    )

    print("Dataset cargado")
    print("Muestras:", packed.shape[0])
    print("Palabras por muestra:", packed.shape[1])

    X_reconstructed = unpack_bits_matrix(
        packed,
        total_bits=total_bits
    )

    print("Reconstrucción completada")
    print("Dimensión reconstruida:", X_reconstructed.shape)

    return labels, packed, X_reconstructed


def print_full_sample(packed_data: np.ndarray, sample_idx: int = 0) -> None:
    """
    Imprime todas las palabras binarias de una muestra.

    Args:
        packed_data: Matriz (N, words) con uint32.
        sample_idx: Índice de muestra a visualizar.
    """

    words = packed_data[sample_idx]

    print(f"\n===== SAMPLE {sample_idx} =====")

    for i, word in enumerate(words):

        print(
            f"word[{i:02d}]  "
            f"dec:{word:<12}  "
            f"hex:0x{word:08X}  "
            f"bin:{word:032b}"
        )


def print_first_words(packed_data: np.ndarray, sample_idx: int = 0,
                      num_words: int = 25,) -> None:
    """
    Imprime las primeras palabras empaquetadas.

    Args:
        packed_data: Matriz (N, words) con uint32.
        sample_idx: Índice de muestra.
        num_words: Cantidad de palabras a imprimir.
    """

    words = packed_data[sample_idx][:num_words]

    print(f"\n===== FIRST {num_words} WORDS (sample {sample_idx}) =====")

    for i, word in enumerate(words):

        print(
            f"word[{i:02d}]  "
            f"hex:0x{word:08X}  "
            f"bin:{word:032b}")


def print_first_bits(packed_data: np.ndarray, sample_idx: int = 0,
                     num_bits: int = 800, pack_bits: int = 32,) -> None:
    """
    Imprime los primeros bits reconstruidos desde el binario.

    Args:
        packed_data: Matriz (N, words) con uint32.
        sample_idx: Índice de muestra.
        num_bits: Cantidad de bits a mostrar.
        pack_bits: Bits por palabra.
    """

    words = packed_data[sample_idx]
    bits = []

    for i in range(num_bits):
        word_idx = i // pack_bits
        bit_idx = i % pack_bits
        bit = (words[word_idx] >> bit_idx) & 1
        bits.append(str(bit))

    print(f"\n===== FIRST {num_bits} BITS (sample {sample_idx}) =====")

    for i in range(0, num_bits, pack_bits):

        chunk = bits[i:i + pack_bits]

        print("".join(chunk))


def decode_first_sample_structure(packed_data: np.ndarray, sample_idx: int = 0,
                                  num_words: int = 25,) -> tuple[
                                      np.ndarray, np.ndarray, np.ndarray]:
    """
    Decodifica label, imagen y padding desde las primeras 25 palabras.

    Estructura esperada del vector FF:
        bits 0-9   : one-hot label
        bits 10-793: imagen binaria (784)
        bits 794-799: padding

    Args:
        packed_data: matriz (N, words) con uint32
        sample_idx: índice de muestra a analizar
        num_words: palabras a leer (25)

    Returns:
        label_bits: np.ndarray (10,)
        image_bits: np.ndarray (784,)
        padding_bits: np.ndarray (6,)
    """

    words = packed_data[sample_idx][:num_words]

    bits = []

    for word in words:
        for i in range(32):
            bit = (word >> i) & 1
            bits.append(bit)

    bits = np.array(bits, dtype=np.uint8)

    label_bits = bits[0:10]
    image_bits = bits[10:794]
    padding_bits = bits[794:800]

    return label_bits, image_bits, padding_bits


def inspect_sample_structure(packed_data: np.ndarray, sample_idx: int = 0):
    """
    Imprime label, imagen y padding desde el binario.
    """

    label_bits, image_bits, padding_bits = decode_first_sample_structure(
        packed_data,
        sample_idx
    )

    print("\n===== STRUCTURE INSPECTION =====")

    print("Label bits [0-9]:")
    print(label_bits)

    print("\nImage bits [10-793]:")
    for i in range(len(image_bits)):
        print(image_bits[i], end="")
        if (i + 1) % 28 == 0:
            print()  # Nueva línea cada 28 bits

    print("\nPadding bits [794-799]:")
    print(padding_bits)

    label = np.argmax(label_bits)

    print("\nDecoded label:", label)
