# =============================== Librerías ================================= #
from pathlib import Path            # Manejo limpio de rutas de archivos
import numpy as np                  # Cálculo numérico eficiente


# =========================== Variables globales ============================ #
IMG_SIZE = 28
# Tamaño lateral de la imagen MNIST: 28x28

NUM_PIXELS = IMG_SIZE * IMG_SIZE
# Total de píxeles por imagen: 784

NUM_CLASSES = 10
# Número de clases en MNIST: dígitos del 0 al 9

FF_INPUT_BITS = NUM_PIXELS + NUM_CLASSES
# Entrada total para FF: 784 bits de imagen + 10 bits one-hot

PACK_BITS = 32
# Cantidad de bits por palabra de empaquetado

PIX_PACK_WORDS = (NUM_PIXELS + PACK_BITS - 1) // PACK_BITS
# Número de palabras de 32 bits necesarias para 784 bits

FF_PACK_WORDS = (FF_INPUT_BITS + PACK_BITS - 1) // PACK_BITS
# Número de palabras de 32 bits necesarias para 794 bits


# =============================== Funciones ================================= #
def load_packed_ff_binary_dataset(
    file_path: str,
    packed_words: int,
) -> np.ndarray:
    """
    Lee un archivo binario FF con estructura:

    [packed_words uint32][packed_words uint32]...

    Es decir, cada muestra contiene únicamente las palabras del vector
    FF empaquetado. No existe un byte adicional de label.

    Args:
        file_path: Ruta del archivo binario.
        packed_words: Cantidad de palabras uint32 por muestra.

    Returns:
        packed_rows: Matriz empaquetada con forma (N, packed_words).
    """
    file_path = Path(file_path)
    # Convierte la ruta a objeto Path

    raw_words = np.fromfile(file_path, dtype="<u4")
    # Lee todo el archivo como palabras uint32 little-endian

    if raw_words.size == 0:
        # Verifica si el archivo está vacío
        raise ValueError("El archivo binario está vacío.")

    if raw_words.size % packed_words != 0:
        # Verifica que la cantidad total de words sea múltiplo exacto
        raise ValueError(
            "El tamaño del archivo no es múltiplo de packed_words. "
            "Revise el formato del binario."
        )

    packed_rows = raw_words.reshape(-1, packed_words)
    # Reorganiza el vector lineal en una matriz de N filas por packed_words

    return packed_rows
    # Devuelve la matriz empaquetada


def unpack_bits_matrix(
    packed_matrix: np.ndarray,
    total_bits: int,
    pack_bits: int = 32,
) -> np.ndarray:
    """
    Convierte datos empaquetados en su representación binaria original.

    Args:
        packed_matrix: Matriz (N, words) con uint32.
        total_bits: Cantidad total de bits a reconstruir.
        pack_bits: Bits por palabra.

    Returns:
        Matriz binaria (N, total_bits).
    """
    n_samples = packed_matrix.shape[0]
    # Obtiene la cantidad de muestras del dataset

    X_bits = np.zeros((n_samples, total_bits), dtype=np.uint8)
    # Reserva la matriz de salida reconstruida

    for n in range(n_samples):
        # Recorre cada muestra

        for i in range(total_bits):
            # Recorre cada bit a reconstruir

            word_idx = i // pack_bits
            # Calcula en qué palabra se encuentra el bit

            bit_idx = i % pack_bits
            # Calcula la posición del bit dentro de la palabra

            X_bits[n, i] = (packed_matrix[n, word_idx] >> bit_idx) & 1
            # Extrae el bit y lo guarda en la matriz reconstruida

    return X_bits
    # Devuelve la matriz binaria completa


def test_binary_dataset(
    file_path: str,
    packed_words: int,
    total_bits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga un dataset binario FF y reconstruye los vectores binarios.

    Args:
        file_path: Ruta del archivo binario.
        packed_words: Palabras por muestra.
        total_bits: Bits lógicos por muestra.

    Returns:
        packed: Matriz empaquetada (N, packed_words).
        X_reconstructed: Matriz reconstruida (N, total_bits).
    """
    packed = load_packed_ff_binary_dataset(
        file_path=file_path,
        packed_words=packed_words,
    )
    # Carga la matriz de palabras desde el archivo binario

    print("Dataset cargado")
    # Informa que el archivo fue leído

    print("Muestras:", packed.shape[0])
    # Imprime la cantidad total de muestras

    print("Palabras por muestra:", packed.shape[1])
    # Imprime la cantidad de palabras por muestra

    X_reconstructed = unpack_bits_matrix(
        packed_matrix=packed,
        total_bits=total_bits,
        pack_bits=PACK_BITS,
    )
    # Reconstruye los bits lógicos desde las words empaquetadas

    print("Reconstrucción completada")
    # Informa que terminó la reconstrucción

    print("Dimensión reconstruida:", X_reconstructed.shape)
    # Imprime la forma de la matriz reconstruida

    return packed, X_reconstructed
    # Devuelve tanto la matriz empaquetada como la reconstruida


def print_full_sample(
    packed_data: np.ndarray,
    sample_idx: int = 0,
) -> None:
    """
    Imprime todas las palabras binarias de una muestra.

    Args:
        packed_data: Matriz (N, words) con uint32.
        sample_idx: Índice de muestra a visualizar.
    """
    words = packed_data[sample_idx]
    # Selecciona la muestra indicada

    print(f"\n===== SAMPLE {sample_idx} =====")
    # Imprime el encabezado de la muestra

    for i, word in enumerate(words):
        # Recorre cada palabra de la muestra

        print(
            f"word[{i:02d}]  "
            f"dec:{word:<12}  "
            f"hex:0x{word:08X}  "
            f"bin:{word:032b}"
        )
        # Imprime la palabra en decimal, hexadecimal y binario


def print_first_words(
    packed_data: np.ndarray,
    sample_idx: int = 0,
    num_words: int = 25,
) -> None:
    """
    Imprime las primeras palabras empaquetadas de una muestra.

    Args:
        packed_data: Matriz (N, words) con uint32.
        sample_idx: Índice de muestra.
        num_words: Cantidad de palabras a imprimir.
    """
    words = packed_data[sample_idx][:num_words]
    # Selecciona las primeras num_words palabras de la muestra

    print(f"\n===== FIRST {num_words} WORDS (sample {sample_idx}) =====")
    # Imprime el encabezado de la sección

    for i, word in enumerate(words):
        # Recorre las palabras seleccionadas

        print(
            f"word[{i:02d}] = 0x{word:08X}"
        )
        # Imprime cada palabra en hexadecimal


def print_first_bits(
    packed_data: np.ndarray,
    sample_idx: int = 0,
    num_bits: int = 800,
    pack_bits: int = 32,
) -> None:
    """
    Imprime los primeros bits reconstruidos desde el binario.

    Args:
        packed_data: Matriz (N, words) con uint32.
        sample_idx: Índice de muestra.
        num_bits: Cantidad de bits a mostrar.
        pack_bits: Bits por palabra.
    """
    words = packed_data[sample_idx]
    # Obtiene las palabras de la muestra seleccionada

    bits = []
    # Lista temporal donde se almacenan los bits reconstruidos

    for i in range(num_bits):
        # Recorre los bits solicitados

        word_idx = i // pack_bits
        # Calcula en qué palabra se encuentra el bit

        bit_idx = i % pack_bits
        # Calcula la posición del bit dentro de la palabra

        bit = (words[word_idx] >> bit_idx) & 1
        # Extrae el bit correspondiente

        bits.append(str(bit))
        # Convierte el bit a texto y lo agrega a la lista

    print(f"\n===== FIRST {num_bits} BITS (sample {sample_idx}) =====")
    # Imprime el encabezado de la sección

    for i in range(0, num_bits, pack_bits):
        # Recorre los bits por bloques de 32

        chunk = bits[i:i + pack_bits]
        # Extrae un bloque consecutivo de bits

        print("".join(chunk))
        # Imprime el bloque como una cadena continua


def decode_first_sample_structure(
    packed_data: np.ndarray,
    sample_idx: int = 0,
    num_words: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decodifica label, imagen y padding desde las primeras 25 palabras.

    Estructura esperada del vector FF:
        bits 0-9     : one-hot label
        bits 10-793  : imagen binaria (784)
        bits 794-799 : padding (6)

    Args:
        packed_data: Matriz (N, words) con uint32.
        sample_idx: Índice de muestra a analizar.
        num_words: Palabras a leer.

    Returns:
        label_bits: Vector de 10 bits.
        image_bits: Vector de 784 bits.
        padding_bits: Vector de 6 bits.
    """
    words = packed_data[sample_idx][:num_words]
    # Selecciona las primeras palabras de la muestra

    bits = []
    # Lista temporal donde se reconstruirán los 800 bits

    for word in words:
        # Recorre cada palabra

        for i in range(32):
            # Recorre cada bit dentro de la palabra

            bit = (word >> i) & 1
            # Extrae el bit i usando convención LSB-first

            bits.append(bit)
            # Agrega el bit reconstruido a la lista

    bits = np.array(bits, dtype=np.uint8)
    # Convierte la lista final a vector NumPy

    label_bits = bits[0:10]
    # Extrae los 10 bits del one-hot

    image_bits = bits[10:794]
    # Extrae los 784 bits de imagen

    padding_bits = bits[794:800]
    # Extrae los 6 bits de padding

    return label_bits, image_bits, padding_bits
    # Devuelve las tres secciones separadas


def inspect_sample_structure(
    packed_data: np.ndarray,
    sample_idx: int = 0,
) -> None:
    """
    Imprime label, imagen y padding desde el binario.

    Args:
        packed_data: Matriz (N, words) con uint32.
        sample_idx: Índice de la muestra a inspeccionar.
    """
    label_bits, image_bits, padding_bits = decode_first_sample_structure(
        packed_data=packed_data,
        sample_idx=sample_idx,
        num_words=FF_PACK_WORDS,
    )
    # Decodifica la estructura lógica de la muestra seleccionada

    print("\n===== STRUCTURE INSPECTION =====")
    # Imprime el encabezado de inspección

    print("Label bits [0-9]:")
    # Indica la sección de etiqueta

    print(label_bits)
    # Imprime el vector one-hot

    print("\nImage bits [10-793]:")
    # Indica la sección de imagen

    for i in range(len(image_bits)):
        # Recorre todos los bits de imagen

        print(image_bits[i], end="")
        # Imprime el bit sin salto de línea inmediato

        if (i + 1) % 28 == 0:
            # Cada 28 bits se termina una fila lógica de la imagen
            print()

    print("\nPadding bits [794-799]:")
    # Indica la sección de padding

    print(padding_bits)
    # Imprime los 6 bits de padding

    label_sum = int(np.sum(label_bits))
    # Cuenta cuántos bits activos hay en el one-hot

    print("\nOne-hot valid:", label_sum == 1)
    # Informa si la etiqueta es one-hot válida

    decoded_label = int(np.argmax(label_bits))
    # Decodifica la clase por posición del máximo

    print("Decoded label:", decoded_label)
    # Imprime la clase reconstruida
