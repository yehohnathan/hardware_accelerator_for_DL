# ================================ LIBRERÍAS ================================ #
from pathlib import Path
# Importa NumPy para reconstruir y mostrar el contenido del binario.
import numpy as np

# ============================ VARIABLES GLOBALES =========================== #
# Define el tamaño lateral de la imagen MNIST.
IMG_SIZE = 28
# Define la cantidad total de pixeles por muestra.
NUM_PIXELS = IMG_SIZE * IMG_SIZE
# Define la cantidad de bits reservados para la etiqueta one-hot.
NUM_CLASSES = 10
# Define el ancho de cada palabra del archivo binario.
PACK_BITS = 32


# ================================ FUNCIONES ================================ #
def build_layout(pixel_bits: int) -> dict[str, int]:
    """Construye el layout fisico del dataset para el modo solicitado."""
    if pixel_bits not in (1, 4):
        raise ValueError("pixel_bits debe ser 1 o 4.")

    # Calcula el descriptor completo de una muestra empaquetada.
    useful_bits = NUM_CLASSES + (NUM_PIXELS * pixel_bits)
    words_per_sample = (useful_bits + PACK_BITS - 1) // PACK_BITS
    total_bits = words_per_sample * PACK_BITS
    padding_bits = total_bits - useful_bits
    suffix = "1b" if pixel_bits == 1 else "4b"

    return {
        "pixel_bits": pixel_bits,
        "useful_bits": useful_bits,
        "words_per_sample": words_per_sample,
        "total_bits": total_bits,
        "padding_bits": padding_bits,
        "suffix": suffix,
    }


def load_packed_ff_binary_dataset(
    file_path: str | Path,
    packed_words: int,
) -> np.ndarray:
    """
    Lee un archivo binario FF con esta estructura:
    [packed_words uint32][packed_words uint32]...
    """
    # Normaliza la ruta del archivo a leer.
    binary_path = Path(file_path)

    # Carga el archivo como words uint32 little-endian.
    raw_words = np.fromfile(binary_path, dtype="<u4")

    if raw_words.size == 0:
        raise ValueError("El archivo binario esta vacio.")

    if raw_words.size % packed_words != 0:
        raise ValueError(
            "El tamano del archivo no es multiplo de packed_words. "
            "Revise el formato del binario."
        )

    # Reorganiza la lectura en filas, una por muestra.
    return raw_words.reshape(-1, packed_words)


def unpack_bits_row(packed_row: np.ndarray, total_bits: int) -> np.ndarray:
    """Reconstruye una muestra bit a bit desde sus words empaquetadas."""
    # Reserva el vector donde quedaran todos los bits fisicos.
    bits = np.zeros(total_bits, dtype=np.uint8)

    for bit_idx in range(total_bits):
        # Localiza la word y la posicion interna del bit actual.
        word_idx = bit_idx // PACK_BITS
        inner_bit = bit_idx % PACK_BITS
        bits[bit_idx] = (packed_row[word_idx] >> inner_bit) & 1

    return bits


def decode_sample(
    packed_row: np.ndarray,
    pixel_bits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decodifica una muestra a etiqueta, pixeles y padding."""
    # Recupera el layout para interpretar el bloque binario.
    layout = build_layout(pixel_bits)

    # Reconstruye todos los bits fisicos de la muestra.
    bits = unpack_bits_row(packed_row, layout["total_bits"])

    # Separa los bits de la etiqueta one-hot.
    label_bits = bits[0:NUM_CLASSES]

    # Reserva el vector donde quedaran los pixeles logicos.
    pixel_values = np.zeros(NUM_PIXELS, dtype=np.uint8)

    # Recorre el bloque de pixeles respetando el ancho del modo activo.
    pixel_cursor = NUM_CLASSES
    for pixel_idx in range(NUM_PIXELS):
        pixel_value = 0

        for inner_bit in range(pixel_bits):
            pixel_value |= int(bits[pixel_cursor + inner_bit]) << inner_bit

        pixel_values[pixel_idx] = pixel_value
        pixel_cursor += pixel_bits

    # Extrae el padding restante hasta cerrar la muestra completa.
    padding_bits = bits[layout["useful_bits"]: layout["total_bits"]]

    return label_bits, pixel_values, padding_bits


def print_dataset_summary(
    packed_data: np.ndarray,
    pixel_bits: int,
    file_path: str | Path,
) -> None:
    """Imprime un resumen corto del archivo cargado."""
    # Recupera el layout del modo activo para formar el resumen.
    layout = build_layout(pixel_bits)

    print("\n===== DATASET CARGADO =====")
    print(f"file_path       : {Path(file_path)}")
    print(f"input_mode      : {layout['suffix']}")
    print(f"pixel_bits      : {pixel_bits}")
    print(f"useful_bits     : {layout['useful_bits']}")
    print(f"words_per_sample: {layout['words_per_sample']}")
    print(f"padding_bits    : {layout['padding_bits']}")
    print(f"samples         : {packed_data.shape[0]}")


def print_sample_words(
    packed_data: np.ndarray,
    sample_idx: int,
    num_words: int | None = None,
) -> None:
    """Imprime una muestra en decimal, hexadecimal y binario."""
    # Selecciona la fila pedida y la recorta si hace falta.
    words = packed_data[sample_idx]
    if num_words is not None:
        words = words[:num_words]

    print(f"\n===== SAMPLE {sample_idx} =====")

    # Muestra cada word en tres formatos para facilitar la inspeccion.
    for i, word in enumerate(words):
        print(
            f"word[{i:02d}]  "
            f"dec:{int(word):<12}  "
            f"hex:0x{int(word):08X}  "
            f"bin:{int(word):032b}"
        )


def print_digit_matrix(pixel_values: np.ndarray, pixel_bits: int) -> None:
    """Imprime una imagen 28x28 directamente en terminal."""
    for row_idx in range(IMG_SIZE):
        # Extrae una fila logica de la imagen reconstruida.
        row = pixel_values[row_idx * IMG_SIZE:(row_idx + 1) * IMG_SIZE]

        if pixel_bits == 1:
            print("".join(str(int(value)) for value in row))
        else:
            print(" ".join(f"{int(value):02d}" for value in row))


def inspect_sample(
    packed_data: np.ndarray,
    sample_idx: int,
    pixel_bits: int,
    show_words: bool = False,
) -> None:
    """Imprime la informacion completa de una muestra."""
    # Recupera el layout y, si se pide, muestra primero sus words.
    layout = build_layout(pixel_bits)
    if show_words:
        print_sample_words(
            packed_data=packed_data,
            sample_idx=sample_idx,
            num_words=layout["words_per_sample"],
        )

    # Decodifica la muestra y calcula su etiqueta legible.
    label_bits, pixel_values, padding_bits = decode_sample(
        packed_row=packed_data[sample_idx],
        pixel_bits=pixel_bits,
    )
    decoded_label = int(np.argmax(label_bits))

    # Imprime un resumen de la muestra antes de dibujarla.
    print(f"\n===== MUESTRA {sample_idx} =====")
    print(f"input_mode      : {layout['suffix']}")
    print(f"label_bits      : {label_bits}")
    print(f"onehot_valid    : {int(np.sum(label_bits)) == 1}")
    print(f"decoded_label   : {decoded_label}")
    print(f"pixels_nonzero  : {int(np.count_nonzero(pixel_values))}")
    print(f"pixels_sum      : {int(np.sum(pixel_values))}")
    print(f"pixels_max      : {int(np.max(pixel_values))}")
    print(f"padding_bits    : {padding_bits}")
    print("digit_28x28     :")

    # Dibuja la imagen reconstruida en formato 28x28.
    print_digit_matrix(pixel_values, pixel_bits)


def inspect_first_digits(
    packed_data: np.ndarray,
    pixel_bits: int,
    num_digits: int = 2,
    show_words: bool = False,
) -> None:
    """Imprime los primeros digitos del binario como matrices 28x28."""
    # Ajusta la cantidad de muestras al tamaño real del archivo.
    effective_digits = min(num_digits, packed_data.shape[0])

    for sample_idx in range(effective_digits):
        inspect_sample(
            packed_data=packed_data,
            sample_idx=sample_idx,
            pixel_bits=pixel_bits,
            show_words=show_words,
        )
