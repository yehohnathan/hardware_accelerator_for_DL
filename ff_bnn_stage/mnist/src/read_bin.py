# ================================ LIBRERIAS ================================ #
from pathlib import Path

# Importa NumPy para reconstruir y mostrar el contenido del binario.
import numpy as np


# ============================ VARIABLES GLOBALES =========================== #
# Define la palabra magica que identifica el nuevo formato binario.
HEADER_MAGIC = 0x4D4E4953

# Define la version actual de la cabecera del binario.
HEADER_VERSION = 1

# Define la cantidad fija de palabras reservadas para la cabecera.
HEADER_WORDS = 16

# Define el ancho de cada palabra del archivo binario.
PACK_BITS = 32

# Define el orden exacto de los campos guardados en la cabecera.
HEADER_KEYS = (
    "magic",
    "version",
    "header_words",
    "sample_count",
    "image_width",
    "image_height",
    "source_width",
    "source_height",
    "pixel_bits",
    "num_classes",
    "label_bits",
    "useful_bits",
    "words_per_sample",
    "total_bits",
    "padding_bits",
    "resize_applied",
)


# ================================ FUNCIONES ================================ #
def validate_pixel_bits(pixel_bits: int) -> None:
    """
    Valida la cantidad de bits usada para almacenar cada pixel.

    Parametros
    ----------
    pixel_bits : int
        Indica la cantidad de bits solicitada para cada pixel.

    Retorna
    -------
    None
        No retorna ningun valor. Solo valida el parametro recibido.
    """
    if pixel_bits < 1 or pixel_bits > 8:
        raise ValueError("pixel_bits debe estar entre 1 y 8.")


def get_mode_suffix(pixel_bits: int) -> str:
    """
    Construye el sufijo corto usado por el modo almacenado.

    Parametros
    ----------
    pixel_bits : int
        Indica la cantidad de bits usada para almacenar cada pixel.

    Retorna
    -------
    str
        Retorna el sufijo del modo, por ejemplo 1b, 4b o 6b.
    """
    validate_pixel_bits(pixel_bits)
    return f"{pixel_bits}b"


def get_mode_name(pixel_bits: int) -> str:
    """
    Construye el nombre legible del modo almacenado.

    Parametros
    ----------
    pixel_bits : int
        Indica la cantidad de bits usada para almacenar cada pixel.

    Retorna
    -------
    str
        Retorna binary_1bit para el modo binario y quantized_Nbit para
        el resto de modos cuantizados.
    """
    validate_pixel_bits(pixel_bits)

    if pixel_bits == 1:
        return "binary_1bit"

    return f"quantized_{pixel_bits}bit"


def build_layout(
    pixel_bits: int,
    image_width: int,
    image_height: int,
    num_classes: int,
) -> dict[str, int | str]:
    """
    Construye el layout fisico esperado segun la cabecera.

    Parametros
    ----------
    pixel_bits : int
        Indica la cantidad de bits usada para almacenar cada pixel.
    image_width : int
        Indica el ancho de la imagen almacenada en el binario.
    image_height : int
        Indica el alto de la imagen almacenada en el binario.
    num_classes : int
        Indica la cantidad de clases codificadas como one-hot.

    Retorna
    -------
    dict[str, int | str]
        Retorna el diccionario con el layout esperado de la muestra.
    """
    validate_pixel_bits(pixel_bits)

    if image_width <= 0 or image_height <= 0:
        raise ValueError("La resolucion almacenada no es valida.")

    # Calcula el descriptor completo de una muestra empaquetada.
    num_pixels = image_width * image_height
    useful_bits = num_classes + (num_pixels * pixel_bits)
    words_per_sample = (useful_bits + PACK_BITS - 1) // PACK_BITS
    total_bits = words_per_sample * PACK_BITS
    padding_bits = total_bits - useful_bits

    return {
        "pixel_bits": pixel_bits,
        "image_width": image_width,
        "image_height": image_height,
        "num_pixels": num_pixels,
        "num_classes": num_classes,
        "label_bits": num_classes,
        "useful_bits": useful_bits,
        "words_per_sample": words_per_sample,
        "total_bits": total_bits,
        "padding_bits": padding_bits,
        "suffix": get_mode_suffix(pixel_bits),
        "mode_name": get_mode_name(pixel_bits),
    }


def unpack_binary_header(header_words: np.ndarray) -> dict[str, int | str]:
    """
    Reconstruye la cabecera del binario a partir de sus primeras words.

    Parametros
    ----------
    header_words : np.ndarray
        Contiene las words reservadas para la cabecera del archivo.

    Retorna
    -------
    dict[str, int | str]
        Retorna el diccionario de metadatos reconstruido y validado.
    """
    if header_words.size != HEADER_WORDS:
        raise ValueError("La cabecera no tiene el tamano esperado.")

    # Convierte la cabecera al diccionario de metadatos del archivo.
    metadata = {
        key: int(value)
        for key, value in zip(HEADER_KEYS, header_words.tolist())
    }

    if metadata["magic"] != HEADER_MAGIC:
        raise ValueError("El binario no contiene la cabecera esperada.")

    if metadata["version"] != HEADER_VERSION:
        raise ValueError("La version de la cabecera no esta soportada.")

    if metadata["header_words"] != HEADER_WORDS:
        raise ValueError("La cabecera declara un tamano invalido.")

    # Recalcula el layout para validar los campos derivados.
    layout = build_layout(
        pixel_bits=metadata["pixel_bits"],
        image_width=metadata["image_width"],
        image_height=metadata["image_height"],
        num_classes=metadata["num_classes"],
    )

    expected_pairs = {
        "label_bits": int(layout["label_bits"]),
        "useful_bits": int(layout["useful_bits"]),
        "words_per_sample": int(layout["words_per_sample"]),
        "total_bits": int(layout["total_bits"]),
        "padding_bits": int(layout["padding_bits"]),
    }

    for key, expected_value in expected_pairs.items():
        if metadata[key] != expected_value:
            raise ValueError(
                f"La cabecera contiene un valor inconsistente: {key}."
            )

    # Anade los datos derivados para facilitar la inspeccion del archivo.
    metadata["num_pixels"] = int(layout["num_pixels"])
    metadata["suffix"] = str(layout["suffix"])
    metadata["mode_name"] = str(layout["mode_name"])
    metadata["payload_words"] = (
        metadata["sample_count"] * metadata["words_per_sample"]
    )
    return metadata


def load_packed_ff_binary_dataset(
    file_path: str | Path,
) -> tuple[dict[str, int | str], np.ndarray]:
    """
    Lee un archivo binario FF con cabecera y payload empaquetado.

    Parametros
    ----------
    file_path : str | Path
        Indica la ruta del binario que se desea cargar.

    Retorna
    -------
    tuple[dict[str, int | str], np.ndarray]
        Retorna la metadata del archivo y la matriz de muestras empaquetadas.
    """
    # Normaliza la ruta del archivo a leer.
    binary_path = Path(file_path)

    # Carga el archivo como words uint32 little-endian.
    raw_words = np.fromfile(binary_path, dtype="<u4")

    if raw_words.size == 0:
        raise ValueError("El archivo binario esta vacio.")

    if raw_words.size < HEADER_WORDS:
        raise ValueError(
            "El archivo binario no contiene la cabecera completa."
        )

    # Separa la cabecera del resto del payload.
    header_words = raw_words[:HEADER_WORDS]
    payload_words = raw_words[HEADER_WORDS:]

    # Reconstruye la metadata y valida el layout declarado.
    metadata = unpack_binary_header(header_words)

    if payload_words.size != metadata["payload_words"]:
        raise ValueError(
            "El tamano del payload no coincide con la metadata del binario."
        )

    # Reorganiza la lectura en filas, una por muestra.
    packed_data = payload_words.reshape(
        int(metadata["sample_count"]),
        int(metadata["words_per_sample"]),
    )
    return metadata, packed_data


def unpack_bits_row(packed_row: np.ndarray, total_bits: int) -> np.ndarray:
    """
    Reconstruye una muestra bit a bit desde sus words empaquetadas.

    Parametros
    ----------
    packed_row : np.ndarray
        Contiene la muestra empaquetada como words uint32.
    total_bits : int
        Indica la cantidad total de bits fisicos de la muestra.

    Retorna
    -------
    np.ndarray
        Retorna el vector de bits fisicos reconstruido.
    """
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
    metadata: dict[str, int | str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decodifica una muestra a etiqueta, pixeles y padding.

    Parametros
    ----------
    packed_row : np.ndarray
        Contiene la muestra empaquetada como words uint32.
    metadata : dict[str, int | str]
        Contiene la metadata reconstruida de la cabecera del archivo.

    Retorna
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Retorna la etiqueta one-hot, los pixeles y el padding de la muestra.
    """
    # Reconstruye todos los bits fisicos de la muestra.
    bits = unpack_bits_row(packed_row, int(metadata["total_bits"]))

    # Separa los bits de la etiqueta one-hot.
    label_bits = bits[0:int(metadata["label_bits"])]

    # Reserva el vector donde quedaran los pixeles logicos.
    pixel_values = np.zeros(int(metadata["num_pixels"]), dtype=np.uint8)

    # Recorre el bloque de pixeles respetando el ancho del modo activo.
    pixel_cursor = int(metadata["label_bits"])
    for pixel_idx in range(int(metadata["num_pixels"])):
        pixel_value = 0

        for inner_bit in range(int(metadata["pixel_bits"])):
            pixel_value |= int(bits[pixel_cursor + inner_bit]) << inner_bit

        pixel_values[pixel_idx] = pixel_value
        pixel_cursor += int(metadata["pixel_bits"])

    # Extrae el padding restante hasta cerrar la muestra completa.
    padding_bits = bits[
        int(metadata["useful_bits"]):int(metadata["total_bits"])
    ]

    return label_bits, pixel_values, padding_bits


def print_dataset_summary(
    metadata: dict[str, int | str],
    packed_data: np.ndarray,
    file_path: str | Path,
) -> None:
    """
    Imprime un resumen corto del archivo cargado.

    Parametros
    ----------
    metadata : dict[str, int | str]
        Contiene la metadata reconstruida de la cabecera del archivo.
    packed_data : np.ndarray
        Contiene las muestras del payload ya reorganizadas por filas.
    file_path : str | Path
        Indica la ruta del binario que se esta mostrando.

    Retorna
    -------
    None
        No retorna ningun valor. Solo imprime el resumen del archivo.
    """
    print("\n===== DATASET CARGADO =====")
    print(f"file_path       : {Path(file_path)}")
    print(f"input_mode      : {metadata['mode_name']}")
    print(f"samples         : {metadata['sample_count']}")
    print(
        f"resolution      : "
        f"{metadata['image_width']}x{metadata['image_height']}"
    )
    print(
        f"source_size     : "
        f"{metadata['source_width']}x{metadata['source_height']}"
    )
    print(f"resize_applied  : {bool(metadata['resize_applied'])}")
    print(f"pixel_bits      : {metadata['pixel_bits']}")
    print(f"label_bits      : {metadata['label_bits']}")
    print(f"useful_bits     : {metadata['useful_bits']}")
    print(f"words_per_sample: {metadata['words_per_sample']}")
    print(f"padding_bits    : {metadata['padding_bits']}")
    print(f"payload_rows    : {packed_data.shape[0]}")


def print_sample_words(
    packed_data: np.ndarray,
    sample_idx: int,
    num_words: int | None = None,
) -> None:
    """
    Imprime una muestra en decimal, hexadecimal y binario.

    Parametros
    ----------
    packed_data : np.ndarray
        Contiene las muestras del payload ya reorganizadas por filas.
    sample_idx : int
        Indica el indice de la muestra que se desea mostrar.
    num_words : int | None, optional
        Indica cuantas words se desean imprimir.

    Retorna
    -------
    None
        No retorna ningun valor. Solo imprime las words seleccionadas.
    """
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


def print_digit_matrix(
    pixel_values: np.ndarray,
    image_width: int,
    image_height: int,
    pixel_bits: int,
) -> None:
    """
    Imprime una imagen directamente en terminal.

    Parametros
    ----------
    pixel_values : np.ndarray
        Contiene los pixeles reconstruidos de la muestra.
    image_width : int
        Indica el ancho de la imagen a mostrar.
    image_height : int
        Indica el alto de la imagen a mostrar.
    pixel_bits : int
        Indica la cantidad de bits usada para almacenar cada pixel.

    Retorna
    -------
    None
        No retorna ningun valor. Solo imprime la imagen en terminal.
    """
    value_width = len(str((1 << pixel_bits) - 1))

    for row_idx in range(image_height):
        # Extrae una fila logica de la imagen reconstruida.
        row = pixel_values[
            row_idx * image_width:(row_idx + 1) * image_width
        ]

        if pixel_bits == 1:
            print("".join(str(int(value)) for value in row))
        else:
            print(
                " ".join(
                    f"{int(value):0{value_width}d}" for value in row
                )
            )


def inspect_sample(
    packed_data: np.ndarray,
    metadata: dict[str, int | str],
    sample_idx: int,
    show_words: bool = False,
) -> None:
    """
    Imprime la informacion completa de una muestra.

    Parametros
    ----------
    packed_data : np.ndarray
        Contiene las muestras del payload ya reorganizadas por filas.
    metadata : dict[str, int | str]
        Contiene la metadata reconstruida de la cabecera del archivo.
    sample_idx : int
        Indica el indice de la muestra que se desea inspeccionar.
    show_words : bool, optional
        Indica si tambien se deben imprimir las words de la muestra.

    Retorna
    -------
    None
        No retorna ningun valor. Solo imprime la muestra seleccionada.
    """
    # Muestra primero las words si el usuario lo solicita.
    if show_words:
        print_sample_words(
            packed_data=packed_data,
            sample_idx=sample_idx,
            num_words=int(metadata["words_per_sample"]),
        )

    # Decodifica la muestra y calcula su etiqueta legible.
    label_bits, pixel_values, padding_bits = decode_sample(
        packed_row=packed_data[sample_idx],
        metadata=metadata,
    )
    decoded_label = int(np.argmax(label_bits))

    # Imprime un resumen de la muestra antes de dibujarla.
    print(f"\n===== MUESTRA {sample_idx} =====")
    print(f"input_mode      : {metadata['mode_name']}")
    print(
        f"resolution      : "
        f"{metadata['image_width']}x{metadata['image_height']}"
    )
    print(f"label_bits      : {label_bits}")
    print(f"onehot_valid    : {int(np.sum(label_bits)) == 1}")
    print(f"decoded_label   : {decoded_label}")
    print(f"pixels_nonzero  : {int(np.count_nonzero(pixel_values))}")
    print(f"pixels_sum      : {int(np.sum(pixel_values))}")
    print(f"pixels_max      : {int(np.max(pixel_values))}")
    print(f"padding_bits    : {padding_bits}")
    print("digit_matrix    :")

    # Dibuja la imagen reconstruida con la resolucion declarada.
    print_digit_matrix(
        pixel_values=pixel_values,
        image_width=int(metadata["image_width"]),
        image_height=int(metadata["image_height"]),
        pixel_bits=int(metadata["pixel_bits"]),
    )


def inspect_first_digits(
    packed_data: np.ndarray,
    metadata: dict[str, int | str],
    num_digits: int = 2,
    show_words: bool = False,
) -> None:
    """
    Imprime los primeros digitos del binario como matrices.

    Parametros
    ----------
    packed_data : np.ndarray
        Contiene las muestras del payload ya reorganizadas por filas.
    metadata : dict[str, int | str]
        Contiene la metadata reconstruida de la cabecera del archivo.
    num_digits : int, optional
        Indica cuantas muestras se desean mostrar.
    show_words : bool, optional
        Indica si tambien se deben imprimir las words de cada muestra.

    Retorna
    -------
    None
        No retorna ningun valor. Solo imprime las primeras muestras.
    """
    # Ajusta la cantidad de muestras al tamano real del archivo.
    effective_digits = min(num_digits, packed_data.shape[0])

    for sample_idx in range(effective_digits):
        inspect_sample(
            packed_data=packed_data,
            metadata=metadata,
            sample_idx=sample_idx,
            show_words=show_words,
        )
