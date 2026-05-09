# ================================ LIBRERIAS ================================ #
from pathlib import Path

# Importa NumPy para el trabajo vectorial y el empaquetado del dataset.
import numpy as np

# Importa pandas para cargar el CSV de entrenamiento.
import pandas as pd


# ============================ VARIABLES GLOBALES =========================== #
# Define el tamano lateral original de la imagen MNIST.
IMG_SIZE = 28

# Define la cantidad total de pixeles por muestra en el dataset crudo.
NUM_PIXELS = IMG_SIZE * IMG_SIZE

# Define la cantidad de clases disponibles en MNIST.
NUM_CLASSES = 10

# Define el ancho de cada palabra del archivo binario.
PACK_BITS = 32

# Define la palabra magica que identifica el nuevo formato binario.
HEADER_MAGIC = 0x4D4E4953

# Define la version actual de la cabecera del binario.
HEADER_VERSION = 1

# Define la cantidad fija de palabras reservadas para la cabecera.
HEADER_WORDS = 16

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
    Construye el sufijo corto usado por los nombres de archivo.

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
    Construye el nombre legible del modo de entrada.

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
    num_classes: int = NUM_CLASSES,
) -> dict[str, int | str]:
    """
    Construye el layout fisico del dataset para el modo solicitado.

    Parametros
    ----------
    pixel_bits : int
        Indica la cantidad de bits usada para almacenar cada pixel.
    image_width : int
        Indica el ancho de la imagen almacenada en el binario.
    image_height : int
        Indica el alto de la imagen almacenada en el binario.
    num_classes : int, optional
        Indica la cantidad de clases codificadas como one-hot.

    Retorna
    -------
    dict[str, int | str]
        Retorna un diccionario con el layout completo de la muestra.
    """
    validate_pixel_bits(pixel_bits)

    if image_width <= 0 or image_height <= 0:
        raise ValueError("La resolucion debe usar valores mayores que cero.")

    # Calcula la cantidad de pixeles logicos de la imagen reducida.
    num_pixels = image_width * image_height

    # Calcula el tamano util, el tamano fisico y el padding por muestra.
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


def get_default_binary_name(
    pixel_bits: int,
    image_width: int,
    image_height: int,
) -> str:
    """
    Retorna el nombre canonico del binario segun el modo y la resolucion.

    Parametros
    ----------
    pixel_bits : int
        Indica la cantidad de bits usada para almacenar cada pixel.
    image_width : int
        Indica el ancho de la imagen almacenada en el binario.
    image_height : int
        Indica el alto de la imagen almacenada en el binario.

    Retorna
    -------
    str
        Retorna el nombre de archivo que se usara para el binario.
    """
    layout = build_layout(
        pixel_bits=pixel_bits,
        image_width=image_width,
        image_height=image_height,
    )
    return (
        f"mnist_{image_width}x{image_height}_{layout['suffix']}_packed.bin"
    )


def load_train_table(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga el CSV de entrenamiento de MNIST.

    Parametros
    ----------
    path : str | Path
        Indica la ruta del archivo CSV que contiene el dataset.

    Retorna
    -------
    tuple[np.ndarray, np.ndarray]
        Retorna la matriz de imagenes y el vector de etiquetas.
    """
    # Normaliza la ruta para trabajar con objetos Path.
    table_path = Path(path)

    # Lee el CSV completo en memoria.
    df = pd.read_csv(table_path)

    if "label" not in df.columns:
        raise ValueError("No se encontro la columna 'label'.")

    # Selecciona solo las columnas de pixeles.
    pixel_cols = [col for col in df.columns if col.startswith("pixel")]

    if len(pixel_cols) != NUM_PIXELS:
        raise ValueError(
            f"Se esperaban {NUM_PIXELS} columnas de pixeles y "
            f"se encontraron {len(pixel_cols)}."
        )

    # Convierte la tabla al formato numerico usado por el resto del flujo.
    X = df[pixel_cols].to_numpy(dtype=np.uint8)
    y = df["label"].to_numpy(dtype=np.uint8)

    return X, y


def validate_dataset(X: np.ndarray, y: np.ndarray) -> None:
    """
    Verifica que el dataset tenga forma y rangos validos.

    Parametros
    ----------
    X : np.ndarray
        Contiene las imagenes del dataset en formato plano.
    y : np.ndarray
        Contiene las etiquetas del dataset.

    Retorna
    -------
    None
        No retorna ningun valor. Solo valida la consistencia del dataset.
    """
    if X.ndim != 2 or X.shape[1] != NUM_PIXELS:
        raise ValueError(
            f"X debe tener forma (N, {NUM_PIXELS}). Se obtuvo {X.shape}."
        )

    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(
            "Las etiquetas no coinciden con la cantidad de muestras."
        )

    if X.min() < 0 or X.max() > 255:
        raise ValueError("Los pixeles deben estar en el rango [0, 255].")

    if y.min() < 0 or y.max() > 9:
        raise ValueError("Las etiquetas deben estar en el rango [0, 9].")

    # Imprime un resumen corto para confirmar la carga correcta.
    print("Dataset valido.")
    print(f"Muestras: {X.shape[0]}")
    print(f"Dimension por muestra: {X.shape[1]}")
    print(f"Clases presentes: {np.unique(y)}")
    print(f"Rango de pixeles: [{X.min()}, {X.max()}]")


def labels_to_onehot(
    y: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """
    Convierte etiquetas escalares a codificacion one-hot.

    Parametros
    ----------
    y : np.ndarray
        Contiene las etiquetas escalares del dataset.
    num_classes : int, optional
        Indica la cantidad total de clases del problema.

    Retorna
    -------
    np.ndarray
        Retorna la matriz de etiquetas codificadas en one-hot.
    """
    # Reserva una matriz de salida con una columna por clase.
    onehot = np.zeros((y.shape[0], num_classes), dtype=np.uint8)

    # Activa el bit de la clase correspondiente en cada fila.
    onehot[np.arange(y.shape[0]), y] = 1

    return onehot


def build_resize_edges(source_size: int, target_size: int) -> np.ndarray:
    """
    Construye los limites usados para reducir una dimension.

    Parametros
    ----------
    source_size : int
        Indica el tamano original de la dimension a reducir.
    target_size : int
        Indica el tamano final de la dimension reducida.

    Retorna
    -------
    np.ndarray
        Retorna el vector de cortes que delimita cada bloque.
    """
    if target_size <= 0:
        raise ValueError("La resolucion objetivo debe ser mayor que cero.")

    if target_size > source_size:
        raise ValueError(
            "La resolucion objetivo no puede ser mayor que la original."
        )

    # Calcula los cortes que delimitan cada bloque de reescalado.
    edges = np.linspace(
        0,
        source_size,
        num=target_size + 1,
        dtype=np.int32,
    )

    if np.any(np.diff(edges) <= 0):
        raise ValueError("La resolucion objetivo genera bloques invalidos.")

    return edges


def resize_images(
    X: np.ndarray,
    target_width: int,
    target_height: int,
    source_width: int = IMG_SIZE,
    source_height: int = IMG_SIZE,
) -> np.ndarray:
    """
    Redimensiona las imagenes crudas a una resolucion menor.

    Parametros
    ----------
    X : np.ndarray
        Contiene las imagenes del dataset en formato plano.
    target_width : int
        Indica el ancho final de la imagen reducida.
    target_height : int
        Indica el alto final de la imagen reducida.
    source_width : int, optional
        Indica el ancho original de la imagen de entrada.
    source_height : int, optional
        Indica el alto original de la imagen de entrada.

    Retorna
    -------
    np.ndarray
        Retorna las imagenes redimensionadas en formato plano.
    """
    if X.ndim != 2 or X.shape[1] != source_width * source_height:
        raise ValueError(
            "Las imagenes de entrada no coinciden con la resolucion original."
        )

    if target_width == source_width and target_height == source_height:
        return X.copy()

    # Reorganiza las muestras al formato matricial original.
    images = X.reshape(-1, source_height, source_width).astype(np.float32)

    # Calcula los cortes horizontales y verticales del reescalado.
    y_edges = build_resize_edges(source_height, target_height)
    x_edges = build_resize_edges(source_width, target_width)

    # Reserva la salida redimensionada en escala de grises.
    resized = np.zeros(
        (X.shape[0], target_height, target_width),
        dtype=np.float32,
    )

    for out_y in range(target_height):
        # Selecciona el rango vertical del bloque actual.
        y_start = y_edges[out_y]
        y_end = y_edges[out_y + 1]

        for out_x in range(target_width):
            # Selecciona el rango horizontal del bloque actual.
            x_start = x_edges[out_x]
            x_end = x_edges[out_x + 1]

            # Promedia el contenido del bloque para reducir la imagen.
            block = images[:, y_start:y_end, x_start:x_end]
            resized[:, out_y, out_x] = block.mean(axis=(1, 2))

    # Redondea la salida y la vuelve a dejar como vector plano.
    return np.rint(resized).astype(np.uint8).reshape(
        -1,
        target_width * target_height,
    )


def binarize_pixels_1bit(
    X: np.ndarray,
    threshold: int = 127,
) -> np.ndarray:
    """
    Binariza los pixeles usando un umbral fijo.

    Parametros
    ----------
    X : np.ndarray
        Contiene los pixeles que se desean binarizar.
    threshold : int, optional
        Indica el umbral usado para separar 0 y 1.

    Retorna
    -------
    np.ndarray
        Retorna la matriz de pixeles binarios.
    """
    return (X >= threshold).astype(np.uint8)


def quantize_pixels_nbit(X: np.ndarray, pixel_bits: int) -> np.ndarray:
    """
    Cuantiza los pixeles a la cantidad de bits solicitada.

    Parametros
    ----------
    X : np.ndarray
        Contiene los pixeles que se desean cuantizar.
    pixel_bits : int
        Indica la cantidad de bits usada para almacenar cada pixel.

    Retorna
    -------
    np.ndarray
        Retorna la matriz de pixeles cuantizados.
    """
    validate_pixel_bits(pixel_bits)

    if pixel_bits == 1:
        raise ValueError("El modo 1b debe prepararse con binarizacion.")

    if pixel_bits == 8:
        return X.copy()

    # Conserva los bits mas significativos con un corrimiento uniforme.
    shift_bits = 8 - pixel_bits
    return (X >> shift_bits).astype(np.uint8)


def prepare_pixels_for_packing(
    X: np.ndarray,
    pixel_bits: int,
    threshold: int = 127,
) -> np.ndarray:
    """
    Prepara los pixeles segun la cantidad de bits seleccionada.

    Parametros
    ----------
    X : np.ndarray
        Contiene los pixeles que se desean preparar.
    pixel_bits : int
        Indica la cantidad de bits usada para almacenar cada pixel.
    threshold : int, optional
        Indica el umbral usado cuando el modo es binario.

    Retorna
    -------
    np.ndarray
        Retorna la matriz de pixeles lista para empaquetarse.
    """
    validate_pixel_bits(pixel_bits)

    if pixel_bits == 1:
        return binarize_pixels_1bit(X, threshold=threshold)

    return quantize_pixels_nbit(X, pixel_bits=pixel_bits)


def build_ff_bits_matrix(
    y_onehot: np.ndarray,
    pixel_values: np.ndarray,
    pixel_bits: int,
) -> np.ndarray:
    """
    Construye la matriz de bits logicos de cada muestra.

    Parametros
    ----------
    y_onehot : np.ndarray
        Contiene las etiquetas en formato one-hot.
    pixel_values : np.ndarray
        Contiene los pixeles ya preparados para el modo activo.
    pixel_bits : int
        Indica la cantidad de bits usada para almacenar cada pixel.

    Retorna
    -------
    np.ndarray
        Retorna la matriz de bits utiles antes del padding fisico.
    """
    if pixel_bits == 1:
        # Reutiliza directamente los pixeles cuando ya vienen en 0 y 1.
        pixel_bit_matrix = pixel_values.astype(np.uint8)
    else:
        # Extrae los bitplanes en orden little-endian por cada pixel.
        bit_offsets = np.arange(pixel_bits, dtype=np.uint8)
        pixel_bit_matrix = (
            (pixel_values[:, :, None] >> bit_offsets[None, None, :]) & 1
        ).astype(np.uint8)

        # Aplana los bitplanes para obtener el layout lineal del kernel.
        pixel_bit_matrix = pixel_bit_matrix.reshape(
            pixel_values.shape[0],
            pixel_values.shape[1] * pixel_bits,
        )

    # Une la etiqueta y los pixeles en una sola matriz de bits utiles.
    return np.concatenate(
        [y_onehot.astype(np.uint8), pixel_bit_matrix],
        axis=1,
    )


def pack_ff_dataset(
    y_onehot: np.ndarray,
    pixel_values: np.ndarray,
    pixel_bits: int,
    image_width: int,
    image_height: int,
    chunk_size: int = 2048,
) -> np.ndarray:
    """
    Empaqueta el dataset completo en palabras uint32.

    Parametros
    ----------
    y_onehot : np.ndarray
        Contiene las etiquetas en formato one-hot.
    pixel_values : np.ndarray
        Contiene los pixeles ya preparados para el modo activo.
    pixel_bits : int
        Indica la cantidad de bits usada para almacenar cada pixel.
    image_width : int
        Indica el ancho de la imagen almacenada en el binario.
    image_height : int
        Indica el alto de la imagen almacenada en el binario.
    chunk_size : int, optional
        Indica el tamano del bloque usado durante el empaquetado.

    Retorna
    -------
    np.ndarray
        Retorna la matriz de muestras empaquetadas por words.
    """
    # Recupera el layout del modo activo para dimensionar la salida.
    layout = build_layout(
        pixel_bits=pixel_bits,
        image_width=image_width,
        image_height=image_height,
    )

    expected_pixels = image_width * image_height
    if pixel_values.shape[1] != expected_pixels:
        raise ValueError(
            "La cantidad de pixeles no coincide con la resolucion."
        )

    # Reserva la matriz final de palabras empaquetadas.
    packed = np.zeros(
        (pixel_values.shape[0], layout["words_per_sample"]),
        dtype=np.uint32,
    )

    # Precalcula los desplazamientos usados para cada bit de la word.
    shifts = (1 << np.arange(PACK_BITS, dtype=np.uint32)).reshape(
        1,
        1,
        PACK_BITS,
    )

    for start_idx in range(0, pixel_values.shape[0], chunk_size):
        # Limita el trabajo a bloques para no crecer demasiado en memoria.
        end_idx = min(start_idx + chunk_size, pixel_values.shape[0])

        # Construye la matriz de bits utiles del bloque actual.
        chunk_bits = build_ff_bits_matrix(
            y_onehot=y_onehot[start_idx:end_idx],
            pixel_values=pixel_values[start_idx:end_idx],
            pixel_bits=pixel_bits,
        )

        if layout["padding_bits"] > 0:
            # Completa el padding hasta cerrar palabras de 32 bits.
            chunk_bits = np.pad(
                chunk_bits,
                pad_width=((0, 0), (0, layout["padding_bits"])),
                mode="constant",
            )

        # Reorganiza el bloque por muestras, palabras y bits.
        chunk_bits = chunk_bits.reshape(
            -1,
            layout["words_per_sample"],
            PACK_BITS,
        ).astype(np.uint32)

        # Empaqueta cada grupo de 32 bits en una word little-endian.
        packed[start_idx:end_idx] = np.sum(
            chunk_bits * shifts,
            axis=2,
            dtype=np.uint32,
        )

    return packed


def build_binary_metadata(
    sample_count: int,
    pixel_bits: int,
    image_width: int,
    image_height: int,
    source_width: int = IMG_SIZE,
    source_height: int = IMG_SIZE,
) -> dict[str, int | str]:
    """
    Construye el diccionario de metadatos de la cabecera.

    Parametros
    ----------
    sample_count : int
        Indica la cantidad de muestras almacenadas.
    pixel_bits : int
        Indica la cantidad de bits usada para almacenar cada pixel.
    image_width : int
        Indica el ancho de la imagen almacenada en el binario.
    image_height : int
        Indica el alto de la imagen almacenada en el binario.
    source_width : int, optional
        Indica el ancho original de la imagen antes del redimensionado.
    source_height : int, optional
        Indica el alto original de la imagen antes del redimensionado.

    Retorna
    -------
    dict[str, int | str]
        Retorna el diccionario completo de metadatos del binario.
    """
    # Calcula el layout derivado de la resolucion almacenada.
    layout = build_layout(
        pixel_bits=pixel_bits,
        image_width=image_width,
        image_height=image_height,
    )

    # Reune la informacion minima necesaria para interpretar el binario.
    metadata = {
        "magic": HEADER_MAGIC,
        "version": HEADER_VERSION,
        "header_words": HEADER_WORDS,
        "sample_count": sample_count,
        "image_width": image_width,
        "image_height": image_height,
        "source_width": source_width,
        "source_height": source_height,
        "pixel_bits": pixel_bits,
        "num_classes": int(layout["num_classes"]),
        "label_bits": int(layout["label_bits"]),
        "useful_bits": int(layout["useful_bits"]),
        "words_per_sample": int(layout["words_per_sample"]),
        "total_bits": int(layout["total_bits"]),
        "padding_bits": int(layout["padding_bits"]),
        "resize_applied": int(
            image_width != source_width or image_height != source_height
        ),
        "num_pixels": int(layout["num_pixels"]),
        "suffix": str(layout["suffix"]),
        "mode_name": str(layout["mode_name"]),
    }

    # Añade datos derivados que resultan utiles para depuracion.
    metadata["payload_words"] = (
        sample_count * int(layout["words_per_sample"])
    )
    return metadata


def pack_binary_header(metadata: dict[str, int | str]) -> np.ndarray:
    """
    Serializa la cabecera del binario en palabras uint32.

    Parametros
    ----------
    metadata : dict[str, int | str]
        Contiene los metadatos del binario ya construidos.

    Retorna
    -------
    np.ndarray
        Retorna la cabecera serializada como words uint32.
    """
    # Ordena los campos de la cabecera con el formato pactado.
    return np.asarray(
        [int(metadata[key]) for key in HEADER_KEYS],
        dtype=np.uint32,
    )


def verify_vitis_little_endian_compatibility(
    y_onehot: np.ndarray,
    pixel_values: np.ndarray,
    packed_dataset: np.ndarray,
    pixel_bits: int,
    image_width: int,
    image_height: int,
    samples_to_check: int = 4,
) -> None:
    """
    Comprueba que el packing coincida con Vitis 2024.2.

    Parametros
    ----------
    y_onehot : np.ndarray
        Contiene las etiquetas en formato one-hot.
    pixel_values : np.ndarray
        Contiene los pixeles ya preparados para el modo activo.
    packed_dataset : np.ndarray
        Contiene el dataset ya empaquetado por words.
    pixel_bits : int
        Indica la cantidad de bits usada para almacenar cada pixel.
    image_width : int
        Indica el ancho de la imagen almacenada en el binario.
    image_height : int
        Indica el alto de la imagen almacenada en el binario.
    samples_to_check : int, optional
        Indica cuantas muestras se revisan durante la comprobacion.

    Retorna
    -------
    None
        No retorna ningun valor. Solo valida el empaquetado generado.
    """
    # Recupera el layout para interpretar la salida empaquetada.
    layout = build_layout(
        pixel_bits=pixel_bits,
        image_width=image_width,
        image_height=image_height,
    )

    # Reconstruye los bits utiles previos al padding.
    ff_bits = build_ff_bits_matrix(y_onehot, pixel_values, pixel_bits)

    # Limita la verificacion a unas pocas muestras para mantenerla rapida.
    effective_samples = min(samples_to_check, ff_bits.shape[0])

    for sample_idx in range(effective_samples):
        # Toma la muestra original antes del empaquetado fisico.
        sample_bits = ff_bits[sample_idx]

        for word_idx in range(int(layout["words_per_sample"])):
            # Calcula el tramo de bits que le corresponde a la word actual.
            start_bit = word_idx * PACK_BITS
            end_bit = min(start_bit + PACK_BITS, int(layout["useful_bits"]))
            expected_word = np.uint32(0)

            for bit_offset, bit_idx in enumerate(range(start_bit, end_bit)):
                # Coloca cada bit en la posicion LSB-first esperada por Vitis.
                expected_word |= np.uint32(
                    int(sample_bits[bit_idx]) << bit_offset
                )

            if packed_dataset[sample_idx, word_idx] != expected_word:
                raise ValueError(
                    "El packing no sigue la convencion little-endian "
                    "esperada por Vitis 2024.2."
                )

    # Informa que la comprobacion de endianess termino bien.
    print(
        f"Verificacion little-endian OK para modo {layout['suffix']} "
        f"({effective_samples} muestras comprobadas)."
    )


def save_packed_ff_dataset(
    output_path: str | Path,
    packed_dataset: np.ndarray,
    metadata: dict[str, int | str],
) -> Path:
    """
    Guarda la cabecera y el dataset empaquetado en un solo binario.

    Parametros
    ----------
    output_path : str | Path
        Indica la ruta donde se escribira el binario final.
    packed_dataset : np.ndarray
        Contiene el dataset ya empaquetado por words.
    metadata : dict[str, int | str]
        Contiene los metadatos que se guardaran en la cabecera.

    Retorna
    -------
    Path
        Retorna la ruta del archivo binario generado.
    """
    # Normaliza la ruta y crea el directorio si todavia no existe.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Genera la cabecera y aplana el payload para escribirlo de una sola vez.
    header_words = pack_binary_header(metadata)
    payload_words = packed_dataset.reshape(-1)
    file_words = np.concatenate([header_words, payload_words])

    # Escribe todas las words con el orden de bytes esperado por Vitis.
    with open(output_path, "wb") as file:
        file.write(np.asarray(file_words, dtype="<u4").tobytes())

    print(f"Guardado dataset FF empaquetado en: {output_path}")
    return output_path


def save_numpy_debug(
    output_dir: Path,
    prefix: str,
    pixels: np.ndarray,
    y_onehot: np.ndarray,
    packed_dataset: np.ndarray,
) -> None:
    """
    Guarda archivos intermedios .npy para depuracion.

    Parametros
    ----------
    output_dir : Path
        Indica el directorio donde se guardaran los archivos auxiliares.
    prefix : str
        Indica el prefijo comun usado por los archivos de depuracion.
    pixels : np.ndarray
        Contiene los pixeles ya preparados para el modo activo.
    y_onehot : np.ndarray
        Contiene las etiquetas en formato one-hot.
    packed_dataset : np.ndarray
        Contiene el dataset ya empaquetado por words.

    Retorna
    -------
    None
        No retorna ningun valor. Solo escribe los archivos de depuracion.
    """
    # Guarda los tensores intermedios con un prefijo comun.
    np.save(output_dir / f"{prefix}_pixels.npy", pixels)
    np.save(output_dir / f"{prefix}_yonehot.npy", y_onehot)
    np.save(output_dir / f"{prefix}_packed.npy", packed_dataset)


def build_variant_artifacts(
    output_dir: Path,
    y_onehot: np.ndarray,
    pixel_values: np.ndarray,
    pixel_bits: int,
    image_width: int,
    image_height: int,
    source_width: int = IMG_SIZE,
    source_height: int = IMG_SIZE,
) -> dict[str, object]:
    """
    Construye y guarda una variante concreta del dataset.

    Parametros
    ----------
    output_dir : Path
        Indica el directorio donde se guardaran los artefactos.
    y_onehot : np.ndarray
        Contiene las etiquetas en formato one-hot.
    pixel_values : np.ndarray
        Contiene los pixeles ya preparados para el modo activo.
    pixel_bits : int
        Indica la cantidad de bits usada para almacenar cada pixel.
    image_width : int
        Indica el ancho de la imagen almacenada en el binario.
    image_height : int
        Indica el alto de la imagen almacenada en el binario.
    source_width : int, optional
        Indica el ancho original de la imagen antes del redimensionado.
    source_height : int, optional
        Indica el alto original de la imagen antes del redimensionado.

    Retorna
    -------
    dict[str, object]
        Retorna los artefactos principales generados para la variante.
    """
    # Obtiene el layout que corresponde al modo actual.
    layout = build_layout(
        pixel_bits=pixel_bits,
        image_width=image_width,
        image_height=image_height,
    )

    # Construye los metadatos que se guardaran en la cabecera.
    metadata = build_binary_metadata(
        sample_count=pixel_values.shape[0],
        pixel_bits=pixel_bits,
        image_width=image_width,
        image_height=image_height,
        source_width=source_width,
        source_height=source_height,
    )

    # Empaqueta el dataset completo para el modo seleccionado.
    packed_dataset = pack_ff_dataset(
        y_onehot=y_onehot,
        pixel_values=pixel_values,
        pixel_bits=pixel_bits,
        image_width=image_width,
        image_height=image_height,
    )

    # Verifica que el packing siga la convencion little-endian esperada.
    verify_vitis_little_endian_compatibility(
        y_onehot=y_onehot,
        pixel_values=pixel_values,
        packed_dataset=packed_dataset,
        pixel_bits=pixel_bits,
        image_width=image_width,
        image_height=image_height,
    )

    # Guarda el binario principal con su cabecera y su nombre canonico.
    packed_path = save_packed_ff_dataset(
        output_path=output_dir / get_default_binary_name(
            pixel_bits=pixel_bits,
            image_width=image_width,
            image_height=image_height,
        ),
        packed_dataset=packed_dataset,
        metadata=metadata,
    )

    # Guarda tambien los archivos auxiliares de depuracion.
    save_numpy_debug(
        output_dir=output_dir,
        prefix=(
            f"mnist_{image_width}x{image_height}_{layout['suffix']}"
        ),
        pixels=pixel_values,
        y_onehot=y_onehot,
        packed_dataset=packed_dataset,
    )

    # Imprime un resumen corto del layout generado.
    print(
        f"Layout {layout['suffix']}: useful_bits={layout['useful_bits']}, "
        f"words_per_sample={layout['words_per_sample']}, "
        f"padding_bits={layout['padding_bits']}"
    )
    print(
        f"Modo almacenado: {layout['mode_name']} "
        f"con resolucion {image_width}x{image_height}"
    )

    return {
        "layout": layout,
        "metadata": metadata,
        "packed_dataset": packed_dataset,
        "packed_path": packed_path,
    }
