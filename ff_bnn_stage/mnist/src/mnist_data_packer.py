# ================================ LIBRERÍAS ================================ #
from pathlib import Path
# Importa NumPy para el trabajo vectorial y el empaquetado del dataset.
import numpy as np
# Importa pandas para cargar el CSV de entrenamiento.
import pandas as pd

# ============================ VARIABLES GLOBALES =========================== #
# Define el tamaño lateral de la imagen MNIST.
IMG_SIZE = 28
# Define la cantidad total de pixeles por muestra.
NUM_PIXELS = IMG_SIZE * IMG_SIZE
# Define la cantidad de clases disponibles en MNIST.
NUM_CLASSES = 10
# Define el ancho de cada palabra del archivo binario.
PACK_BITS = 32


# ================================ FUNCIONES ================================ #
def build_layout(pixel_bits: int) -> dict[str, int]:
    """Construye el layout fisico del dataset para el modo solicitado."""
    if pixel_bits not in (1, 4):
        raise ValueError("pixel_bits debe ser 1 o 4.")

    # Calcula el tamaño útil, el tamaño físico y el padding por muestra.
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


def get_default_binary_name(pixel_bits: int) -> str:
    """Retorna el nombre canonico del binario segun el modo activo."""
    layout = build_layout(pixel_bits)
    return f"mnist_{layout['suffix']}_packed.bin"


def load_train_table(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga un archivo MNIST con el formato:
    label, pixel0, pixel1, ..., pixel783.
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
    """Verifica que el dataset tenga forma y rangos validos."""
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
    """Convierte etiquetas escalares a codificacion one-hot."""
    # Reserva una matriz de salida con una columna por clase.
    onehot = np.zeros((y.shape[0], num_classes), dtype=np.uint8)

    # Activa el bit de la clase correspondiente en cada fila.
    onehot[np.arange(y.shape[0]), y] = 1

    return onehot


def binarize_pixels_1bit(
    X: np.ndarray,
    threshold: int = 127,
) -> np.ndarray:
    """Binariza los pixeles usando un umbral fijo."""
    return (X >= threshold).astype(np.uint8)


def quantize_pixels_4bit(X: np.ndarray) -> np.ndarray:
    """Cuantiza los pixeles a 4 bits uniformes."""
    return (X >> 4).astype(np.uint8)


def build_ff_bits_matrix(
    y_onehot: np.ndarray,
    pixel_values: np.ndarray,
    pixel_bits: int,
) -> np.ndarray:
    """
    Construye la matriz de bits logicos con este orden:
    [10 bits one-hot][784 * pixel_bits].
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
            NUM_PIXELS * pixel_bits,
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
    chunk_size: int = 2048,
) -> np.ndarray:
    """Empaqueta el dataset completo en palabras uint32."""
    # Recupera el layout del modo activo para dimensionar la salida.
    layout = build_layout(pixel_bits)

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


def verify_vitis_little_endian_compatibility(
    y_onehot: np.ndarray,
    pixel_values: np.ndarray,
    packed_dataset: np.ndarray,
    pixel_bits: int,
    samples_to_check: int = 4,
) -> None:
    """Comprueba que el packing coincida con Vitis 2024.2."""
    # Recupera el layout para interpretar la salida empaquetada.
    layout = build_layout(pixel_bits)

    # Reconstruye los bits utiles previos al padding.
    ff_bits = build_ff_bits_matrix(y_onehot, pixel_values, pixel_bits)

    # Limita la verificacion a unas pocas muestras para mantenerla rapida.
    effective_samples = min(samples_to_check, ff_bits.shape[0])

    for sample_idx in range(effective_samples):
        # Toma la muestra original antes del empaquetado fisico.
        sample_bits = ff_bits[sample_idx]

        for word_idx in range(layout["words_per_sample"]):
            # Calcula el tramo de bits que le corresponde a la word actual.
            start_bit = word_idx * PACK_BITS
            end_bit = min(start_bit + PACK_BITS, layout["useful_bits"])
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
) -> Path:
    """Guarda el dataset empaquetado como words uint32 little-endian."""
    # Normaliza la ruta y crea el directorio si todavia no existe.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Escribe todas las words con el orden de bytes esperado por Vitis.
    with open(output_path, "wb") as file:
        file.write(np.asarray(packed_dataset, dtype="<u4").tobytes())

    print(f"Guardado dataset FF empaquetado en: {output_path}")
    return output_path


def save_numpy_debug(
    output_dir: Path,
    prefix: str,
    pixels: np.ndarray,
    y_onehot: np.ndarray,
    packed_dataset: np.ndarray,
) -> None:
    """Guarda archivos intermedios .npy para depuracion."""
    # Guarda los tensores intermedios con un prefijo comun.
    np.save(output_dir / f"{prefix}_pixels.npy", pixels)
    np.save(output_dir / f"{prefix}_yonehot.npy", y_onehot)
    np.save(output_dir / f"{prefix}_packed.npy", packed_dataset)


def build_variant_artifacts(
    output_dir: Path,
    y_onehot: np.ndarray,
    pixel_values: np.ndarray,
    pixel_bits: int,
) -> dict[str, object]:
    """Construye y guarda una variante concreta del dataset."""
    # Obtiene el layout que corresponde al modo actual.
    layout = build_layout(pixel_bits)

    # Empaqueta el dataset completo para el modo seleccionado.
    packed_dataset = pack_ff_dataset(
        y_onehot=y_onehot,
        pixel_values=pixel_values,
        pixel_bits=pixel_bits,
    )

    # Verifica que el packing siga la convencion little-endian esperada.
    verify_vitis_little_endian_compatibility(
        y_onehot=y_onehot,
        pixel_values=pixel_values,
        packed_dataset=packed_dataset,
        pixel_bits=pixel_bits,
    )

    # Guarda el binario principal con su nombre canonico.
    packed_path = save_packed_ff_dataset(
        output_path=output_dir / get_default_binary_name(pixel_bits),
        packed_dataset=packed_dataset,
    )

    # Guarda tambien los archivos auxiliares de depuracion.
    save_numpy_debug(
        output_dir=output_dir,
        prefix=f"mnist_{layout['suffix']}",
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

    return {
        "layout": layout,
        "packed_dataset": packed_dataset,
        "packed_path": packed_path,
    }
