# =============================== Librerías ================================= #
from pathlib import Path            # Manejo limpio de rutas de archivos
import numpy as np                  # Cálculo numérico eficiente
import pandas as pd                 # Lectura y manejo de tablas
import matplotlib.pyplot as plt     # Visualización de imágenes


# =========================== Variables globales ============================ #
IMG_SIZE = 28
# Tamaño lateral de la imagen MNIST: 28x28

NUM_PIXELS = IMG_SIZE * IMG_SIZE
# Total de píxeles por imagen: 784

NUM_CLASSES = 10
# Número de clases en MNIST: dígitos del 0 al 9

FF_INPUT_BITS = NUM_PIXELS + NUM_CLASSES
# Entrada total FF: 784 bits de imagen + 10 bits one-hot = 794 bits

PACK_BITS = 32
# Cantidad de bits por palabra de empaquetado

PIX_PACK_WORDS = (NUM_PIXELS + PACK_BITS - 1) // PACK_BITS
# Número de palabras de 32 bits necesarias para 784 bits

FF_PACK_WORDS = (FF_INPUT_BITS + PACK_BITS - 1) // PACK_BITS
# Número de palabras de 32 bits necesarias para 794 bits
# En este caso deben ser 25 palabras


# =============================== Funciones ================================= #
def load_train_table(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga un archivo MNIST con formato:
    label, pixel0, pixel1, ..., pixel783

    Args:
        path: Ruta del archivo de entrada.

    Returns:
        X: Matriz de píxeles con forma (N, 784).
        y: Vector de etiquetas con forma (N,).
    """
    path = Path(path)
    # Convierte el texto de ruta a objeto Path

    df = pd.read_csv(path)
    # Lee el archivo CSV completo

    if "label" not in df.columns:
        # Verifica que exista la columna de etiquetas
        raise ValueError("No se encontró la columna 'label'.")

    pixel_cols = [col for col in df.columns if col.startswith("pixel")]
    # Obtiene todas las columnas cuyos nombres empiezan con "pixel"

    if len(pixel_cols) != NUM_PIXELS:
        # Verifica que existan exactamente 784 columnas de píxeles
        raise ValueError(
            f"Se esperaban {NUM_PIXELS} columnas de píxeles y "
            f"se encontraron {len(pixel_cols)}."
        )

    X = df[pixel_cols].to_numpy()
    # Convierte las columnas de píxeles a una matriz NumPy

    y = df["label"].to_numpy()
    # Convierte la columna label a un vector NumPy

    return X, y
    # Devuelve la matriz de imágenes y el vector de etiquetas


def validate_dataset(X: np.ndarray, y: np.ndarray) -> None:
    """
    Verifica que el dataset tenga forma y rangos válidos.

    Args:
        X: Matriz de píxeles con forma esperada (N, 784).
        y: Vector de etiquetas con forma esperada (N,).
    """
    if X.ndim != 2 or X.shape[1] != NUM_PIXELS:
        # Revisa que X sea una matriz 2D con 784 columnas
        raise ValueError(
            f"X debe tener forma (N, {NUM_PIXELS}). Se obtuvo {X.shape}."
        )

    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        # Revisa que y sea un vector y coincida en cantidad de muestras
        raise ValueError(
            "Las etiquetas no coinciden con la cantidad de muestras."
        )

    if X.min() < 0 or X.max() > 255:
        # Verifica que los píxeles estén entre 0 y 255
        raise ValueError("Los píxeles deben estar en el rango [0, 255].")

    if y.min() < 0 or y.max() > 9:
        # Verifica que las etiquetas estén entre 0 y 9
        raise ValueError("Las etiquetas deben estar en el rango [0, 9].")

    print("Dataset válido.")
    # Imprime confirmación de validación

    print(f"Muestras: {X.shape[0]}")
    # Imprime la cantidad de muestras

    print(f"Dimensión por muestra: {X.shape[1]}")
    # Imprime la dimensión por muestra

    print(f"Clases presentes: {np.unique(y)}")
    # Imprime las clases distintas presentes

    print(f"Rango de píxeles: [{X.min()}, {X.max()}]")
    # Imprime el rango de valores de píxeles


def show_binarization_samples(
    X_original: np.ndarray,
    X_binary: np.ndarray,
    num_samples: int = 5,
) -> None:
    """
    Muestra casos aleatorios comparando imagen original vs binarizada.

    Args:
        X_original: Matriz original de píxeles (N, 784).
        X_binary: Matriz binarizada (N, 784).
        num_samples: Cantidad de muestras a mostrar.
    """
    indices = np.random.choice(X_original.shape[0], num_samples, replace=False)
    # Selecciona índices aleatorios sin repetición

    plt.figure(figsize=(10, 4))
    # Crea la figura donde se mostrarán las comparaciones

    for i, idx in enumerate(indices):
        # Recorre cada índice seleccionado

        original_img = X_original[idx].reshape(28, 28)
        # Reconstruye la imagen original en formato 28x28

        binary_img = X_binary[idx].reshape(28, 28)
        # Reconstruye la imagen binarizada en formato 28x28

        plt.subplot(2, num_samples, i + 1)
        # Selecciona la posición del subplot superior

        plt.imshow(original_img, cmap="gray")
        # Muestra la imagen original

        plt.title(f"Original #{idx}")
        # Coloca el título correspondiente

        plt.axis("off")
        # Oculta ejes en la imagen original

        plt.subplot(2, num_samples, i + 1 + num_samples)
        # Selecciona la posición del subplot inferior

        plt.imshow(binary_img, cmap="gray")
        # Muestra la imagen binarizada

        plt.title("Binarizada")
        # Coloca el título correspondiente

        plt.axis("off")
        # Oculta ejes en la imagen binarizada

    plt.suptitle("Comparación de binarización")
    # Coloca un título general a la figura

    plt.tight_layout()
    # Ajusta el espaciado interno de la figura

    plt.show()
    # Muestra la figura en pantalla


def binarize_pixels(
    X: np.ndarray,
    threshold: int = 127,
    show_process: bool = False,
) -> np.ndarray:
    """
    Binariza los píxeles usando un umbral.

    Si pixel >= threshold, devuelve 1.
    Si pixel < threshold, devuelve 0.

    Args:
        X: Matriz original de píxeles con forma (N, 784).
        threshold: Umbral de binarización.
        show_process: Si es True, muestra comparación visual.

    Returns:
        Matriz binaria con valores 0 o 1.
    """
    X_bin = (X >= threshold).astype(np.uint8)
    # Convierte cada píxel a 0 o 1 según el umbral

    if show_process:
        # Si se solicita visualización, se muestran ejemplos
        show_binarization_samples(X, X_bin, num_samples=5)

    return X_bin
    # Devuelve la matriz binarizada


def labels_to_onehot(
    y: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """
    Convierte etiquetas escalares a codificación one-hot.

    Args:
        y: Vector de etiquetas.
        num_classes: Número de clases.

    Returns:
        Matriz one-hot con forma (N, num_classes).
    """
    onehot = np.zeros((y.shape[0], num_classes), dtype=np.uint8)
    # Crea una matriz de ceros de tamaño N x num_classes

    onehot[np.arange(y.shape[0]), y] = 1
    # En cada fila activa el bit correspondiente a la etiqueta

    return onehot
    # Devuelve la matriz one-hot


def concatenate_ff_input(
    X_bin: np.ndarray,
    y_onehot: np.ndarray,
    onehot_position: str = "prefix",
) -> np.ndarray:
    """
    Concatena imagen binaria y etiqueta one-hot.

    Estructuras válidas:
        prefix: [10 bits one-hot || 784 bits imagen]
        suffix: [784 bits imagen || 10 bits one-hot]

    Args:
        X_bin: Matriz binaria de imágenes.
        y_onehot: Matriz de etiquetas one-hot.
        onehot_position: Posición del one-hot en la entrada.

    Returns:
        Matriz concatenada con forma (N, 794).
    """
    if X_bin.shape[0] != y_onehot.shape[0]:
        # Revisa que ambas matrices tengan la misma cantidad de muestras
        raise ValueError(
            "X_bin e y_onehot no tienen la misma cantidad de muestras."
        )

    if onehot_position not in {"prefix", "suffix"}:
        # Verifica que la opción elegida sea válida
        raise ValueError(
            "onehot_position debe ser 'prefix' o 'suffix'."
        )

    if onehot_position == "prefix":
        # Si el one-hot va primero, se concatena antes de la imagen
        return np.concatenate([y_onehot, X_bin], axis=1).astype(np.uint8)

    return np.concatenate([X_bin, y_onehot], axis=1).astype(np.uint8)
    # Si el one-hot va al final, la imagen va primero y la etiqueta después


def pack_bits_row(
    row_bits: np.ndarray,
    pack_bits: int = 32,
) -> np.ndarray:
    """
    Empaqueta un vector binario en palabras de 32 bits.

    El bit i de la entrada se coloca en:
    - palabra: i // 32
    - posición dentro de la palabra: i % 32

    Args:
        row_bits: Vector binario de entrada.
        pack_bits: Cantidad de bits por palabra.

    Returns:
        Vector empaquetado en uint32.
    """
    n_words = (len(row_bits) + pack_bits - 1) // pack_bits
    # Calcula cuántas palabras se necesitan para almacenar todos los bits

    packed = np.zeros(n_words, dtype=np.uint32)
    # Reserva el arreglo de salida inicializado en cero

    for i, bit in enumerate(row_bits):
        # Recorre cada bit del vector original

        if bit:
            # Solo escribe si el bit vale 1

            word_idx = i // pack_bits
            # Calcula en qué palabra cae el bit

            bit_idx = i % pack_bits
            # Calcula la posición del bit dentro de la palabra

            packed[word_idx] |= np.uint32(1 << bit_idx)
            # Activa ese bit en la palabra correspondiente

    return packed
    # Devuelve el vector empaquetado


def pack_bits_matrix(
    X_bin: np.ndarray,
    total_bits: int,
    pack_bits: int = 32,
) -> np.ndarray:
    """
    Empaqueta una matriz binaria fila por fila.

    Args:
        X_bin: Matriz binaria de entrada.
        total_bits: Número de bits esperados por fila.
        pack_bits: Cantidad de bits por palabra.

    Returns:
        Matriz empaquetada en uint32.
    """
    n_words = (total_bits + pack_bits - 1) // pack_bits
    # Calcula cuántas palabras necesita cada fila

    packed = np.zeros((X_bin.shape[0], n_words), dtype=np.uint32)
    # Reserva la matriz de salida

    for n in range(X_bin.shape[0]):
        # Recorre cada muestra

        packed[n] = pack_bits_row(X_bin[n], pack_bits=pack_bits)
        # Empaqueta la fila n y la guarda en la salida

    return packed
    # Devuelve toda la matriz empaquetada


def unpack_bits_row(
    packed_row: np.ndarray,
    total_bits: int,
    pack_bits: int = 32,
) -> np.ndarray:
    """
    Reconstruye el vector binario original a partir de palabras empaquetadas.

    Args:
        packed_row: Vector de palabras uint32.
        total_bits: Número de bits que se desean reconstruir.
        pack_bits: Cantidad de bits por palabra.

    Returns:
        Vector binario reconstruido (0/1).
    """
    bits = np.zeros(total_bits, dtype=np.uint8)
    # Reserva el vector de salida

    for i in range(total_bits):
        # Recorre cada bit a reconstruir

        word_idx = i // pack_bits
        # Identifica la palabra donde está el bit

        bit_idx = i % pack_bits
        # Identifica la posición dentro de la palabra

        bits[i] = (packed_row[word_idx] >> bit_idx) & 1
        # Extrae el bit correspondiente

    return bits
    # Devuelve el vector binario reconstruido


def verify_bit_packing(
    X_original: np.ndarray,
    X_packed: np.ndarray,
    total_bits: int,
    pack_bits: int = 32,
    num_samples: int = 10,
) -> None:
    """
    Verifica que el proceso pack -> unpack conserva los bits originales.

    Args:
        X_original: Matriz binaria original (N, bits).
        X_packed: Matriz empaquetada.
        total_bits: Número de bits por muestra.
        pack_bits: Tamaño de palabra.
        num_samples: Cantidad de muestras a verificar.
    """
    indices = np.random.choice(X_original.shape[0], num_samples, replace=False)
    # Selecciona muestras aleatorias para probar la reversibilidad

    errors = 0
    # Inicializa el contador de errores

    for idx in indices:
        # Recorre cada índice de prueba

        reconstructed = unpack_bits_row(
            X_packed[idx],
            total_bits=total_bits,
            pack_bits=pack_bits,
        )
        # Reconstruye la fila empaquetada

        if not np.array_equal(X_original[idx], reconstructed):
            # Si la reconstrucción no coincide, se reporta error
            print(f"❌ Error en muestra {idx}")
            errors += 1
        else:
            # Si coincide, se reporta éxito
            print(f"✔ Muestra {idx} correcta")

    if errors == 0:
        # Si no hubo errores, se informa verificación exitosa
        print("Verificación exitosa: todos los vectores coinciden.")
    else:
        # Si hubo errores, se informa cuántos se encontraron
        print(f"Se encontraron {errors} errores.")


def save_packed_pixels_dataset(
    output_path: str,
    X_pix_packed: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Guarda un archivo binario con esta estructura por muestra:
    [label: 1 byte][pixeles binarios empaquetados]

    Esta función se mantiene por compatibilidad para el dataset de píxeles
    separado, donde sí puede ser útil conservar la etiqueta escalar aparte.

    Args:
        output_path: Ruta del archivo binario de salida.
        X_pix_packed: Matriz de píxeles empaquetados.
        y: Vector de etiquetas.
    """
    output_path = Path(output_path)
    # Convierte la ruta a objeto Path

    with open(output_path, "wb") as file:
        # Abre el archivo en modo binario de escritura

        for label, packed_words in zip(y, X_pix_packed):
            # Recorre cada etiqueta junto con su muestra empaquetada

            file.write(np.uint8(label).tobytes())
            # Escribe la etiqueta como un byte sin signo

            file.write(np.asarray(packed_words, dtype="<u4").tobytes())
            # Escribe las palabras empaquetadas en formato uint32 little-endian

    print(f"Guardado dataset binario de píxeles en: {output_path}")
    # Informa la ruta de salida


def save_packed_ff_dataset(
    output_path: str,
    X_ff_packed: np.ndarray,
) -> None:
    """
    Guarda un archivo binario FF con esta estructura por muestra:

    [25 words uint32]

    Importante:
    - Ya NO se escribe un byte extra con la etiqueta escalar.
    - El one-hot ya viene embebido en los bits [0:9] de cada muestra.
    - Esto evita el corrimiento de 8 bits que afectaba al hardware.

    Args:
        output_path: Ruta del archivo binario de salida.
        X_ff_packed: Matriz FF empaquetada con forma (N, 25).
    """
    output_path = Path(output_path)
    # Convierte la ruta a objeto Path

    with open(output_path, "wb") as file:
        # Abre el archivo en modo binario de escritura

        for packed_words in X_ff_packed:
            # Recorre cada muestra empaquetada

            file.write(np.asarray(packed_words, dtype="<u4").tobytes())
            # Escribe directamente las 25 palabras uint32 en little-endian

    print(f"Guardado dataset FF empaquetado en: {output_path}")
    # Informa que el archivo fue creado


def save_numpy_debug(
    prefix: str,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    X_bin: np.ndarray | None = None,
    y_onehot: np.ndarray | None = None,
    X_ff: np.ndarray | None = None,
    X_pix_packed: np.ndarray | None = None,
    X_ff_packed: np.ndarray | None = None,
) -> None:
    """
    Guarda archivos .npy para depuración intermedia.

    Args:
        prefix: Prefijo base de nombres.
        X: Datos originales.
        y: Etiquetas originales.
        X_bin: Imágenes binarizadas.
        y_onehot: Etiquetas one-hot.
        X_ff: Entrada FF concatenada.
        X_pix_packed: Píxeles empaquetados.
        X_ff_packed: Entrada FF empaquetada.
    """
    prefix = Path(prefix)
    # Convierte el prefijo a Path para construir nombres fácilmente

    if X is not None:
        # Si se proporcionó X, se guarda en disco
        np.save(prefix.with_name(prefix.name + "_X.npy"), X)

    if y is not None:
        # Si se proporcionó y, se guarda en disco
        np.save(prefix.with_name(prefix.name + "_y.npy"), y)

    if X_bin is not None:
        # Si se proporcionó X_bin, se guarda en disco
        np.save(prefix.with_name(prefix.name + "_Xbin.npy"), X_bin)

    if y_onehot is not None:
        # Si se proporcionó y_onehot, se guarda en disco
        np.save(prefix.with_name(prefix.name + "_yonehot.npy"), y_onehot)

    if X_ff is not None:
        # Si se proporcionó X_ff, se guarda en disco
        np.save(prefix.with_name(prefix.name + "_Xff.npy"), X_ff)

    if X_pix_packed is not None:
        # Si se proporcionó X_pix_packed, se guarda en disco
        np.save(prefix.with_name(prefix.name + "_Xpixpacked.npy"),
                X_pix_packed)

    if X_ff_packed is not None:
        # Si se proporcionó X_ff_packed, se guarda en disco
        np.save(prefix.with_name(prefix.name + "_Xffpacked.npy"), X_ff_packed)

    print("Archivos .npy de depuración guardados.")
    # Informa que el guardado de depuración terminó
