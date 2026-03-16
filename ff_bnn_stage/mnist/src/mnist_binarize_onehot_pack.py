# =============================== Librerías ================================= #
from pathlib import Path            # Manejo limpio de rutas de archivos
import struct                       # Empaquetado binario de datos
import numpy as np                  # Cálculo numérico eficiente
import pandas as pd                 # Lectura y manejo de tablas
import matplotlib.pyplot as plt     # Visualización de imagenes


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
def load_train_table(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga un archivo MNISTcon formato:
    label, pixel0, pixel1, ..., pixel783

    Args:
        path: Ruta del archivo de entrada.

    Returns:
        X: Matriz de píxeles con forma (N, 784).
        y: Vector de etiquetas con forma (N,).
    """
    path = Path(path)       # Convierte el texto de ruta a objeto Path
    df = pd.read_csv(path)  # Lee el archivo .csv

    if "label" not in df.columns:
        # Verifica que exista la columna de etiquetas
        raise ValueError("No se encontró la columna 'label'.")

    # Obtiene todas las columnas cuyos nombres empiezan con "pixel"
    pixel_cols = [col for col in df.columns if col.startswith("pixel")]

    if len(pixel_cols) != NUM_PIXELS:
        # Verifica que existan exactamente 784 columnas de píxeles
        raise ValueError(
            f"Se esperaban {NUM_PIXELS} columnas de píxeles y "
            f"se encontraron {len(pixel_cols)}."
        )

    # Convierte las columnas a una matriz NumPy
    X = df[pixel_cols].to_numpy()
    y = df["label"].to_numpy()

    # Devuelve la matriz de imágenes y el vector de etiquetas
    return X, y


def validate_dataset(X: np.ndarray, y: np.ndarray) -> None:
    """
    Verifica que el dataset tenga forma y rangos válidos.

    Args:
        X: Matriz de píxeles con forma esperada (N, 784).
        y: Vector de etiquetas con forma esperada (N,).
    """
    if X.ndim != 2 or X.shape[1] != NUM_PIXELS:
        # Revisa que X sea una matriz 2D con 784 columnas
        raise ValueError(f"X debe tener forma (N, {NUM_PIXELS}). "
                         f"Se obtuvo {X.shape}.")

    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        # Revisa que y sea un vector y coincida en cantidad de muestras
        raise ValueError("Las etiquetas no coinciden con la"
                         " cantidad de muestras.")

    if X.min() < 0 or X.max() > 255:
        # Verifica que los píxeles estén entre 0 y 255
        raise ValueError("Los píxeles deben estar en el rango [0, 255].")

    if y.min() < 0 or y.max() > 9:
        # Verifica que las etiquetas estén entre 0 y 9
        raise ValueError("Las etiquetas deben estar en el rango [0, 9].")

    # Imprime un resumen del dataset validado
    print("Dataset válido.")
    print(f"Muestras: {X.shape[0]}")
    print(f"Dimensión por muestra: {X.shape[1]}")
    print(f"Clases presentes: {np.unique(y)}")
    print(f"Rango de píxeles: [{X.min()}, {X.max()}]")


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

    # Selecciona índices aleatorios sin repetición
    indices = np.random.choice(X_original.shape[0], num_samples, replace=False)

    plt.figure(figsize=(10, 4))

    for i, idx in enumerate(indices):

        original_img = X_original[idx].reshape(28, 28)
        binary_img = X_binary[idx].reshape(28, 28)

        # Imagen original
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(original_img, cmap="gray")
        plt.title(f"Original #{idx}")
        plt.axis("off")

        # Imagen binarizada
        plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.imshow(binary_img, cmap="gray")
        plt.title("Binarizada")
        plt.axis("off")

    plt.suptitle("Comparación de binarización")
    plt.tight_layout()
    plt.show()


def binarize_pixels(X: np.ndarray, threshold: int = 127,
                    show_process: bool = False,
                    sample_idx: int = 0,) -> np.ndarray:
    """
    Binariza los píxeles usando un umbral.

    Si pixel >= threshold, devuelve 1.
    Si pixel < threshold, devuelve 0.

    Además, si show_process=True, muestra una comparación visual entre la
    imagen original y la binarizada para una muestra seleccionada.

    Args:
        X: Matriz original de píxeles con forma (N, 784).
        threshold: Umbral de binarización.
        show_process: Si es True, muestra comparación visual.
        sample_idx: Índice de la muestra a visualizar.

    Returns:
        Matriz binaria con valores 0 o 1.
    """
    # Matriz binaria donde cada píxel se convierte a 0 o 1 según el umbral
    X_bin = (X >= threshold).astype(np.uint8)

    # Si se solicita, mostrar comparación visual
    if show_process:
        show_binarization_samples(X, X_bin, num_samples=5)

    return X_bin


def labels_to_onehot(y: np.ndarray,
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
    # Crea una matriz de ceros de tamaño N x num_classes
    onehot = np.zeros((y.shape[0], num_classes), dtype=np.uint8)

    # En cada fila, coloca un 1 en la posición correspondiente a la etiqueta
    onehot[np.arange(y.shape[0]), y] = 1

    # Devuelve la matriz codificada en one-hot
    return onehot


def concatenate_ff_input(X_bin: np.ndarray, y_onehot: np.ndarray,
                         onehot_position: str = "prefix",) -> np.ndarray:
    """
    Concatena imagen binaria y etiqueta one-hot.

    Estructura final:
    [784 bits de imagen || 10 bits one-hot]

    Args:
        X_bin: Matriz binaria de imágenes.
        y_onehot: Matriz de etiquetas one-hot.
        onehot_position: Posición del one-hot en la entrada.
            Valores válidos:
                - "prefix"
                - "suffix"

    Returns:
        Matriz concatenada con forma (N, 794).
    """
    if X_bin.shape[0] != y_onehot.shape[0]:
        # Revisa que ambas matrices tengan el mismo número de muestras
        raise ValueError(
            "X_bin e y_onehot no tienen la misma cantidad de muestras.")

    if onehot_position not in {"prefix", "suffix"}:
        raise ValueError(
            "onehot_position debe ser 'prefix' o 'suffix'."
        )

    if onehot_position == "prefix":
        return np.concatenate([y_onehot, X_bin], axis=1).astype(np.uint8)

    # Une la imagen y la etiqueta codificada
    return np.concatenate([y_onehot, X_bin], axis=1).astype(np.uint8)


def pack_bits_row(row_bits: np.ndarray, pack_bits: int = 32) -> np.ndarray:
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

            packed[word_idx] |= (1 << bit_idx)
            # Activa ese bit en la palabra correspondiente

    return packed
    # Devuelve el vector empaquetado


def pack_bits_matrix(X_bin: np.ndarray,
                     total_bits: int,
                     pack_bits: int = 32,) -> np.ndarray:
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


def unpack_bits_row(packed_row: np.ndarray, total_bits: int,
                    pack_bits: int = 32,) -> np.ndarray:
    """
    Reconstruye el vector binario original a partir de palabras empaquetadas.

    Args:
        packed_row: Vector de palabras uint32.
        total_bits: Número de bits que se desean reconstruir.
        pack_bits: Cantidad de bits por palabra (por defecto 32).

    Returns:
        Vector binario reconstruido (0/1).
    """

    # Vector de salida
    bits = np.zeros(total_bits, dtype=np.uint8)

    for i in range(total_bits):

        # Identifica la palabra donde está el bit
        word_idx = i // pack_bits

        # Identifica la posición dentro de la palabra
        bit_idx = i % pack_bits

        # Extrae el bit
        bits[i] = (packed_row[word_idx] >> bit_idx) & 1

    return bits


def verify_bit_packing(X_original: np.ndarray, X_packed: np.ndarray,
                       total_bits: int, pack_bits: int = 32,
                       num_samples: int = 10,) -> None:
    """
    Verifica que el proceso pack → unpack conserva los bits originales.

    Selecciona imágenes aleatorias y compara.

    Args:
        X_original: Matriz binaria original (N, bits).
        X_packed: Matriz empaquetada.
        total_bits: Número de bits por muestra.
        pack_bits: Tamaño de palabra.
        num_samples: Cantidad de muestras a verificar.
    """

    indices = np.random.choice(X_original.shape[0], num_samples, replace=False)

    errors = 0

    for idx in indices:

        reconstructed = unpack_bits_row(
            X_packed[idx],
            total_bits=total_bits,
            pack_bits=pack_bits,
        )

        if not np.array_equal(X_original[idx], reconstructed):

            print(f"❌ Error en muestra {idx}")
            errors += 1

        else:
            print(f"✔ Muestra {idx} correcta")

    if errors == 0:
        print("Verificación exitosa: todos los vectores coinciden.")
    else:
        print(f"Se encontraron {errors} errores.")


def save_packed_pixels_dataset(output_path: str,
                               X_pix_packed: np.ndarray,
                               y: np.ndarray,) -> None:
    """
    Guarda un archivo binario con esta estructura por muestra:
    [label: 1 byte][pixeles binarios empaquetados]

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
            file.write(struct.pack("B", int(label)))
            # Escribe la etiqueta como 1 byte sin signo

            file.write(packed_words.astype(np.uint32).tobytes())
            # Escribe las palabras empaquetadas como bytes

    print(f"Guardado dataset binario de píxeles en: {output_path}")


def save_packed_ff_dataset(output_path: str,
                           X_ff_packed: np.ndarray,
                           y: np.ndarray,) -> None:
    """
    Guarda un archivo binario con esta estructura por muestra:
    [label: 1 byte][entrada FF empaquetada]

    Args:
        output_path: Ruta del archivo binario de salida.
        X_ff_packed: Matriz FF empaquetada.
        y: Vector de etiquetas.
    """
    output_path = Path(output_path)
    # Convierte la ruta a objeto Path

    with open(output_path, "wb") as file:
        # Abre el archivo en modo binario de escritura
        for label, packed_words in zip(y, X_ff_packed):
            # Recorre cada etiqueta junto con su entrada FF empaquetada
            file.write(struct.pack("B", int(label)))
            # Escribe la etiqueta como 1 byte

            file.write(packed_words.astype(np.uint32).tobytes())
            # Escribe las palabras de la entrada FF como bytes

    print(f"Guardado dataset FF empaquetado en: {output_path}")


def save_numpy_debug(prefix: str,
                     X: np.ndarray | None = None,
                     y: np.ndarray | None = None,
                     X_bin: np.ndarray | None = None,
                     y_onehot: np.ndarray | None = None,
                     X_ff: np.ndarray | None = None,
                     X_pix_packed: np.ndarray | None = None,
                     X_ff_packed: np.ndarray | None = None,) -> None:
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
        np.save(prefix.with_name(prefix.name + "_X.npy"),
                X)
        # Guarda la matriz original

    if y is not None:
        np.save(prefix.with_name(prefix.name + "_y.npy"),
                y)
        # Guarda las etiquetas originales

    if X_bin is not None:
        np.save(prefix.with_name(prefix.name + "_Xbin.npy"),
                X_bin)
        # Guarda la versión binarizada

    if y_onehot is not None:
        np.save(prefix.with_name(prefix.name + "_yonehot.npy"),
                y_onehot)
        # Guarda la versión one-hot

    if X_ff is not None:
        np.save(prefix.with_name(prefix.name + "_Xff.npy"),
                X_ff)
        # Guarda la entrada concatenada FF

    if X_pix_packed is not None:
        np.save(prefix.with_name(prefix.name + "_Xpixpacked.npy"),
                X_pix_packed)
        # Guarda los píxeles empaquetados

    if X_ff_packed is not None:
        np.save(prefix.with_name(prefix.name + "_Xffpacked.npy"),
                X_ff_packed)
        # Guarda la entrada FF empaquetada

    print("Archivos .npy de depuración guardados.")
