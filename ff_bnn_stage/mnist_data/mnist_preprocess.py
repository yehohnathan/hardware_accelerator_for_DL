# ============================= mnist_preprocess ============================ #
# Versión para Forward-Forward + BNN: Preprocesamiento de mnist
# 1) Carga MNIST (train) en [0,1] con forma (N,28,28)
# 2) (Opcional) Binariza con umbral 0.5 a {-1,+1} (parámetro booleano)
# 3) Aplana imágenes a vectores (N, d)
# 4) Redimensiona 28x28 a menor dimensión si se solicita
# 5) Valida tamaño uniforme
# 6) Divide en entrenamiento y validación
# 7) Permite seleccionar cantidad de muestras por dígito
#
# Nota: todo se mantiene en memoria y en numpy, sin floats en FPGA.
# ================================ LIBRERÍAS ================================ #
from __future__ import annotations  # Anotaciones como string
from typing import Tuple, Optional  # Colocar dentro de parámetros en funciones
import numpy as np          # Ayuda para convertir de imagenes a tensores
from PIL import Image       # Ayuda en la manipulación de imagenes
from typing import Iterable

try:                        # Importa el dataset de MNIST
    from tensorflow.keras.datasets import mnist  # type: ignore
except Exception:  # pragma: no cover
    mnist = None


# ================================ FUNCIONES ================================ #
def load_mnist_train() -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga MNIST train y normaliza a [0,1], forma (N,28,28).
    """
    # Detecta si no hay mnist cargado en el entorno
    if mnist is None:
        raise RuntimeError("Keras MNIST no disponible en este entorno.")

    # Carga los datos y los normaliza
    (x_tr, y_tr), _ = mnist.load_data()
    x_tr = x_tr.astype(np.float32) / 255.0
    y_tr = y_tr.astype(np.uint8)

    # Retorna las tuplas con la información de mnist
    return x_tr, y_tr


def select_per_digit(x: np.ndarray,
                     y: np.ndarray,
                     n_per_digit: Optional[int],
                     seed: int = 123) -> Tuple[np.ndarray, np.ndarray]:
    """
    Toma la cantidad de muestras por digito (si es None, no recorta).
    """
    # No hay nada con lo que trabajar, se retorna
    if n_per_digit is None:
        return x, y

    # No se puede una cantidad negativa de muestras por dígito
    if n_per_digit <= 0:
        raise ValueError("n_per_digit debe ser > 0.")

    # Generador de números aleatorios fijado
    rng = np.random.default_rng(seed)
    # Lista donde se guarda el índice de las imagenes seleccionadas
    idx_sel = []

    # Itera del 0 al 9, buscando los índice de todas las imagenes
    for k in range(10):
        idx_k = np.where(y == k)[0]
        rng.shuffle(idx_k)
        take = min(n_per_digit, idx_k.size)
        idx_sel.extend(idx_k[:take])
    idx_sel = np.array(idx_sel, dtype=np.int64)

    # Retorna las tuplas con imagenes `x` y sus etiquetas `y`, filtrados y
    # balanceados
    return x[idx_sel], y[idx_sel]


def resize_images(x: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    Redimensiona imágenes (N, H, W) en [0,1] a (N, rows, cols).
    """
    # Si la nueva resolución (n_H, n_W) es igual que la original (H, W), no
    # hace falta procesar nada
    if rows == x.shape[1] and cols == x.shape[2]:
        return x

    # Reserva memoria para el dataset redimencionado, que tendrá la misma
    # cantidad de imagenes pero con tamaño (H, W)
    re_data = np.empty((x.shape[0], rows, cols), dtype=np.float32)

    # Cada imagen de x se va redimencionando y cambiando el tipo de dato.
    for i in range(x.shape[0]):
        img = (x[i] * 255.0).astype(np.uint8)
        pil = Image.fromarray(img, mode="L")
        pil = pil.resize((cols, rows), resample=Image.BILINEAR)
        re_data[i] = np.asarray(pil, dtype=np.float32) / 255.0

    # Retorna el conjunto de imagenes redimencionadas [0, 1]
    return re_data


def validate_uniform_size(x: np.ndarray) -> Tuple[int, int]:
    """
    Valida que todas las imágenes compartan (H,W).
    """
    if x.ndim != 3:
        raise ValueError("x debe tener forma (N, H, W).")
    h, w = x.shape[1], x.shape[2]

    # En numpy, un mismo array ya es uniforme; incluimos chequeo explícito
    if not (x[:, :].shape[1] == h and x[:, :].shape[2] == w):
        raise RuntimeError("Las imágenes deben tener el mismo tamaño.")

    return h, w


def flatten_images(x: np.ndarray) -> np.ndarray:
    """
    Aplana (N, H, W) -> (N, H*W) en orden fila-columna.

    Gracias a este pequeño código, convierte un tensor, donde cada imagen es
    una matriz 2D, a un tensor donde cada imagen es lista.
    """
    n, h, w = x.shape
    return x.reshape(n, h * w)


def binarize_pm1(x_flat: np.ndarray,
                 threshold: float = 0.5) -> np.ndarray:
    """
    Binariza a {-1,+1} usando umbral en [0,1]. Devuelve int8.

    El problema es que no se puede trabajar con datos que van de [0, 1], dado
    que calcular la "bondad" se vuelve más consistente (y más simples) si las
    activaciones binarias {-1,+1}.

    Nota: Averiguar qué hacer para usar datos [0, 1], pero, de todas formas
    con CNN no se puede recurir a datos tan simples como [0, 1].
    """
    return np.where(x_flat >= threshold, 1, -1).astype(np.int8)


def train_valid_split(x: np.ndarray,
                      y: np.ndarray,
                      frac_train: float = 0.8,
                      seed: int = 123) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                                Tuple[np.ndarray, np.ndarray]]:
    """
    Baraja y divide (train, valid) según fracción.
    """
    # Se asegura que la división de datos este entre 0 y 1.
    if not (0.0 < frac_train < 1.0):
        raise ValueError("frac_train debe estar en (0,1).")

    # Divisón en un conjunto de entrenamiento y otro de validación
    n = x.shape[0]                      # Tiene la cantidad de samples totales
    rng = np.random.default_rng(seed)   # Semilla aleatoria fija
    idx = np.arange(n)                  # Genera indices de 0 a n
    rng.shuffle(idx)                    # Pone en orden aleatorio los indices
    k = int(np.floor(n * frac_train))   # Número de samples que son de train
    tr, va = idx[:k], idx[k:]           # entranamiento, validación

    # Retorna la tupla
    return (x[tr], y[tr]), (x[va], y[va])


def _ascii_art_from_vec(vec: np.ndarray,
                        rows: int,
                        cols: int,
                        charset: Optional[str] = None,
                        as_ascii: bool = True) -> list[str]:
    """
    Convierte un vector (rows*cols,) en una cuadrícula de líneas (strings).

    - Si as_ascii=True, genera arte ASCII.
      * {-1,+1} -> 2 niveles por defecto.
      * [0,1]   -> rampa de 10 niveles.
    - Si as_ascii=False, imprime números:
      * enteros con ancho 3 (incluye -1/+1 de BNN).
      * flotantes en [0,1] con 2 decimales.
    """
    mat = vec.reshape(rows, cols)

    # Modo numérico: imprime los valores tal cual, formateados por tipo.
    if not as_ascii:
        lines: list[str] = []
        if np.issubdtype(mat.dtype, np.integer):
            for r in range(rows):
                line = " ".join(f"{int(v):>3d}" for v in mat[r])
                lines.append(line)
        else:
            for r in range(rows):
                line = " ".join(f"{float(v):.2f}" for v in mat[r])
                lines.append(line)
        return lines

    # Modo ASCII: detecta binario {-1,+1} vs escala [0,1]
    is_int_bin = (np.issubdtype(mat.dtype, np.integer) and
                  np.min(mat) >= -1 and np.max(mat) <= 1)

    if charset is None:
        if is_int_bin:
            # Fondo ' ' y trazo '#'. Se usan 2 niveles (extremos).
            charset = " #'"
        else:
            # 10 niveles (oscuro->claro).
            charset = " .:-=+*#%@"

    nlev = len(charset)

    if is_int_bin:
        idx = (mat > 0).astype(np.int32)
        idx *= (nlev - 1)
    else:
        clipped = np.clip(mat, 0.0, 1.0)
        idx = (clipped * (nlev - 1) + 1e-6).astype(np.int32)

    lines: list[str] = []
    for r in range(rows):
        line = "".join(charset[idx[r, c]] for c in range(cols))
        lines.append(line)
    return lines


def print_samples_by_digit(x: np.ndarray,
                           y: np.ndarray,
                           per_digit: int = 2,
                           digits: Optional[Iterable[int]] = None,
                           rows: int = 14,
                           cols: int = 14,
                           split_name: str = "train",
                           charset: Optional[str] = None,
                           as_ascii: bool = True) -> None:
    """
    Imprime 'per_digit' muestras por dígito como ASCII o números.

    - x: (N, d) aplanado; d debe ser rows*cols
    - y: (N,) etiquetas 0..9
    - rows, cols: tamaño original de la imagen
    - as_ascii: True -> arte ASCII; False -> valores numéricos
    """
    if x.ndim != 2 or x.shape[1] != rows * cols:
        msg = (f"x debe ser (N, {rows*cols}). Recibido {x.shape}. "
               "Asegura rows/cols correctos y que x esté aplanado.")
        raise ValueError(msg)

    if digits is None:
        digits = range(10)

    print(f"[{split_name}] x: {x.shape}, y: {y.shape}, dtype: {x.dtype}")
    for k in digits:
        idx_k = np.where(y == k)[0]
        if idx_k.size == 0:
            print(f"  dígito {k}: (sin muestras)")
            continue
        take = min(per_digit, idx_k.size)
        print(f"  dígito {k}: mostrando {take} de {idx_k.size}")
        for j in range(take):
            i = int(idx_k[j])
            art = _ascii_art_from_vec(x[i], rows, cols, charset, as_ascii)
            print(f"    idx={i:6d}  label={int(y[i])}")
            for line in art:
                print("    " + line)


def print_train_valid_samples(xy_train: tuple[np.ndarray, np.ndarray],
                              xy_valid: tuple[np.ndarray, np.ndarray],
                              per_digit: int = 2,
                              digits: Optional[Iterable[int]] = None,
                              rows: int = 14,
                              cols: int = 14,
                              charset: Optional[str] = None,
                              as_ascii: bool = True) -> None:
    """
    Imprime muestras de train y valid como ASCII o números.
    """
    x_tr, y_tr = xy_train
    x_va, y_va = xy_valid
    print_samples_by_digit(x_tr, y_tr, per_digit, digits, rows, cols,
                           split_name="train", charset=charset,
                           as_ascii=as_ascii)
    print_samples_by_digit(x_va, y_va, per_digit, digits, rows, cols,
                           split_name="valid", charset=charset,
                           as_ascii=as_ascii)


# ========================= Clase: mnist_preprocess ========================= #
def mnist_preprocess(rows: int = 28,
                     cols: int = 28,
                     n_per_digit: Optional[int] = None,
                     binarize: bool = False,
                     frac_train: float = 0.8,
                     seed: int = 123
                     ) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                Tuple[np.ndarray, np.ndarray]]:
    """Tareas que realiza:
    - Carga MNIST (train) en [0,1] -> (N,28,28)
    - (opcional) binariza a {-1,+1} con umbral 0.5
    - Redimensiona a (rows, cols)
    - Valida tamaño, aplana -> (N, rows*cols)
    - Divide en train/valid
    - Permite n_per_digit
    """
    x, y = load_mnist_train()
    x, y = select_per_digit(x, y, n_per_digit, seed)
    x = resize_images(x, rows, cols)
    validate_uniform_size(x)
    x = flatten_images(x)  # (N, rows*cols)

    if binarize:
        x = binarize_pm1(x)  # int8 en {-1,+1}
    else:
        x = x.astype(np.float32)  # float32 en [0,1]

    return train_valid_split(x, y, frac_train=frac_train, seed=seed)


# =================================== MAIN ================================== #
if __name__ == "__main__":
    # Ejemplo: 100 por dígito, 14x14, binarizado {-1,+1}
    (x_tr, y_tr), (x_va, y_va) = mnist_preprocess(
        rows=20, cols=20, n_per_digit=100, binarize=True, frac_train=0.8,
        seed=42
    )

    # Ver solo las primeras 2 muestras de cada dígito (0..9)
    # ASCII art de 2 muestras por dígito (0..9) en train y valid
    print_train_valid_samples((x_tr, y_tr), (x_va, y_va),
                              per_digit=2, rows=20, cols=20,
                              as_ascii=True)
    """
    print_train_valid_samples((x_tr, y_tr), (x_va, y_va),
                              per_digit=2, rows=20, cols=20,
                              as_ascii=False)
    """
