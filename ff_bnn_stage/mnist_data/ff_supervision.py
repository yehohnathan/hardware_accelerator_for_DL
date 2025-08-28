# ============================== ff_supervision ============================= #
# Versión para Forward-Forward + BNN: Supervisión (etiqueta en la entrada)
# Utilidades para construir ejemplos positivos/negativos desde vectores 1D
# (MNIST u otros) y entrenar FF en la 1ª capa.
# 1) Concatena one-hot(10) a la entrada (prefix/suffix) según la etiqueta real.
# 2) Genera etiquetas aleatorias (opción: forzar distintas) y concatena tokens.
# 3) Crea ruido con la misma forma que x (int8 {-1,+1} o float32 [0,1]).
# 4) Construye pares FF (x_pos, x_neg) con ratio neg_per_pos y semilla.
# 5) Devuelve vectores de conveniencia:
#    - y_pos/y_neg (etiquetas de tokens), t_pos/t_neg (1/0 para logs).
# 6) Incluye comprobaciones básicas de forma/tamaño para mayor robustez.
# 7) (Opcional) helper para candidatos de inferencia: x + one-hot(k), k=0..9.
# ================================ LIBRERÍAS ================================ #
import mnist_preprocess as mp
import numpy as np          # Ayuda para convertir de imagenes a tensores
from typing import Iterable, Optional


# ================================ FUNCIONES ================================ #
def _token_mode_from_x(x: np.ndarray) -> tuple:
    """
    Determina modo de tokens según dtype/valores de x.
    Retorna (dtype, neg_val, pos_val).
    """
    if np.issubdtype(x.dtype, np.integer):
        # Caso típico BNN: {-1,+1} en int8
        return (np.int8, -1, 1)
    # Caso float32 en [0,1]
    return (np.float32, 0.0, 1.0)


def _make_tokens(labels: np.ndarray,
                 n_classes: int,
                 dtype,
                 neg_val,
                 pos_val) -> np.ndarray:
    """Crea matriz (N, n_classes) con one-hot según valores neg/pos."""
    n = labels.shape[0]
    tok = np.full((n, n_classes), neg_val, dtype=dtype)
    tok[np.arange(n), labels.astype(int)] = pos_val
    return tok


def append_one_hot_correct(x: np.ndarray,
                           y: np.ndarray,
                           n_classes: int = 10,
                           where: str = "suffix") -> tuple[np.ndarray,
                                                           np.ndarray]:
    """
    Concatena el one-hot(10) correcto a cada vector de x según y.

    - x: (N, d) aplanado
    - y: (N,) etiquetas verdaderas 0..9
    - where: 'prefix' o 'suffix' para la posición de los 10 tokens

    Devuelve (x_ff, y) para comodidad aguas abajo.
    """
    dtype, neg_val, pos_val = _token_mode_from_x(x)
    tok = _make_tokens(y, n_classes, dtype, neg_val, pos_val)
    x_ff = (np.concatenate([tok, x], axis=1)
            if where == "prefix" else
            np.concatenate([x, tok], axis=1))
    return x_ff, y


def _random_labels(n: int,
                   n_classes: int,
                   rng: np.random.Generator,
                   forbid: np.ndarray | None = None) -> np.ndarray:
    """
    Etiquetas aleatorias 0..n_classes-1.
    Si 'forbid' se pasa, evita igualdad elemento-a-elemento.
    """
    lbl = rng.integers(0, n_classes, size=n, dtype=np.uint8)
    if forbid is None:
        return lbl
    mask = (lbl == (forbid % n_classes))
    # Reasigna hasta no coincidir con 'forbid'
    while np.any(mask):
        add = rng.integers(1, n_classes, size=mask.sum(), dtype=np.uint8)
        lbl[mask] = (lbl[mask] + add) % n_classes
        mask = (lbl == (forbid % n_classes))
    return lbl


def append_one_hot_random(x: np.ndarray,
                          y_true: np.ndarray,
                          n_classes: int = 10,
                          where: str = "suffix",
                          seed: int = 123,
                          ensure_wrong: bool = True) -> tuple[np.ndarray,
                                                              np.ndarray]:
    """
    Concatena un one-hot(10) aleatorio por muestra.

    - ensure_wrong=True: fuerza etiqueta distinta de y_true (negativo).
    Devuelve (x_ff, y_rand) donde y_rand son las etiquetas usadas.
    """
    rng = np.random.default_rng(seed)
    y_rand = _random_labels(x.shape[0], n_classes, rng,
                            forbid=y_true if ensure_wrong else None)
    dtype, neg_val, pos_val = _token_mode_from_x(x)
    tok = _make_tokens(y_rand, n_classes, dtype, neg_val, pos_val)
    x_ff = (np.concatenate([tok, x], axis=1)
            if where == "prefix" else
            np.concatenate([x, tok], axis=1))
    return x_ff, y_rand


def make_noise_like(x: np.ndarray,
                    seed: int = 123) -> np.ndarray:
    """
    Genera ruido con la misma forma que x.

    - Si x es int8 binario -> {-1,+1} equiprobable.
    - Si x es float32 [0,1] -> uniforme en [0,1].
    """
    rng = np.random.default_rng(seed)
    if np.issubdtype(x.dtype, np.integer):
        # {-1,+1}
        u = rng.integers(0, 2, size=x.shape, dtype=np.int8)
        return (u * 2 - 1).astype(np.int8)
    # [0,1]
    return rng.random(size=x.shape, dtype=np.float32)


def make_ff_pairs(x: np.ndarray,
                  y: np.ndarray,
                  neg_per_pos: int = 1,
                  n_classes: int = 10,
                  where: str = "suffix",
                  seed: int = 123,
                  shuffle_neg: bool = True
                  ) -> tuple[np.ndarray, np.ndarray,
                             np.ndarray, np.ndarray,
                             np.ndarray, np.ndarray]:
    """
    Construye pares para FF (1ª capa) con negativos por etiqueta errónea.
    - Positivo:  x + one-hot(etiqueta correcta)         -> target 1
    - Negativo:  x_perm + one-hot(etiqueta aleatoria!=) -> target 0
      (si shuffle_neg=True, x_perm es una permutación de x)

    Parámetros
    ----------
    x : (N, d) aplanado; dtype int8 ({-1,+1}) o float32 ([0,1])
    y : (N,) etiquetas verdaderas 0..n_classes-1
    neg_per_pos : nº de negativos por positivo (>=1)
    n_classes : nº de clases (MNIST=10)
    where : 'prefix' o 'suffix' para la posición de tokens
    seed : semilla base para RNG
    shuffle_neg : si True, desalinear negativos respecto a positivos

    Retorna
    -------
    x_pos : (N, d+n_classes) positivos con tokens correctos
    y_pos : (N,) copia de y (para logs)
    x_neg : (N*neg_per_pos, d+n_classes) negativos
    y_neg : (N*neg_per_pos,) etiquetas usadas en tokens negativos
    t_pos : (N,) vector de unos (targets FF de conveniencia)
    t_neg : (N*neg_per_pos,) vector de ceros (targets FF)
    """
    if x.ndim != 2:
        raise ValueError("x debe ser 2D (N, d).")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("y debe ser (N,) y coincidir con x.")
    if neg_per_pos < 1:
        raise ValueError("neg_per_pos debe ser >= 1.")

    # Positivos: imagen + etiqueta correcta
    x_pos, y_pos = append_one_hot_correct(
        x, y, n_classes=n_classes, where=where
    )

    # Negativos: barajar base (opcional) y usar etiquetas erróneas
    n = x.shape[0]
    rng_base = np.random.default_rng(seed)
    x_neg_list: list[np.ndarray] = []
    y_neg_list: list[np.ndarray] = []

    for k in range(neg_per_pos):
        sk = seed + 1000 * (k + 1)
        if shuffle_neg:
            perm = rng_base.permutation(n)
            x_base = x[perm]
            y_base = y[perm]
        else:
            x_base = x
            y_base = y
        x_neg_k, y_wrong_k = append_one_hot_random(
            x_base, y_true=y_base, n_classes=n_classes, where=where,
            seed=sk, ensure_wrong=True
        )
        x_neg_list.append(x_neg_k)
        y_neg_list.append(y_wrong_k.astype(np.uint8))

    x_neg = np.vstack(x_neg_list)
    y_neg = np.concatenate(y_neg_list)

    # Targets de conveniencia
    t_pos = np.ones((x_pos.shape[0],), dtype=np.uint8)
    t_neg = np.zeros((x_neg.shape[0],), dtype=np.uint8)

    return x_pos, y_pos, x_neg, y_neg, t_pos, t_neg


def print_ff_samples_by_digit(x_ff: np.ndarray,
                              y: np.ndarray,
                              per_digit: int = 2,
                              digits: Optional[Iterable[int]] = None,
                              rows: int = 14,
                              cols: int = 14,
                              n_classes: int = 10,
                              where: str = "suffix",
                              split_name: str = "train",
                              charset: Optional[str] = None,
                              as_ascii: bool = True,
                              map_bin_to01: bool = True) -> None:
    """
    Imprime 'per_digit' muestras por dígito mostrando primero el one-hot(10)
    concatenado y luego la imagen (ASCII o numérica).

    - x_ff: (N, d+n_classes) con tokens concatenados (prefix/suffix)
    - y:    (N,) etiquetas 0..9
    - rows, cols: tamaño de la imagen a reconstruir
    - where: 'prefix' o 'suffix' según dónde estén los tokens
    - as_ascii: True -> ASCII; False -> números
    - map_bin_to01: si los tokens son {-1,+1}, mostrarlos como {0,1}
    """
    if x_ff.ndim != 2:
        raise ValueError("x_ff debe ser 2D (N, d+n_classes).")
    if y.ndim != 1 or y.shape[0] != x_ff.shape[0]:
        raise ValueError("y debe ser (N,) y coincidir con x_ff.")
    if where not in ("prefix", "suffix"):
        raise ValueError("where debe ser 'prefix' o 'suffix'.")

    d_expected = rows * cols + n_classes
    if x_ff.shape[1] != d_expected:
        msg = (f"x_ff debe ser (N, {d_expected}). Recibido {x_ff.shape}. "
               "Asegura rows/cols y n_classes correctos.")
        raise ValueError(msg)

    if where == "prefix":
        tok = x_ff[:, :n_classes]
        x = x_ff[:, n_classes:]
    else:
        tok = x_ff[:, -n_classes:]
        x = x_ff[:, :-n_classes]

    if digits is None:
        digits = range(10)

    print(f"[{split_name}] x_ff: {x_ff.shape}, dtype: {x_ff.dtype}")
    for k in digits:
        idx_k = np.where(y == k)[0]
        if idx_k.size == 0:
            print(f"  dígito {k}: (sin muestras)")
            continue
        take = min(per_digit, idx_k.size)
        print(f"  dígito {k}: mostrando {take} de {idx_k.size}")

        for j in range(take):
            i = int(idx_k[j])
            t = tok[i]

            # Normalizar visualización de tokens
            if map_bin_to01 and np.issubdtype(t.dtype, np.integer):
                t_show = (t > 0).astype(np.int32)
            else:
                t_show = t

            # Mejor esfuerzo para estimar índice del token activo
            t_arg = int(np.argmax(t))

            print(f"    idx={i:6d}  label={int(y[i])}  "
                  f"token_idx={t_arg}")
            print(f"    one_hot: {t_show.tolist()}")

            art = mp._ascii_art_from_vec(x[i], rows, cols, charset, as_ascii)
            for line in art:
                print("    " + line)
