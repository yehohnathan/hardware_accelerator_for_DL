# ================================ ff_loader ================================ #
# Ruta A (lectura): utilidades para abrir meta.json + *.bin generados por
# export_routeA_binary. Soporta:
#   - dtype="bitpacked": desempaquetar a {0,1} (uint8) o mapear a {-1,+1}.
#   - dtype="float32": leer filas contiguas (N, D).
#   - streaming por lotes para no cargar todo en memoria.
# Mantiene PEP8 y líneas <= 79 caracteres.
# ========================================================================== #
from __future__ import annotations
import mnist_preprocess as mp
from typing import Dict, Iterator, Tuple, Literal
import json
import os
import numpy as np


# ================================ Lectura META ============================= #
def load_meta(out_dir: str) -> Dict:
    """
    Lee out_dir/meta.json y devuelve un dict con la configuración.
    """
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================== Lectores base ============================= #

def _expected_sizes(meta: Dict,
                    which: Literal["pos", "neg"]) -> Tuple[int, int]:
    """
    Retorna (N, D) esperados para el dataset 'pos' o 'neg'.
    """
    N = int(meta["n_pos"] if which == "pos" else meta["n_neg"])
    D = int(meta["input_dim"])
    return N, D


def _read_labels(path: str, n: int) -> np.ndarray:
    """
    Lee 'n' bytes de etiquetas (uint8).
    """
    arr = np.fromfile(path, dtype=np.uint8, count=n)
    if arr.size != n:
        raise IOError("labels: tamaño inesperado.")
    return arr


def _read_inputs_float32(path: str, n: int, d: int) -> np.ndarray:
    """
    Lee 'n*d' float32 y devuelve (n, d).
    """
    arr = np.fromfile(path, dtype=np.float32, count=n * d)
    if arr.size != n * d:
        raise IOError("inputs float32: tamaño inesperado.")
    return arr.reshape(n, d)


def _read_inputs_bitpacked(path: str,
                           n: int,
                           d: int,
                           bytes_per_row: int) -> np.ndarray:
    """
    Lee 'n' filas bit-empaquetadas y devuelve (n, d) en {0,1} (uint8).
    - bitorder little: feature 0 es LSB del primer byte.
    """
    total = n * bytes_per_row
    raw = np.fromfile(path, dtype=np.uint8, count=total)
    if raw.size != total:
        raise IOError("inputs bitpacked: tamaño inesperado.")
    rows = raw.reshape(n, bytes_per_row)
    bits = np.unpackbits(rows, axis=1, bitorder='little')
    return bits[:, :d].astype(np.uint8, copy=False)


# =============================== API de alto nivel ======================== #

def load_dataset(out_dir: str,
                 which: Literal["pos", "neg"] = "pos",
                 to_pm1: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Carga X y etiquetas T (1 para pos, 0 para neg) desde out_dir.
    Devuelve (X, T, meta).
    - to_pm1=True: si bitpacked, mapea {0,1}->{-1,+1} (int8).
    """
    meta = load_meta(out_dir)
    n, d = _expected_sizes(meta, which)

    if which == "pos":
        f_in = os.path.join(out_dir, "inputs_pos.bin")
        f_lb = os.path.join(out_dir, "labels_pos.bin")
        t_val = 1
    else:
        f_in = os.path.join(out_dir, "inputs_neg.bin")
        f_lb = os.path.join(out_dir, "labels_neg.bin")
        t_val = 0

    dtype = meta["dtype"]
    if dtype == "bitpacked":
        bpr = int(meta["bytes_per_row"])
        x01 = _read_inputs_bitpacked(f_in, n, d, bpr)
        x = x01 if not to_pm1 else (x01 * 2 - 1).astype(np.int8)
    elif dtype == "float32":
        x = _read_inputs_float32(f_in, n, d)
    else:
        raise ValueError(f"dtype no soportado: {dtype}")

    # targets FF (útiles para logs/chequeos)
    t = np.full((n,), t_val, dtype=np.uint8)

    # Si existen labels_*.bin, se priorizan (por si se generaron aparte)
    if os.path.exists(f_lb):
        t = _read_labels(f_lb, n)

    return x, t, meta


def split_tokens(x_ff: np.ndarray,
                 token_dim: int = 10,
                 where: Literal["prefix", "suffix"] = "suffix"
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separa tokens (one-hot) e imagen de un vector (N, D_img+token_dim).
    Devuelve (tok, x_img).
    """
    if x_ff.ndim != 2:
        raise ValueError("x_ff debe ser 2D (N, D).")
    if x_ff.shape[1] < token_dim:
        raise ValueError("x_ff no contiene los tokens esperados.")
    if where == "prefix":
        tok = x_ff[:, :token_dim]
        x_img = x_ff[:, token_dim:]
    else:
        tok = x_ff[:, -token_dim:]
        x_img = x_ff[:, :-token_dim]
    return tok, x_img


# ============================ Iteración por lotes ========================= #

def batch_ranges(n: int,
                 batch_size: int,
                 drop_last: bool) -> Iterator[Tuple[int, int]]:
    """
    Genera rangos [i,j) por lote, compatible con meta.
    """
    full = n // batch_size
    for b in range(full):
        i = b * batch_size
        yield i, i + batch_size
    if not drop_last and (n % batch_size) != 0:
        yield full * batch_size, n


def stream_batches(out_dir: str,
                   which: Literal["pos", "neg"] = "pos",
                   to_pm1: bool = False
                   ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Itera lotes desde disco según meta.json, devolviendo (Xb, Tb).
    Conviene cuando N es grande.
    """
    meta = load_meta(out_dir)
    n, d = _expected_sizes(meta, which)
    bsz = int(meta["batch_size"])
    drop = bool(meta["drop_last"])
    dtype = meta["dtype"]

    if which == "pos":
        f_in = os.path.join(out_dir, "inputs_pos.bin")
        f_lb = os.path.join(out_dir, "labels_pos.bin")
        t_val = 1
    else:
        f_in = os.path.join(out_dir, "inputs_neg.bin")
        f_lb = os.path.join(out_dir, "labels_neg.bin")
        t_val = 0

    # Abrimos una vista memmap sobre el archivo de entrada
    if dtype == "bitpacked":
        bpr = int(meta["bytes_per_row"])
        total = n * bpr
        mm = np.memmap(f_in, mode="r", dtype=np.uint8, shape=(total,))
        for i, j in batch_ranges(n, bsz, drop):
            sl = mm[i * bpr:j * bpr]
            rows = sl.reshape(j - i, bpr)
            x01 = np.unpackbits(rows, axis=1, bitorder='little')
            xb = x01[:, :d].astype(np.uint8, copy=False)
            if to_pm1:
                xb = (xb * 2 - 1).astype(np.int8)
            # labels
            if os.path.exists(f_lb):
                tb = _read_labels(f_lb, n)[i:j]
            else:
                tb = np.full((j - i,), t_val, dtype=np.uint8)
            yield xb, tb
    elif dtype == "float32":
        mm = np.memmap(f_in, mode="r", dtype=np.float32, shape=(n * d,))
        for i, j in batch_ranges(n, bsz, drop):
            sl = mm[i * d:j * d]
            xb = np.asarray(sl, dtype=np.float32).reshape(j - i, d)
            if os.path.exists(f_lb):
                tb = _read_labels(f_lb, n)[i:j]
            else:
                tb = np.full((j - i,), t_val, dtype=np.uint8)
            yield xb, tb
    else:
        raise ValueError(f"dtype no soportado: {dtype}")


# ======================= Visualización rápida (Ruta A) ===================== #
def _get_rows_cols_where(meta: Dict) -> Tuple[int, int, str, int]:
    """
    Obtiene (rows, cols, where, token_dim) desde meta. Si no están
    presentes, intenta deducirlos (asume token_dim=10).
    """
    where = meta.get("where_tokens", "suffix")
    token_dim = int(meta.get("token_dim", 10))
    d = int(meta["input_dim"])

    rows = meta.get("rows", None)
    cols = meta.get("cols", None)

    if rows is None or cols is None:
        # Intento de deducción: si (d - token_dim) es cuadrado perfecto
        img_dim = d - token_dim
        s = int(round(img_dim ** 0.5))
        if s * s != img_dim:
            raise ValueError("No se pudo deducir rows/cols desde meta.")
        rows = s
        cols = s

    if where not in ("prefix", "suffix"):
        raise ValueError("where_tokens inválido en meta.")
    return int(rows), int(cols), where, token_dim


def print_first_vectors(x_ff: np.ndarray,
                        name: str,
                        n: int = 3) -> None:
    """
    Imprime los primeros 'n' vectores 1D completos (mucho texto).
    """
    n = min(n, x_ff.shape[0])
    print(f"\n=== {name}: primeros {n} vectores 1D ===")
    for i in range(n):
        v = x_ff[i]
        print(f"[{name}] idx={i} len={v.size}")
        # Imprime el vector completo (cuidado: es largo)
        np.set_printoptions(edgeitems=v.size, threshold=v.size)
        print(v)
    # Restaurar opciones
    np.set_printoptions(edgeitems=3, threshold=1000)


def print_head_ff_ascii(x_ff: np.ndarray,
                        meta: Dict,
                        name: str = "POS",
                        n: int = 3,
                        charset: str | None = None,
                        as_ascii: bool = True,
                        map_bin_to01: bool = True) -> None:
    """
    Muestra 'n' primeras muestras: token one-hot y la imagen como ASCII.
    """
    rows, cols, where, token_dim = _get_rows_cols_where(meta)

    if where == "prefix":
        tok = x_ff[:n, :token_dim]
        img = x_ff[:n, token_dim:]
    else:
        tok = x_ff[:n, -token_dim:]
        img = x_ff[:n, :-token_dim]

    print(f"\n=== {name}: primeros {img.shape[0]} (ASCII) ===")
    for i in range(img.shape[0]):
        t = tok[i]
        if map_bin_to01 and np.issubdtype(t.dtype, np.integer):
            t_show = (t > 0).astype(np.int32)
        else:
            t_show = t
        t_arg = int(np.argmax(t))
        print(f"{name} idx={i:4d}  token_idx={t_arg}")
        print(f"one_hot: {t_show.tolist()}")

        art = mp._ascii_art_from_vec(img[i], rows, cols,
                                     charset=charset, as_ascii=as_ascii)
        for line in art:
            print("  " + line)


# =================================== MAIN ================================== #
if __name__ == "__main__":
    out = "build/pack_routeA"  # <- tu carpeta real

    # 1) Cargar datasets completos (¡ojo memoria si N es grande!)
    if os.path.exists(os.path.join(out, "meta.json")):
        x_pos, t_pos, meta = load_dataset(out, which="pos", to_pm1=False)
        x_neg, t_neg, _ = load_dataset(out, which="neg", to_pm1=False)
        print("POS:", x_pos.shape, x_pos.dtype, t_pos.shape)
        print("NEG:", x_neg.shape, x_neg.dtype, t_neg.shape)

        # 1) Ver los primeros 3 vectores crudos (1D)
        print_first_vectors(x_pos, name="POS", n=3)
        print_first_vectors(x_neg, name="NEG", n=3)

        # 2) Ver ASCII art de esos 3 (usa rows/cols/where del meta)
        print_head_ff_ascii(x_pos, meta, name="POS", n=3,
                            as_ascii=True, map_bin_to01=True)
        print_head_ff_ascii(x_neg, meta, name="NEG", n=3,
                            as_ascii=True, map_bin_to01=True)
    else:
        print("No se encontró meta.json en", out)
