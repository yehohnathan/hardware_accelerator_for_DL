# =============================== ff_packaging ============================== #
# Empaquetado para FPGA (Ruta A) – Forward-Forward + BNN/CNN
#
# ¿Qué hace?
# 1) Crea lotes (batches) de tamaño B a partir de x_pos/x_neg
#    (salida de make_ff_pairs).
# 2) Si la entrada es binaria {-1,+1} (o 0/1), bit-pack a palabras de
#    32/64 bits (little-endian).
#    - Orden por filas (row-major).
#    - El primer feature (índice 0) es el bit menos significativo (LSB)
#      del primer byte.
#    - Los 4/8 bytes forman una palabra uint32/uint64 en little-endian.
#    - Si D no es múltiplo de word_bits, se rellena con ceros al final de cada
#      fila.
# 3) Guarda archivos:
#      - inputs_pos.bin, inputs_neg.bin
#      - labels_pos.bin, labels_neg.bin
#      - meta.json
# ========================================================================== #
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Iterator, Tuple, Optional, Literal
import json
# import math
import os
import numpy as np


# ================================ Metadatos ================================ #
@dataclass
class PackInfo:
    format: str
    dtype: str
    input_dim: int
    batch_size: int
    drop_last: bool
    n_pos: int
    n_neg: int
    n_batches_pos: int
    n_batches_neg: int
    files: Dict[str, str]
    # bit-packed
    word_bits: Optional[int] = None
    bytes_per_row: Optional[int] = None
    features_padded: Optional[int] = None
    bitorder_in_byte: Optional[str] = None
    endianness: Optional[str] = None
    row_major: bool = True
    rows: Optional[int] = None
    cols: Optional[int] = None
    token_dim: Optional[int] = 10
    where_tokens: Optional[str] = None  # "prefix" | "suffix"


# ================================ FUNCIONES ================================ #
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _is_pm1_binary(x: np.ndarray) -> bool:
    """True si x es entero y todos los valores están en {-1, +1} (o {0,1})."""
    if not np.issubdtype(x.dtype, np.integer):
        return False
    mn, mx = int(np.min(x)), int(np.max(x))
    return (mn >= -1 and mx <= 1) and all(v in (-1, 0, 1) for v in (mn, mx))


def _to01_from_pm1(
        x: np.ndarray) -> np.ndarray:
    """
    {-1,+1} -> {0,1} (uint8). Si ya está en {0,1}, lo normaliza a uint8.
    Lanza si encuentra otros valores enteros fuera de {-1,0,+1}.
    """
    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError("Se esperaba dtype entero para mapear a {0,1}.")
    mn, mx = int(np.min(x)), int(np.max(x))
    if mn < -1 or mx > 1:
        raise ValueError("Valores fuera de rango esperado {-1,0,+1} para "
                         "empaquetado binario.")
    if mn >= 0:
        return x.astype(np.uint8, copy=False)  # ya {0,1}
    # caso {-1,+1} -> {0,1}
    return (x > 0).astype(np.uint8)


def _batch_ranges(
        n: int,
        B: int,
        drop_last: bool) -> Iterator[Tuple[int, int]]:
    """Genera rangos [i, j) de tamaño B (último parcial si drop_last=False)."""
    full = n // B
    for b in range(full):
        i = b * B
        yield i, i + B
    if not drop_last and (n % B) != 0:
        yield full * B, n


def _pack_bits_rows(x01: np.ndarray,
                    word_bits: Literal[32, 64]) -> Tuple[np.ndarray, int]:
    """
    Empaqueta por filas un array {0,1} (N, D) a bytes, agrupados en palabras
    de 32/64 bits.

    - Devuelve (bytes2D, bytes_per_row).
    - 'bytes2D' es (N, bytes_per_row) en dtype=uint8, contiguo en C.
    - bitorder='little': el feature 0 es LSB del primer byte.
    - Se rellena cada fila hasta múltiplo de word_bits con ceros al final.
    """
    if x01.dtype != np.uint8:
        x01 = x01.astype(np.uint8, copy=False)

    N, D = x01.shape
    word_bytes = word_bits // 8
    words_per_row = (D + word_bits - 1) // word_bits
    bytes_per_row = words_per_row * word_bytes

    # packbits por filas -> (N, ceil(D/8)) bytes
    packed = np.packbits(x01, axis=1, bitorder='little')  # uint8

    # Asegurar longitud múltiplo del tamaño de palabra en bytes
    cur_bytes = packed.shape[1]
    if cur_bytes < bytes_per_row:
        pad = bytes_per_row - cur_bytes
        packed = np.pad(packed, pad_width=((0, 0), (0, pad)), mode='constant')

    # Contiguo en C
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    return packed, bytes_per_row


def _write_labels_stream(f, n: int, value: int, chunk: int = 4096) -> None:
    """Escribe n etiquetas uint8 con el mismo valor (0 o 1) en chunks."""
    buf = (np.full((min(chunk, n),), value, dtype=np.uint8)).tobytes()
    total = 0
    # Escribimos en bloques repetidos para no alocar n grande.
    while total < n:
        to_write = min(n - total, len(buf))
        f.write(buf[:to_write])
        total += to_write


def _write_inputs_batch(f, Xb: np.ndarray,
                        binary_pack: bool,
                        word_bits: Literal[32, 64]) -> int:
    """
    Escribe un batch en el archivo 'f'.
    - Si binary_pack=True: bit-pack y escribe bytes; retorna bytes escritos
      por muestra.
    - Si False: escribe float32 en fila mayor; retorna bytes escritos por
      muestra.
    """
    if binary_pack:
        x01 = _to01_from_pm1(Xb)
        packed, bytes_per_row = _pack_bits_rows(x01, word_bits=word_bits)
        f.write(packed.tobytes(order='C'))
        return bytes_per_row
    else:
        Xf = np.ascontiguousarray(Xb, dtype=np.float32)
        f.write(Xf.tobytes(order='C'))
        return Xf.shape[1] * 4  # float32


def export_routeA_binary(x_pos: np.ndarray,
                         x_neg: np.ndarray,
                         out_dir: str,
                         batch_size: int = 64,
                         word_bits: Literal[32, 64] = 32,
                         drop_last: bool = False,
                         rows: Optional[int] = None,
                         cols: Optional[int] = None,
                         where_tokens: str = "suffix",
                         token_dim: int = 10,) -> Dict:
    """
    Exporta archivos binarios (Ruta A) para entrenamiento FF en FPGA.

    Parámetros
    ----------
    x_pos : (N_pos, D) ndarray
    - Entradas positivas con tokens one-hot concatenados (prefix o suffix).
    - dtype int8 en {-1,+1}/{0,1}     -> se bit-packea.
    - dtype float32 en [0,1]          -> se guarda tal cual.
    x_neg : (N_neg, D) ndarray
    - Entradas negativas (mismas reglas y D).
    out_dir : str
    - Carpeta de salida. Se crea si no existe.
    batch_size : int
    - Tamaño de lote B. Si drop_last=False, el último lote puede ser parcial.
    word_bits : 32 | 64
    - Ancho de palabra para bit-packing cuando la entrada es binaria.
    drop_last : bool
    - Si True, descarta el último lote parcial.

    Retorna
    -------
    meta : dict
    - Metadatos útiles para lectura en C++/HLS.
    - También se guardan en meta.json.

    Archivos generados
    ------------------
    - out_dir/inputs_pos.bin  (por filas, concatenando lotes)
    - out_dir/inputs_neg.bin
    - out_dir/labels_pos.bin  (uint8: 1 por muestra)
    - out_dir/labels_neg.bin  (uint8: 0 por muestra)
    - out_dir/meta.json
    """
    if x_pos.ndim != 2 or x_neg.ndim != 2:
        raise ValueError("x_pos y x_neg deben ser 2D (N, D).")
    if x_pos.shape[1] != x_neg.shape[1]:
        raise ValueError("x_pos y x_neg deben compartir la misma dimensión D.")
    if where_tokens not in ("prefix", "suffix"):
        raise ValueError("where_tokens debe ser 'prefix' o 'suffix'.")
    if token_dim <= 0:
        raise ValueError("token_dim debe ser > 0.")

    D = int(x_pos.shape[1])
    Np = int(x_pos.shape[0])
    Nn = int(x_neg.shape[0])

    if batch_size <= 0:
        raise ValueError("batch_size debe ser > 0.")
    if word_bits not in (32, 64):
        raise ValueError("word_bits debe ser 32 o 64.")

    # Detectar modo de empaquetado
    pos_is_bin = _is_pm1_binary(x_pos)
    neg_is_bin = _is_pm1_binary(x_neg)
    if pos_is_bin != neg_is_bin:
        raise ValueError("x_pos y x_neg deben ser ambos binarios o ambos"
                         " float32.")
    binary_pack = pos_is_bin

    # Batches efectivos
    n_batches_pos = Np // batch_size + (
        0 if (drop_last or Np % batch_size == 0) else 1)
    n_batches_neg = Nn // batch_size + (
        0 if (drop_last or Nn % batch_size == 0) else 1)

    # Tamaños efectivos (si drop_last=True se truncan)
    Np_eff = (n_batches_pos * batch_size) if drop_last else Np
    Nn_eff = (n_batches_neg * batch_size) if drop_last else Nn

    _ensure_dir(out_dir)
    f_in_pos = os.path.join(out_dir, "inputs_pos.bin")
    f_in_neg = os.path.join(out_dir, "inputs_neg.bin")
    f_lb_pos = os.path.join(out_dir, "labels_pos.bin")
    f_lb_neg = os.path.join(out_dir, "labels_neg.bin")

    # Abrimos 4 streams de salida
    with open(f_in_pos, "wb") as finp, \
         open(f_in_neg, "wb") as finn, \
         open(f_lb_pos, "wb") as flbp, \
         open(f_lb_neg, "wb") as flbn:

        bytes_per_row: Optional[int] = None  # se fijará en el primer batch
        # POSITIVOS
        for i, j in _batch_ranges(Np, batch_size, drop_last):
            Xb = x_pos[i:j]
            if Xb.shape[0] == 0:
                continue
            bpr = _write_inputs_batch(
                finp, Xb, binary_pack=binary_pack, word_bits=word_bits)
            if bytes_per_row is None:
                bytes_per_row = bpr
            _write_labels_stream(flbp, Xb.shape[0], value=1)

        # NEGATIVOS
        for i, j in _batch_ranges(Nn, batch_size, drop_last):
            Xb = x_neg[i:j]
            if Xb.shape[0] == 0:
                continue
            bpr = _write_inputs_batch(
                finn, Xb, binary_pack=binary_pack, word_bits=word_bits)
            if bytes_per_row is None:
                bytes_per_row = bpr
            _write_labels_stream(flbn, Xb.shape[0], value=0)

    # Metadatos
    files = {
        "inputs_pos": os.path.basename(f_in_pos),
        "inputs_neg": os.path.basename(f_in_neg),
        "labels_pos": os.path.basename(f_lb_pos),
        "labels_neg": os.path.basename(f_lb_neg),
        "meta": "meta.json",
    }

    if binary_pack:
        word_bytes = word_bits // 8
        words_per_row = (D + word_bits - 1) // word_bits
        features_padded = words_per_row * word_bits
        meta = PackInfo(
            format="ff_routeA_v1",
            dtype="bitpacked",
            input_dim=D,
            batch_size=batch_size,
            drop_last=drop_last,
            n_pos=Np_eff,
            n_neg=Nn_eff,
            n_batches_pos=n_batches_pos,
            n_batches_neg=n_batches_neg,
            files=files,
            word_bits=word_bits,
            bytes_per_row=words_per_row * word_bytes,
            features_padded=features_padded,
            bitorder_in_byte="lsb0",
            endianness="little",
            row_major=True,
            rows=rows,
            cols=cols,
            token_dim=token_dim,
            where_tokens=where_tokens,
        )
    else:
        meta = PackInfo(
            format="ff_routeA_v1",
            dtype="float32",
            input_dim=D,
            batch_size=batch_size,
            drop_last=drop_last,
            n_pos=Np_eff,
            n_neg=Nn_eff,
            n_batches_pos=n_batches_pos,
            n_batches_neg=n_batches_neg,
            files=files,
            word_bits=None,
            bytes_per_row=D * 4,
            features_padded=None,
            bitorder_in_byte=None,
            endianness="little",
            row_major=True,
            rows=rows,
            cols=cols,
            token_dim=token_dim,
            where_tokens=where_tokens,
        )

    # Guardar meta.json
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2, ensure_ascii=False)

    return asdict(meta)
