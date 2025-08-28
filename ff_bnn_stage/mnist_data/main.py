# ================================ LIBRERÍAS ================================ #
import mnist_preprocess as mp
import ff_supervision as sp
import ff_packaging as pc
import json
# import numpy as np

# =================================== MAIN ================================== #
if __name__ == "__main__":
    # Uso de referencia (descomenta si quieres probar en tu entorno):
    # 1) Prepara datos
    (x_tr, y_tr), (x_va, y_va) = mp.mnist_preprocess(
        rows=28, cols=28, n_per_digit=80, binarize=True, frac_train=0.8,
        seed=42)

    # 2) Pares FF
    x_pos, y_pos, x_neg, y_neg, t_pos, t_neg = sp.make_ff_pairs(
        x_tr, y_tr, neg_per_pos=1, n_classes=10, where="prefix", seed=5,
        shuffle_neg=True)

    # 3) Empaquetar Ruta A
    meta = pc.export_routeA_binary(x_pos=x_pos,
                                   x_neg=x_neg,
                                   out_dir="build/pack_routeA",
                                   batch_size=64,
                                   word_bits=32,
                                   drop_last=False,
                                   rows=28,
                                   cols=28,
                                   where_tokens="prefix",)

    print("META:", json.dumps(meta, indent=2, ensure_ascii=False))


"""
# Supón que ya hiciste el preprocesamiento:
(x_tr, y_tr), (x_va, y_va) = mp.mnist_preprocess(
    rows=28, cols=28, n_per_digit=80, binarize=True,
    frac_train=0.8, seed=42
)

# Supón que ya tienes x_tr (N,d) y y_tr (N,)
x_pos, y_pos, x_neg, y_neg, t_pos, t_neg = sp.make_ff_pairs(
    x_tr, y_tr, neg_per_pos=1, n_classes=10, where="prefix", seed=5,
    shuffle_neg=True
)

# Prueba de números: x_tr VS x_pos vs x_neg
num_test = 3
print(f"\n\nORIGINAL 1: {len(x_tr[num_test])}")
np.set_printoptions(threshold=np.inf)
print(x_tr[num_test])
mp.print_samples_by_digit(
    x=np.expand_dims(x_tr[num_test], axis=0),   # (1, d)
    y=np.array([y_tr[num_test]]),               # (1,)
    per_digit=1,                         # 1 muestra
    digits=[y_tr[num_test]],                    # solo ese dígito
    rows=28, cols=28,
    as_ascii=True
)

np.set_printoptions(threshold=np.inf)
print(f"\n\nORIGINAL 2: {len(x_pos[num_test])}")
print(x_pos[num_test])
# Mostrar la primera imagen con one-hot concatenado
sp.print_ff_samples_by_digit(
    x_ff=np.expand_dims(x_pos[num_test], axis=0),  # (1, d+10)
    y=np.array([y_pos[num_test]]),                 # (1,)
    per_digit=1,
    digits=[y_pos[num_test]],
    rows=28, cols=28,
    n_classes=10,
    where="prefix",    # asegúrate de usar lo mismo que en make_ff_pairs
    as_ascii=True
)

np.set_printoptions(threshold=np.inf)
print(f"\n\nORIGINAL 2: {len(x_neg[num_test])}")
print(x_neg[num_test])
# Mostrar la primera imagen con one-hot concatenado
sp.print_ff_samples_by_digit(
    x_ff=np.expand_dims(x_neg[num_test], axis=0),  # (1, d+10)
    y=np.array([y_neg[num_test]]),                 # (1,)
    per_digit=1,
    digits=[y_neg[num_test]],
    rows=28, cols=28,
    n_classes=10,
    where="prefix",    # asegúrate de usar lo mismo que en make_ff_pairs
    as_ascii=True
)
"""
