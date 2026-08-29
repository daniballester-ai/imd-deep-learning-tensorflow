"""Carrega e prepara os 3 subconjuntos EMNIST (dígitos 1-5, letras A-E, V/F).

Notas de design (ver design.md da mudança atividade02-cnn-pytorch-emnist):
- EMNIST vem rotacionado/espelhado em relação ao MNIST -> aplicamos correção
  (transpose) antes de qualquer outro processamento.
- Imagens são redimensionadas de 28x28 grayscale para 32x32x3 (canal
  replicado), para casar com a entrada usada no artigo Silva Filho et al.
- V/F não existe nativamente no EMNIST -> usamos o subconjunto de letras
  filtrando apenas as classes V e F (ver design.md - Decisões).
"""

import numpy as np
import pandas as pd
import torch

DATA_DIR = "./data"
IMG_SIZE = 32
MAX_PER_CLASS = 3000  # amostras por classe (tamanho tratável nesta sessão; ver README)
RANDOM_SEED = 42


def _fix_emnist_orientation(images):
    # images: (N, 28, 28) uint8. EMNIST vem transposto/espelhado.
    return np.transpose(images, (0, 2, 1))


def _to_32x32x3(images):
    # images: (N, 28, 28) uint8 -> tensor (N, 3, 32, 32) float32 em [0,1]
    t = torch.from_numpy(images).float() / 255.0
    t = t.unsqueeze(1)  # (N, 1, 28, 28)
    t = torch.nn.functional.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                                         align_corners=False)
    t = t.repeat(1, 3, 1, 1)  # (N, 3, 32, 32)
    return t


def _load_csv_subset(csv_path, label_filter, max_per_class, seed=RANDOM_SEED):
    df = pd.read_csv(csv_path, header=None)
    labels_all = df.iloc[:, 0].to_numpy()
    mask = np.isin(labels_all, list(label_filter.keys()))
    df = df[mask]
    labels_all = df.iloc[:, 0].to_numpy()

    rng = np.random.default_rng(seed)
    keep_idx = []
    for lbl in label_filter:
        idx = np.where(labels_all == lbl)[0]
        rng.shuffle(idx)
        keep_idx.append(idx[:max_per_class])
    keep_idx = np.concatenate(keep_idx)
    rng.shuffle(keep_idx)

    df = df.iloc[keep_idx]
    labels = df.iloc[:, 0].to_numpy()
    pixels = df.iloc[:, 1:].to_numpy(dtype=np.uint8).reshape(-1, 28, 28)

    # Remapeia rótulos originais do EMNIST para índices de classe 0..K-1
    class_index = {orig: i for i, orig in enumerate(sorted(label_filter.keys()))}
    y = np.array([class_index[l] for l in labels], dtype=np.int64)

    pixels = _fix_emnist_orientation(pixels)
    x = _to_32x32x3(pixels)
    y = torch.from_numpy(y)
    return x, y, class_index


def _split_70_15_15(x, y, seed=RANDOM_SEED):
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx]), (x[test_idx], y[test_idx])


def load_digits_subset(max_per_class=MAX_PER_CLASS):
    """Dígitos {1,2,3,4,5} - mapping EMNIST digits: label == valor do dígito."""
    label_filter = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
    x, y, class_index = _load_csv_subset(f"{DATA_DIR}/emnist-digits-train.csv", label_filter, max_per_class)
    return (*_split_70_15_15(x, y), class_index)


def load_letters_subset(max_per_class=MAX_PER_CLASS):
    """Letras {A,B,C,D,E} - mapping EMNIST letters: label 1=A .. 5=E."""
    label_filter = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
    x, y, class_index = _load_csv_subset(f"{DATA_DIR}/emnist-letters-train.csv", label_filter, max_per_class)
    return (*_split_70_15_15(x, y), class_index)


def load_vf_subset(max_per_class=MAX_PER_CLASS):
    """V/F - derivado do subconjunto de letras do EMNIST (V=label 22, F=label 6)."""
    label_filter = {22: "V", 6: "F"}
    x, y, class_index = _load_csv_subset(f"{DATA_DIR}/emnist-letters-train.csv", label_filter, max_per_class)
    return (*_split_70_15_15(x, y), class_index)


if __name__ == "__main__":
    for name, loader in [("digitos", load_digits_subset), ("letras", load_letters_subset), ("vf", load_vf_subset)]:
        (xtr, ytr), (xval, yval), (xte, yte), class_index = loader()
        print(f"\n=== {name} ===")
        print("classes:", class_index)
        print("treino:", xtr.shape, "validação:", xval.shape, "teste:", xte.shape)
        print("contagem treino por classe:", torch.bincount(ytr).tolist())
