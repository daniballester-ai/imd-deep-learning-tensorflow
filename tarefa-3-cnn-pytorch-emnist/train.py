"""Treina as 3 CNNs (dígitos, letras, V/F) reproduzindo as arquiteturas do
artigo Silva Filho et al. (2022), exporta os modelos e monta o quadro
comparativo de parâmetros/tamanho exigido pela Tarefa 3."""

import json
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import load_digits_subset, load_letters_subset, load_vf_subset
from models import CNN3Blocos, CNN4Blocos

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
MAX_EPOCHS = 20
PATIENCE = 3  # early stopping (proxy da margem de 0,1% do artigo)
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)


def make_loaders(train, val, test):
    (xtr, ytr), (xval, yval), (xte, yte) = train, val, test
    train_dl = DataLoader(TensorDataset(xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(TensorDataset(xval, yval), batch_size=BATCH_SIZE)
    test_dl = DataLoader(TensorDataset(xte, yte), batch_size=BATCH_SIZE)
    return train_dl, val_dl, test_dl


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total


def train_model(model, train_dl, val_dl, test_dl, name):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_loss, val_acc = evaluate(model, val_dl, criterion)
        print(f"[{name}] epoch {epoch:2d} - val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-3:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"[{name}] early stopping na época {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_dl, criterion)
    print(f"[{name}] TESTE -> loss={test_loss:.4f} acc={test_acc:.4f}")
    return model, test_loss, test_acc


def model_stats(model, path):
    torch.save(model.state_dict(), path)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # os.path.getsize logo após o save pode ler metadados desatualizados em
    # sistemas de arquivos sincronizados (ex.: Google Drive) - medir o
    # tamanho real serializando em memória evita essa corrida.
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = len(buffer.getvalue()) / (1024 * 1024)
    return n_params, size_mb


if __name__ == "__main__":
    results = {}
    os.makedirs("models_out", exist_ok=True)

    t0 = time.time()

    # --- Dígitos (1-5) - Estrutura 4 ---
    train, val, test, class_index = load_digits_subset()
    train_dl, val_dl, test_dl = make_loaders(train, val, test)
    model = CNN4Blocos(num_classes=len(class_index))
    model, test_loss, test_acc = train_model(model, train_dl, val_dl, test_dl, "digitos")
    n_params, size_mb = model_stats(model, "models_out/cnn_digitos.pt")
    results["digitos"] = {
        "classes": class_index, "test_loss": test_loss, "test_accuracy": test_acc,
        "n_params": n_params, "size_mb": size_mb,
    }

    # --- Letras (A-E) - Estrutura 4 ---
    train, val, test, class_index = load_letters_subset()
    train_dl, val_dl, test_dl = make_loaders(train, val, test)
    model = CNN4Blocos(num_classes=len(class_index))
    model, test_loss, test_acc = train_model(model, train_dl, val_dl, test_dl, "letras")
    n_params, size_mb = model_stats(model, "models_out/cnn_letras.pt")
    results["letras"] = {
        "classes": class_index, "test_loss": test_loss, "test_accuracy": test_acc,
        "n_params": n_params, "size_mb": size_mb,
    }

    # --- V/F - Estrutura 3 ---
    train, val, test, class_index = load_vf_subset()
    train_dl, val_dl, test_dl = make_loaders(train, val, test)
    model = CNN3Blocos(num_classes=len(class_index))
    model, test_loss, test_acc = train_model(model, train_dl, val_dl, test_dl, "vf")
    n_params, size_mb = model_stats(model, "models_out/cnn_vf.pt")
    results["vf"] = {
        "classes": class_index, "test_loss": test_loss, "test_accuracy": test_acc,
        "n_params": n_params, "size_mb": size_mb,
    }

    elapsed = time.time() - t0
    print(f"\nTempo total: {elapsed:.1f}s")

    print("\n=== Quadro comparativo ===")
    print(f"{'Modelo':<10} {'Params':>10} {'Tamanho (MB)':>14} {'Acc. teste':>12} {'Perda teste':>12}")
    for name, r in results.items():
        print(f"{name:<10} {r['n_params']:>10} {r['size_mb']:>14.3f} {r['test_accuracy']:>12.4f} {r['test_loss']:>12.4f}")

    with open("training_results.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "elapsed_seconds": elapsed}, f, indent=2, ensure_ascii=False)
    print("\nResultados salvos em training_results.json")
