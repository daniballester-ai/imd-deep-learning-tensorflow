"""Segunda técnica de redução de tamanho de modelo (análise do item 3.2):
reduzir o número de filtros convolucionais (32/64/64/64 -> 16/32/32/32) e
comparar tamanho/acurácia com o modelo original, usando o subconjunto de
dígitos como referência empírica (mesma divisão de dados e treinamento)."""

import io
import json

import torch
from torch import nn

from data_utils import load_digits_subset
from models import CNN4Blocos, ConvBlock
from train import DEVICE, evaluate, make_loaders, train_model


class ReducedCNN(nn.Module):
    """Mesma estrutura de 4 blocos conv+maxpool, mas com metade dos filtros
    em cada camada (16 -> 32 -> 32 -> 32 em vez de 32 -> 64 -> 64 -> 64)."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 16), ConvBlock(16, 32), ConvBlock(32, 32), ConvBlock(32, 32)
        )
        with torch.no_grad():
            flat_size = self.features(torch.zeros(1, 3, 32, 32)).numel()
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(flat_size, num_classes)

    def forward(self, x):
        return self.classifier(self.flatten(self.features(x)))


def size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue()) / (1024 * 1024)


if __name__ == "__main__":
    train, val, test, class_index = load_digits_subset()
    train_dl, val_dl, test_dl = make_loaders(train, val, test)

    original = CNN4Blocos(num_classes=len(class_index))
    original, orig_loss, orig_acc = train_model(original, train_dl, val_dl, test_dl, "digitos-original")
    orig_params = sum(p.numel() for p in original.parameters() if p.requires_grad)
    orig_size = size_mb(original)

    reduced = ReducedCNN(num_classes=len(class_index))
    reduced, red_loss, red_acc = train_model(reduced, train_dl, val_dl, test_dl, "digitos-reduzido")
    red_params = sum(p.numel() for p in reduced.parameters() if p.requires_grad)
    red_size = size_mb(reduced)

    print("\n=== Comparação: filtros originais vs reduzidos (dígitos) ===")
    print(f"Original  -> params={orig_params}, tamanho={orig_size:.4f} MB, acc_teste={orig_acc:.4f}")
    print(f"Reduzido  -> params={red_params}, tamanho={red_size:.4f} MB, acc_teste={red_acc:.4f}")
    print(f"Redução de tamanho: {(1 - red_size/orig_size)*100:.1f}%")
    print(f"Variação de acurácia: {(red_acc - orig_acc)*100:+.2f} p.p.")

    results = {
        "original": {"params": orig_params, "size_mb": orig_size, "test_accuracy": orig_acc, "test_loss": orig_loss},
        "reduced": {"params": red_params, "size_mb": red_size, "test_accuracy": red_acc, "test_loss": red_loss},
        "size_reduction_pct": (1 - red_size / orig_size) * 100,
        "accuracy_delta_pp": (red_acc - orig_acc) * 100,
    }
    with open("filter_reduction_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nResultados salvos em filter_reduction_results.json")
