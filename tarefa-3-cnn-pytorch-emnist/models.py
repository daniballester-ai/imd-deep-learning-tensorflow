"""Arquiteturas CNN reproduzindo as "melhores estruturas" do artigo
Silva Filho et al. (2022) sobre o Multiprova Corretor.

- Estrutura 4 (dígitos e letras, no artigo): 4 blocos [conv 3x3 ReLU + maxpool 2x2],
  32 filtros no 1º bloco e 64 nos 3 seguintes, entrada 32x32x3, flatten + densa final.
- Estrutura 3 (V/F, no artigo): mesmo padrão com apenas 3 blocos.
"""

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))


class ArticleCNN(nn.Module):
    """CNN genérica com N blocos conv+maxpool (32 filtros no 1º, 64 nos seguintes),
    seguida de flatten + camada densa final, replicando a estrutura base do artigo."""

    def __init__(self, n_blocks, num_classes, input_size=32):
        super().__init__()
        assert n_blocks >= 1
        blocks = [ConvBlock(3, 32)]
        for _ in range(n_blocks - 1):
            blocks.append(ConvBlock(32 if len(blocks) == 1 else 64, 64))
        self.features = nn.Sequential(*blocks)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            flat_size = self.features(dummy).numel()

        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(flat_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)


def CNN4Blocos(num_classes):
    """Estrutura 4 do artigo: 4 blocos conv+maxpool (dígitos e letras)."""
    return ArticleCNN(n_blocks=4, num_classes=num_classes)


def CNN3Blocos(num_classes):
    """Estrutura 3 do artigo: 3 blocos conv+maxpool (V/F)."""
    return ArticleCNN(n_blocks=3, num_classes=num_classes)


def count_conv_maxpool_blocks(model):
    return sum(1 for m in model.modules() if isinstance(m, ConvBlock))


if __name__ == "__main__":
    m4 = CNN4Blocos(num_classes=5)
    m3 = CNN3Blocos(num_classes=2)
    print("CNN4Blocos:")
    print(m4)
    print("blocos conv+maxpool:", count_conv_maxpool_blocks(m4))
    print("\nCNN3Blocos:")
    print(m3)
    print("blocos conv+maxpool:", count_conv_maxpool_blocks(m3))
