"""Aplica quantização dinâmica pós-treino ao modelo de letras (o maior dos 3)
e mede a redução de tamanho obtida, para a análise de redução de modelo
exigida pelo item 3.2 da Tarefa 3."""

import io
import json

import torch
from torch import nn

from models import CNN4Blocos

with open("training_results.json", encoding="utf-8") as f:
    results = json.load(f)["results"]

model = CNN4Blocos(num_classes=len(results["letras"]["classes"]))
model.load_state_dict(torch.load("models_out/cnn_letras.pt", weights_only=True))
model.eval()

quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

buffer = io.BytesIO()
torch.save(quantized_model.state_dict(), buffer)
size_mb_quantized = len(buffer.getvalue()) / (1024 * 1024)
torch.save(quantized_model.state_dict(), "models_out/cnn_letras_quantized.pt")

size_mb_original = results["letras"]["size_mb"]
reduction_pct = (1 - size_mb_quantized / size_mb_original) * 100

print(f"Tamanho original (letras):    {size_mb_original:.4f} MB")
print(f"Tamanho quantizado (letras):  {size_mb_quantized:.4f} MB")
print(f"Redução: {reduction_pct:.1f}%")

quant_results = {
    "model": "letras",
    "technique": "dynamic_quantization_qint8_linear",
    "size_mb_original": size_mb_original,
    "size_mb_quantized": size_mb_quantized,
    "reduction_pct": reduction_pct,
}
with open("quantization_results.json", "w", encoding="utf-8") as f:
    json.dump(quant_results, f, indent=2, ensure_ascii=False)
print("\nResultados salvos em quantization_results.json")
