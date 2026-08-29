# 🔍 Atividade 02 — Tarefa 2: Otimizar MNIST com Optuna

**Programa de Pós-Graduação em Tecnologia da Informação (PPgTI) — IMD/UFRN**
*Doutorado Profissional*

**Disciplina:** Aprendizado Profundo — Prof. Josenalde Oliveira
**Aluna:** Danielle Ballester

Sintonia de hiperparâmetros de uma rede Keras para o MNIST usando [Optuna](https://optuna.org/), a partir do notebook base do professor: [`mnist_keras.ipynb`](https://github.com/josenalde/deeplearning/blob/main/src/mnist_keras.ipynb).

⬅️ [Voltar ao índice da Atividade 02](../README.md)

---

## 📓 Notebook

- **Notebook publicado no Kaggle (executado; resultados oficiais desta entrega):** [https://www.kaggle.com/code/danielleballester/mnist-optuna-tuning-ppgti-atividade-02](https://www.kaggle.com/code/danielleballester/mnist-optuna-tuning-ppgti-atividade-02)

## ⚙️ Espaço de busca

- Número de camadas densas: 1 a 3
- Unidades por camada: 32 a 256 (escala log)
- Otimizador: Adam, SGD, RMSprop
- Taxa de aprendizagem: 1e-4 a 1e-2 (escala log)
- Batch size: 32, 64 ou 128
- Dropout: 0 a 0,5

Amostrador: `TPESampler`. Pruner: `MedianPruner` (aborta trials com desempenho abaixo da mediana dos trials já concluídos). Divisão dos dados: 50.000 treino / 10.000 validação (usada só pelo Optuna) / 10.000 teste (reservado só para a avaliação final, sem vazamento de dados).

## 📊 Resultados oficiais (execução no Kaggle, 20 trials, seed=42)

| Métrica                                                         | Valor                                    |
| ---------------------------------------------------------------- | ---------------------------------------- |
| Trials executados                                                | 20 (2 interrompidos pelo pruner:#7, #10) |
| Melhor trial                                                     | #15                                      |
| Melhor val_accuracy (Optuna)                                     | 0,9771                                   |
| **Modelo final — acurácia de teste**                     | **0,9740**                         |
| Modelo final — perda de teste                                   | 0,0819                                   |
| Baseline (1 camada×128, Adam, lr padrão) — acurácia de teste | 0,9735                                   |
| Baseline — perda de teste                                       | 0,0883                                   |

Melhores hiperparâmetros encontrados: 2 camadas densas (177 → 130 unidades), otimizador **Adam**, learning rate ≈ **0,00134**, batch size **128**, dropout ≈ **0,0037**.

Tabela completa de todos os 20 trials: [`optuna_trials.csv`](optuna_trials.csv)

## 💬 Comentário sobre o ajuste final

O modelo sintonizado pelo Optuna superou o baseline fixo em acurácia de teste (97,40% vs 97,35%, **+0,05 p.p.**) e também em perda de teste (0,0819 vs 0,0883), usando uma arquitetura com 2 camadas densas em vez de 1, learning rate menor (0,00134) e dropout praticamente nulo.

O ganho de acurácia é pequeno neste dataset — o MNIST já é "fácil" mesmo para redes simples de 1 camada — mas o processo demonstra a mecânica completa de sintonia automática de hiperparâmetros com Optuna: espaço de busca sobre arquitetura/otimizador/learning rate/batch size/dropout, amostrador TPE, pruning de trials fracos (2 de 20 abortados nesta execução) e avaliação final do melhor modelo estritamente no conjunto de teste, nunca visto durante a busca (evitando vazamento de dados entre a etapa de tuning e a avaliação final). O ganho relativo tende a ser maior em datasets/arquiteturas mais complexos, onde a escolha manual de hiperparâmetros é mais custosa e menos intuitiva do que neste caso.
