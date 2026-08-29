# 🧩 Atividade 02 — Tarefa 3: Reprodução em PyTorch das CNNs do artigo Silva Filho et al. (2022)

**Programa de Pós-Graduação em Tecnologia da Informação (PPgTI) — IMD/UFRN**
*Doutorado Profissional*

**Disciplina:** Aprendizado Profundo — Prof. Josenalde Oliveira
**Aluna:** Danielle Ballester

Reprodução em PyTorch das melhores arquiteturas de CNN identificadas no artigo [Silva Filho et al. (2022) — Classificação de Caracteres Manuscritos para Correção Automática do Sistema Multiprova](https://www.researchgate.net/publication/365834221), aplicadas a subconjuntos do [EMNIST](https://www.kaggle.com/datasets/crawford/emnist) (dígitos 1-5, letras A-E, V/F).

⬅️ [Voltar ao índice da Atividade 02](../README.md)

---

## 📓 Notebook

**Notebook publicado no Kaggle (executado; resultados oficiais desta entrega):** https://www.kaggle.com/code/danielleballester/cnn-pytorch-emnist-ppgti-atividade-02

## 🗂️ Dados: subconjuntos EMNIST

| Subconjunto | Origem no EMNIST | Classes | Amostras/classe (treino) |
|---|---|---|---|
| Dígitos | `emnist-digits` (label = valor do dígito) | 1, 2, 3, 4, 5 | ~2.100 |
| Letras | `emnist-letters` (label 1=A ... 26=Z) | A, B, C, D, E | ~2.100 |
| V/F | `emnist-letters` (label 22=V, label 6=F) | V, F | ~2.100 |

O EMNIST **não possui uma classe nativa "V ou F"** — o subconjunto V/F foi derivado filtrando apenas as classes de letras V e F do `emnist-letters`, tratando o problema como classificação binária (decisão registrada em `design.md` da mudança OpenSpec correspondente).

Pré-processamento: correção de orientação do EMNIST (as imagens vêm transpostas/espelhadas em relação ao MNIST), redimensionamento de 28×28 grayscale para 32×32×3 (canal replicado), e divisão 70% treino / 15% validação / 15% teste.

## 🏗️ Arquiteturas (reproduzindo o artigo)

- **Estrutura 4** (dígitos e letras): 4 blocos `Conv2d(3x3) + ReLU + MaxPool(2x2)` — 32 filtros no 1º bloco, 64 nos 3 seguintes — seguidos de `Flatten` + `Linear` final. Otimizador Adam.
- **Estrutura 3** (V/F): mesmo padrão, com apenas 3 blocos conv+maxpool.

Ver [`models.py`](models.py) para a implementação (`CNN4Blocos`, `CNN3Blocos`).

## 📊 Resultados oficiais (execução no Kaggle)

| Modelo | Classes | Parâmetros treináveis | Tamanho (.pt) | Acurácia teste | Perda teste |
|---|---|---|---|---|---|
| Dígitos (Estrutura 4) | 5 | 94.533 | 0,3644 MB | **99,29%** | 0,0210 |
| Letras (Estrutura 4) | 5 | 94.533 | 0,3644 MB | **97,16%** | 0,0804 |
| V/F (Estrutura 3) | 2 | 58.370 | 0,2260 MB | **99,67%** | 0,0081 |

Resultados agregados: [`training_results.json`](training_results.json) · modelos exportados: [`models_out/`](models_out) (`.pt`)

Essas acurácias são compatíveis com a faixa relatada no artigo original (93–99%), ainda que os valores exatos não sejam diretamente comparáveis — o artigo foi treinado em um dataset proprietário de cartões-resposta reais do Multiprova, enquanto aqui usamos o EMNIST como substituto acadêmico acessível (ver `design.md` — Risks/Trade-offs).

## 🗜️ Redução do tamanho dos modelos

Duas técnicas foram comparadas (análise completa no notebook):

| Técnica | Aplicada em | Tamanho antes | Tamanho depois | Redução | Δ Acurácia |
|---|---|---|---|---|---|
| Quantização dinâmica pós-treino (`quantize_dynamic`, só `nn.Linear`) | Modelo de letras | 0,3644 MB | 0,3616 MB | **0,8%** | não reavaliado (afeta só a camada densa) |
| Redução de filtros (32→64→64→64 para 16→32→32→32) | Modelo de dígitos | 0,3644 MB | 0,0963 MB | **73,6%** | **+0,22 p.p.** (99,29% → 99,51%) |

**Por que a quantização teve efeito mínimo:** a quantização dinâmica em modo eager do PyTorch só suporta `nn.Linear`/`nn.LSTM`, não `nn.Conv2d` — e nesta arquitetura quase todos os parâmetros estão nas camadas convolucionais, não na camada densa final. Para reduzir de fato o tamanho das convoluções seria necessário quantização estática (calibrada com dados) ou *quantization-aware training*.

**Por que a redução de filtros funcionou bem:** reduzir os filtros pela metade em cada bloco corta diretamente os parâmetros das camadas que concentram o peso do modelo, e a capacidade residual ainda é suficiente para o problema (poucas classes, imagens simples) — daí a acurácia não cair (e até subir ligeiramente nesta execução).

**Outras técnicas discutidas (não aplicadas empiricamente, por tempo):**
- **Poda estruturada (pruning):** remove canais/filtros de menor magnitude após o treino — efeito parecido ao da redução de filtros, mas decidido a partir do modelo já treinado em vez de definido a priori na arquitetura.
- **Destilação de conhecimento (distillation):** treina um modelo "aluno" menor para imitar as saídas do modelo "professor", tendendo a preservar mais acurácia do que só reduzir a arquitetura e retreinar do zero.

## 📁 Estrutura de arquivos

- [`data_utils.py`](data_utils.py) — carregamento e pré-processamento dos 3 subconjuntos EMNIST
- [`models.py`](models.py) — arquiteturas `CNN4Blocos` / `CNN3Blocos`
- [`train.py`](train.py) — treinamento, avaliação e exportação dos 3 modelos
- [`quantize.py`](quantize.py) — quantização dinâmica pós-treino
- [`reduce_filters.py`](reduce_filters.py) — comparação empírica com filtros reduzidos
- [`build_notebook.py`](build_notebook.py) — gera o notebook Kaggle a partir dos scripts acima
- [`models_out/`](models_out) — modelos treinados exportados (`.pt`)
