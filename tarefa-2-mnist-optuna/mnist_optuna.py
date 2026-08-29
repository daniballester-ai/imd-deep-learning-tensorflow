"""
Tarefa 2 - Otimizar MNIST (Optuna) - Atividade 02, Aprendizado Profundo (PPGTI-UFRN)

Baseado no notebook do professor: https://github.com/josenalde/deeplearning/blob/main/src/mnist_keras.ipynb

Este script é a fonte de verdade do notebook publicado no Kaggle (`mnist_optuna.ipynb`),
gerado a partir das mesmas células. Rodar em um ambiente com TensorFlow + Optuna instalados.
"""

import json
import time

import numpy as np
import optuna
import tensorflow as tf
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_SEED = 42
N_TRIALS = 20
EPOCHS_PER_TRIAL = 5
FINAL_EPOCHS = 15

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. Dados
# ---------------------------------------------------------------------------
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Split de validação interna para o Optuna (conjunto de teste fica reservado
# só para a avaliação final do melhor modelo, evitando vazamento de dados).
val_size = 10000
x_val, y_val = x_train_full[:val_size], y_train_full[:val_size]
x_train, y_train = x_train_full[val_size:], y_train_full[val_size:]

print(f"Treino: {x_train.shape}, Validação: {x_val.shape}, Teste: {x_test.shape}")


# ---------------------------------------------------------------------------
# 2. Modelo parametrizado + função objetivo do Optuna
# ---------------------------------------------------------------------------
def build_model(trial_params):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    for i in range(trial_params["n_layers"]):
        model.add(
            layers.Dense(trial_params[f"n_units_l{i}"], activation="relu")
        )
        if trial_params["dropout"] > 0:
            model.add(layers.Dropout(trial_params["dropout"]))
    model.add(layers.Dense(10, activation="softmax"))

    optimizer_name = trial_params["optimizer"]
    lr = trial_params["lr"]
    if optimizer_name == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_name == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=lr)
    else:
        optimizer = keras.optimizers.RMSprop(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def objective(trial):
    params = {
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
    }
    for i in range(params["n_layers"]):
        params[f"n_units_l{i}"] = trial.suggest_int(f"n_units_l{i}", 32, 256, log=True)

    model = build_model(params)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS_PER_TRIAL,
        batch_size=params["batch_size"],
        callbacks=[early_stop],
        verbose=0,
    )

    val_accuracy = max(history.history["val_accuracy"])
    # Reporta ao pruner (MedianPruner decide, com base nos trials já
    # concluídos, se este trial deve ser abortado antes de terminar).
    trial.report(val_accuracy, step=EPOCHS_PER_TRIAL)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return val_accuracy


# ---------------------------------------------------------------------------
# 3. Estudo Optuna
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_SEED),
        pruner=MedianPruner(n_warmup_steps=2),
    )

    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS)
    elapsed = time.time() - t0

    print(f"\nEstudo concluído em {elapsed:.1f}s, {N_TRIALS} trials")
    print("Melhor trial:", study.best_trial.number)
    print("Melhor val_accuracy:", study.best_value)
    print("Melhores hiperparâmetros:", study.best_params)

    trials_df = study.trials_dataframe()
    trials_df.to_csv("optuna_trials.csv", index=False)
    print("\nTabela de trials salva em optuna_trials.csv")
    print(trials_df[["number", "value", "params_n_layers", "params_optimizer",
                      "params_lr", "params_batch_size", "params_dropout", "state"]]
          .to_string(index=False))

    # -----------------------------------------------------------------------
    # 4. Modelo final com os melhores hiperparâmetros
    # -----------------------------------------------------------------------
    best_params = dict(study.best_params)
    final_model = build_model(best_params)
    final_early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    final_history = final_model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=FINAL_EPOCHS,
        batch_size=best_params["batch_size"],
        callbacks=[final_early_stop],
        verbose=0,
    )

    test_loss, test_accuracy = final_model.evaluate(x_test, y_test, verbose=0)
    print(f"\nModelo final -> perda de teste: {test_loss:.4f}, acurácia de teste: {test_accuracy:.4f}")

    # Baseline simples (arquitetura fixa do notebook do professor: 1 camada
    # densa de 128 unidades, Adam, lr padrão) para comparação.
    baseline_model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    baseline_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                            metrics=["accuracy"])
    baseline_model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=FINAL_EPOCHS, batch_size=32, verbose=0,
                        callbacks=[keras.callbacks.EarlyStopping(
                            monitor="val_loss", patience=3, restore_best_weights=True)])
    baseline_test_loss, baseline_test_accuracy = baseline_model.evaluate(
        x_test, y_test, verbose=0)
    print(f"Baseline (1x128, Adam) -> perda de teste: {baseline_test_loss:.4f}, "
          f"acurácia de teste: {baseline_test_accuracy:.4f}")

    results = {
        "n_trials": N_TRIALS,
        "best_trial": study.best_trial.number,
        "best_val_accuracy": study.best_value,
        "best_params": best_params,
        "final_test_loss": float(test_loss),
        "final_test_accuracy": float(test_accuracy),
        "baseline_test_loss": float(baseline_test_loss),
        "baseline_test_accuracy": float(baseline_test_accuracy),
        "elapsed_seconds": elapsed,
    }
    with open("optuna_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nResultados salvos em optuna_results.json")
