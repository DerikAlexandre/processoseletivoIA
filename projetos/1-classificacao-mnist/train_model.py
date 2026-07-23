import os

# Força o treinamento apenas em CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Pasta onde este script está localizado
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.h5")

# Reprodutibilidade
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def main():
    # ---------------------------------------------------------
    # 1. Carregamento do dataset MNIST
    # ---------------------------------------------------------
    print("Carregando o dataset MNIST...")

    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    # ---------------------------------------------------------
    # 2. Normalização e ajuste do formato
    # ---------------------------------------------------------
    # De 0-255 para 0-1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # De (28, 28) para (28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # ---------------------------------------------------------
    # 3. Split explícito de treino e validação
    # ---------------------------------------------------------
    # Embaralha os dados antes da separação
    indices = np.random.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]

    validation_size = 6000

    x_val = x_train[:validation_size]
    y_val = y_train[:validation_size]

    x_train = x_train[validation_size:]
    y_train = y_train[validation_size:]

    print(f"Amostras de treino: {len(x_train)}")
    print(f"Amostras de validação: {len(x_val)}")
    print(f"Amostras de teste: {len(x_test)}")

    # ---------------------------------------------------------
    # 4. Construção da CNN
    # ---------------------------------------------------------
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),

            # Bloco convolucional 1
            layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Bloco convolucional 2
            layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Bloco convolucional 3
            layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Classificação
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),

            # 10 classes: dígitos de 0 até 9
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # ---------------------------------------------------------
    # 5. Early Stopping
    # ---------------------------------------------------------
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    )

    # ---------------------------------------------------------
    # 6. Treinamento
    # ---------------------------------------------------------
    print("\nIniciando o treinamento...\n")

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=15,
        batch_size=128,
        callbacks=[early_stopping],
        verbose=1,
    )

    # ---------------------------------------------------------
    # 7. Avaliação final na validação
    # ---------------------------------------------------------
    val_loss, val_accuracy = model.evaluate(
        x_val,
        y_val,
        verbose=0,
    )

    print("\n----------------------------------------")
    print(f"Perda final de validação: {val_loss:.4f}")
    print(f"Acurácia final de validação: {val_accuracy:.4f}")
    print(f"Acurácia em porcentagem: {val_accuracy * 100:.2f}%")
    print("----------------------------------------")

    # Avaliação no teste, apenas como informação adicional
    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0,
    )

    print(f"Acurácia no conjunto de teste: {test_accuracy * 100:.2f}%")

    # ---------------------------------------------------------
    # 8. Salvamento do modelo
    # ---------------------------------------------------------
    model.save(MODEL_PATH)

    print(f"\nModelo salvo com sucesso em:")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()