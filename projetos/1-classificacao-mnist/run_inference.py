import os

# Força a execução apenas em CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

N_SAMPLES = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.tflite")


def main():
    # ---------------------------------------------------------
    # 1. Verificação do model.tflite
    # ---------------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            'O arquivo "model.tflite" não foi encontrado. '
            'Execute primeiro train_model.py e optimize_model.py.'
        )

    # ---------------------------------------------------------
    # 2. Carregamento do modelo otimizado
    # ---------------------------------------------------------
    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Informações da entrada:")
    print(f"Shape: {input_details[0]['shape']}")
    print(f"Tipo: {input_details[0]['dtype']}")

    # ---------------------------------------------------------
    # 3. Carregamento do conjunto de teste
    # ---------------------------------------------------------
    (_, _), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    # Normalização para [0, 1]
    x_test = x_test.astype("float32") / 255.0

    # De (28, 28) para (28, 28, 1)
    x_test = np.expand_dims(x_test, axis=-1)

    # ---------------------------------------------------------
    # 4. Inferência em cinco amostras
    # ---------------------------------------------------------
    print(
        f"\nRodando inferência em {N_SAMPLES} amostras "
        "usando model.tflite:\n"
    )

    correct_predictions = 0

    for i in range(N_SAMPLES):
        # Adiciona dimensão do batch:
        # (28, 28, 1) -> (1, 28, 28, 1)
        sample = np.expand_dims(
            x_test[i],
            axis=0,
        )

        sample = sample.astype(
            input_details[0]["dtype"]
        )

        # Coloca a imagem na entrada do modelo
        interpreter.set_tensor(
            input_details[0]["index"],
            sample,
        )

        # Executa a inferência
        interpreter.invoke()

        # Obtém a saída
        prediction = interpreter.get_tensor(
            output_details[0]["index"]
        )[0]

        predicted_class = int(np.argmax(prediction))
        real_class = int(y_test[i])

        correct = predicted_class == real_class

        if correct:
            correct_predictions += 1
            status = "ACERTOU"
        else:
            status = "ERROU"

        print(
            f"Amostra {i + 1}: "
            f"predito={predicted_class} | "
            f"real={real_class} | "
            f"{status}"
        )

    print("\n----------------------------------------")
    print(
        f"Acertos nas amostras: "
        f"{correct_predictions}/{N_SAMPLES}"
    )
    print("----------------------------------------")


if __name__ == "__main__":
    main()