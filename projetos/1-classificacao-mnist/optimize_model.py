import os

# Força a execução apenas em CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

# Pasta onde este script está localizado
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

H5_MODEL_PATH = os.path.join(SCRIPT_DIR, "model.h5")
TFLITE_MODEL_PATH = os.path.join(SCRIPT_DIR, "model.tflite")


def get_file_size_mb(file_path):
    """Retorna o tamanho do arquivo em megabytes."""
    return os.path.getsize(file_path) / (1024 * 1024)


def main():
    # ---------------------------------------------------------
    # 1. Verificação do model.h5
    # ---------------------------------------------------------
    if not os.path.exists(H5_MODEL_PATH):
        raise FileNotFoundError(
            'O arquivo "model.h5" não foi encontrado. '
            'Execute primeiro o arquivo train_model.py.'
        )

    print("Carregando model.h5...")

    model = tf.keras.models.load_model(
        H5_MODEL_PATH,
        compile=False,
    )

    # ---------------------------------------------------------
    # 2. Conversão para TensorFlow Lite
    # ---------------------------------------------------------
    print("Convertendo o modelo para TensorFlow Lite...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # ---------------------------------------------------------
    # 3. Dynamic Range Quantization
    # ---------------------------------------------------------
    # Otimiza principalmente os pesos do modelo.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    # ---------------------------------------------------------
    # 4. Salvamento do model.tflite
    # ---------------------------------------------------------
    with open(TFLITE_MODEL_PATH, "wb") as file:
        file.write(tflite_model)

    original_size = get_file_size_mb(H5_MODEL_PATH)
    optimized_size = get_file_size_mb(TFLITE_MODEL_PATH)

    reduction = (
        (original_size - optimized_size) / original_size
    ) * 100

    print("\n----------------------------------------")
    print("Conversão concluída com sucesso!")
    print("Técnica: Dynamic Range Quantization")
    print(f"Tamanho do model.h5: {original_size:.2f} MB")
    print(f"Tamanho do model.tflite: {optimized_size:.2f} MB")
    print(f"Redução aproximada: {reduction:.2f}%")
    print("----------------------------------------")

    print(f"\nModelo otimizado salvo em:")
    print(TFLITE_MODEL_PATH)


if __name__ == "__main__":
    main()