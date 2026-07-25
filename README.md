## 📝 Relatório do Candidato

👤 **Nome Completo:** Derik Alexandre Alves de Andrade

### 1️⃣ Resumo da Arquitetura do Modelo

O projeto utiliza uma CNN com três blocos convolucionais. Cada bloco é formado por uma camada `Conv2D`, uma camada de `BatchNormalization` e uma camada de `MaxPooling2D`.

A quantidade de filtros aumenta em cada bloco, sendo 32 no primeiro, 64 no segundo e 128 no terceiro.
Após os blocos convolucionais, são utilizadas uma camada `Flatten`, uma camada densa com 128 neurônios, uma camada `Dropout` de 0.4 e uma camada de saída com 10 neurônios e ativação `softmax`, correspondente aos dígitos de 0 a 9.

As imagens do MNIST são normalizadas para valores entre 0 e 1 e têm o formato ajustado para `28x28x1`.
Também é feita uma separação manual dos dados, deixando 54.000 imagens para treinamento e 6.000 para validação.

O treinamento é configurado para no máximo 15 épocas. O `EarlyStopping` acompanha a perda de validação e utiliza paciência de duas épocas. O treinamento foi encerrado na época 7 e os pesos da época 5 foram restaurados, pois essa época apresentou o melhor resultado de validação.

Todo o treinamento foi realizado utilizando apenas a CPU.

### 2️⃣ Bibliotecas Utilizadas

As principais bibliotecas utilizadas foram:

* `TensorFlow`, foi utilizado para criação, treinamento, salvamento e conversão do modelo;
* `Keras`, por meio do TensorFlow, para construção das camadas da rede;
* `NumPy`, para normalização e organização das imagens;
* `os`, para trabalhar com os caminhos dos arquivos e configurar o uso da CPU.

### 3️⃣ Técnica de Otimização do Modelo

A técnica utilizada para otimizar o modelo foi a **Dynamic Range Quantization**, aplicada durante a conversão para TensorFlow Lite.

No código, a otimização foi configurada da seguinte forma:

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```

Depois da aplicação dessa técnica, o arquivo `model.h5` foi convertido para `model.tflite`.

O objetivo da otimização foi diminuir o tamanho do modelo, tornando-o mais adequado para dispositivos com menos memória e menor capacidade de processamento.

### 4️⃣ Resultados Obtidos

Os resultados obtidos foram:

| Resultado                      |       Valor |
| ------------------------------ | ----------: |
| Acurácia de validação          |  **99,00%** |
| Perda de validação             |  **0,0365** |
| Acurácia no conjunto de teste  |  **98,92%** |
| Tamanho do `model.h5`          | **2,84 MB** |
| Tamanho do `model.tflite`      | **0,24 MB** |
| Redução de tamanho             |  **91,39%** |
| Acertos no teste de inferência |  **5 de 5** |

A conversão reduziu bastante o tamanho do arquivo. Mesmo após a otimização, o modelo continuou funcionando corretamente nas amostras testadas.

### 5️⃣ Exemplo de Inferência

Saída obtida ao executar o arquivo `run_inference.py`:

```text
Informações da entrada:
Shape: [ 1 28 28  1]
Tipo: <class 'numpy.float32'>

Rodando inferência em 5 amostras usando model.tflite:

Amostra 1: predito=7 | real=7 | ACERTOU
Amostra 2: predito=2 | real=2 | ACERTOU
Amostra 3: predito=1 | real=1 | ACERTOU
Amostra 4: predito=0 | real=0 | ACERTOU
Amostra 5: predito=4 | real=4 | ACERTOU

----------------------------------------
Acertos nas amostras: 5/5
----------------------------------------
```

Nas cinco amostras testadas, o valor previsto pelo modelo foi igual ao valor real.

Com esse teste, foi possível confirmar que o arquivo `model.tflite` foi carregado corretamente e realizou as classificações esperadas nos testes.
