# AI_image_health

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)
[![author](https://img.shields.io/badge/author-RafaelGallo-red.svg)](https://github.com/RafaelGallo?tab=repositories) 
[![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-374/) 
[![](https://img.shields.io/badge/R-3.6.0-red.svg)](https://www.r-project.org/)
[![](https://img.shields.io/badge/ggplot2-white.svg)](https://ggplot2.tidyverse.org/)
[![](https://img.shields.io/badge/dplyr-blue.svg)](https://dplyr.tidyverse.org/)
[![](https://img.shields.io/badge/readr-green.svg)](https://readr.tidyverse.org/)
[![](https://img.shields.io/badge/ggvis-black.svg)](https://ggvis.tidyverse.org/)
[![](https://img.shields.io/badge/Shiny-red.svg)](https://shiny.tidyverse.org/)
[![](https://img.shields.io/badge/plotly-green.svg)](https://plotly.com/)
[![](https://img.shields.io/badge/XGBoost-red.svg)](https://xgboost.readthedocs.io/en/stable/#)
[![](https://img.shields.io/badge/Tensorflow-orange.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Keras-red.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/CUDA-gree.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Caret-orange.svg)](https://caret.tidyverse.org/)
[![](https://img.shields.io/badge/Pandas-blue.svg)](https://pandas.pydata.org/) 
[![](https://img.shields.io/badge/Matplotlib-blue.svg)](https://matplotlib.org/)
[![](https://img.shields.io/badge/Seaborn-green.svg)](https://seaborn.pydata.org/)
[![](https://img.shields.io/badge/Matplotlib-orange.svg)](https://scikit-learn.org/stable/) 
[![](https://img.shields.io/badge/Scikit_Learn-green.svg)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/Numpy-white.svg)](https://numpy.org/)
[![](https://img.shields.io/badge/PowerBI-red.svg)](https://powerbi.microsoft.com/pt-br/)

![Logo](https://img.freepik.com/vetores-gratis/infografico-medico-tecnologico-com-detalhes_23-2148521123.jpg?w=740&t=st=1685905608~exp=1685906208~hmac=00c2c1160cebd6283d2e2f1b2739edd80617ca1efcf5e52986c9687944cebcc2)

Um Estudo sobre a Identificação de Casos de Câncer por meio da Classificação de Imagens Utilizando Redes Neurais Convolucionais com TensorFlow e Keras

# Introdução
A detecção precoce de casos de câncer é fundamental para aumentar as chances de sucesso no tratamento e salvar vidas. Neste contexto, a área da saúde tem explorado as possibilidades oferecidas pelo deep learning e pelas redes neurais convolucionais para a classificação de imagens médicas. Este estudo tem como objetivo investigar a aplicação dessas técnicas, utilizando as bibliotecas TensorFlow e Keras, a fim de identificar casos de câncer em imagens médicas.

# Metodologia
Para realizar a identificação de casos de câncer, empregamos uma rede neural convolucional, uma arquitetura de aprendizado profundo que se destaca na análise de dados de imagem. Utilizamos as bibliotecas TensorFlow e Keras, que oferecem uma combinação poderosa para construir e treinar modelos de deep learning.

Nosso conjunto de dados consistiu em uma ampla variedade de imagens médicas, incluindo radiografias, ressonâncias magnéticas e tomografias de pacientes com e sem câncer. Realizamos um processo de pré-processamento nas imagens para normalizá-las e redimensioná-las, a fim de garantir uma entrada consistente para o modelo. Em seguida, dividimos o conjunto de dados em conjuntos de treinamento e teste para avaliar a eficiência do modelo em dados não vistos anteriormente.

# Arquitetura
A arquitetura da rede neural convolucional foi construída com várias camadas convolucionais, camadas de pooling e camadas totalmente conectadas. Utilizamos a função de perda de entropia cruzada para otimizar os pesos do modelo, empregando o algoritmo de otimização Adam durante o treinamento. Foram realizadas múltiplas épocas de treinamento para aprimorar gradualmente a precisão do modelo.

# Conclusão
Os resultados deste estudo demonstraram a eficácia das redes neurais convolucionais no auxílio à identificação de casos de câncer por meio da análise de imagens médicas. A aplicação das bibliotecas TensorFlow e Keras proporcionou uma abordagem flexível e de alto desempenho para a implementação de modelos de deep learning voltados à área da saúde.

Por meio da análise dos dados de teste, observamos uma precisão significativa na detecção de casos de câncer nas imagens médicas. Esses resultados evidenciam o potencial das redes neurais convolucionais para auxiliar os profissionais de saúde na tomada de decisões precisas e rápidas, possibilitando um diagnóstico precoce e um tratamento mais eficaz. Este estudo destaca a importância de explorar as técnicas de deep learning na área da saúde, especificamente na detecção de câncer por meio da classificação de imagens. Com o apoio das redes neurais convolucionais e das ferramentas fornecidas pelo TensorFlow e Keras, podemos aprimorar ainda mais a precisão e a eficiência desses modelos, contribuindo assim para um avanço significativo na área médica. À medida que a tecnologia continua a evoluir, é essencial investir em pesquisas e desenvolvolvimento para aprimorar os modelos de classificação de imagens médicas. A combinação do deep learning, redes neurais convolucionais e das bibliotecas TensorFlow e Keras oferece um caminho promissor para avançar no diagnóstico e tratamento de doenças como o câncer.

Além disso, é importante ressaltar que a utilização dessas tecnologias não tem o objetivo de substituir a expertise dos profissionais de saúde, mas sim de fornecer um suporte adicional na interpretação e análise das imagens médicas. Os modelos de classificação de imagens podem auxiliar na triagem e na identificação de padrões complexos, permitindo que os especialistas concentrem seus esforços em áreas de maior relevância clínica. No entanto, é crucial mencionar que a implementação desses modelos requer um cuidado especial em relação à qualidade dos dados, à interpretabilidade dos resultados e à ética na utilização das informações médicas. É fundamental garantir a privacidade e a segurança dos pacientes, bem como validar e aprimorar continuamente os modelos em diferentes contextos clínicos. 

Em conclusão, este estudo evidencia os avanços e as possibilidades proporcionadas pela aplicação de redes neurais convolucionais, deep learning e das bibliotecas TensorFlow e Keras na área da saúde, mais especificamente na identificação de casos de câncer por meio da classificação de imagens médicas. Com o desenvolvimento contínuo dessas tecnologias, espera-se uma melhoria significativa no diagnóstico precoce e no tratamento de doenças, possibilitando uma abordagem mais assertiva e eficiente na área da oncologia e, consequentemente, um impacto positivo na vida dos pacientes.

## Stack utilizada

**Programação** Python

**Deep learning**: Tensorflow, keras, opencv

**Análise de dados**: Seaborn, Matplotlib.

## Dataset

| Dataset               | Link                                                |
| ----------------- | ---------------------------------------------------------------- |
| RSNA Breast Cancer Detection | [Projeto - Notebook](https://www.kaggle.com/datasets/theoviel/rsna-breast-cancer-512-pngs)|
| Blood Cancer - Image | [Projeto - Notebook](https://www.kaggle.com/datasets/akhiljethwa/blood-cancer-image-dataset/code)|
| Melanoma Skin Cancer | [Projeto - Notebook](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)|
| Skin Cancer | [Projeto - Notebook](https://www.kaggle.com/datasets/pattnaiksatyajit/skin-cancer)|
| Lung and Colon Cancer Histopathological | [Projeto - Notebook](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)|

## Variáveis de Ambiente

Para rodar esse projeto, você vai precisar adicionar as seguintes variáveis de ambiente no seu .env

`API_KEY`

`ANOTHER_API_KEY`


## Instalação

Instalação das bibliotecas para esse projeto no python.

```bash
  conda install pandas 
  conda install scikitlearn
  conda install numpy
  conda install scipy
  conda install matplotlib

  python==3.6.4
  numpy==1.13.3
  scipy==1.0.0
  matplotlib==2.1.2
```
Instalação do Python É altamente recomendável usar o anaconda para instalar o python. Clique aqui para ir para a página de download do Anaconda https://www.anaconda.com/download. Certifique-se de baixar a versão Python 3.6. Se você estiver em uma máquina Windows: Abra o executável após a conclusão do download e siga as instruções. 

Assim que a instalação for concluída, abra o prompt do Anaconda no menu iniciar. Isso abrirá um terminal com o python ativado. Se você estiver em uma máquina Linux: Abra um terminal e navegue até o diretório onde o Anaconda foi baixado. 
Altere a permissão para o arquivo baixado para que ele possa ser executado. Portanto, se o nome do arquivo baixado for Anaconda3-5.1.0-Linux-x86_64.sh, use o seguinte comando: chmod a x Anaconda3-5.1.0-Linux-x86_64.sh.

Agora execute o script de instalação usando.


Depois de instalar o python, crie um novo ambiente python com todos os requisitos usando o seguinte comando

```bash
conda env create -f environment.yml
```
Após a configuração do novo ambiente, ative-o usando (windows)
```bash
activate "Nome do projeto"
```
ou se você estiver em uma máquina Linux
```bash
source "Nome do projeto" 
```
Agora que temos nosso ambiente Python todo configurado, podemos começar a trabalhar nas atribuições. Para fazer isso, navegue até o diretório onde as atribuições foram instaladas e inicie o notebook jupyter a partir do terminal usando o comando
```bash
jupyter notebook
```

## Demo rede neural convolucional

```
# Importação das bibliotecas
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Carregar o conjunto de dados
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Pré-processamento dos dados
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Definir a arquitetura do modelo
model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ]
)

# Compilar o modelo
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Treinar o modelo
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1)

# Avaliar o modelo com os dados de teste
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Acurácia do teste:", test_acc)


## Aplicação em R
library(keras)

# Carregar o conjunto de dados
cifar <- dataset_cifar10()
x_train <- cifar$train$x
y_train <- cifar$train$y
x_test <- cifar$test$x
y_test <- cifar$test$y

# Pré-processamento dos dados
x_train <- array_reshape(x_train, c(nrow(x_train), 32, 32, 3))
x_test <- array_reshape(x_test, c(nrow(x_test), 32, 32, 3))
x_train <- x_train / 255
x_test <- x_test / 255
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# Definir a arquitetura do modelo
model <- keras_model_sequential()
model %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(32, 32, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 10, activation = "softmax")

# Compilar o modelo
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)

# Treinar o modelo
model %>% fit(
  x_train, y_train,
  batch_size = 64,
  epochs = 10,
  validation_split = 0.1
)

# Avaliar o modelo com os dados de teste
model %>% evaluate(x_test, y_test)



```

## Melhorias
Que melhorias você fez no seu código? 
- Ex: refatorações, melhorias de performance, acessibilidade, etc


## Suporte
Para suporte, mande um email para rafaelhenriquegallo@gmail.com
