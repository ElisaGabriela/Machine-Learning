# Transfer Learning

Este projeto explora a aplicação de Transfer Learning utilizando a rede neural convolucional VGG16 no dataset Animals-10. Foram testados dois métodos principais:

* Feature Extraction: Utilizando as camadas convolucionais do VGG16 como extrator de features, mantendo os pesos congelados e treinando apenas a camada de classificação final.
* Fine-Tuning: Ajustando os pesos de algumas camadas convolucionais superiores do VGG16 para melhorar a capacidade de generalização do modelo.

Para visualizar o passo a passo das execuções dos métodos acesse a nota do medium aqui : [🧾](https://medium.com/@elisa.lucena.127/tranfer-learning-como-reciclar-modelos-de-ml-%EF%B8%8F-bdb11f0907e2)
## Dataset
O dataset utilizado foi o Animals-10, disponível no Kaggle, que contém aproximadamente 28 mil imagens de 10 categorias de animais:

* Cão
* Gato
* Cavalo
* Aranha
* Borboleta
* Galinha
* Ovelha
* Vaca
* Esquilo
* Elefante

![img](https://cdn-images-1.medium.com/max/1100/1*W3e-jXTWFtwwW2fArKgasg.png)

## Pré-processamento de dados

As imagens do dataset apresentam diferentes tamanhos e resoluções, o que pode afetar o desempenho do modelo. Para padronizar os dados, redimensionamos todas as imagens para 224x224 pixels, que é o tamanho esperado pelo modelo VGG16. Além disso, normalizamos os valores dos pixels para a faixa [0,1] e aplicamos a padronização com a média e desvio padrão do ImageNet, garantindo compatibilidade com os pesos pré-treinados.

Vamos usar técnicas de data augmentation para aumentar a variabilidade do conjunto de treino. Esse conjunto de técnicas melhoram o desempenho dos modelos, aproveitando ao máximo os dados disponíveis. Evita overfitting, melhora a acurácia e aumenta a diversidade dos dados de treinamento. 

As técnicas utilizadas são: rotação, deslocamento, zoom e espelhamento.

## VGG16

![img](https://media.geeksforgeeks.org/wp-content/uploads/20200219152207/new41.jpg)

## Feature Extraction 👽

Feature extraction é uma técnica de transfer learning que utiliza os pesos de uma rede neural convolucional pré-treinada para extrair padrões relevantes das imagens.

A principal característica desse método é congelar as camadas convolucionais e substituir a camada final da rede. Com isso, é possível reutilizar as features aprendidas em um novo conjunto de dados.

![img](https://cdn-images-1.medium.com/max/1100/0*vOQvqcaiyAA9Wz6T.jpg)

Os resultados obtidos podem ser vistos na figura abaixo:

![img](https://cdn-images-1.medium.com/max/1100/1*d6wYzV0VmhVMz87YrrJnwA.png)

## Fine-Tuning 📏

Enquanto o Feature Extraction congela as camadas convolucionais da rede pré-treinada e substitui apenas a camada final para adaptação ao novo conjunto de dados, o Fine-Tuning descongela algumas camadas convolucionais (geralmente as últimas) e permite que elas sejam atualizadas durante o treinamento, possibilitando um ajuste fino da rede no novo domínio.

![img](https://cdn-images-1.medium.com/max/1100/0*DiUp4XsDfMEdQTth.jpg)

Os resultados obtidos podem ser vistos na figura abaixo:

![img](https://cdn-images-1.medium.com/max/1100/1*Y8JOEUAcxUqf0Gp5dblU0A.png)

## Escolha do método

A decisão entre fine-tuning e feature extraction depende da complexidade da nova tarefa, do tamanho do conjunto de dados disponível e da similaridade entre as tarefas prévia e atual. Em geral, o fine-tuning tende a oferecer melhor desempenho quando há dados suficientes disponíveis, enquanto o Feature Extraction é mais apropriado para conjuntos de dados menores ou quando o treinamento do modelo completo é computacionalmente custoso.

![img](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*PjFqIsF5KA6vTyEpnQm4Vg.jpeg)





