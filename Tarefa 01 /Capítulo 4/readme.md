# Capítulo 4: Treinamento de Dados


**Link do vídeo no Youtube: https://youtu.be/BGtNYqZ-1XU?si=2BpReEGBanaKwH8v**


O quarto capítulo do livro Designing Machine Learning Systems explora os desafios e as melhores práticas para o uso de dados no treinamento de modelos. Chip Huyen descreve os dados como “complexos e imprevisíveis”, destacando a importância de uma abordagem cuidadosa.

O capítulo começa abordando a amostragem, uma técnica essencial para tornar o processamento mais rápido e acessível. Ele diferencia entre amostragem não probabilística (ex.: por conveniência) e probabilística, com tipos como amostragem estratificada e por importância. A amostragem de reservoir é útil em dados de streaming, enquanto a amostragem ponderada ajusta as probabilidades de seleção.

Em seguida, Huyen explica o processo de rotulagem de dados para modelos supervisionados. A rotulagem manual é cara e suscetível a ambiguidades. O capítulo destaca a importância da linhagem dos dados (rastrear a origem dos rótulos) e os rótulos naturais, que permitem uma avaliação automática. O tempo de feedback (loop de feedback) também afeta o processo de rotulagem.

Quando há falta de rótulos, o capítulo sugere métodos como supervisão fraca, semissupervisão, transfer learning e aprendizado ativo, permitindo formas alternativas e menos dependentes de rotulagem manual.

Para classes desbalanceadas, o capítulo recomenda métricas adequadas e técnicas de reamostragem (oversampling e undersampling) para balancear os dados. Métodos de nivelamento também ajustam os pesos das instâncias sem alterar a distribuição.

Por fim, o capítulo trata da aumento de dados (data augmentation) para melhorar a robustez dos modelos, incluindo:

* Transformações de preservação de rótulo, que modificam as amostras sem alterar os rótulos.
* Perturbação para robustez contra ataques adversários, adicionando ruído aos dados de treinamento.
* Síntese de dados para aumentar a quantidade e variedade dos dados de treinamento.
