# Capítulo 3: Fundamentos de Engenharia de Dados

**Link do video no youtube: https://www.youtube.com/watch?v=h42yjWwEakM**

O terceiro capítulo de Designing Machine Learning Systems explora os aspectos fundamentais da engenharia de dados, incluindo as fontes, formatos e processamento de dados. Chip Huyen explica que a coleta e o armazenamento de dados variam de acordo com a origem (dados first-party, second-party, third-party) e que esses dados demandam diferentes níveis de verificação e limpeza.

O capítulo descreve os principais formatos de armazenamento de dados, como JSON, CSV e Parquet, e destaca os modelos de dados (relacional, orientado a documentos, orientado a grafos), ressaltando que a escolha de modelo e formato impacta tanto a estrutura do sistema quanto os problemas que ele pode resolver.

Para o armazenamento e processamento, o capítulo diferencia entre bancos transacionais e analíticos, bem como o processo de ETL (Extração, Transformação e Carga) – essencial para a coleta, preparação e carregamento de dados no formato desejado. Ele introduz também o conceito de dataflow, descrevendo três modos principais: banco de dados compartilhado, APIs REST e transporte em tempo real. Cada método possui suas vantagens e limitações, com base em fatores como a necessidade de baixa latência ou a interoperabilidade entre diferentes sistemas.

Por fim, o capítulo aborda o processamento em lote versus o processamento de fluxo, explicando que o processamento em tempo real é ideal para dados com alta rotatividade, enquanto o processamento em lote é mais adequado para cálculos periódicos e dados de menor frequência.
