# Capítulo 5: Engenharia de Features

**Link do vídeo no youtube : https://youtu.be/xuLNSnXJff8**

O capítulo 5 do livro de Chip Huyen foca na engenharia de features, destacando a diferença entre features aprendidas, típicas do aprendizado profundo, e as projetadas, que são mais comuns em aplicações de machine learning em produção. A autora discute várias operações de engenharia de features, começando com o tratamento de valores ausentes, que pode ser feito por meio da exclusão de dados ou imputação com média, mediana ou moda, sem uma solução única que sirva para todos os casos.

O capítulo também aborda a importância do escalonamento de features para garantir intervalos semelhantes e a discretização, que converte features contínuas em discretas. Para a codificação de features categóricas, o uso do hashing trick é sugerido, reconhecendo o risco de colisões, mas enfatizando que o impacto dessas colisões é geralmente pequeno. Técnicas como feature crossing e embeddings posicionais são apresentadas como formas de melhorar a modelagem de relações não lineares e a compreensão da ordem de elementos em sequências.

Um tema central é o data leakage, que ocorre quando informações do rótulo influenciam as features durante a predição. O autor oferece recomendações para evitar esse fenômeno, como dividir dados por tempo e escalonar após essa divisão. Além disso, discute-se a importância e a generalização das features, ressaltando que um número excessivo de features pode levar a problemas de sobreajuste, aumento da latência e a necessidade de uma seleção cuidadosa para garantir a eficácia do modelo.

Em síntese, a engenharia de features é essencial para o desempenho de modelos de machine learning, e o capítulo fornece diretrizes práticas para sua aplicação.
