# Tarefa 3: Explorando o Capítulo 6 – "Deep Learning with PyTorch"

##### Nome: Elisa Gabriela Machado de Lucena

* [Link do artigo no medium](https://medium.com/@elisa.lucena.127/ml-com-pytorch-adaptive-learning-rates-momentum-e-learning-rate-schedulers-211db8b9e367)

* [Notebook do desenvolvimento do trabalho](https://github.com/ElisaGabriela/Machine-Learning/blob/main/Tarefa%2003/Tarefa3.ipynb)


Este repositório tem como objetivo explorar alguns conceitos do capítulo 6 do livro 'Deep Learning with PyTorch" de Daniel Godoy:

* EWMA meets gradients: Como as médias móveis exponencialmente ponderadas são usadas para suavizar os gradientes e seu impacto na atualização de parâmetros em otimizadores modernos;
* Adam: Funcionamento do otimizador adam, explicando o papel dos seus parâmetros na adaptação dos gradientes e na estabilidade do treinamento;
* Vizualização dos gradientes adaptados: Vizualizações que ilustram como os gradientes adaptados evoluem durante o treinamento, detacando os efeitos das médias móveis e da escala dos gradientes;
* SGD: Funcionamento do SGD básico e sua evolução para as variantes com momentum e nesterov, explicando a intuição por trás dessas melhorias;
* Learning rate schedulers: Como diferentes schedulers de learning rate podem ser aplicados, explicando a teoria e implementando exemplos práticos;
* Hora da prática: Desenvolvimento de um exemplo completo com base nos tópicos acima, integrando EWMA, adam, SGD e Learning rate schedulers.

# Learning Rate

Bom, first things first, vamos falar sobre a learning rate (ou taxa de aprendizagem). Carinhosamente apelidado de LR, é um hiperparâmetro usado no treinamento de redes neurais e em algoritmos de otimização. De maneira simples: ele que define quão rápido o modelo aprende.

De maneira mais formal: ele controla o tamanho do passo que o modelo dá na direção oposta ao gradiente durante a atualização dos pesos da rede. Sugerimos que se você nunca ouviu falar de learning rate, volte algumas casas e dê uma olhada mais aprofundada nesse conceito.

Escolher o LR ideal para o seu problema não é uma tarefa simples… muitas vezes essa escolha fica na mão da tentativa e erro.

É comum reduzir a taxa de aprendizado por um fator de 3 ou um fator de 10. Assim, seus valores de taxa de aprendizado poderiam ser algo como [0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4, 1e-4] (usando um fator de 3) ou [0.1, 0.01, 1e-3, 1e-4, 1e-5] (usando um fator de 10).

![Curva de perda em função da Taxa de Aprendizado](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*NQE0mUfcUYB9vMagbs4OAg.png)

![Efeito de diferentes taxas de aprendizado no custo durante o treinamento](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*tFHuyFDeiPrE6Op6KXTBrg.png)

Bom, se você é do tipo que presta atenção aos detalhes, deve ter reparado que eu comecei comentando que o LR também estava presente nos otimizadores.

Podemos atribuir um scheduler a um otimizador, de modo que ele atualize a taxa de aprendizado ao longo do treinamento. Vamos nos aprofundar mais nos schedulers de taxa de aprendizado daqui a alguns tópicos.

Dentro das possibilidades do learning rate a gente tem o Adaptive Learning Rate (Taxa de aprendizagem adaptativa), que como o nome sugere, é dinamicamente ajustada com base na performance e no gradiente da função custo.

Você já ouviu falar do otimizador Adam, certo? Ele é um bom exemplo do Adaptive Learning Rate! ele começa com a taxa de aprendizado fornecida como argumento, mas adapta essa taxa ao longo do treinamento, ajustando-a de forma diferente para cada parâmetro do modelo. Ou mais ou menos isso, na verdade, o Adam não adapta diretamente a taxa de aprendizado — ele adapta os gradientes. Mas, como a atualização dos parâmetros é feita pela multiplicação da taxa de aprendizado e do gradiente, essa diferença se torna irrelevante na prática.

O Adam combina as características de dois outros otimizadores:

* SGD com momentum: Ele usa uma média móvel dos gradientes em vez dos gradientes diretamente (isso é conhecido como o primeiro momento em termos estatísticos);
* RMSProp: Ele escala os gradientes usando uma média móvel dos gradientes ao quadrado (o segundo momento, ou variância não centralizada).
  
Complicou? Bom, vamos conhecer um pouco sobre essa média móvel, ou melhor ainda, sobre a média móvel exponencialmente ponderada, e depois a gente se aprofunda melhor no Adam.

# EWMA

EWMA é uma média móvel exponencialmente ponderada, e como o nome sugere, ela é diferente da média simples… Ela fornece um peso maior para valores mais recentes, se destacando na suavização dos dados.

No Machine Learning, usamos o EWMA para suavizar os gradientes durante o treinamento. Pode parecer simples, mas impede que gradientes muito instáveis atrapalhem a convergência do modelo. Essa abordagem é especialmente útil em algoritmos de otimização como o Adam, que dependem de cálculos de médias móveis para ajustar a taxa de aprendizado de forma adaptativa.

Visualizar o comportamento dos gradientes suavizados com EWMA pode ajudar a entender melhor a estabilidade do treinamento e identificar possíveis problemas, como gradientes explosivos ou o desaparecimento de gradientes.

Vamos entender o EWMA comparando com a média móvel simples (MA).

A Média Móvel Simples (MA) calcula a média aritmética de um conjunto de valores ao longo de um período fixo. Ela é usada para suavizar flutuações nos dados e destacar tendências.

Por sua vez, a Média Móvel Exponencialmente Ponderada (EWMA) atribui pesos exponenciais decrescentes aos valores passados, dando mais importância aos valores mais recentes.

Quando α está próximo de 1: os valores mais recentes recebem mais peso, resultando em uma curva mais sensível a mudanças. Quando α está próximo de 0: os valores antigos ainda influenciam bastante, resultando em uma curva mais suave e lenta para reagir a mudanças.

Vamos tentar visualizar essa diferença com um exemplo simples:

![gráfico EWMA](https://miro.medium.com/v2/resize:fit:786/format:webp/1*13t8PurrEYDSOMuF-IsLdw.png)

O gráfico compara a Média Móvel Exponencialmente Ponderada (EWMA) e a Média Móvel Simples (MA) usando 5 períodos. O eixo Y diz respeito ao peso atribuido a cada ponto de dado anterior ao calcular a média, enquanto o eixo X representa o número de passos (ou períodos) anterires ao ponto atual (quanto maior o lag, mas distante no passado está o dado).

Pela imagem a gente analisa o seguinte: a EWMA atribui pesos decrescentes exponencialmente ao longo do tempo. Os valores mais recentes têm mais influência no cálculo da média, enquanto os valores mais antigos têm menos peso. Isso quer dizer que a EWMA reage mais rapidamente a mudanças recentes nos dados, pois dá mais peso ao ponto atual.

Já a MA, atribui pesos iguais para os últimos 5 períodos. Assim, os últimos 5 pontos contribuem igualmente para o cálculo da média, enquanto os mais antigos não têm influência. Ou seja, quando comparado a EWMA, A MA reage mais lentamente e suaviza os dados de forma mais uniforme, sem priorizar os valores recentes.

Essa diferença é importante no contexto de otimização de redes neurais, pois a EWMA ajuda a estabilizar o treinamento ao suavizar os gradientes de forma mais dinâmica.

# O problema do viés

O Bias-Corrected EWMA (Média Móvel Exponencialmente Ponderada Corrigida para Viés) é uma variação da EWMA que corrige o viés introduzido no início do cálculo de uma média móvel exponencialmente ponderada.

Quando o cálculo de uma EWMA começa, os primeiros valores podem estar enviesados para valores menores, pois a série temporal ainda não acumulou informações suficientes. Isso ocorre porque o cálculo depende fortemente dos valores iniciais e a média tende a subestimar os valores reais no início. Para corrigir esse viés, multiplicamos o termo da EWMA por um fator de correção de viés. Essa correção é especialmente usada em algoritmos como o Adam, para melhorar a estabilidade no início do treinamento, já que o otimizador utiliza EWMA tanto para os gradientes quanto para os gradientes ao quadrado.

![gráfico bias](https://github.com/user-attachments/assets/9b1beda2-97cc-47cf-bc5b-ddf4259ff02d)

A Figura acima mostra um exemplo prático da comparação da Média Móvel Simples (MA), da EWMA e a EWMA com correção do viés (Bias). Como esperado, a EWMA sem correção (linha vermelha tracejada) está bem distante no início, enquanto a média móvel regular (linha preta tracejada) acompanha os valores reais de forma muito mais próxima. No entanto, a EWMA corrigida faz um ótimo trabalho ao acompanhar os valores reais desde o início. De fato, após 19 dias, as duas EWMAs são quase indistinguíveis.

# Adam

Voltando para o Adam, vamos entender o que tem por trás. Já sabemos que o otimizador Adam utiliza a média móvel exponencialmente ponderada (EWMA) para ajustar os gradientes durante o treinamento, vamos dar uma olhada em como isso acontece.

Para cada parâmetro do modelo, o Adam calcula duas EWMA:

* EWMA dos gradientes — para suavizar os gradientes.
* EWMA do quadrado dos gradientes — para escalonar os gradientes e controlar a variância.


Essas médias são corrigidas para viés no início do treinamento.

β1 e β2 são hiperparâmetros que controlam o decaimento exponencial. Os valores típicos são 0.9 e 0.999.

Essa abordagem ajuda a tornar o processo de otimização mais estável e eficiente, especialmente para problemas complexos com gradientes ruidosos.

Vamos falar de seis argumentos para o Adam no PyTorch, a maioria já mencionado por aqui.

* Primeiro o params , que são os parâmetros do modelo que o otimizador irá atualizar durante o treinamento. Normalmente, é passado como model.parameters() ao inicializar o otimizador.
* o lr(learning rate), a taxa de aprendizado, que controla o tamanho dos passos na direção do gradiente durante a atualização dos parâmetros. Valor padrão: 1e-3 (0.001).
* beta1: controla a suavização dos gradientes.
* beta2: controla a suavização dos gradientes ao quadrado (para a escala).
* eps(epsilon), um pequeno valor constante (normalmente 1e-8) adicionado ao denominador para evitar divisão por zero e garantir estabilidade numérica.

Esses quatro argumentos são os principais para o funcionamento básico do Adam, enquanto weight_decay e amsgrad são argumentos opcionais que controlam regularização e variações do algoritmo.

* weight_decay vai adicionar uma regularização L2 para prevenir overfitting, reduzindo o valor dos pesos ao longo do tempo.
* amsgrad é uma variante do Adam que ajusta a forma de calcular o gradiente adaptado para maior estabilidade.
  
Esses ajustes tornam o Adam mais robusto e controlado, especialmente em treinamentos de redes neurais profundas.

# Visualizando os Gradientes Adaptativos

Agora que a gente entende melhor os gradientes, que tal gerar uma vizualização? Os códigos utilizados para gerar as visualizações são do livro Deep Learning with PyTorch Step-by-Step. Apesar do livro possuir seu próprio repositório no Github, o código comentado passo a passo pode ser visto [aqui](https://github.com/ElisaGabriela/Machine-Learning/blob/main/Tarefa%2003/Tarefa3.ipynb).


Vamos usar um problema simples de regressão linear, executar o loop de treinamento para que possamos registrar os gradientes. Vamos ilustrar os efeitos de diferentes parâmetros na minimização da perda.

![image](https://github.com/user-attachments/assets/816b7623-3427-4f80-9a8b-f198ca8cb32c)


No gráfico à esquerda, vemos que o EWMA corrigido por viés dos gradientes (em vermelho) suaviza os gradientes. No centro, o EWMA corrigido por viés dos gradientes ao quadrado é usado para escalar os gradientes suavizados. À direita, ambos os EWMAs são combinados para calcular os gradientes adaptados.

Internamente, o Adam mantém dois valores para cada parâmetro, exp_avg e exp_avg_sq, representando os EWMAs (não corrigidos) para gradientes e gradientes ao quadrado, respectivamente.


Agora que tal comparar com o SGD? Temos discutido como a atualização do parâmetro é diferente, mas agora é hora de mostrar como isso afeta o treinamento do modelo. Vamos visualizar o caminho percorrido por cada otimizador para trazer os dois parâmetros (mais próximos) de seus valores ótimos.


![image](https://github.com/user-attachments/assets/450029c2-8f84-4a92-ace7-03aab140ce27)

No gráfico à esquerda, temos o caminho típico e bem comportado (e lento) percorrido pelo SGD. Você pode ver que ele oscila um pouco devido ao ruído introduzido pelo uso de mini-lotes. No gráfico à direita, vemos o efeito do uso das médias móveis exponenciais: por um lado, é mais suave e se move mais rápido; por outro, ele ultrapassa o alvo e precisa mudar de direção várias vezes enquanto se aproxima do objetivo. Está se adaptando à superfície de perda, por assim dizer.

Falando em perdas, também podemos comparar as trajetórias das perdas de treinamento e validação para cada otimizador.


![image](https://github.com/user-attachments/assets/5b1da362-8c33-471b-b9b6-a27ae1290ada)

Lembre-se, as perdas são calculadas ao final de cada época, com base na média das perdas dos mini-lotes. No gráfico à esquerda, mesmo que o SGD oscile um pouco, podemos ver que cada época apresenta uma perda menor que a anterior. No gráfico à direita, o overshooting fica claramente visível como um aumento na perda de treinamento. Mas também é evidente que o Adam alcança uma perda menor, pois se aproximou mais do valor ótimo (o ponto vermelho no gráfico anterior).

Em problemas reais, onde é praticamente impossível traçar a superfície de perda, podemos olhar para as perdas como um “resumo executivo” do que está acontecendo. Perdas de treinamento às vezes podem aumentar antes de cair novamente, e isso é esperado.

# Stochastic Gradient Descent (SGD) 

Bora falar um pouqinho sobre SGD…O Gradiente Descendente Estocástico A.K.A SGD é um algoritmo de otimização que atualiza os pesos de um modelo utilizando uma amostra aleatória dos dados de treinamento. O SGD do PyTorch tem alguns argumentos:

* params: parâmetros do modelo
* lr: taxa de aprendizado
* weight_decay: penalidade L2
* momentum: fator de momentum, o próprio argumento beta do SGD.
* dampening: fator de amortecimento para o momentum
* nesterov: habilita o momentum de Nesterov, que é uma versão mais inteligente do momentum regular.

## SGD com Momentum

O SGD com Momentum é uma variação do SGD, que busca acelerar a convergência e reduzir as oscilações durante o processo de otimização. Inspirado no conceito de física, onde um objeto em movimento tende a continuar sua trajetória com base na velocidade adquirida, o SGD com Momentum utiliza o gradiente acumulado das iterações anteriores para atualizar os parâmetros, ao invés de se basear apenas no gradiente atual.

Apesar de parecer semelhante ao uso de uma média exponencialmente ponderada (EWMA) para os gradientes, o SGD com Momentum não faz uma média, mas sim uma soma cumulativa de gradientes “descontados”. Gradientes passados contribuem para a soma, mas são progressivamente “descontados” à medida que envelhecem, e esse desconto é controlado pelo fator beta. Esse fator de amortecimento, comumente configurado em 0.9, determina o impacto do “passado” nas atualizações. O gradiente mais recente tem sua contribuição reduzida pelo fator de amortecimento, enquanto os gradientes anteriores influenciam cada vez menos à medida que se tornam mais distantes.

No entanto, apesar de gradientes antigos contribuírem de forma decrescente, gradientes recentes ainda têm grande influência. Isso resulta em um comportamento em que o algoritmo tende a fazer atualizações rápidas, e pode acabar “passando do ponto” ao se mover muito rápido, o que exige ajustes ao longo do caminho para corrigir o curso. Esse efeito pode ser visualizado como uma bola rolando por uma colina, ganhando velocidade até ultrapassar o ponto de mínimo e precisando recuar para alcançar o objetivo.

![image](https://github.com/user-attachments/assets/39124a5e-caf9-4b8a-a37e-4a41302d68bf)

Por sua vez, o ADAM incorpora não apenas o momento (média do gradiente) como o SGD com Momentum, mas também uma média móvel do quadrado do gradiente. Isso permite que o ADAM adapte dinamicamente as taxas de aprendizado para cada parâmetro, proporcionando uma abordagem mais robusta e eficaz em situações de grande variabilidade nos gradientes. No entanto, essa capacidade de adaptação do ADAM pode resultar em uma convergência mais rápida para um mínimo, porém nem sempre para um mínimo de boa qualidade, especialmente em problemas de aprendizado profundo, onde diferentes mínimos podem ter desempenhos variados.

Embora o ADAM seja eficaz e frequentemente mais estável, o SGD com Momentum tem o potencial de explorar melhor a superfície de perda, especialmente quando combinado com um learning rate scheduler. Enquanto o ADAM tende a encontrar mínimos rapidamente, o Momentum, com sua “oscilação” controlada, pode levar a um mínimo de melhor qualidade. Em resumo, ambos os algoritmos têm como objetivo melhorar a convergência, mas o ADAM oferece um desempenho mais consistente e requer menos ajuste de parâmetros, enquanto o SGD com Momentum pode ser preferido em contextos onde se deseja um controle mais refinado sobre o processo de otimização.

## SGD com Nesterov

O Nesterov Momentum é uma variação do método de momentum que introduz uma técnica de “olhar à frente”, antecipando a direção do movimento para melhorar a convergência. Em vez de calcular o gradiente no ponto atual, o método de Nesterov calcula o gradiente após o passo do momentum, ou seja, faz uma previsão de onde o ponto de atualização estará no próximo passo, ajustando a direção de forma mais eficiente. No SGD com Momentum tradicional, no passo t, o momentum é calculado com base no gradiente do passo t e no momentum do passo t−1. Para o próximo passo t+1, o algoritmo utiliza o gradiente no ponto t+1 e o momentum calculado no passo anterior para atualizar os parâmetros. Em contraste, o Nesterov faz uma previsão mais inteligente: antes de atualizar o parâmetro no passo t, o algoritmo “antecipa” a atualização e calcula o gradiente não no ponto t, mas em um ponto projetado para t+1. Isso é feito utilizando o gradiente atual e o momentum acumulado até o momento, considerando que o gradiente no próximo passo t+1 será uma estimativa do gradiente no passo atual t.


![image](https://github.com/user-attachments/assets/19c6c298-10d3-436d-b1ff-b504442c9cbe)

Em termos práticos, o Nesterov Momentum tenta calcular o momentum um passo à frente, antecipando a direção em que o gradiente provavelmente se moverá. Isso é possível porque, embora o gradiente no próximo passo não seja conhecido, uma estimativa razoável é usar o gradiente no ponto atual, dada a suposição de que o movimento entre os pontos não mudará drasticamente. Ao calcular o momentum antecipadamente e usá-lo para atualizar os parâmetros, o método melhora a velocidade de convergência, já que os parâmetros são ajustados de forma mais precisa, refletindo melhor a direção desejada.

Em resumo, a principal diferença do Nesterov Momentum é a forma como ele calcula o gradiente, não apenas considerando o ponto atual, mas também antecipando o movimento. Isso permite que o Nesterov se mova mais rapidamente em direção ao mínimo, pois ele já se ajusta ao futuro comportamento do gradiente. Os gradientes, embora calculados da mesma forma em todas as variações do momentum, resultam em diferentes trajetórias de atualização, uma vez que a posição na superfície de perda onde o gradiente é calculado varia.


![image](https://github.com/user-attachments/assets/7707e15f-0e0d-41c9-8853-14992a81623f)

Essa abordagem de “olhar à frente” pode ser especialmente útil em modelos de aprendizado profundo, onde a superfície de perda é complexa e pode ter muitos mínimos locais. Ao antecipar o gradiente de forma mais inteligente, o Nesterov Momentum pode escapar de mínimos locais e alcançar um mínimo de melhor qualidade mais rapidamente.

![image](https://github.com/user-attachments/assets/8ea6c83b-fe17-418b-a8e0-153eb8b3da7e)

Se você ainda não está convencido pelo Momentum, seja o regular ou o de Nesterov, podemos adicionar um outro fator à discussão: o uso de um learning rate scheduler, que pode tornar o processo ainda mais eficaz ao ajustar dinamicamente a taxa de aprendizado durante a otimização.

# Learning Rate Schedulers 

Este é um tema muito amplo e repleto de novos conceitos. Vamos apenas introduzir os agendadores de taxa de aprendizado (tradução de learning rate schedulers).

É possível programar mudanças na taxa de aprendizado durante o treinamento, em vez de apenas adaptar os gradientes. Por exemplo, você pode querer reduzir a taxa de aprendizado em uma ordem de magnitude a cada T épocas, de forma que o treinamento seja mais rápido no início e desacelere após algum tempo, para evitar problemas de convergência.

Isso é exatamente o que um Learning Rate Scheduler (agendador de taxa de aprendizado) faz: ele atualiza a taxa de aprendizado do otimizador.

Sendo mais clara: a taxa de aprendizado inicial é mantida por algumas épocas e depois é multiplicada por um fator (ex.: 0.1). Isso permite que o treinamento comece rápido e desacelere com o tempo para evitar problemas de convergência.

“Todos os agendadores reduzem a taxa de aprendizado?”

Não, nem todos. Antigamente, era comum reduzir a taxa de aprendizado ao longo do treinamento, mas essa ideia foi desafiada por taxas cíclicas (cyclical learning rates). Existem várias abordagens de agendamento, e muitas delas estão disponíveis no PyTorch.

Dividimos os agendadores em três grupos:

* Agendadores que atualizam a taxa a cada T épocas (como no exemplo acima).
* Agendadores que atualizam a taxa quando a validation loss parece estagnada.
* Agendadores que atualizam a taxa após cada mini-batch.

## Epoch Schedulers
Estes agendadores têm o método step() chamado no final de cada época.

* StepLR: Multiplica a taxa por um fator (gamma) a cada step_size épocas.
* MultiStepLR: Multiplica a taxa por gamma nas épocas especificadas em uma lista de milestones.
* ExponentialLR: Multiplica a taxa por gamma em todas as épocas.
* LambdaLR: Permite personalizar a atualização com uma função que recebe a época como argumento.
* CosineAnnealingLR: Usa cosine annealing para atualizar a taxa.


![image](https://github.com/user-attachments/assets/14d34279-9540-4fd2-ac8d-a4a7bc601021)

## Validation Loss Scheduler

O agendador ReduceLROnPlateau não segue um cronograma fixo. Em vez disso, reduz a taxa de aprendizado quando a perda de validação não melhora por um número definido de épocas (argumento patience). Por exemplo, se a perda estagnar por 5 épocas, a taxa será reduzida na próxima.


![image](https://github.com/user-attachments/assets/c8899651-81c7-4c94-bec0-1fde9dbcd4c0)

## Mini-Batch Schedulers

Esses têm o método step() chamado após cada mini-batch.

* CyclicLR: Alterna entre uma taxa mínima (base_lr) e máxima (max_lr) ao longo de ciclos. Modos como triangular2 ou exp_range permitem ajustar a amplitude dos ciclos.
* OneCycleLR: Aumenta a taxa até um valor máximo e a reduz para um valor bem baixo em um único ciclo.
* CosineAnnealingWarmRestarts: Requer o número da época (incluindo frações de mini-batches) para aplicar cosine annealing.


  ![image](https://github.com/user-attachments/assets/ab1d9c1e-9a33-4165-b629-909677e38ab5)

# Hora da prática ⏰

Vamos integrar todos os conceitos que vimos hoje: EWMA, adam, SDG e learning rate schedulers.

O fluxo de treinamento vai ser o seguinte:

* Pré-processamento dos dados: Normalização e preparação dos loaders.
* Inicialização do modelo: Definição da arquitetura e escolha do otimizador (Adam ou SGD com momentum).
* Configuração do scheduler: Definir um learning rate scheduler para ajuste dinâmico da taxa de aprendizado.
* Treinamento: Atualizar pesos com o otimizador, aplicar EWMA nos gradientes e monitorar a performance.
* Avaliação: Calcular acurácia e perda no conjunto de validação.
  
O livro do Godoy trás uma análise muito semelhante na seção “Putting It All Together”, vale a pena dar uma olhada! Tentei construir um exemplo mais didático, então vamos lá…

O primeiro passo é escolher nosso dataset. Se queremos um exemplo de CNN simples, nada melhor do que usar o MNIST, um dos datasets mais usados para exemplos de visão computacional.

O MNIST é um grande conjunto de dados (treinamento com 60.000 exemplos e um conjunto de teste com 10.000 exemplos) de dígitos manuscritos. Os dígitos foram normalizados em termos de tamanho e centralizados em uma imagem de tamanho fixo.


![image](https://github.com/user-attachments/assets/f17aae6c-d433-4ca0-8a5a-9c1731926142)

Vamos utilizar o DataLoader e uma normalização baseada em estatísticas do dataset.


    ~~~python
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    ~~~

O nosso próximo passo é a inicialização do nosso modelo. Como o foco desta seção é mostra o uso das ferramentas, vamos aplicar uma arquitetura bem simples.


    ~~~python
    class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.model(x)
    ~~~

Agora vamos para o que interessa! Vamos definir o Adam e o SGD para comparação e incluir um scheduler da nossa escolha (aqui dá para brincar e testar todos que estudamos).


    ~~~python
    model = SimpleNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    ~~~

Não podemos esquecer da suavização dos gradientes com EWMA.


    ~~~python
    def ewma_gradients(model, beta=0.9):
    avg_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            if name not in avg_grads:
                avg_grads[name] = param.grad.clone()
            else:
                avg_grads[name] = beta * avg_grads[name] + (1 - beta) * param.grad
    return avg_grads
    ~~~

Agora vamos treinar nosso modelo e obter nossas acurácias para cada época.

    ~~~python
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            avg_grads = ewma_gradients(model)
            optimizer.step()
        
        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    ~~~

Bom, com isso já temos um modelo — mesmo que simples — em pleno funcionamento! Vamos ver como ele ele foi na prática?

Primeiro, vamos ver nossa curva de acurácia!

![image](https://github.com/user-attachments/assets/e9c98057-7afe-4317-b417-c61527bd744f)

Para um modelo simples, os resultados foram bem bacanas! 97,42% de acurácia no treinamento :)

Vamos fazer mais algumas análises… que tal dar uma olhada na evolução da taxa de aprendizado?

![image](https://github.com/user-attachments/assets/aa579b6f-8521-47be-a636-3b7967e0c588)

A gente consegue ver que a taxa de aprendizado começa relativamente alta e decresce em etapas ao longo das épocas. Isso se deve ao scheduler escolhido, que reduz a taxa de aprendizado progressivamente, o que pode ajudar o modelo a refinar os pesos ao longo do treinamento, prevenindo oscilações excessivas no final. Podemos testar outros tipos de schedulers para comparar seu comportamento. Legal, né?

E o EWMA? Vamos ver se a suavização funcionou? Para isso podemos fazer um plot comparando os gradientes “crus” e a suavização.

![image](https://github.com/user-attachments/assets/7641c43a-c7cc-41be-8a46-3c49d4e7cd06)

Inicialmente, os gradientes são elevados, mas reduzem com o tempo, indicando que o modelo está se ajustando e atualizando os pesos de forma mais sutil ao longo do tempo. A curva suavizada mostra uma diminuição mais estável dos gradientes, o que sugere um processo de convergência controlado, do jeito que a gente queria 🥳🪩.

# Referências
* Deep Learning with PyTorch Step-by-Step — Daniel Godoy
* GitHub do livro: https://github.com/dvgodoy/PyTorchStepByStep
* https://medium.com/aimonks/understanding-exponentially-weighted-moving-averages-ewma-cc8e78b01184
* GitHub da disciplina de Machine Learning do PPGEEC: https://github.com/ivanovitchm/PPGEEC2318
* https://www.datacamp.com/pt/tutorial/adam-optimizer-tutorial
* https://www.geeksforgeeks.org/impact-of-learning-rate-on-a-model/

