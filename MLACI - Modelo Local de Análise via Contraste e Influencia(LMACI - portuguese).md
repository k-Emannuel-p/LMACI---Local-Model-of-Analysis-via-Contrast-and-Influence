# MLACI - Modelo Local de Análise via Contraste e Influência

**Data de Criação do Documento:** 21 de outubro de 2025
**Autor:** Imaru
**E-mail de Contato:** KeplerDevelopment.help@gmail.com

---

## Resumo

Este artigo apresenta o LMACI (Modelo Local de Análise via Contraste e Influência), um modelo de IA projetado para análise textual local e eficiente. Seu objetivo é identificar relações textuais a partir de qualquer texto, sem treinamento prévio, de maneira altamente eficiente e precisa.

O LMACI opera através de uma "análise por contraste", que envolve medir como a remoção de um token específico altera o significado geral de um texto. Esse processo cria gradientes de conexão, identificando o núcleo da mensagem, os tokens mais importantes e medindo a importância relacional de um token para outro (análise token a token). O principal diferencial do LMACI é sua capacidade de realizar essa análise com notável eficiência e velocidade (tempo de resposta inferior a 1s).

---

## Introdução

O LMACI (Modelo Local de Análise via Contraste e Influência) é um modelo de IA de análise textual desenvolvido para análise textual local e eficiente. Sua função é identificar relações textuais de qualquer texto, em qualquer idioma, sem treinamento prévio ou conexão com a internet, e fazê-lo de forma eficiente.

Sua criação começou em fevereiro de 2025, quando eu (Imaru) estava tentando criar um algoritmo capaz de identificar as palavras centrais em uma frase, em qualquer idioma, com base apenas em um arquivo de embeddings. Eu queria isso para construir uma IA que pudesse gerar texto natural e fluente (o que levou ao nascimento do MAM — Markov Attention Model — descrito no documento “Attention, Markov! MAM”). Após um mês de planejamento, simplificação e testes desse sistema, tive uma ideia muito interessante que se tornou a base do que hoje é o LMACI: o **AS**.

## AS - Atenção e Síntese

O AS foi o primeiro modelo de análise que desenvolvi. Ele se baseava em um tipo de análise que chamei de **GCA** (Análise de Contraste Global). Isso significa que o modelo busca entender a importância de uma palavra (daí “*Atenção*” no nome, pois mede a importância da palavra) comparando seu efeito sobre outras palavras quando é removida de uma frase, e comparando o contraste entre a frase original e a frase sem o token (daí “*Síntese*”, pois compara o efeito na síntese da frase).

O processo começa com a tokenização, que se refere à conversão do texto a ser analisado em uma lista de palavras ou partes de palavras (tokens). Vamos usar a frase “O carro azul é muito rápido” como exemplo. Seus tokens são: “O”, “carro”, “azul”, “é”, “muito”, “rápido”.

O próximo passo é atribuir um **embedding** a cada token correspondente. Um embedding é um vetor de números reais que codifica o significado de uma palavra, de modo que palavras mais próximas no espaço vetorial devem ser semanticamente semelhantes.[[1]](https://en.wikipedia.org/wiki/Word_embedding) Eles vêm de um arquivo externo que contém um grande número de embeddings de alta dimensão, como GloVe, Word2Vec, FastText, etc.

Os embeddings para nossa frase seriam, por exemplo:

“O” → [0.54, 0.50, -0.12, 0.08, 0.33…]

“carro” → [0.82, -0.25, 0.76, -0.41, 0.19…]

“azul” → [-0.03, 0.94, 0.11, -0.62, 0.47...]

“é” → [0.29, -0.44, 0.05, 0.88, -0.09…]

“muito” → [0.67, 0.13, -0.58, 0.21, 0.74…]

“rápido” → [-0.45, 0.38, 0.90, 0.07, -0.52…]

(**Estes embeddings são puramente ilustrativos.**)

Em seguida, chegamos à parte mais fundamental do AS, onde sua análise realmente começa. Primeiro, calculamos um "embedding médio", que sintetiza a frase em um único embedding. Isso é feito somando o embedding de cada token na frase e, em seguida, dividindo o resultado pelo número de vetores. Por exemplo, se a frase fosse “O céu é azul”, esse embedding médio capturaria as características semânticas de um “céu azul”. Da mesma forma, se a frase fosse “O gato é inteligente e ágil”, esse embedding capturaria as características semânticas de um “gato inteligente e ágil”.

Aqui está como o calculamos:

$$
S = \frac{ \sum_{i=1}^{n} e_i }{n}
$$

Onde:

- *S* é o embedding final, que sintetiza a frase.
- *n* é o número total de tokens.
- $e_i$ é o embedding de um token na frase, de 1 a n ($e_1 + e_2 + e_3… e_n$).

Em seguida, recalculamos o mesmo valor, mas desta vez removendo cada token um por um:

$$
S(i) = \frac{(\sum_{j = 0}^{n} e_j) - e_i}{n-1}
$$

Onde:

- $S(i)$ é o novo embedding médio com um token da frase removido, que poderia ser $e_1, e_2, e_3... e_n$.
- $j$ é o token atual (iterando por cada token de 1 a *n*).

Agora, para cada token, o modelo recalcula o embedding médio da frase sem aquele token e, em seguida, calcula a **similaridade de cossenos** entre o embedding médio original e aquele com o token removido. Essa "perturbação de token" nos permite medir o impacto que um token tem dentro de uma frase.

Por exemplo, na frase **“O carro azul é muito rápido.”**:

- Remover **“carro”** elimina o sujeito, quebrando a estrutura principal da frase.
- Remover **“rápido”** elimina uma qualidade importante, mas a ideia de "O carro azul é muito" ainda sugere uma característica de intensidade.
- Remover **“azul”** elimina uma característica secundária; a frase “O carro é muito rápido” ainda preserva a informação mais relevante.
- Remover “é” ou “O” é trivial; a mensagem “Carro azul muito rápido” ainda existe mesmo sem eles. Esses tipos de palavras são chamados de *stop-words*.
- Remover “.” não muda completamente o significado da frase, mas pode alterar a percepção se é uma pergunta, afirmação ou exclamação.

O AS faz exatamente isso, mas utiliza embeddings e operações vetoriais essenciais para alcançar esse resultado.

Aqui está a fórmula que o AS usa:

$$
sim_{cos} = \frac{\mathbf{S(i)} \cdot \mathbf{S}}{\|\mathbf{S(i)}\| \|\mathbf{S}\|}
$$

Quanto menor o resultado de $sim_{cos}$, mais importante é o token removido ($i$), e vice-versa.

No entanto, identifiquei um problema significativo no AS. Embora funcionasse muito bem para frases com significados e ideias consistentes, como afirmações ou perguntas, ele falhava significativamente quando recebia frases com múltiplas ideias ou contraditórias. Percebi que isso ocorria porque, quando o modelo calculava o embedding médio da frase, se ela contivesse várias ideias ou conflitantes, o significado seria completamente diluído.

Para resolver esse problema, desenvolvi um novo método de análise que não media mais o impacto de um token em uma frase, mas sim a influência de um token sobre outro. Chamei esse método de **Pairwise**.

## Pairwise e a Criação do EAS (Enhanced Attention Synthesis)

O Pairwise foi um novo método de análise que desenvolvi para medir a **influência** de um token sobre outro, em vez de apenas sua similaridade.

A análise funciona da seguinte forma: para medir o impacto que o token **A** tem sobre o token **B**, primeiro calculamos um embedding médio que representa o contexto do par (A+B). Em seguida, medimos a similaridade de cossenos entre esse embedding contextual (A+B) e o embedding original do token **B**. O resultado nos diz o quanto a presença de **A** alterou ou reforçou o significado de **B**.

Isso nos permite capturar relações locais complexas. Por exemplo, na frase “Eu **não gosto** de doces”:

- Ao analisar o par ("não", "gosto"), seu embedding médio seria semanticamente muito distante de "gosto" sozinho. A similaridade de cossenos seria baixa, indicando que "não" tem uma **alta influência** e inverte o significado de "gosto".

Esse processo é repetido para cada par de tokens na frase. Embora essa análise tenha uma complexidade computacional de O(N²), pois cada token é avaliado em relação a todos os outros, ela fornece um mapa detalhado das microrrelações dentro do texto. No EAS, essa complexidade é otimizada para tornar a análise mais rápida e eficiente, como veremos a seguir.

Alguns meses se passaram e tive uma nova ideia. Baseava-se no conceito de que, como o AS é um modelo GCA, se eu o combinasse com o Pairwise, poderia criar um modelo de Análise via Contraste e Influência (ACI) que seria muito superior, identificando mais do que apenas ATs (Tokens de Atenção).

Assim, nasceu o **EAS** — um modelo que superaria o AS e abriria caminho para novas aplicações onde o LMACI se tornaria uma peça fundamental, como tradução, geração de texto, análise de discurso, marketing, redação, SEO e busca e organização inteligentes.

## EAS - Enhanced Attention and Synthesis (Síntese de Atenção Aprimorada)

O EAS é um modelo ACI que, a partir de qualquer texto em qualquer idioma, pode identificar não apenas os ATs (como o AS faz), mas também retornar *gradientes*, que são mapas das relações entre cada AT e os outros tokens na frase. Por exemplo, na frase “O gato azul morava na casa vermelha, e ele voou sobre o céu”, os ATs seriam: “gato”, “azul”, “morava”, “casa”, “vermelha”, “voou”, “céu”.

Para cada AT, o EAS retorna um gradiente que conecta os outros tokens (incluindo outros ATs) a ele:

- “gato” → [”O”: 0.23, “azul”: 0.87, “morava”: 0.63 …]
- “azul”: → [”O”: 0.11, “gato”: 0.88, “morava: “0.46 …]

…

Ele também calcula um gradiente final, que atribui um peso final com base no valor que cada palavra tem em cada gradiente, calculando uma média dos pesos que ela recebeu em todos os gradientes em que apareceu. Isso é muito mais preciso do que os valores que a análise do AS retornaria.

O resultado é algo assim:

- “O”: 0.145
- “gato”: 0.885
- “azul”: 0.878
- “morava”: 0.51

…

Em termos mais simples, ele não apenas nos diz quais tokens são mais importantes para a frase, mas também como eles se conectam com outras palavras, mapeando a estrutura de conexões entre palavras por meio de *gradientes*.

O processo do EAS começa aplicando a análise do AS ao texto. Isso fornece os ATs identificados pelo AS. No entanto, como vimos anteriormente, o AS sofre com o problema da diluição do embedding médio. Nosso passo para resolver isso é calcular a influência Pairwise de cada AT identificado pelo AS sobre os outros tokens na frase, incluindo outros ATs.

Esse processo cria os gradientes de AT. Agora, também precisamos calcular um gradiente final, que tornará a análise mais precisa do que o AS é capaz de fornecer. Para fazer isso, para cada palavra, nós a procuramos nos gradientes de AT. Somamos seus pesos de cada gradiente e dividimos o resultado pelo número de vezes que ela apareceu em todos os gradientes.

A fórmula para calcular o peso final de cada token é a seguinte:

$$
W_{final}(t_k)=\frac{1}{m} \sum_{i=1}^{m}P(at_i,t_k)
$$

### **Explicação da Fórmula**

- $W_{final}(t_k)$: Isso é o que queremos calcular. Representa o **peso final** (a importância consolidada) de qualquer token na frase, que chamamos de $t_k$.
- $m$: Este é o **número total de Tokens de Atenção (ATs)** que o AS encontrou na frase. Se o AS identificou 6 ATs, então $m = 6$.
- $\sum_{i=1}^{m}$: Este é o símbolo de somatório. Indica que somaremos os valores para cada Token de Atenção, do primeiro ($i = 1$) ao último ($i = m$).
- $P(at_i, t_k)$: Esta é a parte mais importante. Representa o **valor do gradiente Pairwise** que mede a influência do Token de Atenção $at_i$ sobre nosso token $t_k$. É o peso que $t_k$ recebeu no gradiente do AT específico, $at_i$.

Em outras palavras, a fórmula simplesmente calcula a **influência média** que todos os Tokens de Atenção da frase exercem sobre cada palavra individual. Um token será considerado altamente importante se receber consistentemente altas pontuações de influência dos múltiplos "pontos de vista" semânticos representados pelos Tokens de Atenção. Isso cria uma pontuação final muito mais robusta e contextualizada do que a análise global isolada do AS.

Com isso, eu (Umaru), concluo o artigo e agradeço a todos que leram até o final.

Nota: O modelo era anteriormente conhecido como LOSAM (Lightweight and Offline Semantic Analyzer Model), mas foi renomeado para LMACI para este artigo.