# LMACI - Local Model of Analysis via Contrast and Influence

**Document Creation Date:** June 26, 2025
**Author:** Immanuel Bokkey
**Contact E-mail:** KeplerDevelopment.help@gmail.com

---

## Abstract

This paper introduces LMACI (Local Model of Analysis via Contrast and Influence), an AI model designed for efficient, local textual analysis. Its purpose is to identify textual relationships from any given text, without prior training, in a highly efficient and accurate manner.

LMACI operates through an "analysis via contrast," which involves measuring how the removal of a specific token alters the overall meaning of a text. This process creates connection gradients, identifying the core of the message, the most important tokens, and measuring the relational importance from one token to another (token-to-token analysis). The key differentiator of LMACI is its ability to perform this analysis with remarkable efficiency and speed (response time of less than 1s).

---

## Introduction

LMACI (Local Model of Analysis via Contrast and Influence) is an AI text analysis model developed for efficient, local textual analysis. Its function is to identify textual relationships from any text, in any language, without prior training or an internet connection, and to do so efficiently.

Its creation began in February 2025, when I (Immanuel) was trying to create an algorithm capable of identifying the central words in a sentence, in any language, based solely on an embeddings file. I wanted this to build an AI that could generate natural and fluent text (which led to the birth of MAM — Markov Attention Model — described in the document “Attention, Markov! MAM"]). After a month of planning, simplifying, and testing this system, I had a very interesting idea that became the foundation of what is now LMACI: **AS**.

## AS - Attention and Synthesis

AS was the first analysis model I developed. It was based on a type of analysis I called **GCA** (Global Contrast Analysis). This means the model seeks to understand the importance of a word (hence “*Attention*” in the name, as it measures word importance) by comparing its effect on other words when it is removed from a sentence, and by comparing the contrast between the original sentence and the sentence without the token (hence “*Synthesis*,” as it compares the effect on the sentence's synthesis).

The process begins with tokenization, which refers to converting the text to be analyzed into a list of words or parts of words (tokens). Let's use the sentence “The blue car is very fast” as an example. Its tokens are: “The”, “blue”, “car”, “is”, “very”, “fast”.

The next step is to assign an **embedding** to each corresponding token. An embedding is a vector of real numbers that encodes the meaning of a word such that words that are closer in the vector space are expected to be similar in meaning.[[1]](https://en.wikipedia.org/wiki/Word_embedding) These come from an external file containing a large number of high-dimensional embeddings, such as GloVe, Word2Vec, FastText, etc.

The embeddings for our sentence would be, for example:

“The” → [0.54, 0.50, -0.12, 0.08, 0.33…]

“car” → [0.82, -0.25, 0.76, -0.41, 0.19…]

“blue” → [-0.03, 0.94, 0.11, -0.62, 0.47...]

“is” → [0.29, -0.44, 0.05, 0.88, -0.09…]

“very” → [0.67, 0.13, -0.58, 0.21, 0.74…]

“fast” → [-0.45, 0.38, 0.90, 0.07, -0.52…]

(**These embeddings are purely illustrative.**)

Next, we arrive at the most fundamental part of AS, where its analysis truly begins. First, we calculate an "average embedding," which synthesizes the sentence into a single embedding. This is done by summing the embedding of each token in the sentence and then dividing the result by the number of vectors. For example, if the sentence were “The sky is blue,” this average embedding would capture the semantic characteristics of a “blue sky.” Similarly, if the sentence were “The cat is smart and swift,” this embedding would capture the semantic characteristics of a “smart and swift cat.”

Here is how we calculate it:

$$
S = \frac{ \sum_{i=1}^{n} e_i }{n}
$$

Where:

- *S* is the final embedding, which synthesizes the sentence.
- *n* is the total number of tokens.
- $e_i$ is the embedding of a token in the sentence, from 1 to n ($e_1 + e_2 + e_3… e_n$).

Next, we recalculate the same value, but this time removing each token one by one:

$$
S(i) = \frac{(\sum_{j = 0}^{n} e_j) - e_i}{n-1}
$$

Where:

- $S(i)$ is the new average embedding with one token from the sentence removed, which could be $e_1, e_2, e,3... e_n$.
- $j$ is the current token (iterating through each token from 1 to *n*).

Now, for each token, the model recalculates the average embedding of the sentence without that token and then computes the **cosine similarity** between the original average embedding and the one with the token removed. This "token perturbation" allows us to measure the impact that a token has within a sentence.

For example, in the sentence **“The blue car is very fast.”**:

- Removing **“car”** eliminates the subject, breaking the main structure of the sentence.
- Removing **“fast”** eliminates an important quality, but the idea of "The blue car is very" still suggests a characteristic of intensity.
- Removing **“blue”** eliminates a secondary characteristic; the sentence “The car is very fast” still preserves the most relevant information.
- Removing “is” or “The” is trivial; the message “Blue car very fast” still exists even without them. These types of words are called *stop-words.*
- Removing “.” does not completely change the sentence's meaning but may alter the perception of whether it is a question, statement, or exclamation.

AS does exactly this, but it leverages embeddings and essential vector operations to achieve this result.

Here is the formula AS uses:

$$
sim_{cos} = \frac{\mathbf{S(i)} \cdot \mathbf{S}}{\|\mathbf{S(i)}\| \|\mathbf{S}\|}
$$

The lower the $sim_{cos}$ result, the more important the removed token ($i$) is, and vice-versa.

However, I identified a significant problem in AS. Although it worked very well for sentences with consistent meanings and ideas, such as statements or questions, it failed significantly when it received sentences with multiple or contradictory ideas. I realized this occurred because when the model calculated the average embedding of the sentence, if it contained several or conflicting ideas, the meaning would be completely diluted.

To solve this problem, I developed a new analysis method that no longer measured the impact of a token on a sentence, but rather the influence of one token on another. I called this method **Pairwise**.

## Pairwise and the Creation of EAS (Enhanced Attention Synthesis)

Pairwise was a new analysis method I developed to measure the **influence** of one token on another, rather than just their similarity.

The analysis works as follows: to measure the impact that token **A** has on token **B**, we first calculate an average embedding that represents the context of the pair (A+B). Then, we measure the cosine similarity between this contextual embedding (A+B) and the original embedding of token **B**. The result tells us how much the presence of **A** altered or reinforced the meaning of **B**.

This allows us to capture complex local relationships. For example, in the sentence “I **do not like** sweets”:

- When analyzing the pair ("not", "like"), their average embedding would be semantically very distant from "like" alone. The cosine similarity would be low, indicating that "not" has a **high influence** and inverts the meaning of "like".

This process is repeated for every pair of tokens in the sentence. Although this analysis has a computational complexity of O(N²), as each token is evaluated against all others, it provides a detailed map of the micro-relationships within the text. In EAS, this complexity is optimized to make the analysis faster and more efficient, as we will see next.

A few months passed, and I had a new idea. It was based on the concept that since AS is a GCA model, if I combined it with Pairwise, I could create a model of Analysis via Contrast and Influence (ACI) that would be far superior, identifying more than just ATs (Attention Tokens).

Thus, **EAS** was born—a model that would surpass AS and pave the way for new applications where LMACI would become a fundamental piece, such as translation, text generation, discourse analysis, marketing, writing, SEO, and intelligent search and organization.

## EAS - Enhanced Attention and Synthesis

EAS is an ACI model that, from any text in any language, can identify not only ATs (as AS does) but also return *gradients*, which are maps of the relationships between each AT and the other tokens in the sentence. For example, in the sentence “The blue cat lived in the red house, and it flew over the sky,” the ATs would be: “cat”, “blue”, “lived”, “house”, “red”, “flew”, “sky”.

For each AT, EAS returns a gradient that connects the other tokens (including other ATs) to it:

- “cat” → [”The”: 0.23, “blue”: 0.87, “lived”: 0.63 …]
- “blue”: → [”The”: 0.11, “cat”: 0.88, “lived: “0.46 …]

…

It also calculates a final gradient, which assigns a final weight based on the value each word has in every gradient, by calculating an average of the weights it received across all the gradients it appeared in. This is much more precise than the values the AS analysis would return.

The result is something like this:

- “The”: 0.145
- “cat”: 0.885
- “blue”: 0.878
- “lived”: 0.51

…

In simpler terms, it not only tells us which tokens are most important to the sentence but also how they connect with other words, mapping out the structure of connections between words through *gradients*.

The EAS process begins by applying the AS analysis to the text. This provides the ATs identified by AS. However, as we saw earlier, AS suffers from the problem of average embedding dilution. Our step to solve this is to calculate the Pairwise influence of each AT identified by AS on the other tokens in the sentence, including other ATs.

This process creates the AT gradients. Now, we also need to calculate a final gradient, which will make the analysis more precise than what AS is capable of providing. To do this, for each word, we look for it in the AT gradients. We sum its weights from each gradient and divide the result by the number of times it appeared across all gradients.

The formula to calculate the final weight of each token is as follows:

$$
W_{final}(t_k)=\frac{1}{m} \sum_{i=1}^{m}P(at_i,t_k)
$$

### **Formula Explanation**

- $W_{final}(t_k)$: This is what we want to calculate. It represents the **final weight** (the consolidated importance) of any given token in the sentence, which we call $t_k$.
- $m$: This is the **total number of Attention Tokens (ATs)** that AS found in the sentence. If AS identified 6 ATs, then $m = 6$.
- $\sum_{i=1}^{m}$: This is the summation symbol. It indicates that we will sum the values for each Attention Token, from the first ($i = 1$) to the last ($i = m$).
- $P(at_i, t_k)$: This is the most important part. It represents the **Pairwise gradient value** that measures the influence of the Attention Token $at_i$ on our token $t_k$. It is the weight that $t_k$ received in the gradient of the specific AT, $at_i$.

In other words, the formula simply calculates the **average influence** that all of the sentence's Attention Tokens exert on each individual word. A token will be considered highly important if it consistently receives high influence scores from the multiple semantic "viewpoints" represented by the Attention Tokens. This creates a final score that is much more robust and contextualized than the isolated global analysis of AS.

With this, I conclude the paper and thank all who have read it to the end.

Note: The model was formerly known as LOSAM (Lightweight and Offline Semantic Analyzer Model), but has been renamed to LMACI for this paper.
