/**
 * LOSAM.js - Biblioteca modular para processamento de texto e análise de embeddings
 * Versão 1.0.6 (Saída de resultados limpa, sem vetores de embedding)
 */

// Armazenamento de embeddings
let embeddingsMap = new Map();
let lastProcessedData = null;

/**
 * Carrega embeddings a partir de um arquivo
 * @param {string} filePath - Caminho para o arquivo de embeddings
 * @returns {Promise<Map<string, Float16Array>>} - Mapa de embeddings carregados
 */
export async function loadEmbeddings(filePath) {
    try {
        let text;
        // Em ambiente Node.js, usar 'fs' para ler arquivos locais
        if (typeof process !== 'undefined' && process.versions && process.versions.node) {
            const fs = (await import('fs')).promises;
            const path = (await import('path')).default;
            const fullPath = path.resolve(filePath);

            text = await fs.readFile(fullPath, 'utf8');
        } else {
            // Em ambiente de navegador ou Web Worker, usar 'fetch'
            const response = await fetch(filePath);
            text = await response.text();
        }

        const lines = text.split("\n");
        const embeddings = new Map();

        for (const line of lines) {
            const parts = line.trim().split(" ");
            if (parts.length < 2) continue;
            const word = parts[0];
            const vector = new Float16Array(parts.slice(1).map(Number));
            embeddings.set(word, vector);
        }

        embeddingsMap = embeddings;
        return embeddings;
    } catch (error) {
        console.error("Erro ao carregar embeddings:", error);
        return new Map();
    }
}

/**
 * Carrega embeddings a partir de uma string de texto bruto.
 * @param {string} text - O conteúdo de texto do arquivo de embeddings.
 * @returns {Map<string, Float16Array>} - Mapa de embeddings carregados.
 */
export function loadEmbeddingsFromText(text) {
    try {
        const lines = text.split("\n");
        const embeddings = new Map();

        for (const line of lines) {
            const parts = line.trim().split(" ");
            if (parts.length < 2) continue;
            const word = parts[0];
            const vector = new Float16Array(parts.slice(1).map(Number));
            embeddings.set(word, vector);
        }

        embeddingsMap = embeddings; // Atualiza o mapa global
        return embeddings;
    } catch (error) {
        console.error("Erro ao processar texto de embeddings:", error);
        return new Map();
    }
}

/**
 * Tokeniza um texto em palavras individuais
 * @param {string} text - Texto a ser tokenizado
 * @returns {Array<string>} - Array de tokens
 */
export function tokenizeText(text) {
    // Preserves emojis and separates tokens
    const cleanedText = text
        .replace(/([\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}])/gu, ' $1 ')
        .replace(/[.,!?;:]/g, " $& ") // Punctuation
        .replace(/\s+/g, " ") // Multiple spaces
        .trim();

    const tokens = cleanedText.split(/\s+/).filter(token => token.length > 0);
    return tokens;
}

/**
 * Calculates the dot product between two vectors
 * @param {Float16Array} vecA - First vector
 * @param {Float16Array} vecB - Second vector
 * @returns {number} - Dot product
 */
export function dotProduct(vecA, vecB) {
    let product = 0;
    for (let i = 0; i < vecA.length; i++) {
        product += vecA[i] * vecB[i];
    }
    return product;
}

/**
 * Calculates the magnitude of a vector
 * @param {Float16Array} vec - Vector
 * @returns {number} - Magnitude
 */
export function magnitude(vec) {
    let sumOfSquares = 0;
    for (let i = 0; i < vec.length; i++) {
        sumOfSquares += vec[i] * vec[i];
    }
    return Math.sqrt(sumOfSquares);
}

/**
 * Calcula a similaridade de Jaccard (sobreposição) entre dois arrays de tokens.
 * Retorna um score entre 0 (nenhum token em comum) e 1 (conjuntos de tokens idênticos).
 * @param {Array<string>} tokensA - O primeiro array de tokens.
 * @param {Array<string>} tokensB - O segundo array de tokens.
 * @returns {number} - Um score de similaridade entre 0 e 1.
 */
export function calculateOverlap(tokensA, tokensB) {
    if (!tokensA || !tokensB || tokensA.length === 0 || tokensB.length === 0) {
        return 0;
    }

    const setA = new Set(tokensA);
    const setB = new Set(tokensB);

    const intersection = new Set([...setA].filter(token => setB.has(token)));
    const intersectionSize = intersection.size;

    const unionSize = setA.size + setB.size - intersectionSize;

    if (unionSize === 0) {
        return 0;
    }

    return intersectionSize / unionSize;
}

/**
 * Calculates the cosine similarity between two vectors
 * @param {Float16Array} vecA - First vector
 * @param {Float16Array} vecB - Second vector
 * @returns {number} - Cosine similarity (between -1 and 1)
 */
export function cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length === 0 || vecB.length === 0 || vecA.length !== vecB.length) return 0;
    const magA = magnitude(vecA);
    const magB = magnitude(vecB);
    if (magA === 0 || magB === 0) return 0;
    return dotProduct(vecA, vecB) / (magA * magB);
}

/**
 * Calculates the average of a set of embeddings
 * @param {Array<Float16Array>} embeddingsList - List of embeddings
 * @returns {Float16Array|null} - Average embedding or null if the list is empty
 */
export function averageEmbeddings(embeddingsList) {
    if (!embeddingsList || embeddingsList.length === 0) return null;
    const numDimensions = embeddingsList[0].length;
    const avgEmbedding = new Float16Array(numDimensions).fill(0);
    for (const emb of embeddingsList) {
        for (let i = 0; i < numDimensions; i++) {
            avgEmbedding[i] += emb[i];
        }
    }
    for (let i = 0; i < numDimensions; i++) {
        avgEmbedding[i] /= embeddingsList.length;
    }
    return avgEmbedding;
}

/**
 * Calculates the average embedding for a list of words.
 * @param {Array<string>} words - List of words.
 * @param {Map<string, Float16Array>} embeddingsMap - The embeddings map.
 * @returns {Float16Array|null} - Average embedding or null if no word has an embedding.
 */
export function getAverageEmbeddingForWords(words, embeddingsMap) {
    if (!words || words.length === 0) return null;

    const validEmbeddings = [];
    for (const word of words) {
        const embedding = embeddingsMap.get(word.toLowerCase());
        if (embedding) {
            validEmbeddings.push(embedding);
        }
    }
    return averageEmbeddings(validEmbeddings);
}

/**
 * (Internal) Executes the core AS logic and returns full data with embeddings.
 * @private
 */
async function _internalRunAS(inputText, customEmbeddings, minImportantTokens = 3) {
    const startTime = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const currentEmbeddingsMap = customEmbeddings || embeddingsMap;
    const originalTokens = tokenizeText(inputText);

    if (originalTokens.length === 0) {
        return { summary: "No tokens to process.", relationsAnalysis: "N/A", tokens: [], tokenCount: 0, executionTime: 0 };
    }

    const validTokenObjects = originalTokens
        .map((token, index) => ({ token, originalIndex: index, embedding: currentEmbeddingsMap.get(token.toLowerCase()) || null }))
        .filter(obj => obj.embedding !== null);

    if (validTokenObjects.length === 0) {
        return { summary: "No known tokens found in embeddings.", relationsAnalysis: "No known tokens for analysis.", tokens: originalTokens, tokenCount: originalTokens.length, executionTime: 0 };
    }

    const baseSentenceEmbedding = averageEmbeddings(validTokenObjects.map(obj => obj.embedding));
    if (!baseSentenceEmbedding) {
        return { summary: "Error calculating base sentence embedding.", relationsAnalysis: "Internal error.", tokens: originalTokens, tokenCount: originalTokens.length, executionTime: 0 };
    }

    const tokenImpacts = [];
    for (let i = 0; i < validTokenObjects.length; i++) {
        const currentTokenObj = validTokenObjects[i];
        const variationEmbeddings = validTokenObjects
            .filter((_, j) => i !== j)
            .map(obj => obj.embedding);

        let impact = 0;
        if (variationEmbeddings.length > 0) {
            const variationSentenceEmbedding = averageEmbeddings(variationEmbeddings);
            if (variationSentenceEmbedding) {
                impact = 1 - cosineSimilarity(baseSentenceEmbedding, variationSentenceEmbedding);
            }
        } else {
            impact = 1;
        }
        tokenImpacts.push({ ...currentTokenObj, impact });
    }

    const averageImpact = tokenImpacts.length > 0 ? tokenImpacts.reduce((sum, tip) => sum + tip.impact, 0) / tokenImpacts.length : 0;
    let importantTokenDetails = tokenImpacts
        .filter(tip => tip.impact > averageImpact);

    // If the count of tokens above average is less than the desired minimum, adjust the selection.
    if (importantTokenDetails.length < minImportantTokens && tokenImpacts.length > 0) {
        // Determine how many top tokens to select
        const numToSelect = Math.min(minImportantTokens, tokenImpacts.length);
        
        // Create a copy of tokenImpacts and sort it by impact score in descending order
        const sortedByImpact = [...tokenImpacts].sort((a, b) => b.impact - a.impact);
        
        // Select the top N tokens
        importantTokenDetails = sortedByImpact.slice(0, numToSelect);
    }
    
    // Sort the final list of important tokens by their original position in the sentence.
    importantTokenDetails.sort((a, b) => a.originalIndex - b.originalIndex);

    const summaryText = importantTokenDetails.length > 0 ? importantTokenDetails.map(t => t.token).join(" ") : "No tokens highlighted as important.";
    
    let relationsText = "Token impact analysis (higher value means more sentence meaning changes with removal):\n";
    tokenImpacts.sort((a, b) => a.originalIndex - b.originalIndex).forEach(tip => {
        relationsText += `- '${tip.token}' (position ${tip.originalIndex}): impact ${tip.impact.toFixed(4)}\n`;
    });

    if (tokenImpacts.length === 0) {
        relationsText = "Could not calculate token impact (no valid embeddings or very short sentence).";
    }

    const endTime = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const executionTime = endTime - startTime;

    return {
        summary: summaryText,
        relationsAnalysis: relationsText,
        tokens: originalTokens,
        tokenCount: originalTokens.length,
        tokenImpacts: tokenImpacts,
        importantTokens: importantTokenDetails,
        executionTime: executionTime
    };
}


/**
 * Executes the AS (Attention and Synthesis) algorithm on a text.
 * @param {string} inputText - Input text
 * @param {Map<string, Float16Array>} [customEmbeddings] - Custom embeddings (optional, uses embeddingsMap by default)
 * @returns {Promise<Object>} - Analysis result without embedding vectors.
 */
export async function runAS(inputText, customEmbeddings, minImportantTokens = 3) {
    const fullResult = await _internalRunAS(inputText, customEmbeddings, minImportantTokens);

    // Helper to remove embedding property for cleaner output
    const cleanTokenData = ({ token, originalIndex, impact }) => ({ token, originalIndex, impact });

    // Clean the bulky embedding vectors from the final output
    if (fullResult.tokenImpacts) {
        fullResult.tokenImpacts = fullResult.tokenImpacts.map(cleanTokenData);
    }
    if (fullResult.importantTokens) {
        fullResult.importantTokens = fullResult.importantTokens.map(cleanTokenData);
    }

    lastProcessedData = fullResult;
    return fullResult;
}


/**
 * Calculates the importance of other tokens in relation to a specific token in a text.
 * Importance is measured by cosine similarity between embeddings.
 * @param {string} inputText - The input text.
 * @param {string} targetToken - The specific token for which importance will be calculated.
 * @param {Map<string, Float16Array>} [customEmbeddings] - Custom embeddings (optional, uses embeddingsMap by default).
 * @returns {Array<Object>} - An array of objects, each containing 'token', 'originalIndex', and 'similarity'
 * with the target token. Returns an empty array if the target token is not found
 * or if there are no valid embeddings.
 */
export function calculateTokenImportance(inputText, targetToken, customEmbeddings) {
    const currentEmbeddingsMap = customEmbeddings || embeddingsMap;
    const originalTokens = tokenizeText(inputText);
    const importanceResults = [];

    if (originalTokens.length === 0) {
        console.warn("No tokens to process in the input text.");
        return [];
    }

    const targetTokenLower = targetToken.toLowerCase();
    const targetEmbedding = currentEmbeddingsMap.get(targetTokenLower);

    if (!targetEmbedding) {
        console.warn(`Embedding for target token '${targetToken}' not found.`);
        return [];
    }

    // Iterates over all tokens, except the target token
    for (let i = 0; i < originalTokens.length; i++) {
        const currentToken = originalTokens[i];
        const currentTokenLower = currentToken.toLowerCase();

        // Skips the target token itself
        if (currentTokenLower === targetTokenLower && originalTokens.indexOf(targetToken, i + 1) === -1) {
            continue; // If it's the last occurrence of the target token, don't compare with itself
        }

        const currentEmbedding = currentEmbeddingsMap.get(currentTokenLower);

        if (currentEmbedding) {
            const similarity = cosineSimilarity(targetEmbedding, currentEmbedding);
            importanceResults.push({
                token: currentToken,
                originalIndex: i,
                similarity: similarity
            });
        } else {
            // Optional: include tokens without embedding with 0 similarity or an indicator
            importanceResults.push({
                token: currentToken,
                originalIndex: i,
                similarity: 0, // Or NaN, or a "no embedding" indicator
                note: "No embedding"
            });
        }
    }

    // Sorts results by similarity (from most to least important)
    return importanceResults.sort((a, b) => b.similarity - a.similarity);
}

/**
 * Calculates the contextual importance of a specific token based on contiguous n-gram perturbation.
 * Importance is measured by the change in cosine similarity of the average embedding of n-grams
 * when the target token is removed.
 * @param {string} inputText - The input text.
 * @param {string} targetToken - The specific token for which importance will be calculated.
 * @param {number} [maxNgramSize=3] - The maximum size of n-grams to consider (contiguous).
 * @param {Map<string, Float16Array>} [customEmbeddings] - Custom embeddings (optional, uses embeddingsMap by default).
 * @returns {Object} - An object containing the total importance, and details of impacts by n-gram.
 * Returns an object with 'totalImportance: 0' and 'nGramImpacts: []' if the target token is not found
 * or if there are no valid embeddings.
 */
export function calculateContiguousNgramImportance(inputText, targetToken, maxNgramSize = 6, customEmbeddings) {
    const currentEmbeddingsMap = customEmbeddings || embeddingsMap;
    const originalTokens = tokenizeText(inputText);
    const targetTokenLower = targetToken.toLowerCase();
    const targetEmbedding = currentEmbeddingsMap.get(targetTokenLower);

    if (!targetEmbedding) {
        console.warn(`Embedding for target token '${targetToken}' not found.`);
        return { totalImportance: 0, nGramImpacts: [] };
    }
    if (originalTokens.length === 0) {
        console.warn("No tokens to process in the input text.");
        return { totalImportance: 0, nGramImpacts: [] };
    }

    const nGramImpacts = [];

    // Find all occurrences of the target token
    const targetTokenIndices = originalTokens.map((token, idx) => token.toLowerCase() === targetTokenLower ? idx : -1).filter(idx => idx !== -1);

    for (const targetIdx of targetTokenIndices) {
        for (let n = 1; n <= maxNgramSize; n++) {
            // Define the start and end of the n-gram, ensuring it's within bounds
            const nGramStart = Math.max(0, targetIdx - Math.floor((n - 1) / 2));
            const nGramEnd = Math.min(originalTokens.length, nGramStart + n);
            const nGramTokens = originalTokens.slice(nGramStart, nGramEnd);

            // Calculate the average embedding of the original n-gram
            const originalNgramEmbedding = getAverageEmbeddingForWords(nGramTokens, currentEmbeddingsMap);

            if (!originalNgramEmbedding) continue; // Skip if n-gram has no valid embeddings

            // Create a perturbed n-gram by removing the target token
            const perturbedNgramTokens = nGramTokens.filter((_, idx) => (nGramStart + idx) !== targetIdx);

            let impact = 0;
            if (perturbedNgramTokens.length > 0) {
                const perturbedNgramEmbedding = getAverageEmbeddingForWords(perturbedNgramTokens, currentEmbeddingsMap);
                if (perturbedNgramEmbedding) {
                    impact = 1 - cosineSimilarity(originalNgramEmbedding, perturbedNgramEmbedding);
                }
            } else {
                // If removing the token leaves an empty n-gram, it means the token was the only one
                // in that n-gram, so its impact is high.
                impact = 1;
            }
            nGramImpacts.push({ nGram: nGramTokens.join(' '), n: n, targetTokenIndex: targetIdx, impact: impact });
        }
    }

    const totalImportance = nGramImpacts.reduce((sum, item) => sum + item.impact, 0);

    return { totalImportance, nGramImpacts };
}

/**
 * Calculates the pairwise influence between all tokens in a text.
 * Influence of token A on B is measured by comparing the embedding of B
 * with the average embedding of the pair (A, B).
 * @param {string} inputText - The input text.
 * @param {Map<string, Float16Array>} [customEmbeddings] - Custom embeddings (optional, uses embeddingsMap by default).
 * @returns {Array<Object>} - An array of objects, each containing 'influencer', 'influenced', and 'influence'.
 */
export function calculatePairwiseImportance(inputText, customEmbeddings) {
    const currentEmbeddingsMap = customEmbeddings || embeddingsMap;
    const originalTokens = tokenizeText(inputText);
    const pairwiseInfluences = [];

    const validTokenObjects = originalTokens
        .map((token, index) => ({
            token,
            originalIndex: index,
            embedding: currentEmbeddingsMap.get(token.toLowerCase()) || null
        }))
        .filter(obj => obj.embedding !== null);

    for (let i = 0; i < validTokenObjects.length; i++) {
        for (let j = 0; j < validTokenObjects.length; j++) {
            if (i === j) continue; // A token does not influence itself

            const tokenObjA = validTokenObjects[i]; // The influencer
            const tokenObjB = validTokenObjects[j]; // The influenced

            const contextualEmbedding = averageEmbeddings([tokenObjA.embedding, tokenObjB.embedding]);
            
            let influence = 0;
            if (contextualEmbedding) {
                // Low similarity means high influence (the meaning changed a lot)
                influence = 1 - cosineSimilarity(contextualEmbedding, tokenObjB.embedding);
            }
            
            pairwiseInfluences.push({
                influencer: tokenObjA.token,
                influenced: tokenObjB.token,
                influence: influence
            });
        }
    }
    return pairwiseInfluences.sort((a, b) => b.influence - a.influence);
}

/**
 * Executes the EAS (Enhanced Attention and Synthesis) algorithm on a text.
 * This mode combines global attention (AS) with pairwise influence analysis
 * to create a more robust, contextualized importance score for each token.
 * @param {string} inputText - Input text.
 * @param {Map<string, Float16Array>} [customEmbeddings] - Custom embeddings (optional, uses embeddingsMap by default).
 * @returns {Promise<Object>} - Analysis result including Attention Tokens, influence gradients, and final scores.
 */
export async function runEnhancedAS(inputText, customEmbeddings) {
    const startTime = typeof performance !== 'undefined' ? performance.now() : Date.now();

    // 1. Run the internal AS to get full data with embeddings for calculations
    const globalASResult = await _internalRunAS(inputText, customEmbeddings);
    const attentionTokens = globalASResult.importantTokens;

    if (!attentionTokens || attentionTokens.length === 0) {
        return {
            summary: "EAS analysis could not be completed: no attention tokens found.",
            globalAS: globalASResult, // It will be cleaned later
            attentionTokens: [],
            gradients: {},
            finalWeights: [],
            executionTime: (typeof performance !== 'undefined' ? performance.now() : Date.now()) - startTime
        };
    }
    
    const allValidTokens = globalASResult.tokenImpacts; // Use the full list with embeddings

    // 2. For each AT, create a gradient of its influence on all other tokens
    const atGradients = {};
    const finalScoresSum = new Map();
    const finalScoresCount = new Map();

    for (const at of attentionTokens) {
        const atEmbedding = at.embedding;
        const atToken = at.token;
        atGradients[atToken] = [];

        for (const otherToken of allValidTokens) {
            // Use originalIndex to avoid comparing a token with itself
            if (at.originalIndex === otherToken.originalIndex) continue;

            const contextualEmbedding = averageEmbeddings([atEmbedding, otherToken.embedding]);
            let influence = 0;
            if (contextualEmbedding) {
                influence = 1 - cosineSimilarity(contextualEmbedding, otherToken.embedding);
            }
            
            atGradients[atToken].push({ token: otherToken.token, influence: influence });
            
            finalScoresSum.set(otherToken.token, (finalScoresSum.get(otherToken.token) || 0) + influence);
            finalScoresCount.set(otherToken.token, (finalScoresCount.get(otherToken.token) || 0) + 1);
        }
    }

    // 3. Calculate the final weight for every token
    const finalWeights = [];
    for (const [token, sum] of finalScoresSum.entries()) {
        const count = finalScoresCount.get(token);
        const averageWeight = count > 0 ? sum / count : 0;
        finalWeights.push({ token: token, weight: averageWeight });
    }
    
    for (const at of attentionTokens) {
        if (!finalScoresSum.has(at.token)) {
             finalWeights.push({ token: at.token, weight: at.impact });
        }
    }
    
    finalWeights.sort((a, b) => b.weight - a.weight);
    
    const summaryText = `Top tokens by influence: ${finalWeights.slice(0, 5).map(t => `${t.token} (${t.weight.toFixed(3)})`).join(', ')}.`;

    const endTime = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const executionTime = endTime - startTime;

    // Helper to remove embedding property for cleaner output
    const cleanTokenData = ({ token, originalIndex, impact }) => ({ token, originalIndex, impact });

    // Clean the nested AS result before including it in the final output
    if (globalASResult.tokenImpacts) {
        globalASResult.tokenImpacts = globalASResult.tokenImpacts.map(cleanTokenData);
    }
    if (globalASResult.importantTokens) {
        globalASResult.importantTokens = globalASResult.importantTokens.map(cleanTokenData);
    }

    const result = {
        summary: summaryText,
        globalAS: globalASResult,
        attentionTokens: attentionTokens.map(t => t.token),
        gradients: atGradients,
        finalWeights: finalWeights,
        executionTime: executionTime
    };

    lastProcessedData = result;
    return result;
}


/**
 * Returns the last processed data.
 * @returns {Object|null} - Last processed data.
 */
export function getLastProcessedData() {
    return lastProcessedData;
}

/**
 * Returns the current embeddings map.
 * @returns {Map<string, Float16Array>} - Current embeddings map.
 */
export function getEmbeddingsMap() {
    return embeddingsMap;
}

// Exporta todas as funções como módulo ESM
export default {
    loadEmbeddings,
    loadEmbeddingsFromText,
    tokenizeText,
    dotProduct,
    magnitude,
    calculateOverlap,
    cosineSimilarity,
    averageEmbeddings,
    getAverageEmbeddingForWords,
    runAS,
    calculateTokenImportance,
    calculateContiguousNgramImportance,
    calculatePairwiseImportance,
    runEnhancedAS,
    getLastProcessedData,
    getEmbeddingsMap
};