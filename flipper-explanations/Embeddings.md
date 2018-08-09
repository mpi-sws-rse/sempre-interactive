Author: Akshal Aniche

# Using word embeddings to augment extended parsing
We are using word embeddings to compute word similarity and augmenting the computation of the similarity of RHS in extended parsing. 

During the computation of the longest common subsequence, instead of having a discreet increment (add 1 to the length of the LCS if both tokens are equal, skip one of the tokens otherwise), we are possibly having a continuous increment if the tokens aren't equal (maximum of the subsequence obtained by skipping either of the tokens, or by incrementing by the similarity between the two tokens). 

(see function `longestCommonSubsequence()` in `edu.stanford.nlp.sempre.interactive.InteractiveEmbeddingsBeamParser.java`)

# Similarity measure
The similarity between words can be computed in one of five ways:
- Lin et al. algorithm (LIN)
- Wu-Palmer algorithm (WPU)
- Hirst-St-Onge algorithm (HSO)
- Path algorithm (PATH)
- word vector cosine similarity (W2V)

## WordNet embeddings
LIN, WPU, HSO, PATH depend on WordNet embeddings. 

They require the JAR files: [`jaw-jaw-1.0.2.jar`](https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/jawjaw/jawjaw-1.0.2.jar) and [`ws4j-1.0.1.jar`](https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/ws4j/ws4j-1.0.1.jar)
(see function `computeSimilarityWordNet()` in `edu.stanford.nlp.sempre.interactive.InteractiveEmbeddingsBeamParser.java`)


## Word vector embeddings
W2V requires word vector models, for example [GloVe](https://nlp.stanford.edu/projects/glove/).

You will need to download a database of pre-trained vectors, or train vectors yourself, and provide a path to the database of the file as an option to the 'InteractiveEmbeddingsBeamParser'.

The similarity computation for the longest common subsequence is handled by the function `computeSimilarityWordVector()` in `edu.stanford.nlp.sempre.interactive.InteractiveEmbeddingsBeamParser.java`, and the word vector embedding is handled in the package `edu.stanford.nlp.sempre.interactive.embeddings`

# Running the application with word embeddings
Run the program with the option `-Parser "interactive.InteractiveEmbeddingsBeamParser"` to select the correct Parser.

The similarity measure used is selected with the option `-InteractiveEmbeddingsBeamParser.sim`, with one of `LIN`, `WPU`, `HSO`, `PATH`, `W2V`.

The path to the word vector embeddings is given with the option  `-InteractiveEmbeddingsBeamParser.embeddingsPath`. (Necessary when using W2V)

It is assumed that `-Parser.aggressivePartialParsing` is `true`

For example: 

To use the word vector embeddings, type the following (assuming "./glove/glove.6B.50d.txt" is the relative path to a word vector embeddings file) :
> ./interactive/run @mode=voxelurn -Parser.aggressivePartialParsing -Parser "interactive.InteractiveEmbeddingsBeamParser" -InteractiveEmbeddingsBeamParser.sim W2V  -InteractiveEmbeddingsBeamParser.embeddingsPath "./glove/glove.6B.50d.txt"

To use the LIN algorithm, type the following :
> ./interactive/run @mode=voxelurn -Parser.aggressivePartialParsing -Parser "interactive.InteractiveEmbeddingsBeamParser" -InteractiveEmbeddingsBeamParser.sim LIN 