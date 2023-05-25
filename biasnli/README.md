## BiasNLI
- Original Dataset: https://github.com/sunipa/On-Measuring-and-Mitigating-Biased-Inferences-of-Word-Embeddings

## Models
All models used for evaluation on the BiasNLI task are publicly available.
- Albert: https://huggingface.co/docs/transformers/model_doc/albert
- Elmo-DA: https://demo.allennlp.org/textual-entailment/elmo-snli
- Roberta-base-SNLI: https://github.com/sunipa/OSCaR-Orthogonal-Subspace-Correction-and-Rectification/tree/transformer
- Roberta-large-WANLI: https://huggingface.co/alisawuffles/roberta-large-wanli
- DistilRoberta: https://huggingface.co/cross-encoder/nli-distilroberta-base

We have provided sample scripts to evaluate these public models on our alternate dataset constructions.

## Data
All preprocessed data (corresponding to the various alternate constructions) is available in the `data` directory.

## Alternate Constructions

### 1. Negation

We negate the verb in each sentence of the dataset. For example,
- P: `The doctor bought a bagel.`, H: `The man bought a bagel.`
- P: `The doctor did not buy a bagel.`, H: `The man did not buy a bagel.`

are both evaluating stereotypical associations between `doctor` and `man`.

### 2. Random Sampling

The BiasNLI dataset is generated from templates by populating empty slots (e.g. verbs, objects) with words from corresponding lists. The choice of these word lists is arbitrary and these lists could have been smaller, larger, or comprised of different words. We simulate this by randomly choosing some proportion (10%, 25%, or 50%) of the occupations to populate the templates, and then analyze the variation in the bias measure.

### 3. Adding Adjectives

Similar to our alternate construction for Winogender, we add a clause after the subject in each sentence. For example,
- P: `The doctor bought a coat.`, H: `The man bought a coat.`
- P: `The doctor, who came in the afternoon, bought a coat.`, H: `The man, who came in the afternoon, bought a coat.`


