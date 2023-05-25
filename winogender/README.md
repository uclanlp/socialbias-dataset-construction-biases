## Winogender
- Original Dataset: https://github.com/rudinger/winogender-schemas

## Models
All models used for evaluation on the Winogender task are publicly available.
-  ai2spanbert: https://demo.allennlp.org/coreference-resolution
- UnifiedQA: https://github.com/allenai/unifiedqa
- Longformer: https://github.com/shtoshni/fast-coref

We have provided sample scripts to evaluate these public models on our alternate dataset constructions.

## Data
All preprocessed data (corresponding to the various alternate constructions) is available in the `data` directory.

## Alternate Constructions

### 1. Addition of Clauses

For Winogender, we add clauses like `who just returned from the beach` to the different entities in the sentence. For instance, the sentence `The customer left the bartender a big tip because he was feeling generous.` becomes `The customer, who just returned from the beach, left the bartender a big tip because he was feeling generous.`

### 2. Synonym Substitution

We substitute with synonyms such that it does not change the meaning of the sentence. Winogender has $720$ sentences generated from $120$ templates, making manual substitution of synonyms in the templates feasible. For example, the sentence `The supervisor gave the employee feedback on his stellar performance.` is replaced by `The supervisor gave the employee feedback on his amazing performance.`

### 3. Adding Adjectives
We add descriptors in the form of adjectives that do not add information about which entity the pronoun or noun would refer to. We do it in four distinct ways:
-  (i) adding the descriptor to the occupation mentioned, e.g. doctor  (e.g., `doctor` to `good doctor`).
- (ii) adding it to the occupation as a separate clause (e.g., `doctor` to `the doctor who was good`).
- (iii) adding the descriptor to the participant mentioned, e.g., `client` (similar to (i)).
- (iv) adding it to the participant as a separate clause (similar to (ii)).


