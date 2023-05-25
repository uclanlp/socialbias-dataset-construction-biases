# Dataset Construction Biases of Social Bias Benchmarks
This repository contains the all datasets and pre-processing/evaluation code for our ACL 2023 paper titled `The Tail Wagging the Dog: Dataset Construction Biases of Social Bias Benchmarks`. Here is a link to the [preprint](https://arxiv.org/abs/2210.10040/).


## Abstract
- How reliably can we trust the scores  obtained from social bias benchmarks as faithful indicators of problematic social biases in a given model?
- In this work, we study this question by contrasting social biases with non-social biases that stem from choices made during dataset construction (which might not even  be discernible to the human eye).
- To do so, we empirically simulate various alternative constructions for a given benchmark based on seemingly innocuous modifications (such as paraphrasing or random-sampling) that maintain the essence of their social bias.
- On two well-known social bias benchmarks ([Winogender](https://arxiv.org/abs/1804.09301) and [BiasNLI](https://arxiv.org/abs/1908.09369)), we observe that these shallow modifications have a surprising effect on the resulting degree of bias across various models and consequently the relative ordering of these models when ranked by measured bias.
- We hope these troubling observations motivate more robust measures of social biases.

## Usage
- All the datasets are publicly available. We include fully preprocessed datasets (corresponding to the various alternate constructions) in this repository.
- All the models used are publicly available. They are extremely simple to use, and we include sample scripts used to run relevant experiments using these models.
- Please refer to the folders `winogender` and `biasnli` and their corresponding READMEs for further details.