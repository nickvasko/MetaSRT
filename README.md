# MetaSRT

MetaSRT: Meta-Learning Framework for Sentence Representation Transfer

## Overview

Code is adapted from [ConSERT: A Contrastive Framework for Self-Supervised Sentence Representation Transfer](https://arxiv.org/abs/2105.11741)
paper in ACL 2021. Please review their [GitHub](https://github.com/yym6472/ConSERT) 
for requirements and explanation of coding framework.

#### Contributions

1. Test whether contrastive training combined with direct STS supervision outperforms original 
best model from [Sentence-BERT](https://aclanthology.org/D19-1410.pdf).

## Results for English STS Tasks 

##### Reimplementations of ConSERT experiments

| ID | Model                 | STS12 | STS13 | STS14 | STS15 | STS16 | STSb | SICK-R | Avg.  |
|----|-----------------------|:-----:|:-----:|:-----:|:-----:|:-----:|:----:|:------:|:-----:|
| -  | BERT<sub>base</sub>   | 35.20 | 59.53 | 49.37 | 63.39 | 62.73 | 48.18 | 58.60 | 53.86 |
| 1  | ConSERT               | 64.59 | 78.57 | 69.09 | 79.66 | 75.86 | 73.93 | 67.18 | 72.70 |
| 2  | SBERT-NLI             | 70.23 | 75.56 | 71.97 | 78.65 | 73.63 | 76.47 | 73.05 | 74.22 |
| 3  | ConSERT-*joint*       | 70.91 | 80.08 | 75.26 | 82.03 | 76.73 | 79.12 | 77.90 | 77.43 |
| 4  | ConSERT-*sup-unsup*   | 73.18 | 85.15 | 77.28 | 82.79 | 78.48 | 81.86 | 75.23 | 79.14 |
| 5  | ConSERT-*joint-unsup* | 74.38 | 84.05 | 77.32 | 83.79 | 78.63 | 81.50 | 76.61 | 79.47 |

##### New ConSERT models:

The original Sentence-BERT paper achieved best results with NLI supervision followed by 
STS supervision on the STSb training dataset. The ConSERT paper left this unexplored and
it is unclear in what order to perform STS supervision.
- **Framework 1**: NLI joint supervision &#8594; Unsupervised STS &#8594; STS Supervision 
- **Framework 2**: NLI joint supervision &#8594; STS Supervision &#8594; Unsupervised STS

**Framework 2** is more practical in that the model is trained STS training examples, then 
contrastive learning pushes the model to the correct representation space for the unseen data. 

| ID | Model                                     | STS12 | STS13 | STS14 | STS15 | STS16 | STSb | SICK-R | Avg.  |
|----|-------------------------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|:----:|:------:|:-----:|
| 1  | SBERT-STS                                 |  |  |  |  |  |  |  |  |
| 2  | SBERT-NLI-STS                             |  |  |  |  |  |  |  |  |
| 3  | ConSERT-*joint-unsup-STS*                 |  |  |  |  |  |  |  |  |
| 4  | ConSERT-*joint-STS<sub>joint</sub>*       |  |  |  |  |  |  |  |  |
| 5  | ConSERT-*joint-STS<sub>joint</sub>-unsup* |  |  |  |  |  |  |  |  |

Notes:
1. All the models are trained from `bert-base-uncased`.
2. For the unsupervised transfer, we merge all unlabeled texts from 7 STS datasets (STS12-16, STSbenchmark and 
SICK-Relatedness) as the training data (total 89192 sentences), 
and use the STSbenchmark dev split (including 1500 human-annotated sentence pairs) to select the best checkpoint.
3. The sentence representations are obtained by averaging the token embeddings at the last two layers of BERT.
4. All models were trained on a single Tesla T4 with pytorch 1.9.0 and cuda 10.2. 
