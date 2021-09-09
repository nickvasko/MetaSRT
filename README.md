# MetaSRT

MetaSRT: Meta-Learning Framework for Sentence Representation Transfer

## Requirements

Code is adapted from [ConSERT: A Contrastive Framework for Self-Supervised Sentence Representation Transfer](https://arxiv.org/abs/2105.11741)
paper in ACL 2021. Please review their [GitHub](https://github.com/yym6472/ConSERT) 
for requirements and explanation of coding framework.

## Pre-trained Models & Results

### English STS Tasks 

*Reimplementations*

| ID | Model                        | STS12 | STS13 | STS14 | STS15 | STS16 | STSb | SICK-R | Avg. |
|----|------------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|:----:|:------:|:----:|
| -  | bert-base-uncased (baseline) | 35.20 | 59.53 | 49.37 | 63.39 | 62.73 | 48.18 | 58.60 | 53.86 |
| 1  | sup-sbert-base               | 70.23 | 75.56 | 71.97 | 78.65 | 73.63 | 76.47 | 73.05 | 74.22 |
| 2  | unsup-consert-base           | 64.59 | 78.57 | 69.09 | 79.66 | 75.86 | 73.93 | 67.18 | 72.70 |
| 3  | sup-consert-joint-base       | 70.92 | 79.98 | 74.88 | 81.76 | 76.46 | 78.99 | 78.15 | 77.31 |
| 4  | sup-consert-sup-unsup-base   | 73.02 | 84.86 | 77.32 | 82.70 | 78.20 | 81.34 | 75.00 | 78.92 |
| 5  | sup-consert-joint-unsup-base | 74.46 | 84.19 | 77.08 | 83.77 | 78.55 | 81.37 | 77.01 | 79.49 |

Note:
1. All the *base* models are trained from `bert-base-uncased`.
2. For the unsupervised transfer, we merge all unlabeled texts from 7 STS datasets (STS12-16, STSbenchmark and 
SICK-Relatedness) as the training data (total 89192 sentences), 
and use the STSbenchmark dev split (including 1500 human-annotated sentence pairs) to select the best checkpoint.
3. The sentence representations are obtained by averaging the token embeddings at the last two layers of BERT.
4. All models were trained on a single GeForce RTX 3090 with pytorch 1.8.1 and cuda 11.1. 
