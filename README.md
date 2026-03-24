# FairGC: Fairness-aware Graph Condensation

Accepted in IJCNN 2026, the code implementation

## Abstract

Graph condensation (GC) has become a vital strategy for scaling Graph Neural Networks by compressing massive datasets into small, synthetic node sets. While current GC methods effectively maintain predictive accuracy, they are primarily designed for utility and often ignore fairness constraints. Because these techniques are bias-blind, they frequently capture and even amplify demographic disparities found in the original data. This leads to synthetic proxies that are unsuitable for sensitive applications like credit scoring or social recommendations. To solve this problem, we introduce FairGC, a unified framework that embeds fairness directly into the graph distillation process. Our approach consists of three key components. First, a Distribution-Preserving Condensation module synchronizes the joint distributions of labels and sensitive attributes to stop bias from spreading. Second, a Spectral Encoding module uses Laplacian eigen-decomposition to preserve essential global structural patterns. Finally, a Fairness-Enhanced Neural Architecture employs multi-domain fusion and a label-smoothing curriculum to produce equitable predictions. Rigorous evaluations on four real-world datasets, show that FairGC provides a superior balance between accuracy and fairness. Our results confirm that FairGC significantly reduces disparity in Statistical Parity and Equal Opportunity compared to existing state-of-the-art condensation models.

## Dataset

This code supports four diverse real-world graph datasets for fair graph condensation evaluation:

- Pokec-n / Pokec-z: https://github.com/EnyanDai/FairGNN/tree/main/dataset
- Credit: https://archive.ics.uci.edu/ml/datasets/statlog+german+credit+data
- AMiner-L: https://www.aminer.cn/aminernetwork

## Running

First, preprocess the raw data:

`python preprocess_data.py`

second, run the main experiment with the following command example:

`python main.py --dataset credit --reduction_rate 0.01 --gcond_epochs 1000 --lr_feat 0.01 --hidden_dim 64 --nlayer 2 --nheads 1 --tran_dropout 0.1 --feat_dropout 0.2 --prop_dropout 0.6 --lr 0.0005 --k 1 --seed 25`

