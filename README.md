# Long-Short Transformer (Transformer-LS)

This repository hosts the code and models for the paper:

[Long-Short Transformer: Efficient Transformers for Language and Vision](https://arxiv.org/abs/2107.02192)

# Updates
- July 23, 2021: Release the code and models for [ImageNet classification](./Transformer-LS-ImageNet) and [Long-Range Arena](./Transformer-LS-LRA)


# Architecture
![plot](https://user-images.githubusercontent.com/18202259/125551111-28369067-22f1-4615-adaf-611934a9752d.png)
Long-short Transformer substitutes the full self attention of the original Transformer models with an efficient attention that considers both long-range and short-term correlations. Each query attends to tokens from the segment-wise sliding window to capture short-term correlations, and the dynamically projected features to capture long-range correlations. To align the norms of the original and projected feature vectors and improve the efficacy of the aggregation, we normalize the original and project feature vectors with two sets of Layer Normalizations.

# Tasks

- [>>> Transformer-LS for ImageNet classification](./Transformer-LS-ImageNet)
- [>>> Transformer-LS for Long Range Areana](./Transformer-LS-LRA)
- [>>> Transformer-LS for autoregressive language modeling](./Transformer-LS-Autoregressive)
