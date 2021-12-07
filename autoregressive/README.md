# Long-Short Transformer for autoregressive language modeling

This folder contains the source code for char-level language modeling in the Transformer-LS [paper](https://arxiv.org/abs/2107.02192). 

The autoregressive long-short term attention implementation is [here](./model_lib/layer.py).

## Dependencies
From any directory, run the following to install fairseq:
```
git clone https://github.com/pytorch/fairseq.git
git reset --hard 1f7ef9ed1e1061f8c7f88f8b94c7186834398690
cd fairseq
pip install --editable .
```

## Data Preprocessing
First, download and split the datasets for enwik8 and text8 by running `bash data_prepro/get_data.sh` (adapted from [Transformer-XL](https://github.com/kimiyoung/transformer-xl/blob/master/getdata.sh)). Then, run the following to preprocess them into fairseq's binary format.

```bash
fairseq-preprocess --only-source --trainpref datasets/enwik8/train.txt \
    --validpref datasets/enwik8/valid.txt --testpref datasets/enwik8/test.txt \
    --destdir datasets/enwik8/data-bin/ --joined-dictionary --workers 20
    
fairseq-preprocess --only-source --trainpref datasets/text8/train.txt \
    --validpref datasets/text8/valid.txt --testpref datasets/text8/test.txt \
    --destdir datasets/text8/data-bin/ --joined-dictionary --workers 20
```

## Training scripts
Please refer to the scripts under `launch`. Run the scripts under the project directory.
