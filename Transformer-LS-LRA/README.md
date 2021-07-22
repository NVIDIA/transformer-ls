# Long-Short Transformer for Long-Range Arena

This folder contains the source code for Long-Range Arena benchmark in [Transformer-LS paper](https://arxiv.org/abs/2107.02192). 

It is built on this [repository](https://github.com/mlpen/Nystromformer)

The implementation of long-short term attention is [here](./attention_transformer_ls.py).

## Dataset Setup

To run the experiments, first, run the following to create the train, val and test data
  ```angular2html
  cd datasets
  python text.py
  python listops.py
  python retrieval.py
  ```
To download the datasets,  one may refer to the instructions within [Long-Range Arena repository](https://github.com/google-research/long-range-arena#dataset-setup)


## Scripts

Then, create the path for the checkpoints: `mkdir LRA_chks`. Finally, simply execute
  ```angular2html
  bash run_text.sh
  ```
  ```angular2html
  bash run_listops.sh
  ```
  ```angular2html
  bash run_retrieval.sh
  ```

## Reference repositories
- [Nystromformer](https://github.com/mlpen/Nystromformer)
- [Long Range Arena](https://github.com/google-research/long-range-arena)