# Long-Short Transformer for ImageNet classification

This folder contains the source code for ImageNet classification in [Transformer-LS paper](https://arxiv.org/abs/2107.02192). 

The implementation is based on [ViL](https://github.com/microsoft/vision-longformer).

The long-short term attention implementation is [here](./models/layers/transformer_ls.py).

# Experiments

| Model | #Params (M) | Image Size | FLOPs (G) | ImageNet top1 | 
| ----- | :-----:     |   :-----:  | :-----:   |   :-----:     |
| [ViL-LS-medium (224)](https://www.dropbox.com/s/ng14pebstaaydug/model_best.pth) | 39.8  |   224<sup>2</sup>  |   8.7   |   83.8       |
| [ViL-LS-base (224)](https://www.dropbox.com/s/80u5p5eh4txad10/model_best.pth)   | 55.8  |   224<sup>2</sup>  |  13.4   |   84.1       |
| [ViL-LS-medium (384)](https://www.dropbox.com/s/390tzi2ll3sfibl/model_best.pth) | 39.8  |   384<sup>2</sup>  |  28.7   |   84.4       |


## Dataset
This repo supports the following two ways to store the ImageNet data.

- If you specify the dataset name
  ```bash
  DATA.TRAIN "'imagenet-draco'," DATA.TEST "'imagenet-draco',"
  ```
  in the evaluation or training scripts (see the Scripts section), the file structure should look like:
  ```bash
  imagenet
  ├── train-jpeg
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val-jpeg
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```
- If you specify the dataset name
  ```bash
  DATA.TRAIN "'imagenet'," DATA.TEST "'imagenet',"
  ```
  in the evaluation or training scripts, it supports zipped ImageNet format. You may find more details [here](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md#data-preparation).



## Scripts
- To evaluate the checkpoints, first change the path to the ImageNet dataset in the scripts undewr `launch/eval`, then execute the following:
  ```bash
  # directory for checkpoints
  mkdir checkpoints
  
  # evaluating medium 224
  mkdir checkpoints/LS_medium_224
  wget -O checkpoints/LS_medium_224/model_best.pth https://www.dropbox.com/s/ng14pebstaaydug/model_best.pth
  bash launch/eval/eval_medium_224.sh
  
  # evaluating base 224
  mkdir checkpoints/LS_base_224
  wget -O checkpoints/LS_base_224/model_best.pth https://www.dropbox.com/s/80u5p5eh4txad10/model_best.pth
  bash launch/eval/eval_base_224.sh
  
  # evaluating medium 384
  mkdir checkpoints/LS_medium_384
  wget -O checkpoints/LS_medium_384/model_best.pth https://www.dropbox.com/s/390tzi2ll3sfibl/model_best.pth
  bash launch/eval/eval_medium_384.sh
  ```

- To train the models, run the following scripts (modify the dataset path accordingly):
  ```bash
  bash launch/train/train_medium_224.sh
  bash launch/train/train_base_224.sh
  bash launch/train/train_medium_384.sh
  ```