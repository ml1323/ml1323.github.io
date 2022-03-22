## MUSE-VAE

### Multi-Scale VAE for Environment-Aware Long Term Trajectory Prediction
**CVPR 2022**
> Mihee Lee, Samuel S. Sohn, Seonghyeon Moon, Sejong Yoon, Mubbasir Kapadia, Vladimir Pavlovic


[Paper](https://arxiv.org/abs/2201.07189)
&nbsp;&nbsp;&nbsp;
[Website](https://ml1323.github.io/MUSE-VAE)



## Pretrained Models
+ You can download pretrained models for PFSD from
**[PFSD models](https://drive.google.com/file/d/1HWzOzskjDjdYCJLiHpHM1L7Dg-NPMhoy/view?usp=sharing)**

+ Place the unzipped directory under the `ckpts` directory as follows.
```bash
ckpts
    |- pretrained_models_pfsd
```

## Datasets
+ The pre-processed version of our new dataset, PathFinding Simulation Dataset (PFSD), is available for download at
**[Preprocessed PFSD](https://drive.google.com/file/d/1Wm5CTBrxozg9zMKvS2l9M3XtHhWyy3g9/view?usp=sharing)**

+ Place the unzipped pkl files under the `datasets/pfsd` directory as follows.
```bash
datasets
    |- pfsd
```

+ Raw data and details of PFSD can be found
**[here](https://ml1323.github.io/MUSE-VAE/tree-of-codes)**


## Running models
+ You can use the script `scripts/pfsd/eval.sh` to get the evaluation results for PFSD reported in the paper.
```bash
sh eval.sh
```

+ You can use the scripts starting with `train` under `scripts/pfsd/` to train each of the network.
```bash
sh train_lg_cvae.sh
sh train_sg_net.sh
sh train_micro.sh
```
