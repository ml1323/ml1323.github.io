## MUSE-VAE : Multi-Scale VAE for Environment-Aware Long Term Trajectory Prediction

> Mihee Lee, Samuel S. Sohn, Seonghyeon Moon, Sejong Yoon, Mubbasir Kapadia, Vladimir Pavlovic
> CVPR 2022

[Paper](https://arxiv.org/abs/2201.07189)
[Website](https://ml1323.github.io/MUSE-VAE)



## Pretrained Models
* You can download pretrained models for PFSD from
**[PFSD models](https://drive.google.com/file/d/1QGGgYNomsQf2bCrR3OXBDi_1yWLHJm6y/view?usp=sharing)**
* Place the unzipped directory under the `datasets` directory as follows.
  ```
  datasets
    |- pretrained_models_pfsd
  ```

## Datasets
You can download our new dataset, PathFinding Simulation Dataset (PFSD)
**[Raw data](https://drive.google.com/file/d/1QGGgYNomsQf2bCrR3OXBDi_1yWLHJm6y/view?usp=sharing)**
**[Preprocessed data for MUSE-VAE](https://drive.google.com/file/d/1Wm5CTBrxozg9zMKvS2l9M3XtHhWyy3g9/view?usp=sharing)**
Details regarding PFSD can be found [here](https://ml1323.github.io/MUSE-VAE/tree-of-codes)


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
