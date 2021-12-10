# Graph Neural Controlled Differential Equations for Traffic Forecasting

[![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=STG-NCDE&color=red&logo=arxiv)](https://arxiv.org/abs/2112.03558)

## Introduction

This is the repository of our accepted AAAI 2022 paper "Graph Neural Controlled Differential Equations for Traffic Forecasting". Paper is available on [arxiv](https://arxiv.org/abs/2112.03558).

## Citation
If you find this code useful, you may cite us as:

```
@inproceedings{choi2022STGNCDE,
  title={Graph Neural Controlled Differential Equations for Traffic Forecasting},
  author={Jeongwhan Choi AND Hwangyong Choi AND Jeehyun Hwang AND Noseong Park},
  booktitle={AAAI},
  year={2022}
}
```

## Setup Python environment for STG-NCDE
Install python environment
```{bash}
$ conda env create -f environment.yml 
```


## Reproducibility
### Usage
#### In terminal
- Run the shell file (at the root of the project)

```{bash}
$ bash run.sh
```
- Run the python file (at the `model` folder)
```{bash}
$ cd model

$ python Run_cde.py --dataset='PEMSD4' --model='GCDE' --model_type='type1' --embed_dim=10 --hid_dim=64 --hid_hid_dim=64 --num_layers=2 --lr_init=0.001 --weight_decay=1e-3 --epochs=200 --tensorboard --comment="" --device=0
```