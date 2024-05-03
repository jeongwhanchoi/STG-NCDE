# Graph Neural Controlled Differential Equations for Traffic Forecasting
![GitHub Repo stars](https://img.shields.io/github/stars/jeongwhanchoi/STG-NCDE?style=social) ![Twitter Follow](https://img.shields.io/twitter/follow/jeongwhan_choi?style=social)
[![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=STG-NCDE&color=red&logo=arxiv)](https://arxiv.org/abs/2112.03558)[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fjeongwhanchoi%2FSTG-NCDE&count_bg=%233D59C8&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-neural-controlled-differential/traffic-prediction-on-pemsd7-l)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd7-l?p=graph-neural-controlled-differential)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-neural-controlled-differential/traffic-prediction-on-pemsd7-m)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd7-m?p=graph-neural-controlled-differential)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-neural-controlled-differential/traffic-prediction-on-pemsd3)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd3?p=graph-neural-controlled-differential) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-neural-controlled-differential/traffic-prediction-on-pemsd7)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd7?p=graph-neural-controlled-differential) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-neural-controlled-differential/traffic-prediction-on-pemsd4)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd4?p=graph-neural-controlled-differential) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-neural-controlled-differential/traffic-prediction-on-pemsd8)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd8?p=graph-neural-controlled-differential)

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

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=jeongwhanchoi/STG-NCDE&type=Date)](https://star-history.com/#jeongwhanchoi/STG-NCDE&Date)
