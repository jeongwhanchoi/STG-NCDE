# Graph Neural Controlled Differential Equations for Traffic Forecasting

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