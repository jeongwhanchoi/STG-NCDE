import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
import time

from model.BasicTrainer_cde import Trainer
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader_cde
from lib.TrainInits import print_model_parameters
import os
from os.path import join
from Make_model import make_model
from torch.utils.tensorboard import SummaryWriter
#*************************************************************************#
Mode = 'train'
DEBUG = 'False'
DATASET = 'PEMSD4'      #PEMSD4 or PEMSD8
MODEL = 'GCDE'

#get configuration
config_file = './{}_{}.conf'.format(DATASET, MODEL)
#print('Read configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)

from lib.metrics import MAE_torch
def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

#parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default=DATASET, type=str)
args.add_argument('--mode', default=Mode, type=str)
args.add_argument('--device', default=0, type=int, help='indices of GPUs')
args.add_argument('--debug', default=DEBUG, type=eval)
args.add_argument('--model', default=MODEL, type=str)
args.add_argument('--cuda', default=True, type=bool)
args.add_argument('--comment', default='', type=str)


#data
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--lag', default=config['data']['lag'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--tod', default=config['data']['tod'], type=eval)
args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
args.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
args.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
#model
args.add_argument('--model_type', default=config['model']['type'], type=str)
args.add_argument('--g_type', default=config['model']['g_type'], type=str)
args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
args.add_argument('--hid_dim', default=config['model']['hid_dim'], type=int)
args.add_argument('--hid_hid_dim', default=config['model']['hid_hid_dim'], type=int)
args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
args.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
args.add_argument('--solver', default='rk4', type=str)

#train
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--weight_decay', default=config['train']['weight_decay'], type=eval)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--teacher_forcing', default=False, type=bool)
#args.add_argument('--tf_decay_steps', default=2000, type=int, help='teacher forcing decay steps')
args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')

args.add_argument('--missing_test', default=False, type=bool)
args.add_argument('--missing_rate', default=0.1, type=float)

#test
args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
args.add_argument('--model_path', default='', type=str)
#log
args.add_argument('--log_dir', default='../runs', type=str)
args.add_argument('--log_step', default=config['log']['log_step'], type=int)
args.add_argument('--plot', default=config['log']['plot'], type=eval)
args.add_argument('--tensorboard',action='store_true',help='tensorboard')

args = args.parse_args()
init_seed(args.seed)

GPU_NUM = args.device
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU

print(args)

#config log path
save_name = time.strftime("%m-%d-%Hh%Mm")+args.comment+"_"+ args.dataset+"_"+ args.model+"_"+ args.model_type+"_"+"embed{"+str(args.embed_dim)+"}"+"hid{"+str(args.hid_dim)+"}"+"hidhid{"+str(args.hid_hid_dim)+"}"+"lyrs{"+str(args.num_layers)+"}"+"lr{"+str(args.lr_init)+"}"+"wd{"+str(args.weight_decay)+"}"
path = '../runs'

log_dir = join(path, args.dataset, save_name)
args.log_dir = log_dir
if (os.path.exists(args.log_dir)):
        print('has model save path')
else:
    os.makedirs(args.log_dir)

if args.tensorboard:
    w : SummaryWriter = SummaryWriter(args.log_dir)
else:
    w = None

#init model
if args.model_type=='type1':
    model, vector_field_f, vector_field_g = make_model(args)
elif args.model_type=='type1_temporal':
    model, vector_field_f = make_model(args)
elif args.model_type=='type1_spatial':
    model, vector_field_g = make_model(args)
else:
    raise ValueError("Check args.model_type")

model = model.to(args.device)

if args.model_type=='type1_temporal':
    vector_field_f = vector_field_f.to(args.device)
    vector_field_g = None
elif args.model_type=='type1_spatial':
    vector_field_f = None
    vector_field_g = vector_field_g.to(args.device)
else:
    vector_field_f = vector_field_f.to(args.device)
    vector_field_g = vector_field_g.to(args.device)

print(model)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
print_model_parameters(model, only_num=False)

#load dataset
train_loader, val_loader, test_loader, scaler, times = get_dataloader_cde(args,
                                                               normalizer=args.normalizer,
                                                               tod=args.tod, dow=False,
                                                               weather=False, single=False)

#init loss function, optimizer
if args.loss_func == 'mask_mae':
    loss = masked_mae_loss(scaler, mask_value=0.0)
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
elif args.loss_func == 'huber_loss':
    loss = torch.nn.HuberLoss(delta=1.0).to(args.device)
else:
    raise ValueError

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init,
                             weight_decay=args.weight_decay)

#learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)

#start training
trainer = Trainer(model, vector_field_f, vector_field_g, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                  args, lr_scheduler, args.device, times,
                  w)
if args.mode == 'train':
    trainer.train()
elif args.mode == 'test':
    model.load_state_dict(torch.load('./pre-trained/{}.pth'.format(args.dataset)))
    print("Load saved model")
    trainer.test(model, trainer.args, test_loader, scaler, trainer.logger, times)
else:
    raise ValueError
