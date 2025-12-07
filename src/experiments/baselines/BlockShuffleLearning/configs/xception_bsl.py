import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms


device_id = 0
device = torch.device('cuda:{}'.format(device_id)) if torch.cuda.is_available() else torch.device('cpu')
is_train = True
is_resume = False
load_model = \
'../checkpoints/xception_bsl/xception_58_97.072049_220626173708.pth'
print('Init model...')
from models.xception import XceptionBSL
net = XceptionBSL(num_class=1, is_train=is_train, is_bs_adv=True, is_rs_adv=True).eval().to(device)

from tools.train_bsl import TrainBSL
train = TrainBSL(net, device)
train.net_name = 'xception'
train.correct_str_test = '{:.2f}'

if is_train:
    train.correct_str = '{0:.2f} {1:.2f} {2:.2f} {3:.2f}'
    save_path = '../checkpoints/xception_bsl_rvf10k/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, os.path.split(__file__)[-1])
    shutil.copy(__file__, file_path)
    train.save_path = save_path
    criterion = nn.BCEWithLogitsLoss().to(device)
    criterion_bs = nn.BCEWithLogitsLoss().to(device)
    criterion_rs = nn.SmoothL1Loss().to(device)
    optimizer = optim.SGD(net.get_trainable_parameters(), lr=1e-2, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, 1e-5)
    train.criterion = criterion
    train.criterion_bs = criterion_bs
    train.criterion_rs = criterion_rs
    train.optimizer = optimizer
    train.lr_scheduler = lr_scheduler

if is_resume:
    print('Load weight...')
    train.resume(load_model)


print('Init data...')

# ------------------------------
# DEFINE DATASET NAME HERE
# ------------------------------
DATASET_NAME = "taming_transformer_VQGAN"   # ← change each run

# ------------------------------
# BUILD DATALOADERS
# ------------------------------
from configs.rvf10k import build_dataloaders
train_loader, val_loader, test_loader = build_dataloaders(DATASET_NAME)








