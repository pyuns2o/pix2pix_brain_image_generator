import os
import numpy as np

import torch
import torch.nn as nn


## 네트워크 grad 설정하기
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def save(ckpt_dir, netG, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'netG': netG.state_dict()}, "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

def load(ckpt_dir, netG):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return netG

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst = [f for f in ckpt_lst if f.endswith('pth')]
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    netG.load_state_dict(dict_model['netG'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return netG, epoch