import os
from argparse import ArgumentParser
import easydict
import numpy as np
import torch

def get_args():
#     parser = ArgumentParser(description='grass_pytorch')
#     parser.add_argument('--box_code_size', type=int, default=12)
#     parser.add_argument('--feature_size', type=int, default=80)
#     parser.add_argument('--hidden_size', type=int, default=200)
#     parser.add_argument('--symmetry_size', type=int, default=8)
#     parser.add_argument('--max_box_num', type=int, default=30)
#     parser.add_argument('--max_sym_num', type=int, default=10)

#     parser.add_argument('--epochs', type=int, default=300)
#     parser.add_argument('--batch_size', type=int, default=123)
#     parser.add_argument('--show_log_every', type=int, default=3)
#     parser.add_argument('--save_log', action='store_true', default=False)
#     parser.add_argument('--save_log_every', type=int, default=3)
#     parser.add_argument('--save_snapshot', action='store_true', default=False)
#     parser.add_argument('--save_snapshot_every', type=int, default=5)
#     parser.add_argument('--no_plot', action='store_true', default=False)
#     parser.add_argument('--lr', type=float, default=.001)
#     parser.add_argument('--lr_decay_by', type=float, default=1)
#     parser.add_argument('--lr_decay_every', type=float, default=1)

#     parser.add_argument('--no_cuda', action='store_true', default=False)
#     parser.add_argument('--gpu', type=int, default=0)
#     parser.add_argument('--data_path', type=str, default='data')
#     parser.add_argument('--save_path', type=str, default='models')
#     parser.add_argument('--resume_snapshot', type=str, default='')
#     args = parser.parse_args()
#     return args

    args = easydict.EasyDict({
        "box_code_size": 512,
        "feature_size": 200,
        "hidden_size": 500,
        "max_box_num": 6126,
        "epochs": 300,
        "batch_size": 10,
        "show_log_every": 1,
        "save_log": False,
        "save_log_every": 3,
        "save_snapshot": True,
        "save_snapshot_every": 5,
        "save_snapshot":'snapshot',
        "no_plot": False,
        "lr":0.001,
        #"lr": 0.1,
        "lr_decay_by":1,
        "lr_decay_every":1,
        "no_cuda": False,
        "gpu":0,
        "data_path":'data',
        "save_path":'models',
        "resume_snapshot":""
    })
    return args



def get_quad_feas(vox,k):

# collect feas in octree order, vox must be (k^n)^3, k is the length of the
# leaf vox, should be power of 2
# label: 0:leaf_mix 1:interior(must be mix)

    n = vox.shape[1]
    
    if n<k:
        raise ValueError('dim must be larger than k');
    
    if (torch.all(vox[0]==1) or torch.all(vox[0]==0)) and (torch.all(vox[1]==1) or torch.all(vox[1]==0)) and \
       (torch.all(vox[2]==1) or torch.all(vox[2]==0)) and (torch.all(vox[3]==1) or torch.all(vox[3]==0)):
        feas_all = vox[:,:k,:k].unsqueeze(0)
        label = torch.tensor([0]).to(torch.device("cuda"))
        return feas_all, label
    
    if n==k:
        feas_all = vox.unsqueeze(0)
        label = torch.tensor([0]).to(torch.device("cuda"))
        return feas_all, label
    
    
    feas1,l1 = get_quad_feas(vox[:,0:int(n/2),0:int(n/2)],k)
    feas2,l2 = get_quad_feas(vox[:,int(n/2):n,0:int(n/2)],k)
    feas3,l3 = get_quad_feas(vox[:,0:int(n/2),int(n/2):n],k)
    feas4,l4 = get_quad_feas(vox[:,int(n/2):n,int(n/2):n],k)
    
    
    feas_all = torch.cat((feas1,feas2,feas3,feas4),0)
    label = torch.cat((l1,l2,l3,l4,torch.tensor([1]).to(torch.device("cuda"))))
    return feas_all, label


