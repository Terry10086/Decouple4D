#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import open3d as o3d
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from random import randint
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.point_utils import addpoint, combine_pointcloud, downsample_point_cloud_open3d, find_indices_in_A
from scene.deformation import deform_network
from scene.segnet import SegNet, SegNet1
from scene.regulation import compute_plane_smoothness
import sys
import pathlib

class Classifier:

    def __init__(self, args, feature_dim: int = 32, num_classes = 2, use_BCE=True):
        feature_dim = 32
        self._mlp = SegNet(args, feature_dim).cuda()
        self._mlp_tjy = SegNet1(args, feature_dim).cuda()
        self._classifier = torch.nn.Conv2d(feature_dim, 1, kernel_size=1).cuda()
        
        
    def training_setup(self, training_args):
        l = [
            {'params': self._mlp.parameters(), 'lr': 1e-3, 'name': 'mlp'},
            {'params': self._classifier.parameters(), 'lr': 1e-3, 'name': 'classifier'},
            ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)     


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif  "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr             
            # elif param_group["name"] == "mlp":
            #     lr = self.mlp_scheduler_args(iteration)
            #     param_group['lr'] = lr

    
    def load_mlp(self, path):
        self._mlp.load_state_dict(torch.load(os.path.join(path, "mlp.pt")))
        self._mlp.cuda()
        self._mlp_tjy.load_state_dict(torch.load(os.path.join(path, "mlp.pt")))
        self._mlp_tjy.cuda()
    
    def load_classifier(self, path):
        self._classifier.load_state_dict(torch.load(os.path.join(path, "classifier.pt")))
        self._classifier.cuda()
        
    def save_mlp(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self._mlp.state_dict(), os.path.join(path, "mlp.pt"))
        torch.save(self._mlp_tjy.state_dict(), os.path.join(path, "mlp_tjy.pt"))
        
    def save_classifier(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self._classifier.state_dict(), os.path.join(path, "classifier.pt"))
        
    def capture(self):      # 保存当下时刻的各个参数.pth
        return (
            self.optimizer.state_dict(),
            self._mlp.state_dict(),
            # self._mlp_tjy.state_dict(),
            self._classifier.state_dict()
        )

    def restore(self, model_args, training_args):       # load checkpoint
        # (opt_dict, mlp_state, _mlp_tjy_state, classifier_state) = model_args
        if len(model_args) == 4:
            opt_dict, mlp_state, _, classifier_state = model_args
        elif len(model_args) == 3:
            opt_dict, mlp_state, classifier_state = model_args
        self.training_setup(training_args)
        self._mlp.load_state_dict(mlp_state)
        # self._mlp_tjy.load_state_dict(_mlp_tjy_state)
        self._classifier.load_state_dict(classifier_state)
        self.optimizer.load_state_dict(opt_dict)



class TrajectoryGRU(nn.Module):
    def __init__(self, feature_dim=32, posbase_pe = 6):
        super().__init__()
        self._gru = nn.GRU(input_size=feature_dim, hidden_size=32, batch_first=False).cuda()
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))

    def forward(self, trajectory_feats):
        """
        trajectory_feats: torch.Tensor, shape [T, N, feature_dim]
        """
        # point_emb = poc_fre(trajectory_feats, self.pos_poc.cuda())
        # traj_feats = trajectory_feats.permute(1, 0, 2).contiguous()  # [N, T, feature_dim]

        output, hidden = self._gru(trajectory_feats)
        # pooled, _ = torch.max(output, dim=0)

        hidden = hidden.squeeze(0)          # [N, feature_dim]

        return hidden  # 返回 [N, feature_dim]，你可以接着调用分类器
    
    def training_setup(self, lr=1e-4):
        self.optimizer = torch.optim.Adam(self._gru.parameters(), lr=lr)
    
    def capture(self):      # 保存当下时刻的各个参数.pth
        return {
            'gru_state': self._gru.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }

    def restore(self, checkpoint):       # load checkpoint
        self.training_setup()  # 先初始化优化器
        self._gru.load_state_dict(checkpoint['gru_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])



def poc_fre(input_data, poc_buf):
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin, input_data_cos], -1)
    return input_data_emb