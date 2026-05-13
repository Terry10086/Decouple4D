import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import sys

class SegNet(nn.Module):
    def __init__(self, args, feature_dim):
        super().__init__()
        timebase_pe = args.timebase_pe
        posbase_pe= args.posebase_pe
        self.input_ch = (3 + (3 * posbase_pe) * 2) # + (3 + (3 * timebase_pe) * 2) # + (1 + (1 * timebase_pe) * 2) # + 48 #  + 768 # + (1 + (1 * timebase_pe) * 2)      # (3 + (3 * posbase_pe) * 2) + 16*3 + (1 + (1 * timebase_pe) * 2)
        self.output_ch = feature_dim
        print("ID Encoding Dimension: ", feature_dim)
        self.W = 128 # 256
        self.D = 4
        self.mlp = nn.ModuleList(
            [nn.Linear(self.input_ch, self.W), nn.ReLU()] +
            sum([[nn.Linear(self.W, self.W), nn.ReLU()] for i in range(self.D-2)], []) +
            [nn.Linear(self.W, self.output_ch)]
        )
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('delta_x_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.apply(initialize_weights)

    def forward(self, point, shs = None, time = None, ):
        point_emb = poc_fre(point, self.pos_poc)
#       shs = poc_fre(shs, self.delta_x_poc)
#       # time_emb = poc_fre(time, self.time_poc)
# 
        # h = torch.cat([point_emb, shs], -1) # torch.cat([point_emb, time_emb, shs.reshape(shs.shape[0], -1)], -1)   torch.cat([point.squeeze(), time_emb], -1)

        h = point_emb
        for i, l in enumerate(self.mlp):
            h = self.mlp[i](h)
    
        return h

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)

def poc_fre(input_data, poc_buf):
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin, input_data_cos], -1)
    return input_data_emb


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import sys

class SegNet1(nn.Module):
    def __init__(self, args, feature_dim):
        super().__init__()
        timebase_pe = args.timebase_pe
        posbase_pe= args.posebase_pe
        self.input_ch = (3 + (3 * posbase_pe) * 2) 
        self.output_ch = feature_dim
        print("ID Encoding Dimension: ", feature_dim)
        self.W = 128
        self.D = 4
        self.mlp = nn.ModuleList(
            [nn.Linear(self.input_ch, self.output_ch)]
        )
        
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.apply(initialize_weights)

    def forward(self, point, shs = None, time = None, ):
        point_emb = poc_fre(point, self.pos_poc)
        h = point_emb
        for i, l in enumerate(self.mlp):
            h = self.mlp[i](h)
    
        return h

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)

def poc_fre(input_data, poc_buf):
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin, input_data_cos], -1)
    return input_data_emb