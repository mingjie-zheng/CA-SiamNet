# -*- coding: utf-8 -*-
'''
This code is borrowed from speedinghzl/CCNet
'''

import torch
import torch.nn as nn

"""
# --------------------------------------------
# CCNet
# --------------------------------------------
# References:
@article{Huang_CCNet,
  author={Huang, Zilong and Wang, Xinggang and Wei, Yunchao and Huang, Lichao and Shi, Humphrey and Liu, Wenyu and Huang, Thomas S.},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={CCNet: Criss-Cross Attention for Semantic Segmentation}, 
  year={2023},
  volume={45},
  number={6},
  pages={6896-6908},
}
# --------------------------------------------
"""


def INF(B,H,W, device):
    return -torch.diag(torch.tensor(float("inf")).to(device).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)
        

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim, device):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
        self.device = device


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1) #(bs*W, H, C)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1) #(bs*H, W, C)
        
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height) #(bs*W, C, H)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width) #(bs*H, C, W)
        
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height) #(bs*W, C, H)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width) #(bs*H, C, W)
        
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width, self.device)).view(m_batchsize,width,height,height).permute(0,2,1,3) #(bs, H, W, H_others)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width) #(bs, H, W, W_others)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3)) #(bs, H, W, H_others+W_others)

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height) #(bs*W, H, H_others)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width) #(bs*H, W, W_others)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1) #(bs, C, H, W)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3) #(bs, C, H ,W)

        return self.gamma*(out_H + out_W) + x


class RCCAModule(nn.Module):
    def __init__(self, in_channels, device):
        super(RCCAModule, self).__init__()
        self.cca = CrissCrossAttention(in_channels, device)
        
    def forward(self, x):
        x = self.cca(x)
        x = self.cca(x)
        return x


if __name__ == '__main__':
    
    model = CrissCrossAttention(1, torch.device('cpu'))
    x = torch.randn((1, 1, 256, 256))
    out = model(x)
    print(out.shape)