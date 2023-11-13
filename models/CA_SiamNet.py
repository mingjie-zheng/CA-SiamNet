# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from models.define_network import define_F, define_Att, define_S

class CA_SiamNet(nn.Module):
    
    def __init__(self, device, is_train=False):
        
        super(CA_SiamNet, self).__init__()
        self.netF = define_F(is_train)
        self.cca = define_Att(device, is_train)
        self.netS = define_S()
    
    def forward(self, imgA, imgB):
        
        imgA = self.netF(imgA)
        imgB = self.netF(imgB)
        
        imgA = self.cca(imgA)
        imgB = self.cca(imgB)  
        
        feature = self.netS(torch.mul(imgA, imgB))
        
        return feature