# -*- coding: utf-8 -*-
'''
This code is borrowed from cszn/KAIR
'''

import functools
from torch.nn import init


# --------------------------------------------
# Feature extractor
# --------------------------------------------
def define_F(is_train):  
    # ----------------------------------------
    # DnCNN
    # ----------------------------------------
    from models.network_dncnn import DnCNN as net
    netF = net(in_nc=1, out_nc=1, nc=64, nb=17, act_mode='Bt')
    
    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if is_train:
        init_weights(netF,
                     init_type='xavier_uniform',
                     init_bn_type='uniform',
                     gain=1.0)

    return netF


# --------------------------------------------
# Recurrent Criss-Cross Attention
# --------------------------------------------
def define_Att(device, is_train):
    
    from models.network_attention import RCCAModule
    netAtt = RCCAModule(in_channels=1, device=device)

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if is_train:
        init_weights(netAtt,
                     init_type='xavier_uniform',
                     init_bn_type='uniform',
                     gain=1.0)
      
    return netAtt


# --------------------------------------------
# SimilarityNet
# --------------------------------------------
def define_S():   
    # ----------------------------------------
    # ResNet-18
    # ----------------------------------------
    from models.network_similarity import Res_Net
    netS = Res_Net('resnet18', num_classes=2)
    return netS


"""
# --------------------------------------------
# weights initialization
# --------------------------------------------
"""


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       1.0
    """
    print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1 or classname.find('ConstrainPre') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
    net.apply(fn)
