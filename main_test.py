# -*- coding: utf-8 -*-
"""
@author: Mingjie Zheng
@email: mingjie.zheng@connect.polyu.hk
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from utils import utils_image as util
import utils.patch_extractor_rank as extractor

from models.CA_SiamNet import CA_SiamNet


def main():
    
    parse = argparse.ArgumentParser()
    parse.add_argument("--img_pth_A", help="Path of image A", default='examples/Samsung_NV15_1/Samsung_NV15_1_46019.JPG')
    parse.add_argument("--img_pth_B", help="Path of image B", default='examples/Samsung_NV15_1/Samsung_NV15_1_46033.JPG')
    parse.add_argument("--model_pth", help="Path of pre-trained model", default='checkpoint/model_best.pth')
    parse.add_argument("--cuda", help="Number of CUDA device", default='0')
    parse.add_argument("--patch_dim", help="Patch size", type=int, default=256)
    parse.add_argument("--patch_num", help="Number of patches", type=int, default=50) 

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    args = parse.parse_args()
    img_pth_A = args.img_pth_A
    img_pth_B = args.img_pth_B
    model_pth = args.model_pth
    cuda = args.cuda
    patch_dim = args.patch_dim
    patch_num = args.patch_num
    
    device = torch.device('cuda:'+cuda if torch.cuda.is_available() else 'cpu')
    
    
    # ----------------------------------------
    # define and load model
    # ----------------------------------------
    model = CA_SiamNet(device, is_train=False)
    
    state_dict = torch.load(model_pth, map_location=device)
    if 'SiamNet' in state_dict.keys():
        model.load_state_dict(state_dict['SiamNet'], strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)
    
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    cal_similarity = nn.Softmax(dim=1).to(device)
    
    
    # ----------------------------------------
    # source device linking
    # ----------------------------------------

    img_name_A = os.path.basename(img_pth_A)
    img_name_B = os.path.basename(img_pth_B)
    
    patches = extractor.patch_extractor_pairs({'img_pth_A': img_pth_A, 'img_pth_B': img_pth_B, 
                                               'patch_dim': patch_dim}, patch_num=patch_num)
    
    scores = []        
    for _, row in patches.iterrows():
        img_A_patch = row['img_A_patch']
        img_B_patch = row['img_B_patch']
        
        with torch.no_grad():
            img_A_patch = util.uint2tensor4(img_A_patch).to(device)
            img_B_patch = util.uint2tensor4(img_B_patch).to(device)
            
            features = model(img_A_patch, img_B_patch)
            results = cal_similarity(features)
            similarity = results[:,1].item()

        scores.append(similarity)
    
    mean_score = np.mean(scores)            
    print('{:s} vs {:s} (patch_num = {}) are predicted from {:s}, the average similairty score is: {:.5f}.'.format(img_name_A, img_name_B, patch_num,    
                                                                                   'the same camera device' if mean_score>0.5 else 'different devices', 
                                                                                   mean_score)) 

    
if __name__ == '__main__':

    main()
