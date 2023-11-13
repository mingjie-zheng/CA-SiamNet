# -*- coding: UTF-8 -*-

from PIL import Image
import numpy as np
import pandas as pd
import os
from scipy.ndimage.filters import correlate
import matplotlib.pyplot as plt

"""
# --------------------------------------------
# Ranking Index Computation
# --------------------------------------------
# References:
@inproceedings{patch_strategy,
  author={Khan, Sahib and Bianchi, Tiziano},
  booktitle={2019 IEEE International Conference on Multimedia and Expo (ICME)},
  title={Fast image clustering based on camera fingerprint ordering},
  year={2019},
  pages={766--771},
}
"""

## Score functions ---
def obtain_score(img):
    
    img = img.astype(np.float64)
    
    G = np.divide(np.sum(img), 255*img.size)
    
    S = np.divide(np.sum(img == 255), img.size)
    
    kernel = [[0, -1, 0], 
              [-1, 4, -1], 
              [0, -1, 0]]
    L = correlate(img, kernel, mode='nearest')  
    T = np.divide(np.sum(pow(L, 2)), np.sum(pow(img, 2)))
    
    if 1 - T > 0:
        return pow(G, 1/2) * pow(1 - S, 1/0.5) * pow(1 - T, 1/2)
    else:
        return 0.0


def patch_extractor(img_A, img_B, dim, patch_num, **kwargs):
    """
    Patch extractor.
    
    Args:
    :param img_A (numpy.ndarray): the image to process. dtype must be either uint8 or float
    
    :param img_B (numpy.ndarray): the other image to process. dtype must be either uint8 or float

    :param dim (tuple | int): the dimensions of the patches (rows,cols).

    :param stride (tuple | int): the stride of each axis starting from top left corner (rows,cols).

    :return list: list of numpy.ndarray of the same type as the input img
    """

    # Arguments parser ---
    if isinstance(dim, int):
        dim = (dim, dim)
    if not isinstance(dim, tuple):
        raise ValueError('dim must be of type: [' + '|'.join([str(int), str(tuple)]) + ']')

    if 'stride' in kwargs.keys():
        stride = kwargs.pop('stride')
        if isinstance(stride, int):
            stride = (stride, stride)
        if not isinstance(stride, tuple):
            raise ValueError('stride must be of type: [' + '|'.join([str(int), str(tuple)]) + ']')
    else:
        stride = dim

    if len(kwargs.keys()):
        for key in kwargs:
            raise('Unrecognized parameter: {:}'.format(key))

    # Patch list ---
    img_A_patch_list = []
    img_B_patch_list = []
    for start_row in np.arange(start=0, stop=img_A.shape[0] - dim[0] + 1, step=stride[0]):
        for start_col in np.arange(start=0, stop=img_A.shape[1] - dim[1] + 1, step=stride[1]):
            patch = img_A[start_row:start_row + dim[0], start_col:start_col + dim[1]]
            img_A_patch_list += [patch]
            patch = img_B[start_row:start_row + dim[0], start_col:start_col + dim[1]]
            img_B_patch_list += [patch]

    # Evaluate patches---
    img_A_score = np.asarray(list(map(obtain_score, img_A_patch_list)))
    img_B_score = np.asarray(list(map(obtain_score, img_B_patch_list)))
    
    data = pd.DataFrame({'img_A_score': img_A_score, 
                         'img_B_score': img_B_score, 
                         'img_A_patch': img_A_patch_list, 
                         'img_B_patch': img_B_patch_list})
    
    score_mean = data[['img_A_score', 'img_B_score']].mean(axis=1)
    data['score_mean'] = score_mean
    
    topleft = data.head(1)
    
    data = data.sort_values('score_mean', ascending=False).loc[:, ('img_A_patch', 'img_B_patch')]
    
    if data.head(patch_num).empty:
        return topleft
    else:
        return data.head(patch_num)


def patch_extractor_pairs(args, patch_num):

    img_pth_A = args.pop('img_pth_A')
    img_pth_B = args.pop('img_pth_B')
    
    if 'patch_dim' in args.keys():
        dim = args.pop('patch_dim')
    else:
        dim = 256
    
    if not 'stride' in args.keys():
        args.update({'stride': dim})

    img_A = np.array(Image.open(img_pth_A).convert('L'))
    img_B = np.array(Image.open(img_pth_B).convert('L'))
    
    if img_A.shape[0] > img_A.shape[1]:
        print('Size warning: img_A: {:s} ({:d}*{:d}).'.format(img_pth_A.split(os.sep)[-1], img_A.shape[0], img_A.shape[1]))
    if img_B.shape[0] > img_B.shape[1]:
        print('Size warning: img_B: {:s} ({:d}*{:d}).'.format(img_pth_B.split(os.sep)[-1], img_B.shape[0], img_B.shape[1]))
    
    
    m = min(img_A.shape[0], img_B.shape[0])
    n = min(img_A.shape[1], img_B.shape[1])

    return patch_extractor(img_A[:m,:n], img_B[:m,:n], dim, patch_num, **args)


if __name__ == "__main__":
    
    img_pth_A = '../examples/Samsung_L74wide_0/Samsung_L74wide_0_43527.JPG'
    img_pth_B = '../examples/Samsung_L74wide_0/Samsung_L74wide_0_43718.JPG'

    args = {'img_pth_A': img_pth_A,
            'img_pth_B': img_pth_B,
            'patch_dim': 256}

    patches = patch_extractor_pairs(args, patch_num=50)
    
    for _, row in patches.iterrows():
        img_A_patch = row['img_A_patch']
        img_B_patch = row['img_B_patch']
        plt.figure()
        plt.imshow(img_A_patch, cmap='gray')
        plt.figure()
        plt.imshow(img_B_patch, cmap='gray')
        
        
        

