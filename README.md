# Unveiling Image Source: Instance-level Camera Device Linking via Context-aware Deep Siamese Network
This is an implementation of context-aware deep Siamese network (CA-SiamNet) for the task of source device linking, which is to determine whether a pair of images was taken by the same camera device.


## License
Copyright (c) 2023 The Hong Kong Polytechnic University.

All rights reserved.

This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this package) 


## Installation
The code requires Python 3.x and PyTorch 1.8.0.

We recommend using a virtual environment to create the experimental environment: 

```
conda create --name venv python=3.6
conda activate venv
```

### Installation with GPU
Install PyTorch using:
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.1 -c pytorch
```

Install the requested libraries using:
```
pip install -r requirements.txt
```

### Installation without GPU
Install PyTorch using:
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly -c pytorch
```

Install the requested libraries using:
```
pip install -r requirements.txt
```


## Usage
To execute the source device linking, run:
```
python main_test.py --img_pth_A <input image A> --img_pth_B <input image B>
```