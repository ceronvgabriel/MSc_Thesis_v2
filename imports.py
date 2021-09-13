#Run if you want to autoreload your personal modules on change
import autoreload
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

#Next is needed in azure vm to autoreload modules in cwd
import os
pwd=os.popen("pwd").read().rstrip()

import sys
sys.path.append(pwd)

'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from matplotlib import pyplot as plt
import numpy
import sklearn


import torchvision
import torchvision.transforms as transforms

import os
import argparse

#from models import *
import utils
from utils import progress_bar
import time
from torchvision import models
import model_actions
import az_manage_proc
import load
import many_inj

from pytorchfi_c.core import fault_injection as pfi_core


print("GPU available: ",torch.cuda.is_available())

print("OS: ",sys.platform)


