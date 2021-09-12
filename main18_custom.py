# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from IPython import get_ipython

# %% [markdown]
# # Import

# %%
#Run if you want to autoreload your personal modules on change
import autoreload
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

#Next is needed in azure vm to autoreload modules in cwd
import os
pwd=os.popen("pwd").read().rstrip()

import sys
sys.path.append(pwd)


# %%
get_ipython().system('ls')

# %%


# %%
import time
import log
def log_time(func,ad_info=""):
    def wrapper(*args, **kwargs):
        print()
        start = time.process_time()
        try:
            best_acc=func(*args, **kwargs)
            total=time.process_time() - start
            log.log("Exec time: " + str(total) + "best_acc: "  + str(best_acc) + " " + ad_info)
        except Exception as e:
            total=time.process_time() - start
            log.log("Exec time: " + str(total) + " ERROR: " + str(e) + " " + ad_info)
    return wrapper


# %%
#Show current devices (GPU)
#print(torch.cuda.current_device())
#from tensorflow.python.client import device_lib
#device_lib.list_local_devices()


# %%
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import torchvision
import torchvision.transforms as transforms

import os
import argparse

#from models import *
from utils import progress_bar
import time
from torchvision import models


# %%


get_ipython().run_line_magic('run', 'main_custom')


# %%
import resnet_18_custom


# %%
#Parameters:

# num_classes: int = 1000, # Output of Fully Connected layer
#norm_layer: Optional[Callable[..., nn.Module]] = None
#inplanes; in git resnet we also have default 64
#layer_stride
#layer_kernel_size

model=resnet_18_custom.resnet18(num_classes=10,layer_stride=4)
default=models.resnet18()


# %%
model


# %%
default


# %%
# run instead of import because of parameters
get_ipython().run_line_magic('run', 'main_custom')


# %%
epochs=200
model_desc="test_stride"
ad_info=" saved: " + model_desc + " epochs: " + str(epochs)
log_time(train_model,ad_info=ad_info)(model,epochs,model_desc)


# %%
#Or use: from functools import partial; to bind function and arguments
log_time(train_model)(model,"test_time")

# %% [markdown]
# # Batch Size mod

# %%
get_ipython().run_line_magic('run', 'main_custom')


# %%
class MyGroupNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyGroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=2, num_channels=num_channels,
                                 eps=1e-5, affine=True)
    
    def forward(self, x):
        x = self.norm(x)
        return x


# %%
model=resnet_18_custom.resnet18(num_classes=10)
default_gn=models.resnet18(num_classes=10,norm_layer=MyGroupNorm)
default=models.resnet18(num_classes=10)


# %%
default_gn


# %%
try:
    epochs=200
    model_desc="default_b_size_256"
    ad_info=" saved: " + model_desc + " epochs: " + str(epochs)
    log_time(train_model,ad_info=ad_info)(default,epochs,model_desc,t_batch_size=256)

    epochs=200
    model_desc="default_b_size_512"
    ad_info=" saved: " + model_desc + " epochs: " + str(epochs)
    log_time(train_model,ad_info=ad_info)(default,epochs,model_desc,t_batch_size=512)
except:
    pass
from az_manage_proc import delete
log.log("end")
delete()

# %% [markdown]
# # Train Test

# %%
default=models.resnet18(num_classes=10)
default.train()


# %%
epochs=1
model_desc="default_test"
ad_info=" saved: " + model_desc + " epochs: " + str(epochs)
log_time(train_model,ad_info=ad_info)(default,epochs,model_desc,t_batch_size=32)

# %% [markdown]
# # Delete Proc

# %%
from az_manage_proc import delete
log.log("end")
delete()


