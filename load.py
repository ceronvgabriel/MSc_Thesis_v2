import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import models
import torchvision.transforms as transforms


from models_c import *
from utils import progress_bar


def load(save_folder,num_classes=10):


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = models.resnet18(num_classes=num_classes)
    #net = ResNet18()
    net = net.to(device)

    if device == 'cuda':
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    net = torch.nn.DataParallel(net)
    checkpoint = torch.load('./checkpoints/' + save_folder + '/ckpt.pth',map_location=torch.device(device))
    net.load_state_dict(checkpoint['net'])
    return net



