'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''

#IF desired function or wrapper is not here, check in log.py

import os
import sys
import time
import math
import numpy as np

import torch.nn as nn
import torch.nn.init as init

from matplotlib import pyplot as plt
import json


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

#MOD: Following two lines where changed from original
import shutil
_, term_width = shutil.get_terminal_size()

#_, term_width = os.popen('stty size', 'r').read().split()

term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar_print(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    print(' [')
    for i in range(cur_len):
        print('=')
    print('>')
    for i in range(rest_len):
        print('.')
    print(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    print(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        print(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        print('\b')
    print(' %d/%d ' % (current+1, total))

    if current < total-1:
        print('\r')
    else:
        print('\n')
    sys.stdout.flush()

def progress_bar(current, total, msg=None):
    TOTAL_BAR_LENGTH = 65.
    last_time = time.time()
    begin_time = last_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def progress_bar_none(current, total, msg=None):
    pass

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

#Gabriel Ceron gaboceron10@gmail.com
from IPython.display import clear_output
import sys
window=5
c=0
def tracefunc(frame, event, arg, indent=[0]):
    # global c
    # c=c+1
    # if c==window:
    #     clear_output(wait=True)
    #     c=0
    if event == "call":
        indent[0] += 2
        print("|" * indent[0]+ ">", frame.f_code.co_name,end="\r")
    elif event == "return":
#       print("<" + "-" * indent[0], "exit function", frame.f_code.co_name)
        indent[0] -= 2
    return tracefunc

def trace(bool):
    if bool==True:
        sys.setprofile(tracefunc)
    if bool == False:
        sys.setprofile(None)

def load(name):
    '''load from /results'''
    with open("./results/"+name, 'r') as fp:
      return json.load(fp)

def save_fig(data,name):
    '''data and figure name'''
    load=log.load(name)
    fig, ax = plt.subplots()
    ax.plot(load)
    fig.savefig("images/"+name+".png")

def save_fig_std(avg,std,name):
    '''data, std shadow and figure name'''
    y=range(len(avg))

    plt.clf()
    
    plt.plot(y,avg)
    
    plt.fill_between(y, avg-std, avg+std, alpha = 0.1)
    
    plt.savefig("images/"+name+".png")
    # fig, ax = plt.subplots()
    # ax.plot(load)
    # fig.savefig("images/"+name+".png")

# def save_plt(plt,name):
#     '''Show fig and save'''
#     plt.show()
#     plt.savefig("images/"+name+".png")

def list_norm(lista):
    minimo=min(lista)
    maximo=max(np.array(lista) - minimo)
    lista_n=[]
    for i in lista:
        lista_n.append((i-minimo)/maximo)
    return lista_n

def save_res(name,func,*args, **kwargs):


    if not os.path.exists("./results/" + os.path.dirname(name)):
            os.makedirs("./results/" + os.path.dirname(name), mode=777)

    res=func(*args, **kwargs)

    # if not os.path.isdir('./results/'+name):
    #     os.makedirs('./results/'+name, mode=0777)

    with open("./results/"+name, 'w+') as fp:
        json.dump(res,fp)
    
    return res

def save(to_save,name):
    os.chmod("./results/",777)
    with open("./results/"+name +".json", 'w+') as fp:
        json.dump(to_save,fp)