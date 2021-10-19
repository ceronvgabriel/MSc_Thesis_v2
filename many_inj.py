
import random
import logging
import torch
from pytorchfi import core
import pdb


"""
helper functions
"""


def random_batch_element(pfi_model):
    return random.randint(0, pfi_model.get_total_batches() - 1)


def random_neuron_location(pfi_model, conv=-1):
    if conv == -1:
        conv = random.randint(0, pfi_model.get_total_conv() - 1)

    c = random.randint(0, pfi_model.get_fmaps_num(conv) - 1)
    h = random.randint(0, pfi_model.get_fmaps_H(conv) - 1)
    w = random.randint(0, pfi_model.get_fmaps_W(conv) - 1)

    return (conv, c, h, w)


def random_weight_location(pfi_model, conv=-1):
    '''Modified for current model i.e default pytorch resnet (shortcut.0) github resnet downsample.0'''
    loc = list()

    if conv == -1:
        corrupt_layer = random.randint(0, pfi_model.get_total_conv() - 1)
    else:
        corrupt_layer = conv
    loc.append(corrupt_layer)

    curr_layer = 0
    for name, param in pfi_model.get_original_model().named_parameters():
        if "conv" in name or "shortcut.0" in name or "downsample.0" in name:
            if curr_layer == corrupt_layer:
                for dim in param.size():
                    loc.append(random.randint(0, dim - 1))
            curr_layer += 1

    assert curr_layer == pfi_model.get_total_conv()
    assert len(loc) == 5

    return tuple(loc)

#this shall work for any resnet architecture, as long as it is compossed of a series of convs and a fc at the end
def random_weight_location_fc(pfi_model, conv=-1):
    '''Modified for current model i.e default pytorch resnet (shortcut.0) github resnet downsample.0'''
    loc = list()

    if conv == -1:
        corrupt_layer = random.randint(0, pfi_model.get_total_conv()) #MOD: it was -1, changed to add 1 layer, FC
    else:
        corrupt_layer = conv
    loc.append(corrupt_layer)

    curr_layer = 0
    for name, param in pfi_model.get_original_model().named_parameters():
        if "conv" in name or "shortcut.0" in name or "downsample.0" in name or "fc.weight" in name:
            if curr_layer == corrupt_layer:
                if "fc.weight" in name:
                    loc.extend([-1,-1,-1,-1]) # set conv values to -1
                    for dim in param.size():
                        loc.append(random.randint(0, dim - 1))

                else:
                    for dim in param.size():
                        loc.append(random.randint(0, dim - 1))
                    loc.extend([-1,-1])# extend FC values
            curr_layer += 1

    # assert curr_layer == pfi_model.get_total_conv() # this is not true anymore
    
    assert len(loc) == 7 # returned list will now have len 7

    return tuple(loc)

def random_value(min_val=-1, max_val=1):
    return random.uniform(min_val, max_val)


#Bit flipper:

def _twos_comp_shifted( val, nbits):
    if val < 0:
        val = (1 << nbits) + val
    else:
        val = _twos_comp(val, nbits)
    return val

def _twos_comp( val, bits):
    # compute the 2's complement of int value val
    if (val & (1 << (bits - 1))) != 0:  # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)  # compute negative value
    return val  # return positive value as is

def _flip_bit_signed( orig_value, max_value, bit_pos):
    # quantum value
    save_type = orig_value.dtype
    total_bits = bits
    logging.info("orig value:", orig_value)

    quantum = int((orig_value / max_value) * ((2.0 ** (total_bits - 1))))
    twos_comple = _twos_comp_shifted(quantum, total_bits)  # signed
    logging.info("quantum:", quantum)
    logging.info("twos_comple:", twos_comple)

    # binary representation
    bits = bin(twos_comple)[2:]
    logging.info("bits:", bits)

    # sign extend 0's
    temp = "0" * (total_bits - len(bits))
    bits = temp + bits
    assert len(bits) == total_bits
    logging.info("sign extend bits", bits)

    # flip a bit
    # use MSB -> LSB indexing
    assert bit_pos < total_bits

    bits_new = list(bits)
    bit_loc = total_bits - bit_pos - 1
    if bits_new[bit_loc] == "0":
        bits_new[bit_loc] = "1"
    else:
        bits_new[bit_loc] = "0"
    bits_str_new = "".join(bits_new)
    logging.info("bits", bits_str_new)

    # GPU contention causes a weird bug...
    if not bits_str_new.isdigit():
        logging.info("Error: Not all the bits are digits (0/1)")

    # convert to quantum
    assert bits_str_new.isdigit()
    new_quantum = int(bits_str_new, 2)
    out = _twos_comp(new_quantum, total_bits)
    logging.info("out", out)

    # get FP equivalent from quantum
    new_value = out * ((2.0 ** (-1 * (total_bits - 1))) * max_value)
    logging.info("new_value", new_value)

    return torch.tensor(new_value, dtype=save_type)


#Many Injections

def many_n_inj(
    pfi_model,n_inj, min_val=-1, max_val=1

):
    batch, conv_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))

    #print("N injections: ",n_inj)

    for ni in range(n_inj):

      for i in range(pfi_model.get_total_batches()):

        (conv, C, H, W) = random_neuron_location(pfi_model)

        err_val = random_value(min_val=min_val, max_val=max_val)

        batch.append(i)
        conv_num.append(conv)
        c_rand.append(C)
        h_rand.append(H)
        w_rand.append(W)
        value.append(err_val)

    return pfi_model.declare_neuron_fi(
        batch=batch, conv_num=conv_num, c=c_rand, h=h_rand, w=w_rand, value=value
    )

def many_n_inj_layer(
    pfi_model,n_inj,layer, min_val=-1, max_val=1
):
    batch, conv_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))

    #print("N injections: ",n_inj)

    for ni in range(n_inj):

      for i in range(pfi_model.get_total_batches()):

          (conv, C, H, W) = random_neuron_location(pfi_model,layer)
          err_val = random_value(min_val=min_val, max_val=max_val)

          batch.append(i)
          conv_num.append(layer)
          c_rand.append(C)
          h_rand.append(H)
          w_rand.append(W)
          value.append(err_val)

    return pfi_model.declare_neuron_fi(
        batch=batch, conv_num=conv_num, c=c_rand, h=h_rand, w=w_rand, value=value
    )

def hello():
  return 0

def many_n_injections_0(
    pfi_model,n_inj, min_val=-1, max_val=1, randLoc=True, randVal=True
):
    batch, conv_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))

    #print("N injections: ",n_inj)

    for ni in range(n_inj):

      if not randLoc:
          (conv, C, H, W) = random_neuron_location(pfi_model)
      if not randVal:
          err_val = random_value(min_val=min_val, max_val=max_val)

      for i in range(pfi_model.get_total_batches()):
          if randLoc:
              (conv, C, H, W) = random_neuron_location(pfi_model)
          if randVal:
              err_val = random_value(min_val=min_val, max_val=max_val)

          batch.append(i)
          conv_num.append(conv)
          c_rand.append(C)
          h_rand.append(H)
          w_rand.append(W)
          value.append(err_val)

    return pfi_model.declare_neuron_fi(
        batch=batch, conv_num=conv_num, c=c_rand, h=h_rand, w=w_rand, value=value
    )

def many_w_injections(
    pfi_model,n_inj, min_val=-1, max_val=1, corrupt_conv=-1
):
    conv_v, k_v, c_v, h_v, w_v, value_v = ([] for i in range(6))

    #print("N injections: ",n_inj)

    for ni in range(n_inj):
      (conv, k, c, kH, kW) = random_weight_location(pfi_model, corrupt_conv)
      faulty_val = random_value(min_val=min_val, max_val=max_val)

      conv_v.append(conv)
      k_v.append(k)
      c_v.append(c)
      h_v.append(kH)
      w_v.append(kW)
      value_v.append(faulty_val)
    #pdb.set_trace()
    return pfi_model.declare_weight_fi(
        conv_num=conv_v, k=k_v, c=c_v, h=h_v, w=w_v, value=value_v
    )

def many_w_bit_injections(
    pfi_model,n_inj, min_val=-1, max_val=1, corrupt_conv=-1,bits=-1
):
    conv_v, k_v, c_v, h_v, w_v = ([] for i in range(5))

    #print("N injections: ",n_inj)

    for ni in range(n_inj):
      (conv, k, c, kH, kW) = random_weight_location(pfi_model, corrupt_conv)

      conv_v.append(conv)
      k_v.append(k)
      c_v.append(c)
      h_v.append(kH)
      w_v.append(kW)
    

    return pfi_model.declare_weight_bit_fi(
        conv_num=conv_v, k=k_v, c=c_v, h=h_v, w=w_v,bits=bits #if we dont pass bit position "bit" (0 to 63) a random bit is selected
    )

def many_w_injections_fc(
    pfi_model,n_inj, min_val=-1, max_val=1, corrupt_conv=-1
):
    conv_v, k_v, c_v, h_v, w_v, value_v, fcx_v,fcy_v = ([] for i in range(8))

    #print("N injections: ",n_inj)

    for ni in range(n_inj):
      (conv, k, c, kH, kW,fcx,fcy) = random_weight_location_fc(pfi_model, corrupt_conv)
      faulty_val = random_value(min_val=min_val, max_val=max_val)

      conv_v.append(conv)
      k_v.append(k)
      c_v.append(c)
      h_v.append(kH)
      w_v.append(kW)
      value_v.append(faulty_val)
      fcx_v.append(fcx)
      fcy_v.append(fcy)
    #pdb.set_trace()

    # print(conv_v)
    # print(fcx_v)
    # print(fcy_v)

    return pfi_model.declare_weight_fi_fc(
        conv_num=conv_v, k=k_v, c=c_v, h=h_v, w=w_v, value=value_v,fcx_v=fcx_v,fcy_v=fcy_v
    )

# tests:

import model_actions
import numpy as np
import numpy
def progressive_inj(pfi_model,i,n,step=10):
    '''i:n steps; n: n experiments'''
    avg_v=[]
    for k in range(i):
        #print(k)
        j=k*step
        acc_v=[]
        for i in range(0,n):
            model_m_inj = many_w_injections(pfi_model,j)
            acc=model_actions.test(model_m_inj)
            acc_v.append(acc)
        av=np.average(acc_v)
        #print(av)
        avg_v.append(av)
    return avg_v

def progressive_inj_zero_nofc(pfi_model,total_inj,step,n_exp):
    '''i:n steps; n: n experiments'''
    avg_v=[]
    std_v=[]
    for k in range(total_inj):
        if k%step==0:
            acc_v=[]
            loss_v=[]
            for i in range(0,n_exp):
                model_m_inj = many_w_injections(pfi_model,k,min_val=0,max_val=0)
                acc,loss=model_actions.test(model_m_inj)
                acc_v.append(acc)
                loss_v.append(loss)
            av=np.average(acc_v)
            st=np.std(acc_v)
            avg_v.append(av)
            std_v.append(st)

    return avg_v, std_v


def progressive_inj_zero(pfi_model,total_inj,step,n_exp=5):
    '''total_inj every step, repeat n_exp each one and get avg and std'''
    avg_v=[]
    std_v=[]
    loss_avg_v=[]
    loss_std_v=[]
    for k in range(total_inj+1):
        if k%step==0:
            print("Injection: " +str(k)+ " of " + str(total_inj))
            acc_v=[]
            loss_v=[]
            for i in range(0,n_exp):
                model_m_inj = many_w_injections_fc(pfi_model,k,min_val=0,max_val=0)
                acc,loss=model_actions.test(model_m_inj)
                
                acc_v.append(acc)
                loss_v.append(loss)
            
            av=np.average(acc_v)
            st=np.std(acc_v)
            avg_v.append(av)
            std_v.append(st)

            loss_avg=np.average(loss_v)
            loss_st=np.std(loss_v)
            loss_avg_v.append(loss_avg)
            loss_std_v.append(loss_st)

    return avg_v, std_v, loss_avg_v, loss_std_v

def progressive_inj_zero_2(pfi_model,total_inj,step,n_exp=5):
    '''total_inj every step, repeat n_exp each one and get avg and std'''
    save_data={}
    save_data["avg"]=[]
    save_data["std"]=[]
    save_data["min"]=[]
    save_data["max"]=[]
    save_data["all_values"]=[]
    save_data["loss_all_values"]=[]

    for k in range(total_inj+1):
        if k%step==0:
            print("Injection: " +str(k)+ " of " + str(total_inj))
            acc_v=[]
            loss_v=[]
            for i in range(0,n_exp):
                model_m_inj = many_w_injections_fc(pfi_model,k,min_val=0,max_val=0)
                acc,loss=model_actions.test(model_m_inj)
                
                acc_v.append(acc)
                loss_v.append(loss)
            
            acc_av=np.average(acc_v)
            acc_st=np.std(acc_v)
            acc_min=min(acc_v)
            acc_max=max(acc_v)

            save_data["avg"].append(acc_av)
            save_data["std"].append(acc_st)
            save_data["min"].append(acc_min)
            save_data["max"].append(acc_max)
            save_data["all_values"].append(acc_v)
            save_data["loss_all_values"].append(loss_v)

    return save_data



