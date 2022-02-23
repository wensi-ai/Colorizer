import sys
sys.path.append("../..")

from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from collections import OrderedDict
from IPython import embed
import cv2
from utils.color_format import rgb2lab

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0)) ),0, 1) * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_subset_dict(in_dict,keys):
    if(len(keys)):
        subset = OrderedDict()
        for key in keys:
            subset[key] = in_dict[key]
    else:
        subset = in_dict
    return subset



def get_colorization_data(data_raw, opt, ab_thresh=5., p=.125, num_points=None):
    data = {}
    data_lab = rgb2lab(data_raw[0], opt)
    data['A'] = data_lab[:,[0,],:,:]
    data['B'] = data_lab[:,1:,:,:]

    if(ab_thresh > 0): # mask out grayscale images
        thresh = 1.*ab_thresh/opt.ab_norm
        mask = torch.sum(torch.abs(torch.max(torch.max(data['B'],dim=3)[0],dim=2)[0]-torch.min(torch.min(data['B'],dim=3)[0],dim=2)[0]),dim=1) >= thresh
        data['A'] = data['A'][mask,:,:,:]
        data['B'] = data['B'][mask,:,:,:]
        # print('Removed %i points'%torch.sum(mask==0).numpy())
        if(torch.sum(mask)==0):
            return None

    return add_color_patches_rand_gt(data, opt, p=p, num_points=num_points)

def add_color_patches_rand_gt(data,opt,p=.125,num_points=None,use_avg=True,samp='normal'):
# Add random color points sampled from ground truth based on:
#   Number of points
#   - if num_points is 0, then sample from geometric distribution, drawn from probability p
#   - if num_points > 0, then sample that number of points
#   Location of points
#   - if samp is 'normal', draw from N(0.5, 0.25) of image
#   - otherwise, draw from U[0, 1] of image
    N,C,H,W = data['B'].shape

    data['hint_B'] = torch.zeros_like(data['B'])
    data['mask_B'] = torch.zeros_like(data['A'])

    for nn in range(N):
        pp = 0
        cont_cond = True
        while(cont_cond):
            if(num_points is None): # draw from geometric
                # embed()
                cont_cond = np.random.rand() < (1-p)
            else: # add certain number of points
                cont_cond = pp < num_points
            if(not cont_cond): # skip out of loop if condition not met
                continue
            print('add hint !!!!!!!!!')
            P = np.random.choice(opt.sample_Ps) # patch size

            # sample location
            if(samp=='normal'): # geometric distribution
                h = int(np.clip(np.random.normal( (H-P+1)/2., (H-P+1)/4.), 0, H-P))
                w = int(np.clip(np.random.normal( (W-P+1)/2., (W-P+1)/4.), 0, W-P))
            else: # uniform distribution
                h = np.random.randint(H-P+1)
                w = np.random.randint(W-P+1)

            # add color point
            if(use_avg):
                # embed()
                data['hint_B'][nn,:,h:h+P,w:w+P] = torch.mean(torch.mean(data['B'][nn,:,h:h+P,w:w+P],dim=2,keepdim=True),dim=1,keepdim=True).view(1,C,1,1)
            else:
                data['hint_B'][nn,:,h:h+P,w:w+P] = data['B'][nn,:,h:h+P,w:w+P]

            data['mask_B'][nn,:,h:h+P,w:w+P] = 1

            # increment counter
            pp+=1

    data['mask_B']-=opt.mask_cent

    return data

def add_color_patch(data,mask,opt,P=1,hw=[128,128],ab=[0,0]):
    # Add a color patch at (h,w) with color (a,b)
    data[:,0,hw[0]:hw[0]+P,hw[1]:hw[1]+P] = 1.*ab[0]/opt.ab_norm
    data[:,1,hw[0]:hw[0]+P,hw[1]:hw[1]+P] = 1.*ab[1]/opt.ab_norm
    mask[:,:,hw[0]:hw[0]+P,hw[1]:hw[1]+P] = 1-opt.mask_cent

    return (data,mask)

def crop_mult(data,mult=16,HWmax=[800,1200]):
    # crop image to a multiple
    H,W = data.shape[2:]
    Hnew = int(min(H/mult*mult,HWmax[0]))
    Wnew = int(min(W/mult*mult,HWmax[1]))
    h = (H-Hnew)/2
    w = (W-Wnew)/2

    return data[:,:,h:h+Hnew,w:w+Wnew]

def encode_ab_ind(data_ab, opt):
    # Encode ab value into an index
    # INPUTS
    #   data_ab   Nx2xHxW \in [-1,1]
    # OUTPUTS
    #   data_q    Nx1xHxW \in [0,Q)

    data_ab_rs = torch.round((data_ab*opt.ab_norm + opt.ab_max)/opt.ab_quant) # normalized bin number
    data_q = data_ab_rs[:,[0],:,:]*opt.A + data_ab_rs[:,[1],:,:]
    return data_q

def decode_ind_ab(data_q, opt):
    # Decode index into ab value
    # INPUTS
    #   data_q      Nx1xHxW \in [0,Q)
    # OUTPUTS
    #   data_ab     Nx2xHxW \in [-1,1]

    data_a = data_q/opt.A
    data_b = data_q - data_a*opt.A
    data_ab = torch.cat((data_a,data_b),dim=1)

    if(data_q.is_cuda):
        type_out = torch.cuda.FloatTensor
    else:
        type_out = torch.FloatTensor
    data_ab = ((data_ab.type(type_out)*opt.ab_quant) - opt.ab_max)/opt.ab_norm

    return data_ab

def decode_max_ab(data_ab_quant, opt):
    # Decode probability distribution by using bin with highest probability
    # INPUTS
    #   data_ab_quant   NxQxHxW \in [0,1]
    # OUTPUTS
    #   data_ab         Nx2xHxW \in [-1,1]

    data_q = torch.argmax(data_ab_quant,dim=1)[:,None,:,:]
    return decode_ind_ab(data_q, opt)

def decode_mean(data_ab_quant, opt):
    # Decode probability distribution by taking mean over all bins
    # INPUTS
    #   data_ab_quant   NxQxHxW \in [0,1]
    # OUTPUTS
    #   data_ab_inf     Nx2xHxW \in [-1,1]

    (N,Q,H,W) = data_ab_quant.shape
    a_range = torch.range(-opt.ab_max, opt.ab_max, step=opt.ab_quant).to(data_ab_quant.device)[None,:,None,None]
    a_range = a_range.type(data_ab_quant.type())

    # reshape to AB space
    data_ab_quant = data_ab_quant.view((N,int(opt.A),int(opt.A),H,W))
    data_a_total = torch.sum(data_ab_quant,dim=2)
    data_b_total = torch.sum(data_ab_quant,dim=1)

    # matrix multiply
    data_a_inf = torch.sum(data_a_total * a_range,dim=1,keepdim=True)
    data_b_inf = torch.sum(data_b_total * a_range,dim=1,keepdim=True)

    data_ab_inf = torch.cat((data_a_inf,data_b_inf),dim=1)/opt.ab_norm

    return data_ab_inf
