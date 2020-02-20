#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Saves features based on volume sensitivity

For usage information, call with --help.

Author: Vinod Subramanian
"""

from __future__ import print_function

import sys
import os
import io
from argparse import ArgumentParser
import json
import pickle 

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
floatX = np.float32
from scipy.special import kl_div

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from progress import progress
from simplecache import cached
import audio
from labels import create_aligned_targets
import model
import augment
import config

def change_volume(audio_data,spl=3):
    rms = (np.sqrt(np.mean(audio_data**2)))
    factor = np.power(10,spl/20.)
    return factor*audio_data
def opts_parser():
    descr = ("Computes predictions with a neural network trained for singing "
             "voice detection.")
    parser = ArgumentParser(description=descr)
    parser.add_argument('featurefolder', metavar='FEATUREFOLDER',
            type=str,
            help='Folder to load the volume sensitive features from (.pkl format)')
    parser.add_argument('featurefolder_ref', metavar='FEATUREREF',
            type=str,
            help='Folder to load the control group features from (.pkl format)')
    return parser

def main():
    # - parse arguments
    parser = opts_parser()
    options = parser.parse_args()
    featurefolder = options.featurefolder
    featurefolder_ref = options.featurefolder_ref
    
    # - different volume changes
    spl_values = np.array([-9,-6,-3,3,6,9])
    
    # - setup tensorboard writer
    writer = SummaryWriter('images')

    # - matrix for layer differences
    layer_distances = np.zeros([spl_values.size,20])
    layer_distances_ref = np.zeros([spl_values.size,20])
    conv_filt_data = np.zeros([spl_values.size,10,3])
    conv_filt_data_ref =  np.zeros([spl_values.size,10,3])
    fc_data = np.zeros([spl_values.size,10,3])
    fc_data_ref = np.zeros([spl_values.size,10,3])

    with open(os.path.join(featurefolder,'data.json')) as f:
        data = json.load(f)
        for p in data:
            fl =p[0]+'-'+str(p[1])+'.pkl'
            with open(os.path.join(featurefolder,fl),'rb') as f:
                d = pickle.load(f)
            
            spl_iter = 0
            for spl in spl_values:
                iter_num = 0
                for i,j in zip(d['orig'],d[str(spl)]):
                    if(iter_num==3):
                        D = np.squeeze(np.linalg.norm(i-j,axis=(2,3)))
                        N1 =  np.squeeze(np.linalg.norm(i,axis=(2,3)))
                        N2 = np.squeeze(np.linalg.norm(j,axis=(2,3)))
                        data = np.transpose((i-j)**2,(1,2,3,0))
                        data = np.reshape(data,(data.shape[0],-1))
                        data = np.sqrt(np.sum(data,axis=-1))
                        srt = np.argsort(data)
                            #print(srt[-5:][::-1],data[srt[-5:]][::-1])
                            #writer.add_images('Original_audio',np.transpose(i,(1,2,3,0)),global_step=image_no,dataformats='NWHC')            
                            #writer.add_images('Volume changed'+str(spl),np.transpose(j,(1,2,3,0)),global_step=image_no,dataformats='NWHC')
                            #writer.add_images('Difference',data,global_step=image_no,dataformats='NHWC')

                        #print(np.sqrt(np.sum((i-j)**2))/i.size,i.size)
                    layer_distances[spl_iter,iter_num] += np.sqrt(np.sum((i-j)**2))/i.size
                    
                    if (i.ndim>3):
                        temp =np.sqrt(np.sum((i-j)**2,axis=(2,3)))
                        conv_filt_data[spl_iter,iter_num,0] += np.mean(temp)/(i.shape[2]+i.shape[3])
                        conv_filt_data[spl_iter,iter_num,1] += np.max(temp)/(i.shape[2]+i.shape[3])
                        conv_filt_data[spl_iter,iter_num,2] += np.std(temp)/(i.shape[2]+i.shape[3])
                    elif(iter_num==17):
                        temp = np.squeeze(np.abs(i-j))
                        srt = np.argsort(np.squeeze(temp))
                        print(srt.shape)
                        print(srt[-5:][::-1],temp[srt[-5:]][::-1],spl)
                        input("Print next layer")
                    else:
                        temp = np.squeeze(np.abs(i-j))
                        fc_data[spl_iter,iter_num-10,0] += np.mean(temp)
                        fc_data[spl_iter,iter_num-10,1] += np.max(temp)
                        fc_data[spl_iter,iter_num-10,2] += np.std(temp)

                    iter_num+=1
                spl_iter+=1
    with open(os.path.join(featurefolder_ref,'data.json')) as f:
        data = json.load(f)
        for p in data:
            fl =p[0]+'-'+str(p[1])+'.pkl'
            with open(os.path.join(featurefolder_ref,fl),'rb') as f:
                d = pickle.load(f)
            spl_iter = 0
            for spl in spl_values:
                iter_num = 0
                for i,j in zip(d['orig'],d[str(spl)]):
                    if(iter_num==3):
                        D = np.squeeze(np.linalg.norm(i-j,axis=(2,3)))
                        N1 =  np.squeeze(np.linalg.norm(i,axis=(2,3)))
                        N2 = np.squeeze(np.linalg.norm(j,axis=(2,3)))
                        data = np.transpose((i-j)**2,(1,2,3,0))
                        data = np.reshape(data,(data.shape[0],-1))
                        data = np.sqrt(np.sum(data,axis=-1))
                        srt = np.argsort(data)
                        #print(srt[-5:][::-1],data[srt[-5:]][::-1])
                        #writer.add_images('Original_audio',np.transpose(i,(1,2,3,0)),global_step=image_no,dataformats='NWHC')            
                        #writer.add_images('Volume changed'+str(spl),np.transpose(j,(1,2,3,0)),global_step=image_no,dataformats='NWHC')
                        #writer.add_images('Difference',data,global_step=image_no,dataformats='NHWC')

                    #print(np.sqrt(np.sum((i-j)**2))/i.size,i.size)
                    layer_distances_ref[spl_iter,iter_num] += np.sqrt(np.sum((i-j)**2))/i.size
                    if (i.ndim>3):
                        temp =np.sqrt(np.sum((i-j)**2,axis=(2,3)))
                        conv_filt_data_ref[spl_iter,iter_num,0] += np.mean(temp)/(i.shape[2]+i.shape[3])
                        conv_filt_data_ref[spl_iter,iter_num,1] += np.max(temp)/(i.shape[2]+i.shape[3])
                        conv_filt_data_ref[spl_iter,iter_num,2] += np.std(temp)/(i.shape[2]+i.shape[3])
                    else:
                        temp = np.squeeze(np.abs(i-j))
                        fc_data_ref[spl_iter,iter_num-10,0] += np.mean(temp)
                        fc_data_ref[spl_iter,iter_num-10,1] += np.max(temp)
                        fc_data_ref[spl_iter,iter_num-10,2] += np.std(temp) 
                    iter_num+=1
                spl_iter+=1
    
    print(conv_filt_data)
    print(conv_filt_data_ref)
    print(fc_data_ref/480)
    print(fc_data/480)
    return
    layer_distances *= 1/480
    layer_distances_ref *= 1/480
    

    x = np.arange(5)  # the label locations
    width = 0.35
    spl_iter=0
    
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


    for spl in spl_values:
        fig, ax = plt.subplots()
        rect1 = ax.bar(x - width/2, np.around(layer_distances[spl_iter,:5],5),width,label='Volume_sensitive')
        rect2 = ax.bar(x + width/2, np.around(layer_distances_ref[spl_iter,:5],5),width,label='reference')
        
        ax.set_ylabel('Avg distance across 480 excerpts')
        ax.set_xticks(x)
        ax.legend()
        
        autolabel(rect1)
        autolabel(rect2)

        writer.add_figure(str(spl)+'dB'+' feature comparison',fig)
        #fig = plt.figure()
        #plt.bar(np.arange(20),layer_distances[spl_iter,:])
        #writer.add_figure(str(spl),fig)
        plt.close()

        fig = plt.figure()
        plt.bar(np.arange(20),layer_distances_ref[spl_iter,:])
        print('control'+str(spl))
        writer.add_figure('control'+str(spl),fig)
        plt.close()  
        spl_iter+=1
    return
if __name__=='__main__':
    main()
