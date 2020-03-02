#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network architecture definition for Singing Voice Detection experiment.

Author: Jan Schlüter
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################################################
#-------------Pytorch implementation of singing voice detection---------------#
###############################################################################
def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            # if param.dim() > 1:
            #     print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            # else:
            #     print(name, ':', num_param)
            total_param += num_param
    # print('Total Parameters: {}'.format(total_param))
    return total_param

class CNNModel(nn.Module):
    def __init__(self,is_zeromean=False):
        super(CNNModel, self).__init__()
        self.is_zeromean = is_zeromean
        self.conv = nn.Sequential(
                    nn.Conv2d(1,64,3),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(64,32,3),
                    nn.LeakyReLU(inplace=True),
                    nn.MaxPool2d(3),
                    nn.Conv2d(32,128,3),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(128,64,3),
                    nn.LeakyReLU(inplace=True),
                    nn.MaxPool2d(3),
                    nn.Flatten())
        
        self.fc = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(64*11*7,256),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(256,64),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(64,1),
                    nn.Sigmoid())
        
        self.param_count = count_parameters(self)
        print(self.param_count)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                                 
    
    def forward(self,x):
        #print(self.training)
        if(self.is_zeromean):
            #print(self.conv[0].weight.shape,self.conv[0].weight[0])
            self.conv[0].weight.data = self.conv[0].weight.data - torch.mean(self.conv[0].weight.data,(2,3),keepdim=True)
            #print(torch.sum(self.conv[0].weight.data,(2,3)))
            #input('continue')
            #print(torch.sum(self.conv[0].weight.data[0]))
        y = self.conv(x)
        return self.fc(y)

















































