#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network architecture definition for Singing Voice Detection experiment.

Author: Jan Schlüter

Gradient reversal taken from https://github.com/jvanvugt/pytorch-domain-adaptation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


###############################################################################
# -------------Pytorch implementation of singing voice detection---------------#
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
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 32, 3),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 128, 3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 64, 3),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(3))

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 11 * 7, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, 1),
            nn.Sigmoid())

        self.uda = nn.Sequential(
            GradientReversal(),
            nn.Dropout(p=0.2),
            nn.Linear(64 * 11 * 7, 256),
            nn.LeakyReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(256, 1)
            #nn.LeakyReLU(inplace=True),
            #nn.Dropout(),

            #nn.Linear(64, 1)
        )

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

    def forward(self, x):
        y = self.conv(x)
        y = y.view(-1, 64 * 11 * 7)
        return self.fc(y), self.uda(y)


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=0.5):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)