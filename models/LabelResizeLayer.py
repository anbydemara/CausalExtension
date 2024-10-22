
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch

import numpy as np
import torch.nn as nn


class InstanceLabelResizeLayer(nn.Module):


    def __init__(self):
        super(InstanceLabelResizeLayer, self).__init__()
        self.minibatch=256

    def forward(self, x,need_backprop):
        # lbs.size() --> ([1])
        feats = x.data.cpu().numpy()
        lbs = need_backprop.data.cpu().numpy()

        resized_lbs = np.ones((feats.shape[0], 1), dtype=np.float32)
        # lbs.shape[0] -> 1
        for i in range(lbs.shape[0]):
            resized_lbs[i*self.minibatch:(i+1)*self.minibatch] = lbs[i]

        y=torch.from_numpy(resized_lbs)

        return y
