"""

Copyright (C) 2021 Hoichan JUNG <hoichanjung@korea.ac.kr> - All Rights Reserved

"""

import numpy as np
import torch 
import kornia as K
import torch.nn as nn
import torch.nn.functional as F


class PSNRLoss(nn.Module):

    def __init__(self, eps=1e-3):
        super(PSNRLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):  
        imdff = torch.clamp(y, 0,1) - torch.clamp(x, 0,1)
        rmse = (imdff**2).mean().sqrt()
        loss = 10 * torch.log10(1/rmse + self.eps)

        return loss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):  
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class EdgeLoss(nn.Module):
    """
    https://kornia.readthedocs.io/en/latest/filters.html
    """
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.loss = CharbonnierLoss()
    
    def forward(self, x, y):  
        loss = self.loss(K.filters.laplacian(x, kernel_size=5), K.filters.laplacian(y, kernel_size=5))
        # loss = self.loss(cv2.Laplacian(x, cv2.CV_16S), cv2.Laplacian(y, cv2.CV_16S))
    
        return loss