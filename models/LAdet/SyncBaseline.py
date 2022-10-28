import torch
import torch.nn as nn
import torch.nn.functional as F

class SyncBaseline(nn.Module):

    def __init__(self, flag = False):
        super(SyncBaseline, self).__init__()
        self.flag = flag
        
    def forward(self, x, t):
        return x[:,-1].unsqueeze(1)