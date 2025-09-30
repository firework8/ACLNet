import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import numpy as np
import random
from functools import partial
 

class Intra_Affinitive_Contrastive_Loss(nn.Module):

    def __init__(self, n_class, memory, n_channel=625, h_channel=256, temperature=0.1, epsilon=0.1):
        
        super(Intra_Affinitive_Contrastive_Loss, self).__init__()
        self.n_class = n_class
        self.n_channel = n_channel
        self.h_channel = h_channel
        self.temperature = temperature
        self.omega = np.exp(-epsilon)
        self.cl_fc = nn.Linear(self.n_channel, self.h_channel)
        
        self.Feature_Bank = []
        self.Label_Bank = []
        self.mem = memory
        

    def forward(self, features, labels):
        
        features = self.cl_fc(features)
        features = features / (torch.norm(features, p=2, dim=1, keepdim=True) + 1e-12)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        device = features.device
        
        for i in range(batch_size):
            self.Feature_Bank.append(features[i].detach())
            self.Label_Bank.append(labels[i])
        
        if len(self.Feature_Bank) < self.mem:
            loss = torch.Tensor([0]).to(device)
        
        if len(self.Feature_Bank) > self.mem:
            for i in range(batch_size):
                self.Feature_Bank.pop(0)
                self.Label_Bank.pop(0)
        
        if len(self.Feature_Bank) == self.mem:
            
            features_mem = torch.stack(self.Feature_Bank[:self.mem-batch_size], dim=0).to(device)
            features = torch.cat((features_mem, features), 0).to(device)
            labels = torch.stack(self.Label_Bank, dim=0).to(device)
            mask = torch.eq(labels, labels.T).float().to(device)
            
            anchor_dot_contrast = torch.div(
                torch.matmul(features, features.T),
                self.temperature)
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
            
            logits_mask = torch.ones_like(mask) - torch.eye(self.mem, dtype=torch.float32).to(device)
            positive_mask = mask * logits_mask
            negative_mask = 1. - mask
            
            alignment = logits
            uniformity = torch.exp(logits) * logits_mask 
            uniformity = self.omega * uniformity * positive_mask + \
                        (uniformity * negative_mask * logits_mask).sum(1, keepdim=True)
            uniformity = torch.log(uniformity + 1e-6)
            
            log_prob = alignment - uniformity
            log_prob = (positive_mask * log_prob).sum(1, keepdim=True) / \
                        torch.max(positive_mask.sum(1, keepdim=True), torch.ones_like(positive_mask.sum(1, keepdim=True)))
            log_prob = log_prob[-batch_size:]
            
            loss = -log_prob
            loss = loss.mean()
            
        return loss
