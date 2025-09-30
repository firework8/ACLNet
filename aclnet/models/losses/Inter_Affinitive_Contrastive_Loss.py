import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import numpy as np
import random
from functools import partial
from collections import Counter


class Inter_Affinitive_Contrastive_Loss(nn.Module):
    
    def __init__(self, n_class, n_channel=625, h_channel=256, tmp=0.125, mom=0.9,
                 pred_threshold=0.0):
        super(Inter_Affinitive_Contrastive_Loss, self).__init__()
        self.n_channel = n_channel
        self.h_channel = h_channel
        self.n_class = n_class

        self.tmp = tmp
        self.mom = mom
        self.pred_threshold = pred_threshold

        self.avg_f = torch.randn(self.h_channel, self.n_class)
        self.cl_fc = nn.Linear(self.n_channel, self.h_channel)
        self.loss = nn.CrossEntropyLoss(reduction='none')
        
        self.Neighborhood_Bank = []
        for i in range(n_class):
            count_dic = {}
            for j in range(n_class):
                count_dic[j] = 0
            self.Neighborhood_Bank.append(count_dic)

        self.class_mask_list = []
        self.affinitive_list = []
        self.state = 'Normal'
        self.current_epoch = 1
        self.start_epoch = 30
        self.save_state = False
        self.epoch_state = False
        self.start_state = True
        self.coincidence_count = 4
        
        
    def onehot(self, label):
        
        lbl = label.clone()
        size = list(lbl.size())
        lbl = lbl.view(-1)
        ones = torch.sparse.torch.eye(self.n_class).to(label.device)
        ones = ones.index_select(0, lbl.long())
        size.append(self.n_class)
        
        return ones.view(*size).float()


    def get_mask(self, lbl, logit):
        
        pred = logit.max(1)[1]
        pred_one = self.onehot(pred)
        lbl_one = self.onehot(lbl)
        logit = torch.softmax(logit, 1)
        tp = lbl_one * pred_one
        tp = tp * (logit > self.pred_threshold).float()
        
        if self.current_epoch > (self.start_epoch - 1):
            batch = len(lbl)
            for i in range(batch):
                if pred[i] != lbl[i]:
                    mask_class_1 = pred[i].item()
                    self.Neighborhood_Bank[lbl[i]][mask_class_1] += 1
                for j in range(batch):
                    if pred[j] == lbl[i] and lbl[j] != lbl[i]:
                        mask_class_2 = lbl[j].item()
                        self.Neighborhood_Bank[lbl[i]][mask_class_2] += 1
        
        return tp

    
    def local_avg(self, f, mask):
        
        b, k = mask.size()
        f = f.permute(1, 0)
        avg_f = self.avg_f.detach().to(f.device)

        mask_sum = mask.sum(0, keepdim=True)
        f_mask = torch.matmul(f, mask)
        f_mask = f_mask / (mask_sum + 1e-12)
        
        has_object = (mask_sum > 1e-8).float()
        has_object[has_object > 0.1] = self.mom
        has_object[has_object <= 0.1] = 1.0
        f_mem = avg_f * has_object + (1 - has_object) * f_mask
        with torch.no_grad():
            self.avg_f = f_mem
        f_mem = f_mem.permute(1, 0)
        
        return f_mem
    
    
    def get_neighborhood(self, lbl):
        
        if self.epoch_state == True:
            
            num_class = self.n_class
            device = lbl.device
            class_mask = torch.zeros((num_class, num_class)).to(device)

            for i in range(num_class):
                counter = self.Neighborhood_Bank[i]
                sorted_num = sorted(counter.items(), key=lambda x: x[1], reverse=True)
                sorted_list = []
                for sorted_i in range(len(sorted_num)):
                    sorted_list.append(sorted_num[sorted_i][0])
                if len(sorted_list) > 10:
                    sorted_list = sorted_list[:10]
                for j in range(len(sorted_list)):
                    class_mask[i][sorted_list[j]] = 1
                sorted_list = sorted(sorted_list)
                self.class_mask_list.append(sorted_list)

            class_mask_source = torch.zeros((num_class, num_class)).to(device)
            for i in range(num_class):
                for j in range(num_class):
                    if class_mask[i][j] == 1:
                        class_mask_source[i][j] = 0.5
            
            for i in range(num_class):
                source = class_mask[i]
                sum_i = len(self.class_mask_list[i])
                source_dic = {}
                for j in range(num_class):
                    if j != i:
                        target = class_mask[j]
                        coincidence = source * target
                        coincidence_num = coincidence.sum()
                        if coincidence_num > self.coincidence_count:
                            source_dic[j] = coincidence_num
                sorted_source_num = sorted(source_dic.items(), key=lambda x: x[1], reverse=True)
                for k in range(len(sorted_source_num)):
                    sum_k = sorted_source_num[k][1]
                    class_mask_source[i][sorted_source_num[k][0]] += sum_k / sum_i
                change_dic = {}
                for n in range(num_class):
                    if class_mask_source[i][n] > 0:
                        change_dic[n] = class_mask_source[i][n]
                indice_list, value_list = zip(*sorted(change_dic.items(), key=lambda x: x[1], reverse=True))
                sorted_list = sorted(indice_list)
                self.affinitive_list.append(sorted_list)
                
            self.epoch_state = False
            self.save_state = True
           
        batch = len(lbl)
        batch_affinitive_list = []
        for i in range(batch):
            batch_affinitive_list.append(self.affinitive_list[lbl[i]])
 
        return batch_affinitive_list


    def save_list(self, **kwargs):
        
        if self.save_state == True:
            self.epoch_state = True
            self.save_state = False
            self.class_mask_list.clear()
            self.affinitive_list.clear()

        if self.start_state == True:
            self.current_epoch += 1
            self.start_state = False
        
        if self.current_epoch > self.start_epoch :
            if self.state == 'Normal':
                self.state = 'Affinitive'
                self.epoch_state = True
        
        return True
 
 
    def get_affinitive_loss(self, feature, f_mem, affinitive_list, lbl):
        
        feature = feature / (torch.norm(feature, p=2, dim=1, keepdim=True) + 1e-12)
        f_mem = f_mem / (torch.norm(f_mem, p=2, dim=-1, keepdim=True) + 1e-12)
        
        batch, _ = feature.size()
        loss_all = []
        for i in range(batch):
            
            list_len = len(affinitive_list[i])
            if list_len <= 10: 
                adaptive_tmp = 0.1
            elif 10 < list_len <= 20:
                adaptive_tmp = 0.5
            else:
                adaptive_tmp = 1.0
            
            if list_len > 0:    
                local_feature = feature[i]
                local_f_mem_1 = []
                local_f_mem_1.append(lbl[i].item())
                local_f_mem_2 = local_f_mem_1 + affinitive_list[i]
                local_f_mem = f_mem[local_f_mem_2]
                
                score = torch.matmul(local_f_mem, local_feature.reshape(-1, 1))
                score_cl = score / adaptive_tmp

                score_cl = score_cl.permute(1, 0).contiguous()
                local_target = torch.tensor([0]).to(feature.device)
                local_loss = self.loss(score_cl, local_target)
                loss_all.append(local_loss)   
            else:
                local_feature = feature[i]
                score = torch.matmul(f_mem, local_feature.reshape(-1, 1))
                score_cl = score / self.tmp
                score_cl = score_cl.permute(1, 0).contiguous()
                local_target = torch.tensor([lbl[i].item()]).to(feature.device)
                local_loss = self.loss(score_cl, local_target)
                loss_all.append(local_loss)
            
        loss = torch.cat(loss_all, dim=0).mean()
        
        return loss
        
    
    def forward(self, feature, lbl, logit):
        
        if self.start_state == False:
            self.start_state = True
        
        feature = self.cl_fc(feature)
        
        mask = self.get_mask(lbl, logit)
        f_mem = self.local_avg(feature, mask)
        
        if self.state == 'Normal':
            loss = torch.Tensor([0]).to(feature.device)
            
        if self.state == 'Affinitive':
            affinitive_list = self.get_neighborhood(lbl)
            loss = self.get_affinitive_loss(feature, f_mem, affinitive_list, lbl)
        
        return loss
    