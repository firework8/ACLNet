import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer

from .init_func import bn_init, conv_branch_init, conv_init

EPS = 1e-4

import sys


class unit_gcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 ratio=0.25,
                 ctr='T',
                 ada='T',
                 subset_wise=False,
                 ada_act='softmax',
                 ctr_act='tanh',
                 norm='BN',
                 act='ReLU'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        num_subsets = A.size(0)
        self.num_subsets = num_subsets
        self.ctr = ctr
        self.ada = ada
        self.ada_act = ada_act
        self.ctr_act = ctr_act
        assert ada_act in ['tanh', 'relu', 'sigmoid', 'softmax']
        assert ctr_act in ['tanh', 'relu', 'sigmoid', 'softmax']

        self.subset_wise = subset_wise

        assert self.ctr in [None, 'NA', 'T']
        assert self.ada in [None, 'NA', 'T']

        if ratio is None:
            ratio = 1 / self.num_subsets
        self.ratio = ratio
        mid_channels = int(ratio * out_channels)
        self.mid_channels = mid_channels

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.act = build_activation_layer(self.act_cfg)

        self.A = nn.Parameter(A.clone())

        # Introduce non-linear
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels * num_subsets, 1),
            build_norm_layer(self.norm_cfg, mid_channels * num_subsets)[1], self.act)
        self.post = nn.Conv2d(mid_channels * num_subsets, out_channels, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-2)

        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))

        if self.ada or self.ctr:
            self.conv1 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)
            self.conv2 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                build_norm_layer(self.norm_cfg, out_channels)[1])
        else:
            self.down = lambda x: x
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape

        res = self.down(x)
        # K V V
        A = self.A

        # 1 (N), K, 1 (C), 1 (T), V, V
        # 1 K 1 1 V V
        A = A[None, :, None, None] 
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v)
        # The shape of pre_x is N, K, C, T, V

        x1, x2 = None, None
        if self.ctr is not None or self.ada is not None:
            # The shape of tmp_x is N, C, T or 1, V
            # N C T V
            tmp_x = x

            # N K C T V
            x1 = self.conv1(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)
            x2 = self.conv2(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)
            
            if not (self.ctr == 'NA' or self.ada == 'NA'):
                # N K C 1 V
                x1 = x1.mean(dim=-2, keepdim=True)
                x2 = x2.mean(dim=-2, keepdim=True)
        
        graph_list = []
        if self.ctr is not None:
            # * The shape of ada_graph is N, K, C[1], T or 1, V, V
            # N K C 1 V V = N K C 1 V 1 - N K C 1 1 V
            diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)
            # N K C 1 V V
            ada_graph = getattr(self, self.ctr_act)(diff)
            
            if self.subset_wise:
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.alpha)
            else:
                ada_graph = ada_graph * self.alpha[0]
            # N K C 1 V V = N K C 1 V V + 1 K 1 1 V V
            A = ada_graph + A
            graph_list.append(ada_graph)

        if self.ada is not None:
            # * The shape of ada_graph is N, K, 1, T[1], V, V
            # N K C 1 V * N K C 1 V = N K 1 1 V V
            ada_graph = torch.einsum('nkctv,nkctw->nktvw', x1, x2)[:, :, None]
            # N K 1 1 V V
            ada_graph = getattr(self, self.ada_act)(ada_graph)
            
            if self.subset_wise:
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.beta)
            else:
                ada_graph = ada_graph * self.beta[0]
            # N K C 1 V V = N K 1 1 V V + N K C 1 V V
            A = ada_graph + A
            graph_list.append(ada_graph)

        if self.ctr is not None or self.ada is not None:
            assert len(A.shape) == 6
            # * C, T can be 1
            if A.shape[2] == 1 and A.shape[3] == 1:
                A = A.squeeze(2).squeeze(2)
                x = torch.einsum('nkctv,nkvw->nkctw', pre_x, A).contiguous()
            elif A.shape[2] == 1:
                A = A.squeeze(2)
                x = torch.einsum('nkctv,nktvw->nkctw', pre_x, A).contiguous()
            elif A.shape[3] == 1:
                A = A.squeeze(3)
                # N K C T V = N K C T V * N K C V V
                x = torch.einsum('nkctv,nkcvw->nkctw', pre_x, A).contiguous()
            else:
                x = torch.einsum('nkctv,nkctvw->nkctw', pre_x, A).contiguous()
        else:
            # * The graph shape is K, V, V
            A = A.squeeze()
            assert len(A.shape) in [2, 3] and A.shape[-2] == A.shape[-1]
            if len(A.shape) == 2:
                A = A[None]
            x = torch.einsum('nkctv,kvw->nkctw', pre_x, A).contiguous()
        
        # N K C T V -> N K*C T V
        x = x.reshape(n, -1, t, v)
        x = self.post(x)
        
        
        get_gcl_graph = graph_list[0] + graph_list[1]
        # N K C 1 V V -> N K C V V
        get_gcl_graph = get_gcl_graph.squeeze(3)
        # N K C V V -> N K*C V V
        get_gcl_graph = get_gcl_graph.reshape(n, -1, v, v)
        # N C V V
        gcl_graph = []
        gcl_graph.append(get_gcl_graph)
        
        
        return self.act(self.bn(x) + res), gcl_graph