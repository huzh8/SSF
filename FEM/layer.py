import math
import copy
import torch
import time
from torch import nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from torch_geometric.nn import knn_graph,knn
from torch_geometric.utils import degree

def cos_dis(X):
        """
        cosine distance
        :param X: (N, d)
        :return: (N, N)
        """
        X = nn.functional.normalize(X)
        XT = X.transpose(0, 1)
        return torch.matmul(X, XT)


class Transform(nn.Module):
    
    def __init__(self, dim_in, k):
        super().__init__()
        self.attention = nn.Linear(100,1)
        self.convKK = nn.Conv1d(k, k * k, dim_in, groups=k)
        self.activation = nn.Softmax(dim=-1)

    def forward(self, region_feats):
        N, k, _ = region_feats.size()  # (N, k, d)
        conved = self.convKK(region_feats)  # (N, k*k, 1)
        multiplier = conved.view(N, k, k)  # (N, k, k)
        multiplier = self.activation(multiplier)  # softmax along last dimension
        transformed_feats = torch.matmul(multiplier, region_feats)  # (N, k, d)
        return transformed_feats

    def forward1(self, region_feats): 
        multiplier = self.attention(region_feats)
        multiplier = multiplier.squeeze(2)
        multiplier = self.activation(multiplier) 
        multiplier = multiplier.unsqueeze(2)
        multiplier = multiplier.repeat(1,1,100)
        transformed_feats = torch.mul(multiplier, region_feats)
        return transformed_feats

class VertexConv(nn.Module):
    
    def __init__(self, dim_in, k):
        super().__init__()
        self.trans = Transform(dim_in, k)                   # (N, k, d) -> (N, k, d)
        self.convK1 = nn.Conv1d(k, 1, 1)                    # (N, k, d) -> (N, 1, d)

    def forward(self, region_feats):
        transformed_feats = self.trans(region_feats)
        pooled_feats = self.convK1(transformed_feats)             # (N, 1, d)
        pooled_feats = pooled_feats.squeeze(1)
        return pooled_feats


class GraphConvolution(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.dim_in = kwargs['dim_in']
        self.dim_out = kwargs['dim_out']
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=kwargs['has_bias'])
        self.dropout = nn.Dropout(p=0.6)
        self.activation = kwargs['activation']
    """
    def _region_aggregate(self, feats, edge_dict):
        N = feats.size()[0]
        pooled_feats = torch.stack([torch.mean(feats[edge_dict[i]], dim=0) for i in range(N)])

        return pooled_feats

    
    def forward(self, feats, edge_dict, ite):
        x = feats 
        x = self.dropout(self.activation(self.fc(x))) 
        x = self._region_aggregate(x, edge_dict)  
        return x
    """

class EdgeConv(nn.Module):
    def __init__(self, dim_ft, hidden):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim_ft, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, ft):
        scores = []
        n_edges = ft.size(1)
        for i in range(n_edges):
            scores.append(self.fc(ft[:, i]))
        scores = torch.softmax(torch.stack(scores, 1), 1)
        return (scores * ft).sum(1)

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.net1=nn.Linear(100,50)
        self.net2=nn.Linear(50,1)
        self.BN=nn.BatchNorm1d(50)
        self.reset_parameters()
    
    def reset_parameters(self):
        init.normal(self.net1.weight)
        init.normal(self.net2.weight)
        init.constant_(self.net1.bias, 0)
        init.constant_(self.net2.bias, 0)
        init.constant(self.BN.weight, 1)
        init.constant(self.BN.bias, 0)

    def forward(self, x):
        x=self.net1(x)
        x=self.BN(x)
        x=F.sigmoid(x)
        x=self.net2(x)
        return x

class DHGLayer(GraphConvolution):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ks = kwargs['structured_neighbor'] # number of sampled nodes in graph adjacency
        self.n_cluster = kwargs['n_cluster']              # number of clusters
        self.n_center = kwargs['n_center']                # a node has #n_center adjacent clusters
        self.kn = kwargs['nearest_neighbor']    # number of the 'k' in k-NN
        self.kc = kwargs['cluster_neighbor']    # number of sampled nodes in a adjacent k-means cluster
        self.wu_knn=kwargs['wu_knn']
        self.wu_kmeans=kwargs['wu_kmeans']
        self.wu_struct=kwargs['wu_struct']
        self.vc_sn = VertexConv(self.dim_in, self.ks+self.kn)    # structured trans
        self.vc_s = VertexConv(self.dim_in, self.ks)    # structured trans
        self.vc_n = VertexConv(self.dim_in, self.kn)    # nearest trans
        self.vc_n_1 = VertexConv(self.dim_in, self.kn)    # nearest trans
        self.vc_c = VertexConv(self.dim_in, self.kc)   # k-means cluster trans
        self.ec = EdgeConv(self.dim_in, hidden=self.dim_in//2)
        self.kmeans = None
        self.structure = None


    def _vertex_conv(self, func, x):
        return func(x)
    
    def _nearest_select(self, feats, x):
        dis = cos_dis(feats)
        _, idx = torch.topk(dis, self.kn, dim=1)
        N = len(idx)
        d = feats.size(1)
        nearest_feature = feats[idx.view(-1)].view(N, self.kn, d)         # (N, kn, d)
        return nearest_feature
    
    def _edge_conv(self, x):
        return self.ec(x)

    def _fc(self, x):
        return self.fc(self.dropout(x))

    def forward(self, feats, edge_dict, ite):
        hyperedges = []
        if ite >= self.wu_knn:
            n_feat = self._nearest_select( feats, feats)
            xn = self._vertex_conv(self.vc_n, n_feat)
            xn  = xn.view(-1, 1, feats.size(1))               
            hyperedges.append(xn)
        x = torch.cat(hyperedges, dim=1)
        x = self._edge_conv(x)                                       
        x = self._fc(x)                                                 
        return x


