import torch
import time
import random
import argparse
import scipy.sparse as sp
from torch import optim
import numpy as np
from utils import single_load_data,single_preprocess_graph, sparse_mx_to_torch_sparse_tensor
from model import SingleGAE
from optimizer import loss_function_AE
class SingleClient:
    def __init__(self, args, edge_path, feat_path, hidden_sizes):
        self.edge_path = edge_path
        self.feat_path = feat_path
        self.adj, self.feature,self.nodes_list = self.load_data()
        self.hidden_sizes = hidden_sizes
        self.hidden_sizes[0]=self.feature.shape[1] 
        self.single_model=SingleGAE(self.hidden_sizes,args).double()
    def load_data(self):
        adj, features,node_list = single_load_data(self.edge_path, self.feat_path)
        return adj, features,node_list
    def prepare_data(self):
        self.n_nodes, self.feat_dim = self.feature.shape
        adj_train = self.adj
        self.adj_norm = single_preprocess_graph(self.adj)
        self.adj_label = adj_train + sp.eye(adj_train.shape[0])
        self.adj_label = torch.DoubleTensor(self.adj_label.toarray())
        self.pos_weight = (self.adj.shape[0] * self.adj.shape[0] - self.adj.sum()) / self.adj.sum()
        
        self.norm = self.adj.shape[0] * self.adj.shape[0] / (self.adj.shape[0] * self.adj.shape[0] - self.adj.sum() * 2)
        
        self.att_norm = self.feature.shape[1] * self.feature.shape[1] / (
                    self.feature.shape[1] * self.feature.shape[1] - self.feature.sum() * 2)
