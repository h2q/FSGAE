import time
from collections import Counter
from copy import deepcopy
import numpy as np
import torch
from torch.nn import functional as F, Parameter
from functorch import jacrev
import cupy as cp
from Kmeans import adapt_hash_kmeans_onmi, kmeans_onmi, unhash_cog_kmeans_onmi
from model import OP_Net
class Aggregator:
    def __init__(self,hidden_sizes,args):
        self.participant_degree = {}
        self.allnodes_num = 0
        self.alledges_num = 0
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        self.hidden_sizes=hidden_sizes
        self.agrs=args
        self.model1 = OP_Net(self.hidden_sizes[0], self.hidden_sizes[1], args, lambda x: x).cuda()
        self.model2 = OP_Net(self.hidden_sizes[1], self.hidden_sizes[2], args, lambda x: x).cuda()
        self.model3 = OP_Net(self.hidden_sizes[2], self.hidden_sizes[3], args, lambda x: x).cuda()
    def get_nodeID_clientID_dict(self, all_overlap_node_dict , all_internal_node_dict,all_all_node_dict,all_edges_num_dict):
        overlapnodeID_clientID_dict = {}
        internalnodeID_clientID_dict = {}
        for key in all_overlap_node_dict.keys():
            for node in all_overlap_node_dict[key]:
                if node in overlapnodeID_clientID_dict.keys():
                    overlapnodeID_clientID_dict[node].append(key)
                else:
                    overlapnodeID_clientID_dict[node] = [key]
        self.all_internalnodeID_list=[]
        for key in all_internal_node_dict.keys():
            self.allnodes_num=self.allnodes_num+len(all_internal_node_dict[key])
            for node in all_internal_node_dict[key]:
                internalnodeID_clientID_dict[node] = key
                self.all_internalnodeID_list.append(node)
        for key in all_edges_num_dict.keys():
            self.alledges_num=self.alledges_num+all_edges_num_dict[key]
        self.alledges_num=self.alledges_num
        self.all_internalnodeID_list=sorted(self.all_internalnodeID_list)
        self.overlapnodeID_clientID_dict = overlapnodeID_clientID_dict
        self.clientID_overlapnodeID_dict = all_overlap_node_dict
        self.internalnodeID_clientID_dict=internalnodeID_clientID_dict
        self.clientID_internalnodeID_dict =  all_internal_node_dict
        self.clientID_allnodeID_dict = all_all_node_dict
    def get_all_h_overlap_dict(self, h_overlap_dict_list):
        all_h_overlap_dict = {}
        index_client = 0
        for h_overlap_dict in h_overlap_dict_list:
            for key in h_overlap_dict.keys():
                if key in all_h_overlap_dict.keys():
                    all_h_overlap_dict[key].append(h_overlap_dict[key]) 
                else:
                    all_h_overlap_dict[key] = [h_overlap_dict[key]]
            index_client = index_client + 1
        return all_h_overlap_dict
    def generate_overlap_node_h_new(self, all_h_overlap_dict):
        for nodeid in all_h_overlap_dict.keys():
            first = 0
            final_h = torch.zeros([1,all_h_overlap_dict[nodeid][0].shape[0]])
            for h in all_h_overlap_dict[nodeid]:
                if first == 0:
                    final_h = h
                    first = 1
                else:
                    final_h = final_h + h
            all_h_overlap_dict[nodeid] = deepcopy(final_h)
        return all_h_overlap_dict
    def get_final_h3(self,all_internalnode_h3_dict,hidden_sizes):
        self.cp_final_hidden_vector=cp.zeros((self.allnodes_num, hidden_sizes[3]))
        for client in all_internalnode_h3_dict.keys():
            for internalnodeID in all_internalnode_h3_dict[client].keys():
                index=self.all_internalnodeID_list.index(internalnodeID)
                self.cp_final_hidden_vector[index]=cp.asarray(all_internalnode_h3_dict[client][internalnodeID].cpu().numpy())
    def cal_recover_mat(self):
        self.cp_recover_matrix=cp.matmul(self.cp_final_hidden_vector, self.cp_final_hidden_vector.T)
    def cal_pos_weight_norm(self):
        self.cp_pos_weight = cp.asarray([(self.allnodes_num * self.allnodes_num - self.alledges_num) / self.alledges_num])
        self.cp_norm = cp.asarray(
            [self.allnodes_num * self.allnodes_num / (self.allnodes_num * self.allnodes_num - self.alledges_num * 2)])
    def aggregate_external_h_w(self,external_h_w_list,layer):
        if layer==2:
            external_h3_w3_dict={}
            for h3_w3_dict in external_h_w_list:
                for external_node in h3_w3_dict.keys():
                    if external_node not in external_h3_w3_dict.keys():
                        external_h3_w3_dict[external_node]=h3_w3_dict[external_node]
                    else:
                        external_h3_w3_dict[external_node] =external_h3_w3_dict[external_node]+h3_w3_dict[external_node]
            return external_h3_w3_dict
        if layer==1:
            external_h2_w2_dict = {}
            for h2_w2_dict in external_h_w_list:
                for external_node in h2_w2_dict.keys():
                    if external_node not in external_h2_w2_dict.keys():
                        external_h2_w2_dict[external_node] = h2_w2_dict[external_node]
                    else:
                        external_h2_w2_dict[external_node] = external_h2_w2_dict[external_node] + h2_w2_dict[external_node]
            return external_h2_w2_dict
        if layer==0:
            external_h1_w1_dict = {}
            for h1_w1_dict in external_h_w_list:
                for external_node in h1_w1_dict.keys():
                    if external_node not in external_h1_w1_dict.keys():
                        external_h1_w1_dict[external_node] = h1_w1_dict[external_node]
                    else:
                        external_h1_w1_dict[external_node] = external_h1_w1_dict[external_node] + h1_w1_dict[external_node]
            return external_h1_w1_dict
    def make_client_allnodeid_Index(self,client):
        nodeid_global_index_dict={}
        for node in self.clientID_allnodeID_dict[client]:
            nodeid_global_index_dict[node]=self.all_internalnodeID_list.index(node)
        return nodeid_global_index_dict
    def aggregate_overlap_l_h(self,overlap_l_h_list,layer):
        if layer==1:
            overlap_l_h2_dict={}
            for node_id_l_h_dict in overlap_l_h_list:
                for node_id in node_id_l_h_dict.keys():
                    if node_id not in overlap_l_h2_dict.keys():
                        overlap_l_h2_dict[node_id] = node_id_l_h_dict[node_id]
                    else:
                        overlap_l_h2_dict[node_id] = overlap_l_h2_dict[node_id] + node_id_l_h_dict[node_id]
            return overlap_l_h2_dict
        if layer==0:
            overlap_l_h1_dict = {}
            for node_id_l_h_dict in overlap_l_h_list:
                for node_id in node_id_l_h_dict.keys():
                    if node_id not in overlap_l_h1_dict.keys():
                        overlap_l_h1_dict[node_id] = node_id_l_h_dict[node_id]
                    else:
                        overlap_l_h1_dict[node_id] = overlap_l_h1_dict[node_id] + node_id_l_h_dict[node_id]
            return overlap_l_h1_dict
    def dict_add(self,hij_h_grad, client_hij_h_dict):
        for key in client_hij_h_dict:
            if key in hij_h_grad.keys():
                hij_h_grad[key]=hij_h_grad[key]+client_hij_h_dict[key]
            else:
                hij_h_grad[key]=client_hij_h_dict[key]
        return hij_h_grad
    def perform_kmeans(self, edge_path, comms_path, file):
        path = '../data/real'
        self.final_hidden_vector=torch.from_numpy(cp.asnumpy(self.cp_final_hidden_vector)).cuda()
        nmi1, ari1, eq1 = adapt_hash_kmeans_onmi(self.all_internalnodeID_list, edge_path, self.final_hidden_vector, comms_path, path,
                                                 file)
        return nmi1, ari1, eq1
    def initial_parameter(self, model_dict):
        self.model1.gc.weight = Parameter(model_dict['gc1.weight'].cuda())
        self.model2.gc.weight = Parameter(model_dict['gc2.weight'].cuda())
        self.model3.gc.weight = Parameter(model_dict['gc3.weight'].cuda())
    def update_l_w(self,weight_grad,layer):
        weight_grad=torch.from_numpy(cp.asnumpy(weight_grad)).cuda()
        if layer==2:
            self.model3.gc.weight.grad = torch.zeros_like(self.model3.gc.weight)
            self.model3.gc.weight.grad.data = weight_grad.clone().data
        if layer == 1:
            self.model2.gc.weight.grad = torch.zeros_like(self.model2.gc.weight)
            self.model2.gc.weight.grad.data = weight_grad.clone().data
        if layer == 0:
            self.model1.gc.weight.grad = torch.zeros_like(self.model1.gc.weight)
            self.model1.gc.weight.grad.data = weight_grad.clone().data
    def get_model_dict(self):
        model_dict={}
        model_dict['gc1.weight']=self.model1.gc.weight.cpu()
        model_dict['gc2.weight']=self.model2.gc.weight.cpu()
        model_dict['gc3.weight']=self.model3.gc.weight.cpu()
        return model_dict
    def aggregate_l_hcut(self,l_hcut_list):
        all_l_hcut_dict={}
        for l_hcut_dict in l_hcut_list:
            for node in l_hcut_dict.keys():
                if node not in all_l_hcut_dict.keys():
                    all_l_hcut_dict[node]=l_hcut_dict[node]
                else:
                    all_l_hcut_dict[node] =all_l_hcut_dict[node]+l_hcut_dict[node]
        return all_l_hcut_dict
