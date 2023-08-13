import time
from copy import deepcopy
import torch
import networkx as nx
from functorch import jacrev
from torch.nn import Parameter
from optimizer import loss_function_AE
from tool import hash
import cupy as cp
import scipy.sparse as sp
from model import  OP_Net
import torch.nn.functional as F
class Participant:
    def __init__(self, edge_path, feat_path, args, hidden_sizes,all_nodes_num):
        self.g = nx.read_edgelist(edge_path, nodetype=int)  
        self.adj = nx.adjacency_matrix(self.g)
        self.edges = torch.DoubleTensor(self.adj.toarray()).sum().cuda()
        self.feat,self.internal_nodes,self.external_nodes,self.all_nodes = self.load_feat_data(self.g, feat_path)
        self.overlapping_node_dict = {}  
        self.overlapping_nodes = set()  
        self.nodeid_line_dict = {}  
        self.hidden_sizes = hidden_sizes
        self.hidden_sizes[0] = self.feat.shape[1]
        self.model1 = OP_Net(self.hidden_sizes[0], self.hidden_sizes[1], args, lambda x: x).cuda()
        self.model2 = OP_Net(self.hidden_sizes[1], self.hidden_sizes[2], args, lambda x: x).cuda()
        self.model3 = OP_Net(self.hidden_sizes[2], self.hidden_sizes[3], args, lambda x: x).cuda()
        self.hash_dict=self.generate_hash_dict()
        self.cp_global_adj_norm =cp.zeros((all_nodes_num,all_nodes_num))
        self.global_all_nodes_num=all_nodes_num
        self.all_nodes_num=all_nodes_num
    def load_feat_data(self, g, feat_path):
        torch.set_printoptions(profile='full')  
        feat = []
        for node in list(g.nodes):
            g.nodes[node]['feat']=None
        feat_len=0
        with open(feat_path, 'r') as f:
            for node in f:
                node = node.strip('\n').split(' ')
                node = [eval(x) for x in node]
                try:
                    g.nodes[node[0]]['feat'] = node[1:]
                    if feat_len==0:
                        feat_len=len(g.nodes[node[0]]['feat'])
                except:
                    pass
        internal_nodes=[]
        hash_internal_nodes=[]
        external_nodes=[]
        hash_external_nodes=[]
        all_nodes=[]
        hash_all_nodes=[]
        for i in g.nodes:
            if g.nodes[i]['feat']==None: 
                g.nodes[i]['feat'] =[0 for item in range(feat_len)]
                external_nodes.append(i)
                hash_external_nodes.append(hash(i))
            else:
                internal_nodes.append(i)
                hash_internal_nodes.append(hash(i))
            feat.append(g.nodes[i]['feat'])
            all_nodes.append(i)
            hash_all_nodes.append(hash(i))
        self.cp_feat = cp.asarray(torch.DoubleTensor(feat).numpy())
        feat_tensor = torch.DoubleTensor(feat).cuda()
        self.hash_internal_nodes=hash_internal_nodes
        self.hash_external_nodes=hash_external_nodes
        self.hash_all_nodes=hash_all_nodes
        return feat_tensor,internal_nodes,external_nodes,all_nodes
    def load_secret_sharing_randommatrix(self, random_matrix):
        self.random_matrix = random_matrix
    def append_overlapping_nodes(self, sub_overlapping_nodes, client):
        self.overlapping_nodes = self.overlapping_nodes.union(sub_overlapping_nodes)
        for node_id in sub_overlapping_nodes:
            self.overlapping_node_dict.setdefault(node_id, []).append(client) 
        self.non_overlap_nodes = set(self.internal_nodes).difference(
            set(self.overlapping_nodes).intersection(self.internal_nodes))
    def generate_hash_dict(self):
        hash_dict = dict()
        for node in self.g.nodes():
            hash_dict[hash(node)] = node
        self.hash_dict = hash_dict
        return hash_dict
    def make_adj_norm(self):
        self.degrees = {}  
        for (node, val) in self.g.degree():
            self.degrees.setdefault(node, val)
            if node in self.external_nodes:
                self.degrees[node]=1
            else:
                self.degrees[node]=self.degrees[node]+1
        self.sqrt_degrees_matrix=torch.zeros((self.adj.shape[0],self.adj.shape[1]))
        for node in self.g.nodes:
            self.sqrt_degrees_matrix[self.nodeid_line_dict[node]][self.nodeid_line_dict[node]]=pow(self.degrees[node],-0.5)
        adj_eye=deepcopy(self.adj_label)
        self.adj_norm=torch.mm(torch.mm(self.sqrt_degrees_matrix,adj_eye),self.sqrt_degrees_matrix)
        self.cp_adj_norm=cp.asarray(self.adj_norm.cpu().numpy())
    def prepare_data(self):
        self.adj_label = self.adj + sp.eye(self.adj.shape[0])
        self.adj_label = torch.DoubleTensor(self.adj_label.toarray()).cuda()
        self.make_adj_norm() 
    def get_parameter(self, model_dict):
        self.model1.gc.weight = Parameter(model_dict['gc1.weight'].cuda())
        self.cupy_gc1_weight = cp.asarray(model_dict['gc1.weight'].numpy())
        self.model2.gc.weight = Parameter(model_dict['gc2.weight'].cuda())
        self.cupy_gc2_weight = cp.asarray(model_dict['gc2.weight'].numpy())
        self.model3.gc.weight = Parameter(model_dict['gc3.weight'].cuda())
        self.cupy_gc3_weight = cp.asarray(model_dict['gc3.weight'].numpy())
    def update_cp_weight(self):
        self.cupy_gc1_weight = cp.asarray(self.model1.gc.weight.cpu().numpy())
        self.cupy_gc2_weight = cp.asarray(self.model2.gc.weight.cpu().numpy())
        self.cupy_gc3_weight = cp.asarray(self.model3.gc.weight.cpu().numpy())
    def get_nodeid_line_dict(self):
        self.nodeid_line_dict = {}
        for node in self.g.nodes:
            self.nodeid_line_dict[node] = list(self.g.nodes).index(node) 
    def get_h_overlap_external_dict(self, hidden_vector):
        h_overlap_dict = {}
        for node in self.overlapping_nodes:
            if node in self.external_nodes:
                h_overlap_dict[hash(node)] = hidden_vector[self.nodeid_line_dict[node]]
        return h_overlap_dict
    def initial_l_h(self):
        self.cp_l_h3 = cp.zeros((self.all_nodes_num, self.hidden_sizes[3]), dtype='float64')
        self.cp_l_h2 = cp.zeros((self.all_nodes_num, self.hidden_sizes[2]), dtype='float64')
        self.cp_l_h2_dict={}
        self.cp_l_h1 = cp.zeros((self.all_nodes_num, self.hidden_sizes[1]), dtype='float64')
        self.cp_l_h1_dict={}
    def save_support_hidden_vector_output(self, support, hidden_vector, layer, output):
        if layer == 0:
            self.support_tensor1 = support
            self.hidden_vector1 = hidden_vector
            self.output1 =output
        if layer == 1:
            self.support_tensor2 = support
            self.hidden_vector2 = hidden_vector
            self.output2 = output
        if layer == 2:
            self.support_tensor3 = support
            self.hidden_vector3 = hidden_vector
            self.output3 = output
    def change_h(self, final_h_dict, layer):
        for hash_node in final_h_dict.keys():
            raw_node=self.hash_dict[hash_node]
            index = self.nodeid_line_dict[raw_node]
            if layer == 0:
                self.old_hidden_vector1=self.hidden_vector1
                tmp_h=final_h_dict[hash_node].data*pow(self.degrees[raw_node],-0.5)
                self.hidden_vector1.data[index] = self.hidden_vector1.data[index]+tmp_h  
            if layer == 1:
                self.old_hidden_vector2=self.hidden_vector2
                tmp_h = final_h_dict[hash_node].data * pow(self.degrees[raw_node], -0.5)
                self.hidden_vector2.data[index] = self.hidden_vector2.data[
                                                      index] + tmp_h  
            if layer == 2:
                tmp_h = final_h_dict[hash_node].data  * pow(self.degrees[raw_node], -0.5)
                self.ah.data[index] = self.ah.data[index] + tmp_h  
        if layer == 0:
            self.hidden_vector1 = F.tanh(self.output1)
            for external_node in self.external_nodes:
                index=self.nodeid_line_dict[external_node]
                self.hidden_vector1.data[index]=torch.zeros_like(self.hidden_vector1.data[index])
            self.cp_hidden_vector1=cp.asarray(self.hidden_vector1.cpu().numpy())
            self.cp_output1=cp.asarray(self.output1.cpu().numpy())
        if layer == 1:
            for external_node in self.external_nodes:
                index=self.nodeid_line_dict[external_node]
                self.hidden_vector2.data[index]=torch.zeros_like(self.hidden_vector2.data[index])
            self.cp_hidden_vector2 = cp.asarray(self.hidden_vector2.cpu().numpy())
        if layer == 2:
            for external_node in self.external_nodes:
                index=self.nodeid_line_dict[external_node]
                self.ah.data[index]=torch.zeros_like(self.ah.data[index])
        del final_h_dict
    def generate_internalnode_ah_dict(self):
        internalnodeID_ah_dict={}
        for internalnodeID in self.internal_nodes:
            internalnodeID_ah_dict[hash(internalnodeID)]=deepcopy(self.ah[self.nodeid_line_dict[internalnodeID],:])
        return internalnodeID_ah_dict
    def get_hash_overlap_node_list(self):
        hash_overlapping_nodes=[]
        for node in self.overlapping_nodes:
            hash_overlapping_nodes.append(hash(node))
        self.hash_overlapping_nodes=hash_overlapping_nodes
        return self.hash_overlapping_nodes
    def get_internal_node_edges(self):
        internal_node_edges=0
        for internalnode in self.internal_nodes:
            internal_node_edges=internal_node_edges+self.degrees[internalnode]-1
        return internal_node_edges
    def prepare_L_A(self,pos_weight,norm):
        self.cp_pos_weight=pos_weight
        self.cp_norm=norm
    def cal_l_h3(self,node0,node1,index0,index1,a_h3_index0,a_h3_index1,pre_a):
        global_Aij_h3=cp.zeros_like(self.cp_l_h3)
        global_Aij_h3[index0,:]=a_h3_index0
        global_Aij_h3[index1,:]=a_h3_index1
        if node0 in self.hash_dict.keys() and node1 in self.hash_dict.keys():
            node0=self.hash_dict[node0]
            node1=self.hash_dict[node1]
            local_index0=self.nodeid_line_dict[node0]
            local_index1=self.nodeid_line_dict[node1]
            if self.adj_label[local_index0][local_index1]==1.0:
                reala = cp.ones((1))
            else:
                reala = cp.zeros((1))
        else:
            reala = cp.zeros((1))
        L_Aij = (-(self.cp_pos_weight * reala * ((cp.exp(-pre_a)) / (1 + cp.exp(-pre_a)))) + (
                    (1 - reala) / ((1 + cp.exp(-pre_a)))))
        L_h3 = cp.multiply(L_Aij, global_Aij_h3)
        if index0==index1:
            self.cp_l_h3=cp.add(self.cp_l_h3,L_h3)
        else:
            self.cp_l_h3 = cp.add(self.cp_l_h3,2*L_h3)
    def cal_l_A(self, node0, node1, pre_a):
        node0 = self.hash_dict[node0]
        node1 = self.hash_dict[node1]
        local_index0 = self.nodeid_line_dict[node0]
        local_index1 = self.nodeid_line_dict[node1]
        if self.adj_label[local_index0][local_index1] == 1.0:
            reala = cp.ones((1))
        else:
            reala = cp.zeros((1))
        L_Aij = (-(self.cp_pos_weight * reala * ((cp.exp(-pre_a)) / (1 + cp.exp(-pre_a)))) + (
                (1 - reala) / ((1 + cp.exp(-pre_a)))))
        return  L_Aij
    def cal_h_w(self, layer):
        if layer==2:
            self.cp_h3_w3=cp.matmul(self.cp_adj_norm,self.cp_hidden_vector2)
        if layer==1:
            self.cp_h2_w2 = cp.matmul(self.cp_adj_norm, self.cp_hidden_vector1)
        if layer==0:
            self.cp_h1_w1 = cp.matmul(self.cp_adj_norm, self.cp_feat)
    def get_overlap_h_w(self, layer):
        if layer==2:
            overlap_node_h3_w3_dict={}
            for overlap_node in self.overlapping_nodes:
                index=self.all_nodes.index(overlap_node)
                overlap_node_h3_w3_dict[hash(overlap_node)]=self.cp_secert_share_enc(self.cp_h3_w3[index],self.random_matrix,self.overlapping_node_dict[overlap_node])
            return overlap_node_h3_w3_dict
        if layer==1:
            overlap_node_h2_w2_dict = {}
            for overlap_node in self.overlapping_nodes:
                index = self.all_nodes.index(overlap_node)
                overlap_node_h2_w2_dict[hash(overlap_node)] = self.cp_secert_share_enc(self.cp_h2_w2[index],self.random_matrix,self.overlapping_node_dict[overlap_node])
            return overlap_node_h2_w2_dict
        if layer==0:
            overlap_node_h1_w1_dict = {}
            for overlap_node in self.overlapping_nodes:
                index = self.all_nodes.index(overlap_node)
                overlap_node_h1_w1_dict[hash(overlap_node)] = self.cp_secert_share_enc(self.cp_h1_w1[index],self.random_matrix,self.overlapping_node_dict[overlap_node])
            return overlap_node_h1_w1_dict
    def update_h_w(self,h_w_dict,layer):
        if layer==2:
            self.cp_internal_h3_w3=cp.zeros((len(self.internal_nodes),self.cp_hidden_vector2.shape[1]))
            for hash_node in h_w_dict.keys():
                all_nodes_index=self.all_nodes.index(self.hash_dict[hash_node])
                internal_node_index=self.internal_nodes.index(self.hash_dict[hash_node])
                self.cp_internal_h3_w3[internal_node_index]=(h_w_dict[hash_node]-self.cp_h3_w3[all_nodes_index])*pow(self.degrees[self.hash_dict[hash_node]], -0.5)+self.cp_h3_w3[all_nodes_index]
            for node in self.internal_nodes:
                if node not in self.overlapping_nodes:
                    all_nodes_index = self.all_nodes.index(node)
                    internal_node_index = self.internal_nodes.index(node)
                    self.cp_internal_h3_w3[internal_node_index] = self.cp_h3_w3[all_nodes_index]
        if layer==1:
            self.cp_internal_h2_w2 = cp.zeros((len(self.internal_nodes), self.cp_hidden_vector1.shape[1]))
            for hash_node in h_w_dict.keys():
                all_nodes_index = self.all_nodes.index(self.hash_dict[hash_node])
                internal_node_index = self.internal_nodes.index(self.hash_dict[hash_node])
                self.cp_internal_h2_w2[internal_node_index] = (h_w_dict[hash_node]-self.cp_h2_w2[all_nodes_index]) * pow(self.degrees[self.hash_dict[hash_node]], -0.5) + \
                                                              self.cp_h2_w2[all_nodes_index]
            for node in self.internal_nodes:
                if node not in self.overlapping_nodes:
                    all_nodes_index = self.all_nodes.index(node)
                    internal_node_index = self.internal_nodes.index(node)
                    self.cp_internal_h2_w2[internal_node_index] = self.cp_h2_w2[all_nodes_index]
        if layer==0:
            self.cp_internal_h1_w1 = cp.zeros((len(self.internal_nodes), self.feat.shape[1]))
            for hash_node in h_w_dict.keys():
                all_nodes_index = self.all_nodes.index(self.hash_dict[hash_node])
                internal_node_index = self.internal_nodes.index(self.hash_dict[hash_node])
                self.cp_internal_h1_w1[internal_node_index] = (h_w_dict[hash_node]-self.cp_h1_w1[all_nodes_index]) * pow(self.degrees[self.hash_dict[hash_node]], -0.5) + self.cp_h1_w1[all_nodes_index]
            for node in self.internal_nodes:
                if node not in self.overlapping_nodes:
                    all_nodes_index = self.all_nodes.index(node)
                    internal_node_index = self.internal_nodes.index(node)
                    self.cp_internal_h1_w1[internal_node_index] = self.cp_h1_w1[all_nodes_index]
    def cal_l_w(self,layer):
        if layer==1:
            self.cp_l_w2=cp.matmul(self.cp_internal_h2_w2.T,self.cp_internal_l_h2)
            self.cp_l_w2==self.cp_secert_share_enc(self.cp_l_w2, self.random_matrix)
        if layer==0:
            h1_o1 = cp.ones((self.cp_internal_output1.shape[0], self.cp_internal_output1.shape[1])) - cp.power(cp.tanh(self.cp_internal_output1),2)
            self.cp_internal_l_o1 = cp.multiply(self.cp_internal_l_h1, h1_o1)
            self.cp_l_w1=cp.matmul(self.cp_internal_h1_w1.T,self.cp_internal_l_o1)
            self.cp_l_w1==self.cp_secert_share_enc(self.cp_l_w1, self.random_matrix)
    def update_l_w(self,weight_grad,layer):
        weight_grad=torch.from_numpy(cp.asnumpy(weight_grad)).cuda()
        if layer == 1:
            self.model2.gc.weight.grad = torch.zeros_like(self.model2.gc.weight)
            self.model2.gc.weight.grad.data = weight_grad.clone().data
        if layer == 0:
            self.model1.gc.weight.grad = torch.zeros_like(self.model1.gc.weight)
            self.model1.gc.weight.grad.data = weight_grad.clone().data
    def make_global_adj_norm(self,hash_nodeid_global_index_dict):
        for hash_node0 in hash_nodeid_global_index_dict.keys():
            local_index0=self.nodeid_line_dict[self.hash_dict[hash_node0]]
            for hash_node1 in hash_nodeid_global_index_dict.keys():
                local_index1 = self.nodeid_line_dict[self.hash_dict[hash_node1]]
                if local_index1<=local_index0:
                    global_index0=hash_nodeid_global_index_dict[hash_node0]
                    global_index1=hash_nodeid_global_index_dict[hash_node1]
                    self.cp_global_adj_norm[global_index0][global_index1]=self.cp_adj_norm[local_index0][local_index1]
                    self.cp_global_adj_norm[global_index1][global_index0]=self.cp_adj_norm[local_index1][local_index0]
        self.nodeid_global_index_dict={}
        for hash_nodeid in hash_nodeid_global_index_dict.keys():
            self.nodeid_global_index_dict[self.hash_dict[hash_nodeid]]=hash_nodeid_global_index_dict[hash_nodeid]
    def make_global_l_h(self,layer):
        if layer == 1:
            self.cp_l_h2 = cp.zeros((self.global_all_nodes_num,self.cp_hidden_vector2.shape[1]))
            for node in self.nodeid_global_index_dict.keys():
                global_index = self.nodeid_global_index_dict[node]
                if node in self.internal_nodes:
                    local_index = self.internal_nodes.index(node)  
                    self.cp_l_h2[global_index] = self.cp_internal_l_h2[local_index]
    def add_non_overlap_l_h(self,layer):
        if layer == 1:
            for node in self.non_overlap_nodes:
                i = self.nodeid_global_index_dict[node]
                for j in range(self.cp_l_h3.shape[1]):
                    A = self.cp_global_adj_norm[i, :].reshape(1, self.cp_global_adj_norm.shape[1])
                    W = (self.cupy_gc3_weight[:, j]).reshape(self.cupy_gc3_weight.shape[0], 1)
                    jacob_ans = cp.matmul(A.T, W.T)
                    tmp = self.cp_l_h3[i][j] * jacob_ans
                    self.cp_l_h2 = cp.add(self.cp_l_h2, tmp)
        if layer == 0:
            for node in self.non_overlap_nodes:
                i = self.nodeid_global_index_dict[node]
                for j in range(self.cp_l_h2.shape[1]):
                    A = self.cp_global_adj_norm[i, :].reshape(1, self.cp_global_adj_norm.shape[1])
                    W = (self.cupy_gc2_weight[:, j]).reshape(self.cupy_gc2_weight.shape[0], 1)
                    jacob_ans = cp.matmul(A.T, W.T)
                    tmp = self.cp_l_h2[i][j] * jacob_ans
                    self.cp_l_h1 = cp.add(self.cp_l_h1, tmp)
    def cal_hij_h(self, layer, internalnode, j):
        i = self.nodeid_global_index_dict[self.hash_dict[internalnode]]
        if layer == 1:
            A = (self.cp_global_adj_norm[i, :]).reshape(1, self.cp_global_adj_norm.shape[1])
            W = (self.cupy_gc3_weight[:, j]).reshape(self.cupy_gc3_weight.shape[0], 1)
            hij_h = cp.matmul(A.T, W.T)
            self.hij_h=self.cp_secert_share_enc(hij_h,self.random_matrix,self.overlapping_node_dict[self.hash_dict[internalnode]])
            return hij_h
        if layer == 0:
            A = self.cp_global_adj_norm[i, :].reshape(1, self.cp_global_adj_norm.shape[1])
            W = (self.cupy_gc2_weight[:, j]).reshape(self.cupy_gc2_weight.shape[0], 1)
            hij_h = cp.matmul(A.T, W.T)
            self.hij_h=self.cp_secert_share_enc(hij_h,self.random_matrix,self.overlapping_node_dict[self.hash_dict[internalnode]])
            return hij_h
    def getnode1_degree(self):
        return pow(self.degrees[1], -0.5)
    def cal_global_l_h(self,hij_h_grad,layer,hash_internal_node,j):
        i=self.nodeid_global_index_dict[self.hash_dict[hash_internal_node]]
        if layer==1:
            internal_index=self.nodeid_line_dict[self.hash_dict[hash_internal_node]]
            A = self.cp_global_adj_norm[i, :].reshape(1, self.cp_global_adj_norm.shape[1])
            W = (self.cupy_gc3_weight[:, j]).reshape(self.cupy_gc3_weight.shape[0], 1)
            local_hij_h_grad = cp.matmul(A.T, W.T)
            for node_index in range(self.cp_adj_norm.shape[1]):
                global_node_index = self.nodeid_global_index_dict[self.all_nodes[node_index]]
                if self.all_nodes[node_index] not in self.internal_nodes and self.cp_adj_norm[internal_index][node_index]!=0:
                    hij_h_grad[global_node_index]=hij_h_grad[global_node_index]*pow(self.degrees[self.hash_dict[hash_internal_node]], -0.5)
                if self.all_nodes[node_index] in self.internal_nodes: 
                    hij_h_grad[global_node_index] = local_hij_h_grad[global_node_index]
            tmp = self.cp_l_h3[i][j] * hij_h_grad
            self.cp_l_h2 = cp.add(self.cp_l_h2, tmp)
        if layer==0:
            internal_index=self.nodeid_line_dict[self.hash_dict[hash_internal_node]]
            A = self.cp_global_adj_norm[i, :].reshape(1, self.cp_global_adj_norm.shape[1])
            W = (self.cupy_gc2_weight[:, j]).reshape(self.cupy_gc2_weight.shape[0], 1)
            local_hij_h_grad = cp.matmul(A.T, W.T)
            for node_index in range(self.cp_adj_norm.shape[1]):
                global_node_index = self.nodeid_global_index_dict[self.all_nodes[node_index]]
                if self.all_nodes[node_index] not in self.internal_nodes and self.cp_adj_norm[internal_index][node_index]!=0:
                    hij_h_grad[global_node_index]=hij_h_grad[global_node_index]*pow(self.degrees[self.hash_dict[hash_internal_node]], -0.5)
                if self.all_nodes[node_index] in self.internal_nodes: 
                    hij_h_grad[global_node_index] = local_hij_h_grad[global_node_index]
            tmp = self.cp_l_h2[i][j] * hij_h_grad
            self.cp_l_h1 = cp.add(self.cp_l_h1, tmp)
    def get_overlap_l_h(self,layer):
        if layer==1:
            overlap_node_l_h2_dict = {}
            for overlap_node in self.overlapping_nodes:
                index = self.nodeid_global_index_dict[overlap_node]
                overlap_node_l_h2_dict[hash(overlap_node)] = self.cp_secert_share_enc(self.cp_l_h2[index],self.random_matrix,self.overlapping_node_dict[overlap_node])
            return overlap_node_l_h2_dict
        if layer==0:
            overlap_node_l_h1_dict = {}
            for overlap_node in self.overlapping_nodes:
                index = self.nodeid_global_index_dict[overlap_node]
                overlap_node_l_h1_dict[hash(overlap_node)] = self.cp_secert_share_enc(self.cp_l_h1[index],self.random_matrix,self.overlapping_node_dict[overlap_node])
            return overlap_node_l_h1_dict
    def update_l_h(self, cor_l_h_dict, layer):
        if layer==2:
            self.cp_internal_l_h3=cp.zeros((len(self.internal_nodes),self.cp_l_h3.shape[1]))
            for hash_node in cor_l_h_dict.keys():
                index=self.internal_nodes.index(self.hash_dict[hash_node])
                self.cp_internal_l_h3[index]=cor_l_h_dict[hash_node]
        if layer==1:
            self.cp_internal_l_h2=cp.zeros((len(self.internal_nodes),self.cp_hidden_vector2.shape[1]))
            for internal_node in self.internal_nodes:
                global_index = self.nodeid_global_index_dict[internal_node]
                index = self.internal_nodes.index(internal_node)
                if internal_node in self.overlapping_nodes:
                    self.cp_internal_l_h2[index]=cor_l_h_dict[hash(internal_node)]
                else:
                    self.cp_internal_l_h2[index] = self.cp_l_h2[global_index]
        if layer==0:
            self.cp_internal_l_h1=cp.zeros((len(self.internal_nodes),self.cp_hidden_vector1.shape[1]))
            self.cp_internal_output1=cp.zeros((len(self.internal_nodes),self.cp_hidden_vector1.shape[1]))
            for internal_node in self.internal_nodes:
                global_index = self.nodeid_global_index_dict[internal_node]
                index = self.internal_nodes.index(internal_node)
                local_index=self.nodeid_line_dict[internal_node]
                self.cp_internal_output1[index] = self.cp_output1[local_index]
                if internal_node in self.overlapping_nodes:
                    self.cp_internal_l_h1[index]=cor_l_h_dict[hash(internal_node)]
                else:
                    self.cp_internal_l_h1[index] = self.cp_l_h1[global_index]
    def cal_ah(self,layer):
        if layer==2:
            self.ah=torch.matmul(self.adj_norm,self.hidden_vector2)
        return self.ah
    def cp_secert_share_enc(self,raw,ramdom_number,only_op=None):
        if only_op!= None:
            random_list=[]
            for op_partcipant in only_op:
                random_list.append(ramdom_number[op_partcipant])
            ramdom_number=random_list
        for number in ramdom_number:
            raw=cp.add(raw,number)
        return raw
    def cal_l_hcut(self,l_ah_dict):
        cp_l_ah = cp.zeros_like(self.cp_hidden_vector2)
        self.nodelist=list(self.nodeid_line_dict.keys())
        for hash_nodeID in l_ah_dict.keys():
            raw_id=self.hash_dict[hash_nodeID]
            index=self.nodelist.index(raw_id)
            cp_l_ah[index] = l_ah_dict[hash_nodeID]
        self.cp_l_hcut=cp.matmul(self.cp_adj_norm.T,cp_l_ah)
    def get_overlap_l_hcut(self):
        overlap_l_hcut_dict={}
        for overlap_node in self.overlapping_nodes:
            index=self.all_nodes.index(overlap_node)
            overlap_l_hcut_dict[hash(overlap_node)]=self.cp_secert_share_enc(self.cp_l_hcut[index],self.random_matrix,self.overlapping_node_dict[overlap_node])
        return overlap_l_hcut_dict
    def update_l_hcut(self,l_hcut_dict):
        for hash_node in l_hcut_dict.keys():
            raw_node=self.hash_dict[hash_node]
            node_index=self.nodelist.index(raw_node)
            self.cp_l_hcut[node_index] = (l_hcut_dict[hash_node] - self.cp_l_hcut[node_index]) * pow(self.degrees[raw_node], -0.5) \
                                         + self.cp_l_hcut[node_index]
        self.cp_internal_l_h2 = cp.zeros((len(self.internal_nodes), self.cp_hidden_vector2.shape[1]))
        for internal_node in self.internal_nodes:
            index = self.internal_nodes.index(internal_node)
            local_all_node_index=self.all_nodes.index(internal_node)
            self.cp_internal_l_h2[index] = self.cp_l_hcut[local_all_node_index]
