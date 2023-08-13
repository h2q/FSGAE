import argparse
import time
from copy import deepcopy
import networkx as nx
from intersect.driver import Intersect
import torch
from torch import optim
import cupy as cp
from participant import Participant
from aggregator import Aggregator
from coordinator import Coordinator
import numpy as np
import os
import datetime
from tool import hash
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--global_lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--epsilon', type=float, default=0.8, help='epsilon of dp.')
    return parser.parse_args()
def list_txt(path, list=None):
    if list != None:
        file = open(path, 'w')
        file.write(str(list))
        file.close()
        return None
    else:
        file = open(path, 'r')
        rdlist = eval(file.read())
        file.close()
        return rdlist
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
    
    file_name_list = ['cora' ]
    hidden_sizes = [128, 128, 64, 64]
    Isreal = True
    time1=datetime.datetime.now()
    time1_str = datetime.datetime.strftime(time1, '%Y%m%d_%H_%M_%S')
    
    for file in file_name_list:
        partnum_list = [4]
        epoch_list = [1]
        args = get_args()
        for n_part in partnum_list:
            for global_epoch in epoch_list:
                seed = 7
                torch.manual_seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.set_printoptions(precision=15)   
                torch.set_default_tensor_type(torch.cuda.DoubleTensor) 
                args = get_args()
                edge_path_list = []
                feat_path_list = []
                if Isreal:
                    edge_path = '../data/real/' + file + '.txt'
                    feat_path = '../data/real/' + file + '_feat.txt'
                    comms_path = '../data/real/real_' + file + '.txt'
                    for i in range(n_part):
                        sub_edge_path = '../data/real/' + str(n_part) + '/' + file + '_' + str(
                            i) + '.txt'
                        sub_feat_path = '../data/real/' + str(n_part) + '/' + file + '_' + str(
                            i) + '_feat' + '.txt'
                        edge_path_list.append(sub_edge_path)
                        feat_path_list.append(sub_feat_path)
                elif Isreal == False:
                    edge_path = '../data/artificial/n/network' + file + '.txt'
                    feat_path = '../data/artificial/n/network' + file + '_feat.txt'
                    comms_path = '../data/artificial/n/community' + file + '.txt'
                    for i in range(n_part):
                        sub_edge_path = '../data/artificial/n/' + str(n_part) + '/network' + file + '_' + str(
                            i) + '.txt'
                        sub_feat_path = '../data/artificial/n/' + str(n_part) + '/network' + file + '_' + str(
                            i) + '_feat' + '.txt'
                        edge_path_list.append(sub_feat_path)
                        feat_path_list.append(sub_feat_path)
                cen_edge_path='../data/real/'+file+'.txt'
                cen_g=nx.read_edgelist(cen_edge_path, nodetype=int)
                all_nodes=nx.number_of_nodes(cen_g)
                ans_fp = open('./split_fl_relu/' + file + '/part_'+str(n_part)+'_epoch_'+str(global_epoch)+'_'+ file + '_' + time1_str + '_cp_ans_for_print.txt', 'a+')
                intersector = Intersect()
                part_list = [Participant(participant_edge_path, participant_feat_path, args, hidden_sizes,all_nodes) for
                             participant_edge_path, participant_feat_path in
                             zip(edge_path_list, feat_path_list)]  
                aggCor = Aggregator(hidden_sizes, args)
                mainCor= Coordinator(hidden_sizes, args)
                for i in range(n_part):
                    for j in range(n_part):
                        if j > i:
                            sub_intersect_nodes = list(intersector.run(part_list[i].g, part_list[j].g))
                            for item in sub_intersect_nodes:
                                sub_intersect_nodes[sub_intersect_nodes.index(item)] = item 
                            part_list[i].append_overlapping_nodes(sub_intersect_nodes, j)
                            part_list[j].append_overlapping_nodes(sub_intersect_nodes, i)
                        else:
                            continue
                random_matrix_degree = np.random.randn(n_part, n_part).astype(np.double)
                for i in range(n_part):
                    for j in range(n_part):
                        if (i == j):
                            random_matrix_degree[i][j] = 0
                        if (i > j):
                            random_matrix_degree[i][j] = -1 * random_matrix_degree[j][i]
                for i in range(n_part):
                    part_list[i].get_nodeid_line_dict() 
                    part_list[i].prepare_data()
                    part_list[i].load_secret_sharing_randommatrix(random_matrix_degree[i])
                
                for i in range(n_part):
                    model_dict = torch.load('../model_parameter/singlemodel_state_' + file + '.txt')
                    part_list[i].get_parameter(model_dict.copy())
                    del model_dict
                model_dict = torch.load('../model_parameter/singlemodel_state_' + file + '.txt')
                mainCor.initial_parameter(model_dict)
                all_overlap_node_dict = {}
                all_internal_node_dict={} 
                all_all_node_dict={}
                all_edges_num_dict={}
                for i in range(n_part):
                    all_overlap_node_dict[i] = part_list[i].get_hash_overlap_node_list()
                    all_internal_node_dict[i]=part_list[i].hash_internal_nodes
                    all_all_node_dict[i] =part_list[i].hash_all_nodes
                    all_edges_num_dict[i]=part_list[i].get_internal_node_edges()
                aggCor.get_nodeID_clientID_dict(all_overlap_node_dict, all_internal_node_dict, all_all_node_dict, all_edges_num_dict) 
                mainCor.get_nodeID_clientID_dict(all_overlap_node_dict, all_internal_node_dict, all_all_node_dict, all_edges_num_dict) 
                clientID_overlapnodeID_dict = {}  
                optimizer_list = []
                for client in range(n_part):
                    optimizer = optim.Adam([{'params': part_list[client].model1.parameters()},
                                            {'params': part_list[client].model2.parameters()},
                                            {'params': part_list[client].model3.parameters()}
                                            ]
                                           , lr=0.01)
                    optimizer_list.append(optimizer)
                total_start_time=time.time()
                with torch.no_grad():
                    for epoch in range(global_epoch):
                        start_time = time.time()
                        torch.set_printoptions(17)
                        
                        forward_start_time = time.time()
                        for layer in range(3):
                            tensor_T_list = []
                            h_overlap_dict_list = []
                            hash_overlapnodeid_list_server = []
                            for i in range(n_part):
                                if layer == 0:
                                    support1, hidden_vector1, output1 = part_list[i].model1(part_list[i].feat, part_list[i].adj_norm)
                                    part_list[i].save_support_hidden_vector_output(support1, hidden_vector1, layer,output1)
                                    hidden_vector = hidden_vector1
                                if layer == 1:
                                    support2, hidden_vector2, output2 = part_list[i].model2(part_list[i].hidden_vector1, part_list[i].adj_norm)
                                    part_list[i].save_support_hidden_vector_output(support2, hidden_vector2, layer,output2)
                                    hidden_vector = hidden_vector2
                                if layer == 2:
                                    hidden_vector=part_list[i].cal_ah(layer)
                                h_overlap_dict= part_list[i].get_h_overlap_external_dict(hidden_vector)
                                h_overlap_dict_list.append(h_overlap_dict)
                            all_h_overlap_dict = aggCor.get_all_h_overlap_dict(h_overlap_dict_list)
                            all_h_overlap_dict=aggCor.generate_overlap_node_h_new(all_h_overlap_dict)
                            for client in aggCor.clientID_overlapnodeID_dict.keys():
                                final_h_dict = {}
                                for need_node in aggCor.clientID_overlapnodeID_dict[client]:
                                    if need_node in aggCor.clientID_internalnodeID_dict[client]:
                                        final_h_dict[need_node] = all_h_overlap_dict[need_node]
                                part_list[client].change_h(final_h_dict, layer)
                                del final_h_dict
                            del all_h_overlap_dict
                        all_internalnode_ah_dict={}
                        for i in range(n_part):
                            all_internalnode_ah_dict[i]=part_list[i].generate_internalnode_ah_dict()
                        mainCor.get_final_ah(all_internalnode_ah_dict, hidden_sizes)
                        mainCor.cal_h()
                        forward_end_time = time.time()
                        
                        
                        
                        mainCor.cal_recover_mat()
                        mainCor.cal_pos_weight_norm()
                        for client in range(n_part):
                            part_list[client].prepare_L_A(mainCor.cp_pos_weight, mainCor.cp_norm)
                            part_list[client].initial_l_h()
                        temp = np.zeros((all_nodes, all_nodes))
                        cp_l_h3=cp.zeros((mainCor.cp_h.shape[0], mainCor.cp_h.shape[1]), dtype='float64')
                        for index, value in np.ndenumerate(temp):
                            if index[0] >= index[1]:
                                A_ij_h3_matrix = cp.zeros((mainCor.cp_h.shape[0], mainCor.cp_h.shape[1]),
                                                          dtype='float64')
                                A_ij_h3_matrix[index[0], :] = cp.add(A_ij_h3_matrix[index[0], :],
                                                                     mainCor.cp_h[index[1], :])
                                A_ij_h3_matrix[index[1], :] = cp.add(A_ij_h3_matrix[index[1], :],
                                                                     mainCor.cp_h[index[0], :])
                                flag = 0
                                cal_client = -1
                                for client in range(n_part):
                                    if (mainCor.all_internalnodeID_list[index[0]] in mainCor.clientID_internalnodeID_dict[
                                        client]) :
                                        part_list[client].cal_l_h3(mainCor.all_internalnodeID_list[index[0]],
                                                                       mainCor.all_internalnodeID_list[index[1]],
                                                                       index[0],
                                                                       index[1], A_ij_h3_matrix[index[0], :],
                                                                       A_ij_h3_matrix[index[1], :],
                                                                       mainCor.cp_recover_matrix[index[0]][index[1]])
                                        break
                                del A_ij_h3_matrix
                        for client in range(n_part):
                            cp_l_h3=cp.add(cp_l_h3,part_list[client].cp_secert_share_enc(part_list[client].cp_l_h3, part_list[client].random_matrix))
                        cp_l_h3 = (cp_l_h3 / (len(mainCor.all_internalnodeID_list) * len(mainCor.all_internalnodeID_list))) * mainCor.cp_norm
                        cp_l_ah,cp_l_w=mainCor.cal_l_ah_w(cp_l_h3)
                        for client in mainCor.clientID_allnodeID_dict.keys():
                            l_ah_dict={}
                            for need_node in mainCor.clientID_internalnodeID_dict[client]:
                                index=mainCor.all_internalnodeID_list.index(need_node)
                                l_ah_dict[need_node] = cp_l_ah[index,:]
                            part_list[client].cal_l_hcut(l_ah_dict)
                        l_hcut_list=[]
                        for i in range(n_part):
                            l_hcut_list.append(part_list[i].get_overlap_l_hcut())
                        all_l_hcut_dict=aggCor.aggregate_l_hcut(l_hcut_list)
                        for client in range(n_part):
                            l_hcut_dict = {}
                            for need_node in aggCor.clientID_internalnodeID_dict[client]:
                                if need_node in aggCor.clientID_overlapnodeID_dict[client]:
                                    l_hcut_dict[need_node] = all_l_hcut_dict[need_node]
                            part_list[client].update_l_hcut(l_hcut_dict)
                        
                        
                        for layer in range(1,-1,-1):
                            
                            l_h_start_time=time.time()
                            if layer == 1:
                                shape1 = hidden_sizes[3]
                                shape2 = hidden_sizes[2]
                                shape3 = hidden_sizes[1]
                            if layer == 0:
                                shape1 = hidden_sizes[2]
                                shape2 = hidden_sizes[1]
                                shape3 = hidden_sizes[0]
                            if layer==0:
                                l_h_list=[]
                                for client in range(n_part):
                                    nodeid_global_index_dict=aggCor.make_client_allnodeid_Index(client)
                                    part_list[client].make_global_adj_norm(nodeid_global_index_dict)
                                    part_list[client].make_global_l_h(layer+1)
                                overlapnum=len(list(aggCor.overlapnodeID_clientID_dict.keys()))
                                tuple_index=[(node,col) for node,col in zip(list(aggCor.overlapnodeID_clientID_dict.keys()) * shape1, list(range(shape1)) * overlapnum)]
                                for overlapnode in aggCor.overlapnodeID_clientID_dict.keys():
                                    node_owner = aggCor.internalnodeID_clientID_dict[overlapnode]
                                    for col in range(shape1):
                                        hij_h_grad = cp.zeros((aggCor.allnodes_num, shape2))
                                        for other_client in aggCor.overlapnodeID_clientID_dict[overlapnode]:
                                            if other_client != node_owner:
                                                hij_h_grad=cp.add(hij_h_grad,part_list[other_client].cal_hij_h(layer,overlapnode,col))
                                        part_list[node_owner].cal_global_l_h(hij_h_grad,layer,overlapnode,col)
                                l_h_overlap_time=time.time()
                                for client in range(n_part):
                                    part_list[client].add_non_overlap_l_h(layer)
                                overlap_l_h_list = []
                                for i in range(n_part):
                                    overlap_l_h_list.append(part_list[i].get_overlap_l_h(layer))
                                overlap_l_h_dict = aggCor.aggregate_overlap_l_h(overlap_l_h_list, layer)
                                for client in aggCor.clientID_internalnodeID_dict.keys():
                                    for_client_agg_l_h_dict = {}
                                    for need_node in aggCor.clientID_internalnodeID_dict[client]:
                                        if need_node in aggCor.clientID_overlapnodeID_dict[client]:
                                            for_client_agg_l_h_dict[need_node] = overlap_l_h_dict[need_node]
                                    part_list[client].update_l_h(for_client_agg_l_h_dict,layer)
                                l_h_end_time=time.time()

                            external_h_w_list = []
                            for i in range(n_part):
                                part_list[i].cal_h_w(layer)
                                external_h_w_list.append(part_list[i].get_overlap_h_w(layer))
                            external_h_w_dict = aggCor.aggregate_external_h_w(external_h_w_list, layer)
                            for client in range(n_part):
                                h_w_dict = {}
                                for need_node in aggCor.clientID_internalnodeID_dict[client]:
                                    if need_node in aggCor.clientID_overlapnodeID_dict[client]:  
                                        h_w_dict[need_node] = external_h_w_dict[need_node]
                                part_list[client].update_h_w(h_w_dict, layer)
                                part_list[client].cal_l_w(layer)
                            global_l_w = cp.zeros((shape3, shape2))
                            for client in range(n_part):
                                if layer==1:
                                    global_l_w = global_l_w + part_list[client].cp_l_w2
                                if layer==0:
                                    global_l_w = global_l_w + part_list[client].cp_l_w1
                            for client in range(n_part):
                                part_list[client].update_l_w(deepcopy(global_l_w), layer)  
                        mainCor.update_l_w(cp_l_w)
                        for client in range(n_part):
                            optimizer_list[client].step()
                            part_list[client].update_cp_weight()
                            optimizer_list[client].zero_grad()
                        backward_end_time = time.time()
                    torch.set_printoptions(17)

                    nmi1, ari1, eq1 = mainCor.perform_kmeans(edge_path, comms_path, file)
                    total_end_time = time.time()
                    
                    
