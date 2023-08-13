import argparse
import pandas as pd
import torch
from torch import optim
from Kmeans import kmeans_onmi
from singleGAE import SingleClient
from optimizer import loss_function_AE
import numpy as np
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
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--global_lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--real', type=str, default=True, help='type of dataset.')
    parser.add_argument('--file', type=str, default='cora', help='type of dataset.')
    parser.add_argument('--epsilon', type=float, default=0.8, help='epsilon of dp.')
    return parser.parse_args()
if __name__ == '__main__':
    file_name_list = ['cora']  
    hidden_sizes = [128, 128,64,64]
    Isreal = True
    for file in file_name_list:
        part_list = [2]
        epoch_list = [150]
        args = get_args()
        for n_part in part_list:
            for global_epoch in epoch_list:
                nmi_list=[]
                ari_list=[]
                #seed_list=[175]
                tes_epoch_list=[]
                for seed in range(10):
                    torch.manual_seed(seed)  
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    torch.set_printoptions(precision=17)
                    torch.set_default_tensor_type(torch.DoubleTensor)
                    args = get_args()
                    args.real = Isreal
                    args.file = file
                    edge_path_list = []
                    feat_path_list = []
                    if args.real:
                        edge_path = '../data/real/' + args.file + '.txt'
                        feat_path = '../data/real/' + args.file + '_feat.txt'
                        comms_path = '../data/real/real_' + args.file + '.txt'
                    elif args.real == False:
                        edge_path = '../data/artificial/'+file+'/'+file+'.txt'
                        feat_path = '../data/artificial/'+file+'/'+file+'_feat.txt'
                        comms_path = '../data/artificial/'+file+'/real_'+file+'.txt'
                    single = SingleClient(args, edge_path, feat_path, hidden_sizes)
                    torch.save(single.single_model.state_dict(),
                               '../model_parameter/singlemodel_state_' + args.file + '.txt')
                    model_dict_path = '../model_parameter/singlemodel_state_' + args.file + '.txt'
                    single.load_data()
                    single.prepare_data()
                    optimizer = optim.Adam(single.single_model.parameters(), lr=0.01)
                    single.adj_norm = single.adj_norm.to_dense()
                    flag=0
                    for epoch in range(global_epoch):
                        single.single_model.train()
                        input1, support1, output1, hidden_vector1, input2, support2, output2, hidden_vector2, input3, support3, output3, hidden_vector3, recover_matrix = single.single_model(
                            single.feature, single.adj_norm)
                        loss = loss_function_AE(preds=recover_matrix.cuda(), labels=single.adj_label.cuda(),
                                                norm=single.norm, pos_weight=single.pos_weight)
                        if flag==0:
                            ori_loss=loss
                            flag=1
                        else:
                            if abs(loss-ori_loss)<1e-3:
                                
                                break;
                            else:
                                ori_loss=loss
                        optimizer.zero_grad()
                        ah2 = torch.matmul(single.adj_norm, hidden_vector2.cpu())
                        loss.backward()
                        optimizer.step()
                        
                    path = './data/real'
                    nmi1, ari1, eq1 = kmeans_onmi(edge_path, hidden_vector3, comms_path, file)
                    
                    nmi_list.append(nmi1)
                    ari_list.append(ari1)
                    tes_epoch_list.append(epoch)
                
                for seed in seed_list:
                    i=seed_list.index(seed)
                    
                    
