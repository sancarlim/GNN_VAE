import numpy as np
import dgl
import random
import pickle
import math
from scipy import spatial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import utils
os.environ['DGLBACKEND'] = 'pytorch'
from torchvision import datasets, transforms
import scipy.sparse as spp
from dgl.data import DGLDataset
from sklearn.preprocessing import StandardScaler


def collate_batch(samples):
    graphs, masks, feats, gt = map(list, zip(*samples))  # samples is a list of pairs (graph, mask) mask es VxTx1
    masks = torch.vstack(masks)
    feats = torch.vstack(feats)
    gt = torch.vstack(gt).float()
    sizes_n = [graph.number_of_nodes() for graph in graphs] # graph sizes
    snorm_n = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_n]
    snorm_n = torch.cat(snorm_n).sqrt()  # graph size normalization 
    sizes_e = [graph.number_of_edges() for graph in graphs] # nb of edges
    snorm_e = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_e]
    snorm_e = torch.cat(snorm_e).sqrt()  # graph size normalization
    batched_graph = dgl.batch(graphs)  # batch graphs
    return batched_graph, masks, snorm_n, snorm_e, feats, gt


class inD_DGLDataset(torch.utils.data.Dataset):

    def __init__(self, train_val, history_frames, future_frames, test=False, data_path=None, classes=(1,2), rel_types=False):
        
        self.train_val=train_val
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.total_frames = history_frames + future_frames
        self.test = test
        self.classes = classes
        self.types = rel_types

        self.raw_dir='/media/14TBDISK/sandra/inD_processed/inD_2.5Hz8_12f_benchmark_train.pkl' #inD_2.5Hz8_12f_benchmark_train.pkl'   #inD_2.5Hz_3s5s.pkl'  
        if self.train_val == 'test':  
            self.raw_dir ='/media/14TBDISK/sandra/inD_processed/inD_2.5Hz8_12f_benchmark_test.pkl'   #inD_2.5Hz8_12f_benchmark_test.pkl'    #rounD_2.5Hz8_8f.pkl'     

        self.process()        

    def load_data(self):
        with open(self.raw_dir, 'rb') as reader:
            [all_feature, self.all_adjacency, self.all_mean_xy, self.all_visible_object_idx]= pickle.load(reader)
        all_feature=np.transpose(all_feature, (0,3,2,1)) #(N,V,T,C)
        self.all_feature=torch.from_numpy(all_feature[:,:,:self.total_frames,:]).type(torch.float32)


    def process(self):
        self.load_data()
        
        total_num = len(self.all_feature)
        print(self.train_val, total_num)
        now_history_frame=self.history_frames-1
        feature_id = [0,1,2,3,4,10]  #pos vel heading obj
        info_feats_id = list(range(5,11))  #recording_id,frame,id, l,w, class
        self.object_type = self.all_feature[:,:,:,-2].int()  # torch Tensor NxVxT
        self.object_type[self.object_type==3] = 1 # truck_bus=1 (car)
        self.object_type[self.object_type==4] = 3 # bic = 3
        
        '''
        mask_car=torch.zeros((total_num,self.all_feature.shape[1],self.total_frames))#.to('cuda') #NxVx10
        for i in range(total_num):
            mask_car_t=torch.Tensor([1 if j in self.classes else 0 for j in self.object_type[i,:,now_history_frame]])#.to('cuda')
            mask_car[i,:]=mask_car_t.view(mask_car.shape[1],1)+torch.zeros(self.total_frames)#.to('cuda') #120x12
        '''
        self.xy_dist=[spatial.distance.cdist(self.node_features[i][:,now_history_frame,:2], self.node_features[i][:,now_history_frame,:2]) for i in range(len(self.all_feature))]  #5010x70x70
        self.vel_l2 = [spatial.distance.cdist(self.node_features[i][:,now_history_frame,3:5].cpu(), self.node_features[i][:,now_history_frame,3:5].cpu()) for i in range(len(self.all_feature))]
        
        #rescale_xy=torch.ones((1,1,1,2))
        #rescale_xy[:,:,:,0] = torch.max(abs(self.all_feature[:,:,:,0]))  #121  - test 119.3
        #rescale_xy[:,:,:,1] = torch.max(abs(self.all_feature[:,:,:,1]))   #77   -  test 79
        rescale_xy=torch.ones((1,1,1,2))*10
        self.all_feature[:,:,:now_history_frame+1,:2] = self.all_feature[:,:,:now_history_frame+1,:2]/rescale_xy

        
        self.node_features = self.all_feature[:,:,:self.history_frames,feature_id]#*mask_car[:,:,:self.history_frames].unsqueeze(-1)  #x,y,heading,vx,vy 5 primeros frames 5s
        self.node_labels=self.all_feature[:,:,self.history_frames:,[0,1]]#*mask_car[:,:,self.history_frames:].unsqueeze(-1)  #x,y 3 ultimos frames    
        self.track_info = self.all_feature[:,:,:,info_feats_id]
        self.output_mask= self.all_feature[:,:,self.history_frames:,-1] ###*mask_car  #mascara only_cars/peds visibles en 6º frame 
        self.output_mask = self.output_mask.unsqueeze_(-1) #(5010,120,T_hist,1)
        
        id_list = list(set(list(range(total_num))))# - set(zero_indeces_list))
        total_valid_num = len(id_list)
        #OPCIÓN A1 / A2
        #self.train_id_list ,self.val_id_list, self.test_id_list = id_list[:round(total_valid_num*0.7)],id_list[round(total_valid_num*0.7):round(total_valid_num*0.9)], id_list[round(total_valid_num*0.9):]
        #self.test_id_list = list(range(np.where(self.track_info[:,0,0,0]==30)[0][0],total_valid_num))
        #id_list = list(set(list(range(total_num))) - set(self.test_id_list))
        #self.train_id_list,self.val_id_list = id_list[:round(total_valid_num*0.8)],id_list[round(total_valid_num*0.8):]
        
        #BENCHMARK        
        
        if self.train_val == 'test':
            self.test_id_list = id_list
        else:
            self.val_id_list = list(range(np.where(self.track_info[:,0,0,0]==0)[0][0],np.where(self.track_info[:,0,0,0]==1)[0][0]))
            self.val_id_list.extend(list(range(np.where(self.track_info[:,0,self.history_frames-1,0]==7)[0][0], np.where(self.track_info[:,0,self.history_frames-1,0]==8)[0][0])))
            self.val_id_list.extend(list(range(np.where(self.track_info[:,0,self.history_frames-1,0]==18)[0][0], np.where(self.track_info[:,0,self.history_frames-1,0]==19)[0][0])))
            self.val_id_list.extend(list(range(np.where(self.track_info[:,0,self.history_frames-1,0]==30)[0][0], np.where(self.track_info[:,0,self.history_frames-1,0]==31)[0][0])))
            self.train_id_list = list(set(id_list)- set(self.val_id_list))

        '''
        #TEST ROUND
        if self.train_val == 'train':
            self.train_id_list = id_list
        else:
            self.val_id_list = list(range(np.where(self.track_info[:,0,0,0]==3)[0][0],np.where(self.track_info[:,0,0,0]==6)[0][0]))
            self.test_id_list = list(range(np.where(self.track_info[:,0,self.history_frames-1,0]==2)[0][0], np.where(self.track_info[:,0,0,0]==3)[0][0]))
            self.test_id_list.extend(list(range(np.where(self.track_info[:,0,self.history_frames-1,0]==0)[0][0], np.where(self.track_info[:,0,self.history_frames-1,0]==1)[0][0])))
            
        '''
        if self.train_val.lower() == 'train':
            self.node_features = self.node_features[self.train_id_list]  #frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading
            self.node_labels = self.node_labels[self.train_id_list]
            self.all_adjacency = self.all_adjacency[self.train_id_list]
            self.all_mean_xy = self.all_mean_xy[self.train_id_list]
            self.object_type =self.object_type[self.train_id_list]
            self.output_mask = self.output_mask[self.train_id_list]
            self.xy_dist = np.array(self.xy_dist)[self.train_id_list]
            self.vel_l2 = np.array(self.vel_l2)[self.train_id_list]
            self.all_visible_object_idx = self.all_visible_object_idx[self.train_id_list]
        elif self.train_val.lower() == 'val':
            self.node_features = self.node_features[self.val_id_list]
            self.node_labels = self.node_labels[self.val_id_list]
            self.all_adjacency = self.all_adjacency[self.val_id_list]
            self.all_mean_xy = self.all_mean_xy[self.val_id_list]
            self.object_type =self.object_type[self.val_id_list]
            self.output_mask = self.output_mask[self.val_id_list]
            self.xy_dist = np.array(self.xy_dist)[self.val_id_list]
            self.vel_l2 = np.array(self.vel_l2)[self.val_id_list]
            self.all_visible_object_idx = self.all_visible_object_idx[self.val_id_list]
        else:
            self.node_features = self.node_features[self.test_id_list]
            self.node_labels = self.node_labels[self.test_id_list]
            self.all_adjacency = self.all_adjacency[self.test_id_list]
            self.object_type =self.object_type[self.test_id_list]
            self.all_mean_xy = self.all_mean_xy[self.test_id_list]
            self.output_mask = self.output_mask[self.test_id_list]
            self.xy_dist = np.array(self.xy_dist)[self.test_id_list]
            self.vel_l2 = np.array(self.vel_l2)[self.test_id_list]
            self.all_visible_object_idx = self.all_visible_object_idx[self.test_id_list]

    def __len__(self):
            return len(self.node_features)

    def __getitem__(self, idx):
        graph = dgl.from_scipy(spp.coo_matrix(self.all_adjacency[idx][:len(self.all_visible_object_idx[idx]),:len(self.all_visible_object_idx[idx])])).int()
        graph = dgl.remove_self_loop(graph)
        edges_uvs=[np.array([graph.edges()[0][i].numpy(),graph.edges()[1][i].numpy()]) for i in range(graph.num_edges())]
        rel_types = [(self.object_type[idx][u,self.history_frames-1]* self.object_type[idx][v,self.history_frames-1])for u,v in edges_uvs]
        graph = dgl.add_self_loop(graph)
        rel_types.extend(torch.zeros_like(rel_types[0], dtype=torch.float32) for i in range(graph.num_nodes()))
        
        feats = self.node_features[idx,self.all_visible_object_idx[idx]] #graph.ndata['x']  (N,Thist,6) - N ~ agents in seq idx = nodes in graph idx
        gt = self.node_labels[idx,self.all_visible_object_idx[idx]]  #graph.ndata['gt']   (N,Tpred,2)
        output_mask = self.output_mask[idx,self.all_visible_object_idx[idx]]
        
        distances = [self.xy_dist[idx][graph.edges()[0][i]][graph.edges()[1][i]] for i in range(graph.num_edges())]
        #rel_vels = [self.vel_l2[idx][graph.edges()[0][i]][graph.edges()[1][i]] for i in range(graph.num_edges())]
        distances = [1/(i) if i!=0 else 1 for i in distances]
        if self.types:
            #rel_vels =  F.softmax(torch.tensor(rel_vels, dtype=torch.float32), dim=0)
            distances = F.softmax(torch.tensor(distances, dtype=torch.float32), dim=0)
            graph.edata['w'] = torch.tensor([[distances[i],rel_types[i]] for i in range(len(rel_types))], dtype=torch.float32)
        else:
            graph.edata['w'] = F.softmax(torch.tensor(distances, dtype=torch.float32), dim=0)



        if self.model_type == 'rgcn' or self.model_type == 'hetero':
            edges_uvs=[np.array([graph.edges()[0][i].numpy(),graph.edges()[1][i].numpy()]) for i in range(graph.num_edges())]
            rel_types = [self.object_type[idx][u,self.history_frames-1]* self.object_type[idx][v,self.history_frames-1] for u,v in edges_uvs]
            rel_types = [r - math.ceil(r/2) for r in rel_types] #0: car-car  1:car-ped  2:ped-ped
            graph.edata['rel_type'] = torch.tensor(rel_types, dtype=torch.uint8)  

            u,v,eid=graph.all_edges(form='all')
            v_canonical = []
            v_canonical.append(v[np.where(np.array(rel_types)==0)])
            v_canonical.append(v[np.where(np.array(rel_types)==1)])
            v_canonical.append(v[np.where(np.array(rel_types)==2)])
            ew_canonical = []
            ew_canonical.append(graph.edata['w'][np.where(np.array(rel_types)==0)])
            ew_canonical.append(graph.edata['w'][np.where(np.array(rel_types)==1)])
            ew_canonical.append(graph.edata['w'][np.where(np.array(rel_types)==2)])
            # calculate norm for each edge type and store in edge
            graph.edata['norm'] = torch.ones(eid.shape[0],1)  
            if self.model_type == 'hetero':
                u_canonical = []
                u_canonical.append(u[np.where(np.array(rel_types)==0)])
                u_canonical.append(u[np.where(np.array(rel_types)==1)])
                u_canonical.append(u[np.where(np.array(rel_types)==2)])
                graph=dgl.heterograph({
                    ('car', 'v2v', 'car'): (u_canonical[0], v_canonical[0]),
                    ('car', 'v2vru', 'ped'): (u_canonical[1], v_canonical[1]),
                    ('ped', 'vru2vru', 'ped'): (u_canonical[2], v_canonical[2]),
                })
                graph.nodes['drug'].data['hv'] = th.ones(3, 1)
                for v, etype in zip(v_canonical, graph.canonical_etypes):
                    _, inverse_index, count = torch.unique(v, return_inverse=True, return_counts=True)
                    degrees = count[inverse_index]
                    norm = torch.ones(v.shape[0]).float() / degrees.float()
                    norm = norm.unsqueeze(1)
                    g.edges[etype].data['norm'] = norm
            else:
                for i,v in enumerate(v_canonical):        
                    _, inverse_index, count = torch.unique(v, return_inverse=True, return_counts=True)
                    degrees = count[inverse_index]
                    norm = torch.ones(v.shape[0]).float() / degrees.float()
                    norm = norm.unsqueeze(1)
                    #g.edges[etype].data['norm'] = norm
                    graph.edata['norm'][np.where(np.array(rel_types)==i)] = norm

        if self.test:
            mean_xy = self.all_mean_xy[idx]
            track_info = self.track_info[idx,self.all_visible_object_idx[idx]]
            object_type = self.object_type[idx,self.all_visible_object_idx[idx],self.history_frames-1]
            return graph, output_mask, track_info, mean_xy, feats, gt, object_type
        else: 
            return graph, output_mask, feats, gt

if __name__ == "__main__":
    history_frames=8
    future_frames=12
    #train_dataset = inD_DGLDataset(train_val='train', history_frames=history_frames, future_frames=future_frames, model_type='gat', classes=(1,2,3,4)) #12281
    val_dataset = inD_DGLDataset(train_val='train', history_frames=history_frames, future_frames=future_frames,  classes=(1,2,3,4))  #3509
    #test_dataset = inD_DGLDataset(train_val='test', history_frames=history_frames, future_frames=future_frames, model_type='gat', classes=(1,2,3,4))  #1754
    #train_dataloader=iter(DataLoader(train_dataset, batch_size=5, shuffle=False, collate_fn=collate_batch) )
    val_dataloader=iter(DataLoader(val_dataset, batch_size=5, shuffle=False, collate_fn=collate_batch) )
    while(1):
        batched_graph, masks, snorm_n, snorm_e, feats, gt = next(val_dataloader)
        print(feats.shape)
    