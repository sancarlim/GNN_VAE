import numpy as np
import dgl
import random
import pickle
from scipy import spatial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
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

    #masks = masks.view(masks.shape[0],-1)
    #masks= masks.view(masks.shape[0]*masks.shape[1],masks.shape[2],masks.shape[3])#.squeeze(0) para TAMAÑO FIJO
    sizes_n = [graph.number_of_nodes() for graph in graphs] # graph sizes
    snorm_n = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_n]
    snorm_n = torch.cat(snorm_n).sqrt()  # graph size normalization 
    sizes_e = [graph.number_of_edges() for graph in graphs] # nb of edges
    snorm_e = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_e]
    snorm_e = torch.cat(snorm_e).sqrt()  # graph size normalization
    batched_graph = dgl.batch(graphs)  # batch graphs
    return batched_graph, masks, snorm_n, snorm_e, feats, gt

class ApolloScape_DGLDataset(torch.utils.data.Dataset):
    def __init__(self, train_val,  test=False, data_path=None, rel_types=False, scale_factor=1):
        self.raw_dir='/media/14TBDISK/sandra/apollo_train_data.pkl'
        self.train_val=train_val
        self.test = test
        self.rel_types = rel_types
        self.scale_factor = scale_factor
        if test:
            self.raw_dir='/home/sandra/PROGRAMAS/DBU_Graph/data/apollo_test_data.pkl'
        self.process() 


    def load_data(self):
        with open(self.raw_dir, 'rb') as reader:
            # Training (N, C, T, V)=(5010, 11, 12, 120), (5010, 120, 120), (5010, 2)
            [all_feature, self.all_adjacency, self.all_mean_xy,_]= pickle.load(reader)
        all_feature=np.transpose(all_feature, (0,3,2,1)) #(N,V,T,C)
        self.all_feature=torch.from_numpy(all_feature).type(torch.float32)


    def process(self):
        #process data to graph, labels, and splitting masks
        self.load_data()
        total_num = len(self.all_feature)
        
        self.last_vis_obj=[]   #contains number of visible objects in each sequence of the training, i.e. objects in frame 5
        #para hacer grafos de tamaño variable
        for idx in range(len(self.all_adjacency)): 
            for i in range(len(self.all_adjacency[idx])): 
                if self.all_adjacency[idx][i,i] == 0:
                    self.last_vis_obj.append(i)
                    break   
        
        feature_id = [3, 4, 9, 2, 10]   #frame,obj,type,x,y,z,l,w,h,heading, QUITO [visible_mask]
            
        now_history_frame=6
        self.object_type = self.all_feature[:,:,:,2].int() # torch Tensor NxVxT  1/2:cars 3:ped 4:bic 5:others
        self.object_type[self.object_type==2] = 1 # car=1 
        self.object_type[self.object_type==5] = 2 # others= 2
        self.object_type[self.object_type==3] = 2 # ped= 2
        self.object_type[self.object_type==4] = 3 #bic = 3

        self.info = self.all_feature[:,:,:,:3] #frame,obj,type for TEST
        '''
        mask_car=np.zeros((total_num,self.all_feature.shape[1],12))#.to('cuda') #NxVx12
        for i in range(total_num):
            mask_car_t=np.array([1  if (j==2 or j==1 or j==3 or j==4) else 0 for j in object_type[i,:,5]])#.to('cuda')
            mask_car[i,:]=np.array(mask_car_t).reshape(mask_car.shape[1],1)+np.zeros(12)#.to('cuda') #120x12
        '''

        rescale_xy=torch.ones((1,1,1,2))*self.scale_factor
        #rescale_xy[:,:,:,0] = torch.max(abs(self.all_feature[:,:,:,3]))
        #rescale_xy[:,:,:,1] = torch.max(abs(self.all_feature[:,:,:,4]))
        self.all_feature[:,:,:now_history_frame,3:5] = self.all_feature[:,:,:now_history_frame,3:5]/rescale_xy  #scale input x,y - positions
        self.node_features = self.all_feature[:,:,:now_history_frame, feature_id] #*(np.expand_dims(mask_car[:,:,:6],axis=-1))).type(torch.float32)  #obj type,x,y 6 primeros frames
        self.node_labels=self.all_feature[:,:,now_history_frame:,[3,4]] #x,y 6 ultimos frames
        #self.node_features[:,:,:,-1] *= mask_car[:,:,:6]   #Pongo 0 en feat 11 [mask] a todos los obj visibles no-car
        
                
        #EDGES weights  #5010x120x120[]
        self.xy_dist=[spatial.distance.cdist(self.node_features[i][:,5,:], self.node_features[i][:,5,:]) for i in range(len(self.all_feature))]  #5010x70x70
        
        if self.test:
            self.output_mask= self.all_feature[:,:,:,-1]#*mask_car[:,:,:6] #mascara obj (car) visibles en 6º frame (5010,120,6,1)
            #zero_indeces_list = [i for i in range(len(self.output_mask )) if np.all(np.array(self.output_mask.squeeze(-1))==0, axis=(1,2))[i] == True ]
            self.test_id_list  = list(set(list(range(total_num)))) #- set(zero_indeces_list))
        else:
            self.output_mask= self.all_feature[:,:,now_history_frame:,-1]#*mask_car #mascara obj (car) visibles en 6º frame (5010,120,6,1)
            self.output_mask = self.output_mask.unsqueeze_(-1)
            # TRAIN VAL SETS
            # Remove empty rows from output mask 
            zero_indeces_list = [i for i in range(len(self.output_mask )) if np.all(np.array(self.output_mask.squeeze(-1))==0, axis=(1,2))[i] == True ]
            id_list = list(set(list(range(total_num))) - set(zero_indeces_list))
            total_valid_num = len(id_list)
            
            self.train_id_list, self.val_id_list = id_list[:round(total_valid_num*0.80)], id_list[round(total_valid_num*0.80):]

            if self.train_val.lower() == 'train':
                self.node_features = self.node_features[self.train_id_list]  #frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading
                self.node_labels = self.node_labels[self.train_id_list]
                self.output_mask = self.output_mask[self.train_id_list]
                self.all_adjacency = self.all_adjacency[self.train_id_list]
                self.all_mean_xy = self.all_mean_xy[self.train_id_list]
                self.xy_dist = torch.tensor(self.xy_dist)[self.train_id_list]
                self.last_vis_obj = torch.tensor(self.last_vis_obj)[self.train_id_list]
            elif self.train_val.lower() == 'val':
                self.node_features = self.node_features[self.val_id_list]
                self.node_labels = self.node_labels[self.val_id_list]
                self.output_mask = self.output_mask[self.val_id_list]
                self.all_adjacency = self.all_adjacency[self.val_id_list]
                self.all_mean_xy = self.all_mean_xy[self.val_id_list]
                self.xy_dist = torch.tensor(self.xy_dist)[self.val_id_list]
                self.last_vis_obj = torch.tensor(self.last_vis_obj)[self.val_id_list]

        #train_id_list = list(np.linspace(0, total_num-1, int(total_num*0.8)).astype(int))
        #val_id_list = list(set(list(range(total_num))) - set(train_id_list))  
        


    def __len__(self):
        return len(self.node_features)

    def __getitem__(self, idx):
        graph = dgl.from_scipy(spp.coo_matrix(self.all_adjacency[idx][:self.last_vis_obj[idx],:self.last_vis_obj[idx]])).int()
        graph = dgl.remove_self_loop(graph)
        edges_uvs=[np.array([graph.edges()[0][i].numpy(),graph.edges()[1][i].numpy()]) for i in range(graph.num_edges())]
        rel_types = [(self.object_type[idx][u,5]* self.object_type[idx][v,5])for u,v in edges_uvs]
        graph = dgl.add_self_loop(graph)
        if len(rel_types)!=0:
            rel_types.extend(torch.zeros_like(rel_types[0], dtype=torch.float32) for i in range(graph.num_nodes()))
        else:
            rel_types=[torch.zeros(1, dtype=torch.float32) for i in range(graph.num_nodes())]
        distances = [self.xy_dist[idx][graph.edges()[0][i]][graph.edges()[1][i]] for i in range(graph.num_edges())]
        distances = [1/(i) if i!=0 else 1 for i in distances]
        if self.rel_types:
            distances = F.softmax(torch.tensor(distances, dtype=torch.float32), dim=0)
            graph.edata['w'] = torch.tensor([[distances[i],rel_types[i]] for i in range(len(rel_types))], dtype=torch.float32)
        else:
            graph.edata['w'] = F.softmax(torch.tensor(distances, dtype=torch.float32), dim=0)

        #graph.ndata['x']=self.node_features[idx,:self.last_vis_obj[idx]] 
        feats = self.node_features[idx,:self.last_vis_obj[idx]] 
        gt=self.node_labels[idx,:self.last_vis_obj[idx]]  #graph.ndata['gt']
        output_mask = self.output_mask[idx,:self.last_vis_obj[idx]]

        if self.test:
            return graph, feats, self.info[idx,:self.last_vis_obj[idx]],self.info[idx,:self.last_vis_obj[idx],2], self.all_mean_xy[idx], output_mask
        
        return graph, output_mask, feats, gt 

if __name__ == "__main__":
    history_frames=6
    future_frames=6
    test_dataset = ApolloScape_DGLDataset(train_val='val', test=False, rel_types=True) 
    test_dataloader=iter(DataLoader(test_dataset, batch_size=512, shuffle=False, collate_fn=collate_batch) )
    
    while(1):
        next(test_dataloader)
    