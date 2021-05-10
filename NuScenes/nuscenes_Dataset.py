import numpy as np
import dgl
import pickle
import math
from scipy import spatial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.transforms import functional as trans_fn
import os
import utils
os.environ['DGLBACKEND'] = 'pytorch'
from torchvision import transforms
import scipy.sparse as spp
from dgl.data import DGLDataset
from sklearn.preprocessing import StandardScaler
import cv2

FREQUENCY = 2
dt = 1 / FREQUENCY
history = 2
future = 6
history_frames = history*FREQUENCY + 1
future_frames = future*FREQUENCY
total_frames = history_frames + future_frames + 1#2s of history + 6s of prediction
max_num_objects = 150 
total_feature_dimension = 16
base_path = '/media/14TBDISK/sandra/nuscenes_processed'

def collate_batch_test(samples):
    graphs, masks, feats, gt, tokens, scene_ids, mean_xy, maps = map(list, zip(*samples))  # samples is a list of pairs (graph, mask) mask es VxTx1
    masks = torch.vstack(masks)
    feats = torch.vstack(feats)
    gt = torch.vstack(gt).float()
    if maps[0] is not None:
        maps = torch.vstack(maps)

    if maps.shape[0] != feats.shape[0]:
        print(scene_ids[0])
    sizes_n = [graph.number_of_nodes() for graph in graphs] # graph sizes
    snorm_n = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_n]
    snorm_n = torch.cat(snorm_n).sqrt()  # graph size normalization 
    sizes_e = [graph.number_of_edges() for graph in graphs] # nb of edges
    snorm_e = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_e]
    snorm_e = torch.cat(snorm_e).sqrt()  # graph size normalization
    batched_graph = dgl.batch(graphs)  # batch graphs
    return batched_graph, masks, snorm_n, snorm_e, feats, gt, tokens[0], scene_ids[0], mean_xy, maps


def collate_batch(samples):
    graphs, masks, feats, gt, maps, scene_id = map(list, zip(*samples))  # samples is a list of tuples
    if maps[0] is not None:
        maps = torch.vstack(maps)
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
    return batched_graph, masks, snorm_n, snorm_e, feats, gt, maps, scene_id[0]


#feats.mean 0.1579 std 12.4354
# feats[:,:,:2].mean() -0.0288 std 26.195
# feats[2] mean 0.2 std 1.79
# 


class nuscenes_Dataset(torch.utils.data.Dataset):

    def __init__(self, train_val_test='train', history_frames=history_frames, future_frames=future_frames, 
                    rel_types=True, challenge_eval=False):
        '''
            :classes:   categories to take into account
            :rel_types: wether to include relationship types in edge features 
        '''
        self.train_val_test=train_val_test
        self.history_frames = history_frames
        self.map_path = os.path.join(base_path, 'hd_maps_challenge_224_ego') 
        self.future_frames = future_frames
        self.types = rel_types
        
        if train_val_test == 'train':
            self.raw_dir =os.path.join(base_path, 'ns_challenge_train.pkl' )#train_val_test = 'train_filter'
        else:
            self.raw_dir = os.path.join(base_path,'ns_challenge_test.pkl')
        
        #self.raw_dir = os.path.join(base_path,'nuscenes_' + train_val_test + '.pkl')
        self.challenge_eval = challenge_eval
        self.transform = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                #transforms.Grayscale(),
                                transforms.Normalize((0.312,0.307,0.377), (0.447,0.447,0.471))
                                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))   #Imagenet
                                #transforms.Normalize((0.35), (0.43)), 
                            ]
                        )
        self.load_data()
        self.process()        

    def load_data(self):
        with open(self.raw_dir, 'rb') as reader:
            [self.all_feature, self.all_adjacency, self.all_mean_xy, self.all_tokens]= pickle.load(reader)
            '''
            self.all_feature = self.all_feature[20:]
            self.all_adjacency = self.all_adjacency[20:]
            self.all_mean_xy = self.all_mean_xy[20:]
            self.all_tokens = self.all_tokens[20:]
            '''
        if self.train_val_test == 'train': 
            with open(os.path.join(base_path,'ns_challenge_val.pkl'), 'rb') as reader:
                [all_feature, all_adjacency, all_mean_xy,all_tokens]= pickle.load(reader)
            self.all_feature = np.vstack((self.all_feature, all_feature))
            self.all_adjacency = np.vstack((self.all_adjacency,all_adjacency))
            self.all_mean_xy = np.vstack((self.all_mean_xy,all_mean_xy))
            self.all_tokens = np.hstack((self.all_tokens,all_tokens ))
        '''
            with open(os.path.join(base_path,'nuscenes_test.pkl'), 'rb') as reader:
                [all_feature, all_adjacency, all_mean_xy,all_tokens]= pickle.load(reader)
        
            self.all_feature = np.vstack((self.all_feature[:,:50], all_feature[:,:50]))
            self.all_adjacency = np.vstack((self.all_adjacency[:,:50,:50],all_adjacency[:,:50,:50]))
            self.all_mean_xy = np.vstack((self.all_mean_xy,all_mean_xy))
            self.all_tokens = np.hstack((self.all_tokens,all_tokens ))
        '''
        self.all_feature=torch.from_numpy(self.all_feature).type(torch.float32)
        
    def process(self):
        '''
        INPUT:
            :all_feature:   x,y (global zero-centralized),heading,velx,vely,accx,accy,head_rate, type, l,w,h, frame_id, scene_id, mask, num_visible_objects (14)
            :all_mean_xy:   mean_xy per sequence for zero centralization
            :all_adjacency: Adjacency matrix per sequence for building graph
            :all_tokens:    Instance token, scene token
        RETURNS:
            :node_feats :  x_y_global, past_x_y, heading,vel,accel,heading_change_rate, type (2+8+5 = 15 in_features)
            :node_labels:  future_xy_local (24)
            :output_mask:  mask (12)
            :track_info :  frame, scene_id, node_token, sample_token (4)
        '''
        
        total_num = len(self.all_feature)
        print(f"{self.train_val_test} split has {total_num} sequences.")
        now_history_frame = self.history_frames - 1
        feature_id = list(range(0,9)) 
        self.track_info = self.all_feature[:,:,:,13:15]
        self.object_type = self.all_feature[:,:,now_history_frame,8].int()
        self.scene_ids = self.all_feature[:,0,now_history_frame,-3].numpy()
        self.all_scenes = np.unique(self.scene_ids)
        self.num_visible_object = self.all_feature[:,0,now_history_frame,-1].int()   #Max=108 (train), 104(val), 83 (test)  #Challenge: test 20 ! train 33!
        self.output_mask= self.all_feature[:,:,self.history_frames:,-2].unsqueeze_(-1)
        
        #rescale_xy[:,:,:,0] = torch.max(abs(self.all_feature[:,:,:,0]))  
        #rescale_xy[:,:,:,1] = torch.max(abs(self.all_feature[:,:,:,1]))  
        #rescale_xy=torch.ones((1,1,1,2))*10
        #self.all_feature[:,:,:self.history_frames,:2] = self.all_feature[:,:,:self.history_frames,:2]/rescale_xy

        ###### Normalize with training statistics to have 0 mean and std=1 #######
        self.node_features = self.all_feature[:,:,:self.history_frames,feature_id]   #xy mean -0.0047 std 8.44 | xyhead 0.002 6.9 | (0,8) 0.0007 4.27 | (0,5) 0.0013 5.398 (test 0.004 3.35)
        #self.node_features[:,:,:,:] = (self.node_features[:,:,:,:] - 0.1579) / 12.4354 # 0.0013) / 4.5471
        self.node_labels = self.all_feature[:,:,self.history_frames:,:2] 
        #self.node_labels[:,:,:,2:] = ( self.node_labels[:,:,:,2:] - 0.014 ) / 0.51    # Normalize heading for z0 loss.

        self.xy_dist=[spatial.distance.cdist(self.all_feature[i][:,now_history_frame,:2], self.node_features[i][:,now_history_frame,:2]) for i in range(len(self.all_feature))]  #5010x70x70
        
    def __len__(self):
            return len(self.all_feature)

    def __getitem__(self, idx):        
        graph = dgl.from_scipy(spp.coo_matrix(self.all_adjacency[idx][:self.num_visible_object[idx],:self.num_visible_object[idx]])).int()
        object_type = self.object_type[idx,:self.num_visible_object[idx]]
        # Compute relation types
        edges_uvs=[np.array([graph.edges()[0][i].numpy(),graph.edges()[1][i].numpy()]) for i in range(graph.num_edges())]
        rel_types = [torch.zeros(1, dtype=torch.int) if u==v else (object_type[u]*object_type[v]) for u,v in edges_uvs]
        
        # Compute distances among neighbors
        distances = [self.xy_dist[idx][graph.edges()[0][i]][graph.edges()[1][i]] for i in range(graph.num_edges())]
        #rel_vels = [self.vel_l2[idx][graph.edges()[0][i]][graph.edges()[1][i]] for i in range(graph.num_edges())]
        distances = [1/(i) if i!=0 else 1 for i in distances]
        if self.types:
            #rel_vels =  F.softmax(torch.tensor(rel_vels, dtype=torch.float32), dim=0)
            distances = F.softmax(torch.tensor(distances, dtype=torch.float32), dim=0)
            graph.edata['w'] = torch.tensor([[distances[i],rel_types[i]] for i in range(len(rel_types))], dtype=torch.float32)
        else:
            graph.edata['w'] = F.softmax(torch.tensor(distances, dtype=torch.float32), dim=0)


        feats = self.node_features[idx, :self.num_visible_object[idx]]
        gt = self.node_labels[idx, :self.num_visible_object[idx]]
        output_mask = self.output_mask[idx, :self.num_visible_object[idx]]

        sample_token=str(self.all_tokens[idx][0,1])
        with open(os.path.join(self.map_path, sample_token + '.pkl'), 'rb') as reader:
            maps = pickle.load(reader)  # [N_agents][3, 112,112] list of tensors
        maps=torch.vstack([self.transform(map_i).unsqueeze(0) for map_i in maps])
        
        #img=((maps-maps.min())*255/(maps.max()-maps.min())).numpy()
        #cv2.imwrite('input_276_0_gray'+sample_token+'.png',cv2.cvtColor(img[0].transpose(1,2,0), cv2.COLOR_RGB2BGR))
        scene_id = int(self.scene_ids[idx])
        if self.challenge_eval:
            return graph, output_mask, feats, gt, self.all_tokens[idx], scene_id, self.all_mean_xy[idx,:2], maps            
            
        return graph, output_mask, feats, gt, maps, int(self.scene_ids[idx])

if __name__ == "__main__":
    
    train_dataset = nuscenes_Dataset(train_val_test='train', challenge_eval=True)  #3509
    #train_dataset = nuscenes_Dataset(train_val_test='train', challenge_eval=False)  #3509
    #test_dataset = nuscenes_Dataset(train_val_test='test', challenge_eval=True)  #1754
    test_dataloader=iter(DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch_test) )
    for batched_graph, masks, snorm_n, snorm_e, feats, gt, tokens, scene_ids, mean_xy, maps in test_dataloader:
        print(feats.shape, maps.shape, scene_ids)
    