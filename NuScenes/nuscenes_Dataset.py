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
from utils import convert_global_coords_to_local, convert_local_coords_to_global
os.environ['DGLBACKEND'] = 'pytorch'
from torchvision import transforms
import scipy.sparse as spp
import cv2
import json

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
    graphs, masks, feats, gt, tokens, scene_ids, mean_xy, maps, global_feats, lanes = map(list, zip(*samples))  
    masks = torch.vstack(masks)
    for lane in lanes[1:]:
        lanes[0].update(lane)
    feats = torch.vstack(feats)
    global_feats = torch.vstack(global_feats)
    gt = torch.vstack(gt).float()
    if maps[0] is not None:
        maps = torch.vstack(maps)
    sizes_n = [graph.number_of_nodes() for graph in graphs] # graph sizes
    snorm_n = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_n]
    snorm_n = torch.cat(snorm_n).sqrt()  # graph size normalization 
    sizes_e = [graph.number_of_edges() for graph in graphs] # nb of edges
    snorm_e = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_e]
    snorm_e = torch.cat(snorm_e).sqrt()  # graph size normalization
    batched_graph = dgl.batch(graphs)  # batch graphs
    return batched_graph, masks, snorm_n, snorm_e, feats, gt, tokens[0], scene_ids[0], mean_xy, maps, global_feats, lanes[0]


def collate_batch_ns(samples):
    graphs, masks, feats, gt, maps, scene_id, tokens, mean_xy, global_feats, lanes = map(list, zip(*samples)) 
    if maps[0] is not None:
        maps = torch.vstack(maps)
    
    if lanes[0] != -1:
        for lane in lanes[1:]:
            lanes[0].update(lane)

    masks = torch.vstack(masks)
    feats = torch.vstack(feats)
    tokens = np.vstack(tokens)
    global_feats = torch.vstack(global_feats)
    gt = torch.vstack(gt).float()
    sizes_n = [graph.number_of_nodes() for graph in graphs] # graph sizes
    snorm_n = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_n]
    snorm_n = torch.cat(snorm_n).sqrt()  # graph size normalization 
    sizes_e = [graph.number_of_edges() for graph in graphs] # nb of edges
    snorm_e = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_e]
    snorm_e = torch.cat(snorm_e).sqrt()  # graph size normalization
    batched_graph = dgl.batch(graphs)  # batch graphs
    return batched_graph, masks, snorm_n, snorm_e, feats, gt, maps, scene_id[0], tokens, mean_xy, global_feats, lanes[0]


class nuscenes_Dataset(torch.utils.data.Dataset):

    def __init__(self, train_val_test='train', history_frames=history_frames, future_frames=future_frames, 
                    rel_types=True, local_frame = True, retrieve_lanes = True, step = 1):
        '''
            :classes:   categories to take into account
            :rel_types: wether to include relationship types in edge features 
        '''
        self.train_val_test=train_val_test
        self.history_frames = history_frames
        self.map_path = os.path.join(base_path, 'hd_maps_3') 
        self.future_frames = future_frames
        self.types = rel_types
        self.local_frame = local_frame
        self.retrieve_lanes = retrieve_lanes
        self.step = step

        if train_val_test == 'train':
            if self.history_frames == 5:
                self.raw_dir =os.path.join(base_path, 'ns_2s6s_train.pkl' )#train_val_test = 'train_filter'
            elif self.history_frames == 8:
                self.raw_dir =os.path.join(base_path, 'ns_4s_train.pkl' )
            else:
                self.raw_dir =os.path.join(base_path, 'ns_3s_train.pkl' )
        else:
            if self.history_frames == 5:
                self.raw_dir = os.path.join(base_path,'ns_2s6s_test.pkl')  #ns_challenge_json_3s_test.pkl
                #self.map_path = os.path.join(base_path, 'hd_maps_challenge_ego') 
            elif self.history_frames == 8:
                self.raw_dir =os.path.join(base_path, 'ns_4s_test.pkl' )
            else:
                self.raw_dir =os.path.join(base_path, 'ns_3s_test.pkl' )
        
        #self.raw_dir = os.path.join(base_path,'ns_step1_train' + train_val_test + '.pkl')
        self.transform = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                #transforms.Grayscale(),
                                #transforms.Normalize((0.312,0.307,0.377), (0.447,0.447,0.471))
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))   #Imagenet
                                #transforms.Normalize((0.35), (0.43)), 
                            ]
                        )
        self.load_data()
        self.process()        

    def load_data(self):
        with open(self.raw_dir, 'rb') as reader:
            [self.all_feature, self.all_adjacency, self.all_mean_xy, self.all_tokens]= pickle.load(reader)
            '''
            self.all_feature = self.all_feature[3660:]
            self.all_adjacency = self.all_adjacency[3660:]
            self.all_mean_xy = self.all_mean_xy[3660:]
            self.all_tokens = self.all_tokens[3660:]
            '''
        
        if self.retrieve_lanes:
            with open(os.path.join(base_path, 'lanes', self.train_val_test + '_np.pkl'), 'rb') as lanes_pkl:
                self.lanes = pickle.load(lanes_pkl)


        if self.train_val_test == 'train': 
            if self.history_frames == 5:
                path = os.path.join(base_path,'ns_2s6s_val.pkl')
            elif self.history_frames == 8:
                path = os.path.join(base_path,'ns_4s_val.pkl')
            else:
                path = os.path.join(base_path,'ns_3s_val.pkl')

            with open(path, 'rb') as reader:
                [all_feature, all_adjacency, all_mean_xy,all_tokens]= pickle.load(reader)

            self.all_feature = np.vstack((self.all_feature, all_feature))
            self.all_adjacency = np.vstack((self.all_adjacency,all_adjacency))
            self.all_mean_xy = np.vstack((self.all_mean_xy,all_mean_xy))
            self.all_tokens = np.hstack((self.all_tokens,all_tokens ))

            if self.retrieve_lanes:
                with open(os.path.join(base_path, 'lanes', 'val_np.pkl'), 'rb') as lanes_pkl:
                    self.lanes.update(pickle.load(lanes_pkl))
        '''
            with open(os.path.join(base_path,'nuscenes_test.pkl'), 'rb') as reader:
                [all_feature, all_adjacency, all_mean_xy,all_tokens]= pickle.load(reader)
        
            self.all_feature = np.vstack((self.all_feature[:,:50], all_feature[:,:50]))
            self.all_adjacency = np.vstack((self.all_adjacency[:,:50,:50],all_adjacency[:,:50,:50]))
            self.all_mean_xy = np.vstack((self.all_mean_xy,all_mean_xy))
            self.all_tokens = np.hstack((self.all_tokens,all_tokens ))
        '''
        step = self.step
        self.all_feature = self.all_feature[::step] 
        self.all_adjacency =  self.all_adjacency[::step] 
        self.all_mean_xy = self.all_mean_xy[::step] 
        self.all_tokens =  self.all_tokens[::step] 
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
        feature_id = list(range(0,5)) + [8] + [-2]
        self.track_info = self.all_feature[:,:,:,13:15]
        self.object_type = self.all_feature[:,:,now_history_frame,8].int()
        self.scene_ids = self.all_feature[:,0,now_history_frame,-3].numpy()
        self.all_scenes = np.unique(self.scene_ids)
        self.num_visible_object = self.all_feature[:,0,now_history_frame,-1].int()   #Max=108 (train), 104(val), 83 (test)  #Challenge: test 20 ! train 33!
        self.output_mask= self.all_feature[:,:,:,-2].unsqueeze_(-1)
        '''
        for t in range(history_frames):
            self.all_feature[:,:,t, 2] -=  self.all_feature[:,:,now_history_frame, 2] 
        '''
        #rescale_xy[:,:,:,0] = torch.max(abs(self.all_feature[:,:,:,0]))  
        #rescale_xy[:,:,:,1] = torch.max(abs(self.all_feature[:,:,:,1]))  
        #rescale_xy=torch.ones((1,1,1,2))*10
        #self.all_feature[:,:,:self.history_frames,:2] = self.all_feature[:,:,:self.history_frames,:2]/rescale_xy
        self.xy_dist=[spatial.distance.cdist(self.all_feature[i][:,now_history_frame,:2], self.all_feature[i][:,now_history_frame,:2]) for i in range(len(self.all_feature))]  #5010x70x70
        
        # Convert history to local coordinates
        if self.local_frame:
            all_feature = self.all_feature.clone()
            for seq in range(self.all_feature.shape[0]):
                index = torch.tensor( [ int(torch.nonzero(mask_i)[0][0]) for mask_i in self.all_feature[seq,:self.num_visible_object[seq]-1,:self.history_frames,-1]])
                all_feature[seq,:self.num_visible_object[seq]-1,:self.history_frames,:2] = torch.tensor([ convert_global_coords_to_local(self.all_feature[seq,i,index[i]:self.history_frames,:2], self.all_feature[seq,i,now_history_frame,:2], self.all_feature[seq,i,now_history_frame,2], self.history_frames) for i in range(self.num_visible_object[seq]-1)])
                # Convert ego feats
                index = torch.nonzero(self.all_feature[seq, self.num_visible_object[seq]-1, :self.history_frames, 0])[0][0]
                all_feature[seq,self.num_visible_object[seq]-1,:self.history_frames,:2] = torch.tensor([ convert_global_coords_to_local(self.all_feature[seq,self.num_visible_object[seq]-1,index:self.history_frames,:2], self.all_feature[seq,self.num_visible_object[seq]-1,now_history_frame,:2], self.all_feature[seq,self.num_visible_object[seq]-1,now_history_frame,2], self.history_frames)])
                
                all_feature[seq,:self.num_visible_object[seq],self.history_frames:,:2] = torch.tensor([ convert_global_coords_to_local(self.all_feature[seq,i,self.history_frames:,:2], self.all_feature[seq,i,now_history_frame,:2], self.all_feature[seq,i,now_history_frame,2], -1) for i in range(self.num_visible_object[seq])])
            
            self.node_features = all_feature[:,:,:now_history_frame,feature_id]   #xy mean -0.0047 std 8.44 | xyhead 0.002 6.9 | (0,8) 0.0007 4.27 | (0,5) 0.0013 5.398 (test 0.004 3.35)
            '''
            self.node_features[:,:,:,1] = (self.node_features[:,:,:,1] + 0.6967) / 2.8636
            self.node_features[:,:,:,0] = self.node_features[:,:,:,0] / 0.2077
            self.node_features[:,:,:,2] = (self.node_features[:,:,:,2] - 0.0215) / 0.6085
            self.node_features[:,:,:,3] = self.node_features[:,:,:,3] / 1.55
            self.node_features[:,:,:,4] = self.node_features[:,:,:,4] / 1.335
            self.node_features[:,:,:,5] = (self.node_features[:,:,:,5] - 0.0917) / 0.2886
            '''
            self.node_labels = all_feature[:,:,self.history_frames:,:2] 
        else:
            ###### Normalize with training statistics to have 0 mean and std=1 #######
            self.node_features = self.all_feature[:,:,:self.history_frames,feature_id]   #xy mean -0.0047 std 8.44 | xyhead 0.002 6.9 | (0,8) 0.0007 4.27 | (0,5) 0.0013 5.398 (test 0.004 3.35)
            #self.node_features = (self.node_features) / 5 #(challenge 1.06) # 5 normal filetered
            self.node_labels = self.all_feature[:,:,self.history_frames:,:3] 
            self.node_labels[:,:,:,2:] = ( self.node_labels[:,:,:,2:] + 0.0015 ) / 1.2944    # Normalize heading for z0 loss.
        
        self.global_features = self.all_feature[:,:,:,feature_id]
        
        
    def __len__(self):
            return len(self.all_feature)

    def __getitem__(self, idx):         
        ### Create graph ###
        graph = dgl.from_scipy(spp.coo_matrix(self.all_adjacency[idx][:self.num_visible_object[idx],:self.num_visible_object[idx]])).int()
        
        ### Edge Data ###

        ##### Compute relation types
        object_type = self.object_type[idx,:self.num_visible_object[idx]]
        edges_uvs=[np.array([graph.edges()[0][i].numpy(),graph.edges()[1][i].numpy()]) for i in range(graph.num_edges())]
        rel_types = [torch.zeros(1, dtype=torch.int) if u==v else (object_type[u]*object_type[v]) for u,v in edges_uvs]
        
        ##### Compute distances among neighbors
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
        output_mask = self.output_mask[idx, :self.num_visible_object[idx], self.history_frames:]
        sample_token=str(self.all_tokens[idx][0,1])
        scene_id = int(self.scene_ids[idx])

        ### Include ego in feats and labels
        feats[-1,:,-1] = 1
        output_mask[-1,:] = 1
        ### Load Lanes in 50m radius for all agents in this sample
        if self.retrieve_lanes:
            lanes = {sample_token: self.lanes[sample_token]}     
        else:
            lanes = -1

        ### Load Maps for all agents in this sample
        with open(os.path.join(self.map_path, sample_token + '.pkl'), 'rb') as reader:
            maps = pickle.load(reader)  # [N_agents][3, 112,112] list of tensors
        maps=torch.vstack([self.transform(map_i).unsqueeze(0) for map_i in maps])

        if maps.shape[0] != feats.shape[0]:
            print('hey')

        ### Load tokens: instance, sample, location, current lane
        tokens = self.all_tokens[idx]

        ### Features in global frame 
        global_feats = self.global_features[idx,:self.num_visible_object[idx],:,:3]
        global_feats[:,:,:2] = global_feats[:,:,:2] + self.all_mean_xy[idx,:2]

        ### Check if all feats == 0
        if self.local_frame:
            empty_agents= []
            for i,feat_i in enumerate(feats):
                if not torch.any(feat_i):
                    empty_agents.append(i)
            idx_with_data = list(set(range(len(feats))) - set(empty_agents))
            feats = feats[idx_with_data] 
            gt = gt[idx_with_data]
            output_mask = output_mask[idx_with_data]   
            global_feats = global_feats[idx_with_data]       
            maps = maps[idx_with_data]
            graph.remove_nodes(empty_agents)
            tokens = tokens[idx_with_data]
        
        """
        ### SAVE HD_MAPS jpg

        img=((maps-maps.min())*255/(maps.max()-maps.min())).numpy()
        for i in range(img.shape[0]):
            path = os.path.join('/media/14TBDISK/sandra/nuscenes_processed/hd_maps_jpg', str(scene_id))
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(os.path.join(path,sample_token+f'agent_{i}.jpg'),cv2.cvtColor(img[i].transpose(1,2,0), cv2.COLOR_RGB2BGR))
        """
        
        return graph, output_mask, feats, gt, maps, int(self.scene_ids[idx]), tokens, self.all_mean_xy[idx,:2], global_feats, lanes

if __name__ == "__main__":
    
    train_dataset = nuscenes_Dataset(train_val_test='test', local_frame=False, retrieve_lanes=False, history_frames=5, step=1)  #3509
    #train_dataset = nuscenes_Dataset(train_val_test='train', challenge_eval=False)  #3509
    #test_dataset = nuscenes_Dataset(train_val_test='test', challenge_eval=True)  #1754
    test_dataloader=iter(DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_batch_ns) )
    for batched_graph, masks, snorm_n, snorm_e, feats, gt, maps,  scene_id, tokens, mean_xy,  global_feats, lanes  in test_dataloader:
        print(feats.shape, maps.shape, scene_id)

    