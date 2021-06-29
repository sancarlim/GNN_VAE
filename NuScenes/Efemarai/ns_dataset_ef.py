import numpy as np
import dgl
import torch
import os
os.environ['DGLBACKEND'] = 'pytorch'
from torchvision import transforms

FREQUENCY = 2
dt = 1 / FREQUENCY
history = 3
future = 5
history_frames = history*FREQUENCY + 1
future_frames = future*FREQUENCY
total_frames = history_frames + future_frames + 1
max_num_objects = 150 
total_feature_dimension = 16


def collate_batch_ef(samples):
    graphs, masks, feats, gt, maps, e_w = map(list, zip(*samples))  # samples is a list of tuples
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
    return batched_graph, masks, snorm_n, snorm_e, feats, gt, maps, e_w[0]


class nuscenes_Dataset_Ef(torch.utils.data.Dataset):

    def __init__(self, future_frames, history_frames):
        '''
            :classes:   categories to take into account
            :rel_types: wether to include relationship types in edge features 
        '''
        self.future_frames = future_frames
        self.history_frames = history_frames
                        
    def __len__(self):
            return 3

    def __getitem__(self, idx):              
        graph = dgl.graph(([0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5]))
        output_masks = torch.rand(6, 1, self.future_frames, 1)
        e_w = torch.rand(6, 2)
        feats = torch.rand(6, self.history_frames-1, 7)
        maps = torch.rand(6, 3, 224, 224)
        gt = torch.rand(6, 1, self.future_frames, 2)

        return graph, output_masks, feats, gt, maps, e_w
