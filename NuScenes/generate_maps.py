
import numpy as np
import torch 
import torch.nn.functional as F
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer
import pickle as pkl
import os

DATAROOT = '/media/14TBDISK/nuscenes'
nuscenes = NuScenes('v1.0-trainval', dataroot=DATAROOT)   
helper = PredictHelper(nuscenes)
base_path = '/media/14TBDISK/sandra/nuscenes_processed'
base_path_map = os.path.join(base_path, 'hd_maps_challenge_ego')

static_layer_rasterizer = StaticLayerRasterizer(helper)
agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=2)
input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

for data_class in ['train']:
    data_dir = os.path.join(base_path, 'ns_challenge_' + data_class + '.pkl')
    with open(data_dir, 'rb') as reader:
        [_, _, _, all_tokens]= pkl.load(reader)
    
    for seq_tokens in all_tokens:
        # Retrieve ego_vehicle pose
        sample_token = seq_tokens[0,1]
        sample_record = nuscenes.get('sample', sample_token)     
        sample_data_record = nuscenes.get('sample_data', sample_record['data']['LIDAR_TOP'])
        poserecord = nuscenes.get('ego_pose', sample_data_record['ego_pose_token'])
        
        maps = np.array( [input_representation.make_input_representation(instance, sample, poserecord) for instance, sample in seq_tokens] )   #[N_agents,500,500,3] uint8 range [0,256] 
        maps = np.array( F.interpolate(torch.tensor(maps.transpose(0,3,1,2)), size=224) ).transpose(0,2,3,1)

        save_path_map = os.path.join(base_path_map, sample_token + '.pkl')
        with open(save_path_map, 'wb') as writer:
            pkl.dump(maps,writer)  
