
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
import json
import os

DATAROOT = '/media/14TBDISK/nuscenes'
nuscenes = NuScenes('v1.0-trainval', dataroot=DATAROOT)   
helper = PredictHelper(nuscenes)
base_path = '/media/14TBDISK/sandra/nuscenes_processed'
base_path_map = os.path.join(base_path, 'hd_maps_3')

static_layer_rasterizer = StaticLayerRasterizer(helper)
agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=2)
input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())



#scene = nuscenes.field2token('scene', 'name','scene-0276')[0]
now_history_frame = 6
#nusc_map = NuScenesMap(map_name = nuscenes.get('log', scene['log_token'])['location'], dataroot=DATAROOT)

for data_class in ['train']:
    data_dir = os.path.join(base_path, 'ns_3s_' + data_class + '.pkl')
    with open(data_dir, 'rb') as reader:
        [all_feature, _, _, all_tokens]= pkl.load(reader)
    '''
    idx = np.where(all_feature[:,0,now_history_frame,-3] == 276)
    for i, seq_tokens in enumerate(all_tokens[idx]):
        feats = all_feature[idx+i]
        sample_token = seq_tokens[0,1]
        lane_info = []
        for instance_feats, instance_token in zip(feats, seq_tokens):
            pose = instance_feats[now_history_frame,:3]

            layers_on_point = nusc_map.layers_on_point(pose[0], pose[1])
            layers_array = np.zeros(4)
            mapping_layers={
                'road_segment': 1,
                'lane': 0,
                'stop_line': 2,
                'carpark':4
            }
                     
            for k,v in layers_on_point.items():
                if v and k in mapping_layers.keys():
                    layers_array[mapping_layers[k]] = 1
                    if k == 'road_segment':
                        layers_array[mapping_layers[k]] = [1 if nusc_map.get('road_segment', 'c6dcc4a6-14df-41ba-8d27-367a698c60c0')['is_intersection'] else 0]
            
            next_obj = nusc_map.get_next_roads(pose[0], pose[1])
            closest_lane = nusc_map.get_closest_lane(pose[0], pose[1], radius=2)
            lane_record = nusc_map.get_arcline_path(closest_lane)
            outgoing_lane = nusc_map.get_outgoing_lane_ids(closest_lane)
    '''
    #lanes = {}
    for seq_tokens in all_tokens: 
        # Retrieve ego_vehicle pose
        sample_token = seq_tokens[0,1]
        sample_record = nuscenes.get('sample', sample_token)     
        sample_data_record = nuscenes.get('sample_data', sample_record['data']['LIDAR_TOP'])
        poserecord = nuscenes.get('ego_pose', sample_data_record['ego_pose_token'])
        poserecord['instance_token'] = sample_data_record['ego_pose_token']
        
        
        if not os.path.exists(os.path.join(base_path_map, sample_token + '.pkl')):
            maps = np.array( [input_representation.make_input_representation(instance, sample_token, poserecord, ego=False) for instance, sample,_,_ in seq_tokens[:-1]] )   #[N_agents,500,500,3] uint8 range [0,256] 
            maps = np.vstack((maps, np.expand_dims( input_representation.make_input_representation(seq_tokens[-1][0], sample_token, poserecord, ego=True), axis=0) ))
            
            maps = np.array( F.interpolate(torch.tensor(maps.transpose(0,3,1,2)), size=224) ).transpose(0,2,3,1)

            save_path_map = os.path.join(base_path_map, sample_token + '.pkl')
            with open(save_path_map, 'wb') as writer:
               pkl.dump(maps,writer)  
        

        ###########
        ## LANES ##
        ###########
        #lanes[sample_token] = {instance: input_representation.get_lanes_representation(instance, sample_token, poserecord, ego=False) for instance, sample,_,_ in seq_tokens[:-1] }   #Array of dicts / Each dict contains mapping of sample_agent's layers
        #lanes[sample_token][seq_tokens[-1][0]] = input_representation.get_lanes_representation(seq_tokens[-1][0], sample_token, poserecord, ego=True)
        
    # path = '/media/14TBDISK/sandra/nuscenes_processed/lanes'
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # with open(os.path.join(path, data_class + '.json'), 'w') as file:
    #     json.dump(lanes, file)