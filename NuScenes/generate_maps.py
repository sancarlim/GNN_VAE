
from matplotlib.pyplot import axis
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
import pickle, json
import os

DATAROOT = '/media/14TBDISK/nuscenes'
nuscenes = NuScenes('v1.0-trainval', dataroot=DATAROOT)   
helper = PredictHelper(nuscenes)
base_path = '/media/14TBDISK/sandra/nuscenes_processed'
base_path_map = os.path.join(base_path, 'hd_maps_ego_history')

static_layer_rasterizer = StaticLayerRasterizer(helper)
agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=2)
input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())


mapping_dict = {
            'YIELD':1,
            'STOP_SIGN':2,
            'PED_CROSSING':3, #truck_bus
            'TURN_STOP':4,
            'TRAFFIC_LIGHT': 5
        }
    
#scene = nuscenes.field2token('scene', 'name','scene-0276')[0]
now_history_frame = 4
#nusc_map = NuScenesMap(map_name = nuscenes.get('log', scene['log_token'])['location'], dataroot=DATAROOT)

for data_class in ['train']:
    data_dir = os.path.join(base_path, 'ns_2s6s_' + data_class + '.pkl')
    with open(data_dir, 'rb') as reader:
        [all_feature, all_adjacency, all_mean_xy, all_tokens]= pkl.load(reader)

    '''
    #idx = np.where(all_feature[:,0,now_history_frame,-3] == 276)
    for i, seq_tokens in enumerate(all_tokens):
        #feats = all_feature[idx+i]
        sample_token = seq_tokens[0,1]
        instances_to_remove = []
        for v, instance_tokens in enumerate(all_tokens[i]):
            instance_token = seq_tokens[v,0]
            try:
                annotation = helper.get_sample_annotation(instance_token, sample_token)
                category = annotation['category_name']
                attribute = nuscenes.get('attribute', annotation['attribute_tokens'][0])['name']
            except:
                category = 'None'
                attribute = 'None'
            if 'object' in category or 'without_rider' in attribute:
                print(category, attribute)
                instances_to_remove.append(v)
        
        num_visible_obj = int(all_feature[i,0,now_history_frame,-3])
        all_feature[i] = np.concatenate( ( np.delete(all_feature[i], instances_to_remove, 0),np.zeros((len(instances_to_remove), all_feature.shape[2],all_feature.shape[-1])) ) , axis=0)
        # Set to 0s v colum and v row
        all_adjacency[i,instances_to_remove] = np.zeros_like(all_adjacency[0,0])
        all_adjacency[i,:,instances_to_remove] = np.zeros_like(all_adjacency[0,0])
        all_tokens[i] = np.delete(all_tokens[i], instances_to_remove, 0)
        all_feature[i,:num_visible_obj-1, :, -3] = num_visible_obj - len(instances_to_remove)
    
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
    
    all_feat_new = np.zeros((all_feature.shape[0], all_feature.shape[1], all_feature.shape[2], 18))
    for i, (seq_feat, mean_xy, seq_tokens) in enumerate(zip(all_feature, all_mean_xy, all_tokens)):
        # V,T,C  i=103    NO DATA
        nusc_map = NuScenesMap(map_name=seq_tokens[0,-1], dataroot=DATAROOT)
        static_feats = []
        num_visible_object = int(seq_feat[0,now_history_frame,-1])
        for agent_feats in seq_feat[:num_visible_object]:
            now_frame_feats = agent_feats[now_history_frame]
            if now_frame_feats[8] != 2:
                x = now_frame_feats[0] + mean_xy[0]
                y = now_frame_feats[1] + mean_xy[1]
                layers = nusc_map.layers_on_point(x, y)
                stop_token = layers['stop_line']
                
                is_intersection = 0
                if len(layers['road_segment']) != 0:
                    is_intersection = int(nusc_map.get('road_segment',layers['road_segment'])['is_intersection'])
                
                stop_type = 0
                if len(stop_token) != 0:
                    stop_type = mapping_dict[nusc_map.get('stop_line',stop_token)['stop_line_type']]
                
                static_feats.append(np.array([is_intersection, stop_type])) 
            else:
                static_feats.append(np.array([-1,-1]))

        all_feat_new[i, :num_visible_object] = np.concatenate((seq_feat[:num_visible_object], 
                np.expand_dims(np.array(static_feats),1).repeat(seq_feat.shape[1], axis=1)), axis=-1)

    all_feat_new = np.array(all_feat_new)
    save_path = '/media/14TBDISK/sandra/nuscenes_processed/ns_3s_' + data_class + '.pkl'
    
    with open(data_dir, 'wb') as writer:
        pickle.dump([all_feature, all_adjacency, all_mean_xy, all_tokens], writer)
    print(f'Processed {all_feature.shape[0]} sequences.')
    '''
    
    
    #lanes = {}
    for idx, seq_tokens in enumerate(all_tokens): 
        # Retrieve ego_vehicle pose
        sample_token = seq_tokens[0,1]
        with open(os.path.join(base_path_map, sample_token + '.pkl'), 'rb') as reader:
            maps = pickle.load(reader)  # [N_agents][3, 112,112] list of tensors
        
        if len(seq_tokens) != len(maps):
            print(f'{idx}: maps {len(maps)}, agents {len(seq_tokens)}')
            sample_record = nuscenes.get('sample', sample_token)     
            sample_data_record = nuscenes.get('sample_data', sample_record['data']['LIDAR_TOP'])
            poserecord = nuscenes.get('ego_pose', sample_data_record['ego_pose_token'])
            poserecord['instance_token'] = sample_data_record['ego_pose_token']

            # Retrieve ego past
            prev_sample_data = sample_data_record
            for history_t in reversed(range(4)):
                prev_sample_data = nuscenes.get('sample_data', prev_sample_data['prev'])
                history_name = 'history_' + str(history_t)
                poserecord[history_name] = nuscenes.get('ego_pose', prev_sample_data['ego_pose_token'])['translation']
                poserecord[history_name].append(nuscenes.get('ego_pose', prev_sample_data['ego_pose_token'])['rotation'])

            #if not os.path.exists(os.path.join(base_path_map, sample_token + '.pkl')):
            maps = np.array( [input_representation.make_input_representation(instance, sample_token, poserecord, ego=False) for instance, sample,_,_ in seq_tokens[:-1]] )   #[N_agents,500,500,3] uint8 range [0,256] 
            
            # Draw ego
            maps = np.vstack((maps, np.expand_dims( input_representation.make_input_representation(seq_tokens[-1][0], sample_token, poserecord, ego=True), axis=0) ))
            
            maps = np.array( F.interpolate(torch.tensor(maps.transpose(0,3,1,2)), size=224) ).transpose(0,2,3,1)

            save_path_map = os.path.join(base_path_map, sample_token + '.pkl')
            with open(save_path_map, 'wb') as writer:
                pkl.dump(maps,writer)  
        
    
        ###########
        ## LANES ##
        ###########
        
    """     lanes[sample_token] = {instance: input_representation.get_lanes_representation(instance, sample_token, poserecord, ego=False) 
                for instance, sample,_,_ in seq_tokens[:-1] if helper.get_sample_annotation(instance, sample)['category_name'].split('.')[0] == 'vehicle' }   #Array of dicts / Each dict contains mapping of sample_agent's layers
        
        lanes[sample_token][seq_tokens[-1][0]] = input_representation.get_lanes_representation(seq_tokens[-1][0], sample_token, poserecord, ego=True)
        
    path = '/media/14TBDISK/sandra/nuscenes_processed/lanes'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, data_class + '.pkl'), 'wb') as file:
        pkl.dump(lanes, file) """