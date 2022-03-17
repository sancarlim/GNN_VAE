import sys
import os
from matplotlib.pyplot import axis
import numpy as np
from numpy.core.fromnumeric import mean
from scipy import spatial 
import pickle
import torch 
import torch.nn.functional as F
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.eval.prediction.config import load_prediction_config
from nuscenes.prediction import PredictHelper
from nuscenes.utils.splits import create_splits_scenes
import pandas as pd
from collections import defaultdict
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw, angle_diff
from torchvision import transforms

from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer


#508 0 sequences???
scene_blacklist = [992]

max_num_objects = 70  #pkl np.arrays with same dimensions
total_feature_dimension = 18 #x,y,heading,vel[x,y],acc[x,y],head_rate, type, l,w,h, frame_id, scene_id, mask, num_visible_objects

FREQUENCY = 2
dt = 1 / FREQUENCY
history = 4
future = 5
history_frames = history*FREQUENCY 
future_frames = future*FREQUENCY
total_frames = history_frames + future_frames + 1 #2s of history + 6s of prediction

# This is the path where you stored your copy of the nuScenes dataset.
DATAROOT = '/media/14TBDISK/nuscenes'
nuscenes = NuScenes('v1.0-trainval', dataroot=DATAROOT)   #850 scenes
# Helper for querying past and future data for an agent.
helper = PredictHelper(nuscenes)
base_path = '/media/14TBDISK/sandra/nuscenes_processed'
base_path_map = os.path.join(base_path, 'hd_maps_3s')

static_layer_rasterizer = StaticLayerRasterizer(helper)
agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=2)
input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())
transform = transforms.Compose(
                            [
                                #transforms.ToTensor(),
                                transforms.Resize((112,112), interpolation=3),
                                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ]
                        )

mapping_dict = {
            'YIELD':1,
            'STOP_SIGN':2,
            'PED_CROSSING':3, #truck_bus
            'TURN_STOP':4,
            'TRAFFIC_LIGHT': 5
        }
    


#DEFINE ATTENTION RADIUS FOR CONNECTING NODES
VEH_VEH_RADIUS= 35
VEH_PED_RADIUS= 20
VEH_BIC_RADIUS= 20
PED_PED_RADIUS= 10 
PED_BIC_RADIUS= 15
BIC_BIC_RADIUS= 25
neighbor_distance = VEH_VEH_RADIUS


def pol2cart(th, r):
    """
    Transform polar to cartesian coordinates.
    :param th: Nx1 ndarray
    :param r: Nx1 ndarray
    :return: Nx2 ndarray
    """

    x = np.multiply(r, np.cos(th))
    y = np.multiply(r, np.sin(th))

    cart = np.array([x, y]).transpose()
    return cart

def cart2pol(cart):
    """
    Transform cartesian to polar coordinates.
    :param cart: Nx2 ndarray
    :return: 2 Nx1 ndarrays
    """
    if cart.shape == (2,):
        cart = np.array([cart])

    x = cart[:, 0]
    y = cart[:, 1]

    th = np.arctan2(y, x)
    r = np.sqrt(np.power(x, 2) + np.power(y, 2))
    return th, r

def calculate_rotated_bboxes(center_points_x, center_points_y, length, width, rotation=0):
    """
    Calculate bounding box vertices from centroid, width and length.
    :param centroid: center point of bbox
    :param length: length of bbox
    :param width: width of bbox
    :param rotation: rotation of main bbox axis (along length)
    :return:
    """

    centroid = np.array([center_points_x, center_points_y]).transpose()

    centroid = np.array(centroid)
    if centroid.shape == (2,):
        centroid = np.array([centroid])

    # Preallocate
    data_length = centroid.shape[0]
    rotated_bbox_vertices = np.empty((data_length, 4, 2))

    # Calculate rotated bounding box vertices
    rotated_bbox_vertices[:, 0, 0] = -length / 2
    rotated_bbox_vertices[:, 0, 1] = -width / 2

    rotated_bbox_vertices[:, 1, 0] = length / 2
    rotated_bbox_vertices[:, 1, 1] = -width / 2

    rotated_bbox_vertices[:, 2, 0] = length / 2
    rotated_bbox_vertices[:, 2, 1] = width / 2

    rotated_bbox_vertices[:, 3, 0] = -length / 2
    rotated_bbox_vertices[:, 3, 1] = width / 2

    for i in range(4):
        th, r = cart2pol(rotated_bbox_vertices[:, i, :])
        rotated_bbox_vertices[:, i, :] = pol2cart(th + rotation, r).squeeze()
        rotated_bbox_vertices[:, i, :] = rotated_bbox_vertices[:, i, :] + centroid

    return rotated_bbox_vertices


def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
    data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
    data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))

    data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name, non_aug_scene=scene)

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        if node.type == 'PEDESTRIAN':
            x = node.data.position.x.copy()
            y = node.data.position.y.copy()

            x, y = rotate_pc(np.array([x, y]), alpha)

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay}

            node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)

            node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep)
        elif node.type == 'VEHICLE':
            x = node.data.position.x.copy()
            y = node.data.position.y.copy()

            heading = getattr(node.data.heading, '°').copy()
            heading += alpha
            heading = (heading + np.pi) % (2.0 * np.pi) - np.pi

            x, y = rotate_pc(np.array([x, y]), alpha)

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

            v = np.stack((vx, vy), axis=-1)
            v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
            heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.))
            heading_x = heading_v[:, 0]
            heading_y = heading_v[:, 1]

            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('velocity', 'norm'): np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1),
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay,
                         ('acceleration', 'norm'): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
                         ('heading', 'x'): heading_x,
                         ('heading', 'y'): heading_y,
                         ('heading', '°'): heading,
                         ('heading', 'd°'): derivative_of(heading, dt, radian=True)}

            node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)

            node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep,
                        non_aug_node=node)

        scene_aug.nodes.append(node)
    return scene_aug


def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    scene_aug.map = scene.map
    return scene_aug


def process_tracks(tracks, start_frame, end_frame, current_frame, nusc_map):
    '''
        Tracks: a list of (n_frames ~40f = 20s) tracks_per_frame ordered by frame.
                Each row (track) contains a dict, where each key corresponds to an array of data from all agents in that frame.
        
        Returns data processed for a sequence of 8s (2s of history, 6s of labels)
    '''
    sample_token = tracks[current_frame]['sample_token'][0]
    visible_node_id_list = tracks[current_frame]["node_id"][:-1]  #All agents in the current frame but ego
    num_visible_object = len(visible_node_id_list) + 1

    #Zero-centralization per frame (sequence)
    mean_xy = [tracks[current_frame]['x_global'].mean(),tracks[current_frame]['y_global'].mean(),0]
    #tracks['position'][:,:2] = tracks['position'][:,:2] - mean_xy

    # You can convert global coords to local frame with: helper.convert_global_coords_to_local(coords,starting_annotation['translation'], starting_annotation['rotation'])
    # x_global y_global are centralized in 0 taking into account all objects positions in the current frame
    xy = tracks[current_frame]['position'][:, :2].astype(float)
    # Compute distance between any pair of objects
    dist_xy = spatial.distance.cdist(xy, xy)  
    # If their distance is less than ATTENTION RADIUS (neighbor_distance), we regard them as neighbors.
    neighbor_matrix = np.zeros((max_num_objects, max_num_objects))
    neighbor_matrix[:num_visible_object,:num_visible_object] = (dist_xy<neighbor_distance).astype(int)

    #Retrieve all past and future trajectories
    '''
    future_xy_local = helper.get_future_for_sample(sample_token, seconds=future, in_agent_frame=True)
    past_xy_local = helper.get_past_for_sample(sample_token, seconds=history, in_agent_frame=True)
    future_xy_local_list=[value for key,value in future_xy_local.items()]
    past_xy_local_list=[value for key,value in past_xy_local.items()]
    '''
    ########## Retrieve features and labels for each agent 

    ''' 
    ############# FIRST OPTION ############
    # Get past and future trajectories
    future_xy_local = np.zeros((num_visible_object, future_frames*2))
    past_xy_local = np.zeros((num_visible_object, 2*history_frames))
    mask = np.zeros((num_visible_object, future_frames))
    for i, node_id in enumerate(track['node_id']):
        future_xy_i=helper.get_future_for_agent(node_id,sample_token, seconds=future, in_agent_frame=True).reshape(-1)
        past_xy_i=helper.get_past_for_agent(node_id,sample_token, seconds=history, in_agent_frame=True).reshape(-1)
        past_xy_local[i,:len(past_xy_i)] = past_xy_i
        future_xy_local[i, :len(future_xy_i)] = future_xy_i # Some agents don't have 6s of future or 2s of history, pad with 0's
        mask[i,:len(future_xy_i)//2] += np.ones((len(future_xy_i)//2))
        
    object_features = np.column_stack((
            track['position'], track['motion'], past_xy_local, future_xy_local, mask, track['info_agent'],
            track['info_sequence'] ))  # 3 + 3 + 8 + 24 + 12 + 4 + 2 = 56   

    inst_sample_tokens = np.column_stack((track['node_id'], track['sample_token']))
    '''

    ############ SECOND OPTION ###############
    object_feature_list = []
    while start_frame < 0:
        object_feature_list.append(np.zeros(shape=(num_visible_object,total_feature_dimension-2)))
        start_frame += 1
    
    #now_all_object_id = set([val for frame in range(start_frame, end_frame) for val in tracks[frame]["node_id"]])  #todos los obj en los15 hist frames
    
    for frame_ind in range(start_frame, end_frame + 1):	
        now_frame_feature_dict = {node_id : (
            list(tracks[frame_ind]['position'][np.where(np.array(tracks[frame_ind]['node_id'])==node_id)[0][0]] - mean_xy)+ 
            list(tracks[frame_ind]['motion'][np.where(np.array(tracks[frame_ind]['node_id'])==node_id)[0][0]]) + 
            list(tracks[frame_ind]['info_agent'][np.where(np.array(tracks[frame_ind]['node_id'])==node_id)[0][0]]) +
            list(tracks[frame_ind]['info_sequence'][0]) + 
            [1] + [num_visible_object]  #mask is 0 for ego
            ) for node_id in tracks[frame_ind]["node_id"] if node_id in visible_node_id_list}
        # if the current object is not at this frame, we return all 0s 
        now_frame_feature = np.array([now_frame_feature_dict.get(vis_id, np.zeros(total_feature_dimension-2)) for vis_id in visible_node_id_list])
        ego_feature = np.array((list(tracks[frame_ind]['position'][-1] - mean_xy) + list(tracks[frame_ind]['motion'][-1]) + list(tracks[frame_ind]['info_agent'][-1]) + list(tracks[frame_ind]['info_sequence'][-1]) + [0] + [num_visible_object])).reshape(1, total_feature_dimension-2)
        now_frame_feature = np.vstack((now_frame_feature, ego_feature))
        object_feature_list.append(now_frame_feature)
        
        if frame_ind == current_frame:
            lanes = []
            static_feats = []
            for v in now_frame_feature:
                if v[8]!=2:
                    x = v[0] + mean_xy[0]
                    y = v[1] + mean_xy[1]
                    lanes.append(nusc_map.get_closest_lane(x, y, radius=2))
                    #road_segment = nusc_map.get_next_roads(x, y)['road_segment'][0]
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
                    lanes.append('-1')

     
    object_feature = np.concatenate((np.array(object_feature_list), np.expand_dims(np.array(static_feats),0).repeat(len(object_feature_list), axis=0)), axis=-1)   # T,V,C
    if object_feature.shape[1] > 50:
        print(object_feature.shape[1])
    assert object_feature.shape[0] == total_frames
    object_frame_feature = np.zeros((max_num_objects, total_frames, total_feature_dimension))  # V, T, C
    object_frame_feature[:num_visible_object] = np.transpose(object_feature, (1,0,2))
    inst_sample_tokens = np.stack((tracks[current_frame]['node_id'], tracks[current_frame]['sample_token'],
                         np.array(lanes)), axis=-1) 
    #visible_object_indexes = [list(now_all_object_id).index(i) for i in visible_node_id_list]
    return object_frame_feature, neighbor_matrix, mean_xy, inst_sample_tokens


def process_scene(scene):
    '''
    Returns a list of (n_frames ~40f = 20s) tracks_per_frame ordered by frame.
    Each row contains a dict, where each key corresponds to an array of data from all agents in that frame.
    '''
    scene_id = int(scene['name'].replace('scene-', ''))   #419 la que data empieza en frame 4 data.frame_id.unique() token '8c84164e752a4ab69d039a07c898f7af'
    location = nuscenes.get('log', ns_scene['log_token'])['location']
    nusc_map = NuScenesMap(map_name=location, dataroot=DATAROOT)
    location = np.array(location)
    data = pd.DataFrame(columns=['scene_id',
                                 'sample_token',
                                 'frame_id',
                                 'type',
                                 'node_id',
                                 'x_global',
                                 'y_global', 
                                 'heading',
                                 'vel_x',
                                 'vel_y',
                                 'acc_x',
                                 'acc_y',
                                 'heading_change_rate',
                                 'length',
                                 'width',
                                 'height'])
    sample_token = scene['first_sample_token']
    sample = nuscenes.get('sample', sample_token)
    frame_id = 0
    mean_xy = []
    while sample['next']: 
        if frame_id != 0:
            sample = nuscenes.get('sample', sample['next'])
            sample_token = sample['token']
        annotations = helper.get_annotations_for_sample(sample_token)
        for i,annotation in enumerate(annotations):
            #print(f'{i} out of {len(annotations)} annotations')
            instance_token = annotation['instance_token']
            category = annotation['category_name']
            if len(annotation['attribute_tokens']):
                attribute = nuscenes.get('attribute', annotation['attribute_tokens'][0])['name']
            else:
                attribute = [None]
            
            if 'pedestrian' in category and 'stroller' not in category and 'wheelchair' not in category and 'sitting_lying_down' not in attribute and 'standing' not in attribute:
                node_type = 2
            elif ('bicycle' in category or 'motorcycle' in category) and 'object' not in category and 'without_rider' not in attribute and 'parked' not in attribute and 'stopped' not in attribute:
                node_type = 3
            elif 'vehicle' in category and not ('bicycle' in category or 'motorcycle' in category) and 'parked' not in attribute and 'stopped' not in attribute:                 
                node_type = 1
            else:
                continue

            #if first sample returns nan
            heading_change_rate = helper.get_heading_change_rate_for_agent(instance_token, sample_token)
            velocity =  helper.get_velocity_for_agent(instance_token, sample_token)
            acceleration = helper.get_acceleration_for_agent(instance_token, sample_token)
            if np.isnan(velocity[0]):
                continue

            data_point = pd.Series({'scene_id': scene_id,
                                    'sample_token': sample_token,
                                    'frame_id': frame_id,
                                    'type': node_type,
                                    'node_id': instance_token,
                                    'x_global': annotation['translation'][0],
                                    'y_global': annotation['translation'][1],
                                    'heading': Quaternion(annotation['rotation']).yaw_pitch_roll[0],
                                    'vel_x': velocity[0],
                                    'vel_y': velocity[1],
                                    'acc_x': acceleration[0],
                                    'acc_y': acceleration[1],
                                    'heading_change_rate': heading_change_rate,
                                    'length': annotation['size'][0],
                                    'width': annotation['size'][1],
                                    'height': annotation['size'][2]}).fillna(0)    #inplace=True         

            data = data.append(data_point, ignore_index=True)

        if not data.empty:
            # Ego Vehicle
            sample_data = nuscenes.get('sample_data', sample['data']['CAM_FRONT'])
            annotation = nuscenes.get('ego_pose', sample_data['ego_pose_token'])
            data_point = pd.Series({'scene_id': scene_id,
                                    'sample_token': sample_token,
                                    'frame_id': frame_id,
                                    'type': 0,
                                    'node_id': sample_data['ego_pose_token'],
                                    'x_global': annotation['translation'][0],
                                    'y_global': annotation['translation'][1],
                                    'heading': Quaternion(annotation['rotation']).yaw_pitch_roll[0],
                                    'vel_x': 0,
                                    'vel_y': 0,
                                    'acc_x': 0,
                                    'acc_y': 0,
                                    'heading_change_rate': 0,
                                    'length': 4,
                                    'width': 1.7,
                                    'height': 1.5})
                                   
            data = data.append(data_point, ignore_index=True)
        
        frame_id += 1
        '''
        #Zero-centralization per frame (sequence)
        mean_xy.append([data['x_global'].mean(),data['y_global'].mean()])
        data[-1]['x_global'] = data['x_global'] - mean_xy[-1][0]
        data['y_global'] = data['y_global'] - mean_xy[-1][1]
        '''

    #data.sort_values('frame_id', inplace=True)
    tracks_per_frame=data.groupby(['frame_id'], sort=True)
    '''
    Tracks is a list of n_frames rows ordered by frame.
    Each row contains a dict, where each key corresponds to an array of data from all agents in that frame.
    '''
    tracks = []
    for frame, track_rows in tracks_per_frame:
        #track_rows contains info of all agents in frame
        track = track_rows.to_dict(orient="list")
        
        for key, value in track.items():
            if key not in ["frame_id", "scene_id", "node_id", "sample_token"]:
                track[key] = np.array(value)
            
        track['info_sequence'] = np.stack([track["frame_id"],track["scene_id"]], axis=-1)
        track['info_agent'] = np.stack([track["type"],track["length"],track["width"],track["height"]], axis=-1)
        track["position"] = np.stack([track["x_global"], track["y_global"], track["heading"]], axis=-1)
        track['motion'] = np.stack([track["vel_x"], track["vel_y"], track["acc_x"],track["acc_y"], track["heading_change_rate"]], axis=-1)
        track["bbox"] = calculate_rotated_bboxes(track["x_global"], track["y_global"],
                                                track["length"], track["width"],
                                                np.deg2rad(track["heading"]))
    
        tracks.append(track)


    frame_id_list = list(range(tracks[0]['frame_id'][0],tracks[-1]['frame_id'][0]+1))      #list(range(data.frame_id.unique()[0], range(data.frame_id.unique()[-1])))
    assert tracks[-1]['frame_id'][0] - tracks[0]['frame_id'][0] == len(tracks)-1, f"{ tracks[-1]['frame_id'][0] - tracks[0]['frame_id'][0]} != {len(frame_id_list)-1} in scene {scene_id},{scene_token}"
    
    all_feature_list = []
    all_adjacency_list = []
    all_mean_list = []
    tokens_list = []
    maps_list = []
    visible_object_indexes_list=[]
    step=1 #iterate over <step> frames
    
    for i in range(1,len(frame_id_list[:-future_frames])):  #, frame in enumerate(frame_id_list[1:-total_frames+1:step]):
        # Avoid sequences where only ego_vehicle is present
        if len(tracks[i]['frame_id']) < 2:
            continue
        current_ind = i #start_ind + history_frames -1   #0,8,16,24
        start_ind = current_ind - history_frames
        end_ind = current_ind + future_frames
        object_frame_feature, neighbor_matrix, mean_xy, inst_sample_tokens = process_tracks(tracks, start_ind, end_ind, current_ind, nusc_map)  
        inst_sample_tokens = np.column_stack((inst_sample_tokens,location.repeat(inst_sample_tokens.shape[0])))
        '''
        #HD MAPs
        # Retrieve ego_vehicle pose
        sample_token = tracks[current_ind]['sample_token'][0]
        sample_record = nuscenes.get('sample', sample_token)     
        sample_data_record = nuscenes.get('sample_data', sample_record['data']['LIDAR_TOP'])
        poserecord = nuscenes.get('ego_pose', sample_data_record['ego_pose_token'])
        poserecord['instance_token'] = sample_data_record['ego_pose_token']

        #maps = [transform(input_representation.make_input_representation(instance, sample_token)) for instance in tracks[current_frame]["node_id"]]   #Tensor [N_agents,3,112,112] float32 [0,1]       
        maps = np.array( [input_representation.make_input_representation(instance, sample_token, poserecord, ego=False) for instance in tracks[current_ind]["node_id"][:-1]] )   #[N_agents,500,500,3] uint8 range [0,256] 
        maps = np.vstack((maps, np.expand_dims( input_representation.make_input_representation(tracks[current_ind]["node_id"][-1], sample_token, poserecord, ego=True), axis=0) ))
        maps = np.array( F.interpolate(torch.tensor(maps.transpose(0,3,1,2)), size=224) ).transpose(0,2,3,1)
        #img=((maps[0]-maps[0].min())*255/(maps[0].max()-maps[0].min())).numpy().transpose(1,2,0)
        #cv2.imwrite('input_276_0_gray'+sample_token+'.png',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        save_path_map = os.path.join(base_path_map, sample_token + '.pkl')
        with open(save_path_map, 'wb') as writer:
            pickle.dump(maps,writer)  
        '''
        all_feature_list.append(object_frame_feature)
        all_adjacency_list.append(neighbor_matrix)	
        all_mean_list.append(mean_xy)
        tokens_list.append(inst_sample_tokens)


    all_adjacency = np.array(all_adjacency_list)
    all_mean = np.array(all_mean_list)                            
    all_feature = np.array(all_feature_list)
    tokens = np.array(tokens_list, dtype=object)
    return all_feature, all_adjacency, all_mean, tokens
    

# Data splits for the CHALLENGE - returns instance and sample token  

# Train: 5883 seq (475 scenes) Train_val: 2219 seq (185 scenes)  Val: 1682 seq (138 scenes) 
ns_scene_names = dict()
ns_scene_names['train'] = get_prediction_challenge_split("train", dataroot=DATAROOT) 
ns_scene_names['val'] =  get_prediction_challenge_split("train_val", dataroot=DATAROOT)
ns_scene_names['test'] = get_prediction_challenge_split("val", dataroot=DATAROOT)


#scenes_df=[]
for data_class in ['train', 'val', 'test']:
    scenes_token_set=set()
    for ann in ns_scene_names[data_class]:
        _, sample_token=ann.split("_")
        sample = nuscenes.get('sample', sample_token)
        scenes_token_set.add(nuscenes.get('scene', sample['scene_token'])['token'])

    all_data = []
    all_adjacency = []
    all_mean_xy = []
    all_tokens = []
    #nuscenes.field2token('scene', 'name','scene-0')[0]
    for scene_token in scenes_token_set:
        ns_scene = nuscenes.get('scene', scene_token)
        scene_id = int(ns_scene['name'].replace('scene-', ''))
        print(f'Scene: {scene_id}')
        #if scene_id in scene_blacklist:  # Some scenes have bad localization
        #    continue
        
        all_feature_sc, all_adjacency_sc, all_mean_sc, tokens_sc = process_scene(nuscenes.get('scene',scene_token))
        
        print(f"Scene {nuscenes.get('scene', scene_token)['name']} processed!")# {all_adjacency_sc.shape[0]} sequences of 8 seconds.")

        all_data.extend(all_feature_sc)
        all_adjacency.extend(all_adjacency_sc)
        all_mean_xy.extend(all_mean_sc)
        all_tokens.extend(tokens_sc)
        #scenes_df.append(scene_df)
        #scene_df.to_csv(os.path.join('./nuscenes_processed/', nuscenes.get('scene', scene_token)['name'] + '.csv'))
    all_data = np.array(all_data)  
    all_adjacency = np.array(all_adjacency) 
    all_mean_xy = np.array(all_mean_xy) 
    all_tokens = np.array(all_tokens)
    save_path = '/media/14TBDISK/sandra/nuscenes_processed/ns_4s_' + data_class + '.pkl'
    with open(save_path, 'wb') as writer:
        pickle.dump([all_data, all_adjacency, all_mean_xy, all_tokens], writer)
    print(f'Processed {all_data.shape[0]} sequences and {len(scenes_token_set)} scenes.')
  
'''
#Usual split: Train 8536 (700 scenes)  Val: 1828 (150 scenes)
splits = create_splits_scenes()
ns_scene_names = dict()
ns_scene_names['train'] =  splits['train']  
ns_scene_names['val'] =  splits['val']
ns_scene_names['test'] = splits['test']

for data_class in ['val']:
    all_data=[]
    all_adjacency=[]
    all_mean_xy=[]
    all_tokens=[]
    for ns_scene_name in ns_scene_names[data_class]:
        scene_token = nuscenes.field2token('scene', 'name', ns_scene_name)
        ns_scene = nuscenes.get('scene', scene_token[0])
        scene_id = int(ns_scene['name'].replace('scene-', ''))
        if scene_id in scene_blacklist:  # Some scenes have bad localization
            continue
        process_scene(ns_scene)
        print(f"Scene {ns_scene_name} processed! ")
        
        all_feature_sc, all_adjacency_sc, all_mean_sc, tokens_sc = process_scene(ns_scene)
        print(f"Scene {ns_scene_name} processed! {all_adjacency_sc.shape[0]} sequences of 8 seconds.")
        all_data.extend(all_feature_sc)
        all_adjacency.extend(all_adjacency_sc)
        all_mean_xy.extend(all_mean_sc)
        all_tokens.extend(tokens_sc)

all_data = np.array(all_data)  
all_adjacency = np.array(all_adjacency) 
all_mean_xy = np.array(all_mean_xy) 
all_tokens = np.array(all_tokens)
save_path = '/media/14TBDISK/sandra/nuscenes_processed/nuscenes_challenge_global_step2_seq_trainandval_filter.pkl' #+ data_class + '.pkl'
with open(save_path, 'wb') as writer:
    pickle.dump([all_data, all_adjacency, all_mean_xy, all_tokens], writer)
print(f'Processed {all_data.shape[0]} sequences and {len(scenes_token_set)} scenes.')
'''


#To return the past/future data for the entire sample (local/global - in_agent_frame=T/F)
#sample_ann = helper.get_annotations_for_sample(sample_token)
#future_xy_global = helper.get_future_for_sample(sample_token, seconds=3, in_agent_frame=False)
#past_xy_global = helper.get_past_for_sample(sample_token, seconds=3, in_agent_frame=False)

# The goal of the nuScenes prediction task is to predict the future trajectories of objects in the nuScenes dataset. 
# A trajectory is a sequence of x-y locations. For this challenge, the predictions are 6-seconds long and sampled at 2 hertz (n_timesteps is 12) and 2s of history.
# The leaderboard will be ranked according to performance on the nuScenes val set. To prevent overfitting on the val set, the top 5 submissions
# on the leaderboard will be asked to send us their code and we will run their model on the test set.
# We release annotations for the train and val set, but not for the test set. We have created a hold out set for validation from 
# the training set called the train_val set.

# 1. Iterate over all scenes (train and val)
# 2. Iterate over each sample
# 3. Iterate over N instances (objects) in each sample
# df_scenes: list of 475 dataframes (475 scenes) (train) 
#     Each dataframe: 40 samples (keyframes)
#          Each sample: N instances