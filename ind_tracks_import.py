import pandas
import glob
import numpy as np
import os
from scipy import spatial 
import pickle


neighbor_distance = 20
max_num_object = 70 #per frame
total_feature_dimension = 12 #pos,heading,vel,recording_id,frame,id, l,w, class, mask

def read_all_recordings_from_csv(base_path="../data/"):
    """
    This methods reads the tracks and meta information for all recordings given the path of the inD dataset.
    :param base_path: Directory containing all csv files of the inD dataset
    :return: a tuple of tracks, static track info and recording meta info
    """
    tracks_files = sorted(glob.glob(base_path + "*_tracks.csv"))
    static_tracks_files = sorted(glob.glob(base_path + "*_tracksMeta.csv"))
    recording_meta_files = sorted(glob.glob(base_path + "*_recordingMeta.csv"))

    all_tracks = []
    all_static_info = []
    all_meta_info = []
    for track_file, static_tracks_file, recording_meta_file in zip(tracks_files,
                                                                   static_tracks_files,
                                                                   recording_meta_files):
        logger.info("Loading csv files {}, {} and {}", track_file, static_tracks_file, recording_meta_file)
        tracks, static_info, meta_info = read_from_csv(track_file, static_tracks_file, recording_meta_file)
        all_tracks.extend(tracks)
        all_static_info.extend(static_info)
        all_meta_info.extend(meta_info)

    return all_tracks, all_static_info, all_meta_info


def read_from_csv(track_file, static_tracks_file, recordings_meta_file):
    """
    This method reads tracks including meta data for a single recording from csv files.
    :param track_file: The input path for the tracks csv file.
    :param static_tracks_file: The input path for the static tracks csv file.
    :param recordings_meta_file: The input path for the recording meta csv file.
    :return: tracks, static track info and recording info
    """
    static_info = read_static_info(static_tracks_file)
    meta_info = read_meta_info(recordings_meta_file)
    tracks = read_tracks(track_file, static_tracks_file)
    return tracks, static_info, meta_info


def read_tracks(track_file, static_info):
    # Read the csv file to a pandas dataframe
    df = pandas.read_csv(track_file)
    df_meta = pandas.read_csv(static_info)
    
    #filter some of the parked vehicles , EXCEPT FOR VISUALIZATION
    
    if test == False:
        max_num_frames = df_meta['numFrames'].max()
        id_parked_objects = list(df_meta[df_meta['numFrames']==max_num_frames].trackId)
        del id_parked_objects[-10:]  #keep 10 parked cars
        df = df[~df['trackId'].isin(id_parked_objects)]
        '''
        #filter out no-car or ped objects
        list_car_obj = list(df_meta[df_meta['class']=='car'].trackId)
        list_ped_obj = list(df_meta[df_meta['class']=='pedestrian'].trackId)
        list_car_obj.extend(list_ped_obj)
        df = df[df['trackId'].isin(list_car_obj)]   #[~df['trackId'].isin(list_no_car_obj)]
        '''

    # To extract every track, group the rows by the track id
    raw_tracks = df.groupby(["frame"], sort=True)
    #ortho_px_to_meter = meta_info["orthoPxToMeter"]
    tracks = []
    for frame, track_rows in raw_tracks:
        track = track_rows.to_dict(orient="list")

        # Convert scalars to single value and lists to numpy arrays
        for key, value in track.items():
            if key in ["frame", "track_Lifetime","recordingId"]:
                track[key] = value[0]
            else:
                track[key] = np.array(value)

        track['info_frame'] = np.array((track["recordingId"], track["frame"]))
        track['info_agent'] = np.stack([track["trackId"],track["length"],track["width"]], axis=-1)
        track["position"] = np.stack([track["xCenter"], track["yCenter"], np.deg2rad(track["heading"])], axis=-1)
        track["velocity"] = np.stack([track["xVelocity"], track["yVelocity"]], axis=-1)
        track["bbox"] = calculate_rotated_bboxes(track["xCenter"], track["yCenter"],
                                                 track["length"], track["width"],
                                                 np.deg2rad(track["heading"]))
        '''
        # Create special version of some values needed for visualization
        track["xCenterVis"] = track["xCenter"] / ortho_px_to_meter
        track["yCenterVis"] = -track["yCenter"] / ortho_px_to_meter
        track["centerVis"] = np.stack([track["xCenter"], -track["yCenter"]], axis=-1) / ortho_px_to_meter
        track["widthVis"] = track["width"] / ortho_px_to_meter
        track["lengthVis"] = track["length"] / ortho_px_to_meter
        track["headingVis"] = track["heading"] * -1
        track["headingVis"][track["headingVis"] < 0] += 360
        track["bboxVis"] = calculate_rotated_bboxes(track["xCenterVis"], track["yCenterVis"],
                                                    track["lengthVis"], track["widthVis"],
                                                    np.deg2rad(track["headingVis"]))
        '''
        #Keep only frames to 1Hz -> 25 ,  2.5Hz ->10, 1.5 -> 20
        ratio = 10 if herz==2.5 else 25
        if frame%ratio == 0:
            tracks.append(track)
    return tracks


def read_static_info(static_tracks_file):
    """
    This method reads the static info file from highD data.
    :param static_tracks_file: the input path for the static csv file.
    :return: the static dictionary - the key is the track_id and the value is the corresponding data for this track
    """
    df = pandas.read_csv(static_tracks_file)
    mapping_dict={
        'class':{
            'car':1,
            'pedestrian':2,
            'truck_bus':3, #truck_bus
            'bicycle':4,
            #'van': 1,
            #'motorcycle': 4,
            #'trailer': 3,
            #'bus': 3
        }
    }

    df=df.replace(mapping_dict)

    return df.to_dict(orient="records")


def read_meta_info(recordings_meta_file):
    """
    This method reads the recording info file from ind data.
    :param recordings_meta_file: the path for the recording meta csv file.
    :return: the meta dictionary
    """
    return pandas.read_csv(recordings_meta_file).to_dict(orient="records")[0]


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

def process_data(tracks,static_info, start_ind, end_ind, observed_last):
    #tracks es una lista de dict donde cada fila es un frame
    #static info es una lista de dict [{column->value},{},...]
    visible_object_id_list = tracks[observed_last]["trackId"] # object_id appears at the last observed frame
    #para ver los obj visibles en esa secuencia miramos el final de la sec
    num_visible_object = visible_object_id_list.size # number of current observed objects

    # compute the mean values of x and y (of all obj detected) for zero-centralization. 
    visible_object_value = tracks[observed_last]['position']
    xy = visible_object_value[:, :2].astype(float)   #x,y
    mean_xy = np.zeros_like(visible_object_value[0], dtype=float)
    m_xy = np.mean(xy, axis=0)
    mean_xy[:2] = m_xy

    # compute distance between any pair of two objects
    dist_xy = spatial.distance.cdist(xy, xy)  #nxn matrix with relative distances
    # if their distance is less than $neighbor_distance, we regard them are neighbors.
    neighbor_matrix = np.zeros((max_num_object, max_num_object))
    neighbor_matrix[:num_visible_object, :num_visible_object] = (dist_xy<neighbor_distance).astype(int)

    now_all_object_id = set([val for frame in range(start_ind, end_ind) for val in tracks[frame]["trackId"]])  #todos los obj en los15 hist frames
    non_visible_object_id_list = list(now_all_object_id - set(visible_object_id_list))  #obj en alguno de los 15 frames pero no el ultimo
    num_non_visible_object = len(non_visible_object_id_list)

    # for all history frames or future frames, we only choose the objects listed in visible_object_id_list
    object_feature_list = []
    # non_visible_object_feature_list = []
    for frame_ind in range(start_ind, end_ind):	
        # we add mark "1" to the end of each row to indicate that this row exists, using list(pra_now_dict[frame_ind][obj_id])+[1] 
        # -mean_xy is used to zero_centralize data
        # now_frame_feature_dict = {obj_id : list(pra_now_dict[frame_ind][obj_id]-mean_xy)+[1] for obj_id in pra_now_dict[frame_ind] if obj_id in visible_object_id_list}
        now_frame_feature_dict = {obj_id : (list(tracks[frame_ind]['position'][int(np.where(tracks[frame_ind]['trackId']==(obj_id))[0])]-mean_xy)+ list(tracks[frame_ind]['velocity'][int(np.where(tracks[frame_ind]['trackId']==(obj_id))[0])]) + list(tracks[frame_ind]['info_frame'])  + list(tracks[frame_ind]['info_agent'][int(np.where(tracks[frame_ind]['trackId']==(obj_id))[0])])+ [static_info[obj_id]['class']] + [1]
         if obj_id in visible_object_id_list else list(tracks[frame_ind]['position'][int(np.where(tracks[frame_ind]['trackId']==(obj_id))[0])]-mean_xy) + list(tracks[frame_ind]['velocity'][int(np.where(tracks[frame_ind]['trackId']==(obj_id))[0])]) + list(tracks[frame_ind]['info_frame'])  + list(tracks[frame_ind]['info_agent'][int(np.where(tracks[frame_ind]['trackId']==(obj_id))[0])]) + [static_info[obj_id]['class']] + [0]) 
         for obj_id in tracks[frame_ind]["trackId"] }
        # if the current object is not at this frame, we return all 0s by using dict.get(_, np.zeros(11))
        now_frame_feature = np.array([now_frame_feature_dict.get(vis_id, np.zeros(total_feature_dimension)) for vis_id in now_all_object_id])
        object_feature_list.append(now_frame_feature)

    # object_feature_list has shape of (frame#, object#, 11) 11 = 10features + 1mark
    object_feature_list = np.array(object_feature_list)

    # object feature with a shape of (frame#, object#, 6) -> (V, T, C)
    object_frame_feature = np.zeros((max_num_object, end_ind-start_ind, total_feature_dimension))
    object_frame_feature[:num_visible_object+num_non_visible_object] = np.transpose(object_feature_list, (1,0,2))
    visible_object_indexes = [list(now_all_object_id).index(i) for i in visible_object_id_list]
    return object_frame_feature, neighbor_matrix, m_xy, visible_object_indexes

def generate_train_data(file_track_path, file_static_path):
    '''
    Read data from $file_path, and split data into clips with $total_frames length (6+6). 
    Return: feature and adjacency_matrix
        feature: (N, C, T, V) 
            N is the number of sequences in file_path 
            C is the dimension of features, 10raw_feature + 1mark(valid data or not)
            T is the temporal length of the data. history_frames + future_frames
            V is the maximum number of objects. zero-padding for less objects. 
    '''
    tracks = read_tracks(file_track_path,file_static_path)
    static_info = read_static_info(file_static_path)
    #maximum_frames = np.max([static_info[track["trackId"]]["finalFrame"] for track in tracks])
    frame_id_set = list(range(len(tracks))) #list con todos frames

    all_feature_list = []
    all_adjacency_list = []
    all_mean_list = []
    visible_object_indexes_list=[]
    step = 8
    for start_ind in frame_id_set[:-total_frames+1:step]:  #[:-total_frames+1:2]#recorre el fichero dividiendo los datos en clips de 8+8 frames a 2.5Hz
        start_ind = int(start_ind)
        end_ind = int(start_ind + total_frames)
        observed_last = start_ind + (history_frames-1)
        object_frame_feature, neighbor_matrix, mean_xy, visible_object_indexes = process_data(tracks, static_info, start_ind, end_ind, observed_last)  #N=1

        all_feature_list.append(object_frame_feature)
        all_adjacency_list.append(neighbor_matrix)	
        all_mean_list.append(mean_xy)
        visible_object_indexes_list.append(visible_object_indexes)

    # (N, V, T, C) --> (N, C, T, V)
    all_feature_list = np.transpose(all_feature_list, (0, 3, 2, 1))
    all_adjacency_list = np.array(all_adjacency_list)
    all_mean_list = np.array(all_mean_list)
    visible_object_indexes_list = np.array(visible_object_indexes_list)
    print(all_feature_list.shape)   #N= nº de secuencias (10 frames) en cada fichero - nºtotal=Nx*nºficheros
    return all_feature_list, all_adjacency_list, all_mean_list,visible_object_indexes_list


def generate_data(file_tracks_list, file_static_list):
    all_data = []
    all_adjacency = []
    all_mean_xy = []
    all_visible_object_indexes=[]

    for track_file, static_file in zip(file_tracks_list, file_static_list):
        now_data, now_adjacency, now_mean_xy, now_visible_object_indexes = generate_train_data(track_file,static_file)
        all_data.extend(now_data)
        all_adjacency.extend(now_adjacency)
        all_mean_xy.extend(now_mean_xy)
        all_visible_object_indexes.extend(now_visible_object_indexes)

    all_data = np.array(all_data) #(N, C, T, V)=(5010, 11, 12, 70) Train
    all_adjacency = np.array(all_adjacency) #(5010, 70, 70) Train
    all_mean_xy = np.array(all_mean_xy) #(5010, 2) Train  MEDIAS xy de cada secuencia de 12 frames
    all_visible_object_indexes = np.array(all_visible_object_indexes)  
    print(all_data.shape[0])
    save_path = '/media/14TBDISK/sandra/inD_processed/inD_test_25m.pkl'
    with open(save_path, 'wb') as writer:
        pickle.dump([all_data, all_adjacency, all_mean_xy, all_visible_object_indexes], writer)
    print('Data successfully saved.')


if __name__ == '__main__':

    test = False
    herz = 2.5
    history_frames = 3 if herz==1 else 8 # 5 second * 1 frame/second
    future_frames = 5 if herz==1 else 12 # 5 second * 1 frame/second
    total_frames = history_frames + future_frames
    input_root_path = '/media/14TBDISK/inD/test_data/'
    tracks_files = sorted(glob.glob(os.path.join(input_root_path , "*_tracks.csv")))
    static_tracks_files = sorted(glob.glob(os.path.join(input_root_path , "*_tracksMeta.csv")))
    recording_meta_files = sorted(glob.glob(os.path.join(input_root_path , "*_recordingMeta.csv")))
    generate_data(tracks_files, static_tracks_files)
    
    
    