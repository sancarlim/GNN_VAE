
import dgl
import torch
from torch.utils.data import DataLoader
import os
import sys
sys.path.append('../../DBU_Graph')
os.environ['DGLBACKEND'] = 'pytorch'
import numpy as np
from nuscenes_Dataset import nuscenes_Dataset
from models.VAE_GNN import VAE_GNN
from models.VAE_PRIOR import VAE_GNN_prior
from models.scout import SCOUT
from models.scout_MTP import SCOUT_MTP
#from VAE_GATED import VAE_GATED
import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace
from utils import str2bool, compute_change_pos
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
import math
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patheffects as pe
from nuscenes.prediction import PredictHelper
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw, angle_diff
from utils import convert_local_coords_to_global

from scipy.ndimage import rotate

FREQUENCY = 2
history = 3
future = 5
history_frames = history*FREQUENCY + 1
future_frames = future*FREQUENCY
input_dim_model = (history_frames-1)*7 #Input features to the model: x,y-global (zero-centralized), heading,vel, accel, heading_rate, type 
output_dim = future_frames*2 +1
base_path='/media/14TBDISK/sandra/nuscenes_processed'
DATAROOT = '/media/14TBDISK/nuscenes'
nuscenes = NuScenes('v1.0-trainval', dataroot=DATAROOT)   #850 scenes

helper = PredictHelper(nuscenes)

layers = ['drivable_area',
          'road_segment',
          'lane',
          'ped_crossing',
          'walkway',
          'stop_line',
          'carpark_area',
          'stop_line',
          'road_divider',
          'lane_divider']
#layers=nusc_map.non_geometric_layers

line_colors = ['#375397', '#F05F78', '#80CBE5', '#ABCB51', '#C8B0B0']
NUM_MODES = 3

ego_car = plt.imread('/home/sandra/PROGRAMAS/DBU_Graph/NuScenes/icons/Car TOP_VIEW ROBOT.png')
cars = [plt.imread('/home/sandra/PROGRAMAS/DBU_Graph/NuScenes/icons/Car TOP_VIEW 375397.png'),
        plt.imread('/home/sandra/PROGRAMAS/DBU_Graph/NuScenes/icons/Car TOP_VIEW F05F78.png'),
        plt.imread('/home/sandra/PROGRAMAS/DBU_Graph/NuScenes/icons/Car TOP_VIEW 80CBE5.png'),
        plt.imread('/home/sandra/PROGRAMAS/DBU_Graph/NuScenes/icons/Car TOP_VIEW ABCB51.png'),
        plt.imread('/home/sandra/PROGRAMAS/DBU_Graph/NuScenes/icons/Car TOP_VIEW C8B0B0.png')]

scene_blacklist = [499, 515, 517]

patch_margin = 50
min_diff_patch = 50


def collate_batch(samples):
    graphs, masks, feats, gt, tokens, scene_ids, mean_xy, maps, global_feats = map(list, zip(*samples))  # samples is a list of pairs (graph, mask) mask es VxTx1
    masks = torch.vstack(masks)
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
    return batched_graph, masks, snorm_n, snorm_e, feats, gt, tokens[0], scene_ids[0], mean_xy, maps, global_feats



class LitGNN(pl.LightningModule):
    def __init__(self, model, model_type, train_dataset, val_dataset, test_dataset, history_frames: int=3, future_frames: int=3, lr: float = 1e-3, 
                    batch_size: int = 64, wd: float = 1e-1, beta: float = 0., delta: float = 1., rel_types: bool = False, 
                    scale_factor=1, scene_id : int = 927, sample : str = None, ckpt : str = None):
        super().__init__()
        self.model= model
        self.history_frames =history_frames
        self.future_frames = future_frames
        self.total_frames = history_frames + future_frames
        self.test_dataset = test_dataset
        self.rel_types = rel_types
        self.scale_factor = scale_factor
        self.scene_id = scene_id
        self.model_type = model_type
        self.sample = sample
        self.cnt = 0
        self.scene = 0
        self.ckpt = ckpt

    def forward(self, graph, feats,e_w,snorm_n,snorm_e):
        # in lightning, forward defines the prediction/inference actions
        pred = self.model.inference(graph, feats,e_w,snorm_n,snorm_e)   
        return pred
    
    def configure_optimizers(self):
        pass
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=12, collate_fn=collate_batch) 
    
    def training_step(self, train_batch, batch_idx):
        pass

    def validation_step(self, val_batch, batch_idx):
        pass
         
    def test_step(self, test_batch, batch_idx):
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos, tokens_eval, scene_id, mean_xy, maps, global_feats = test_batch
        if scene_id != self.scene:
            self.scene = scene_id
            self.cnt = 0 
        sample_token = tokens_eval[0][1]
        
        if self.scene_id is not None and scene_id != self.scene_id:
            return 
        if self.sample is not None and  sample_token != self.sample:
            return
        print(tokens_eval)
        
        rescale_xy=torch.ones((1,1,2), device=self.device)*self.scale_factor
        last_loc = feats[:,-1:,:2].detach().clone() 
        
        feats_vel, labels_vel = compute_change_pos(feats,labels_pos[:,:,:2], hparams.local_frame)
        if not hparams.local_frame:
            feats = torch.cat([feats_vel, feats[:,:,2:]], dim=-1)[:,1:]        
        
        if self.scale_factor == 1:
            pass#last_loc = last_loc*12.4354+0.1579
        else:
            last_loc = last_loc*rescale_xy     
        e_w = batched_graph.edata['w'].float()
        if not self.rel_types:
            e_w= e_w.unsqueeze(1)
        
        # Prediction: Prediction of model [num_modes, n_timesteps, state_dim] = [25, 12, 2]
        prediction_all_agents = []  # [num_agents, num_modes, n_timesteps, state_dim]
        if self.model_type == 'mtp':
            pred = self.model(batched_graph, feats,e_w,snorm_n,snorm_e, maps)
            mode_probs = pred[:, -NUM_MODES:].clone()
            desired_shape = (pred.shape[0], NUM_MODES, -1, 2)
            prediction_all_agents = pred[:, :-NUM_MODES].clone().reshape(desired_shape)
            for j in range(1,labels_pos.shape[1]):
                prediction_all_agents[:,:,j,:] = torch.sum(prediction_all_agents[:,:,j-1:j+1,:],dim=-2) 

            #best_mode = np.argmax(mode_prob.detach().cpu().numpy(), axis = 1)
            #pred = torch.zeros_like(labels)
            #for i, idx in enumerate(best_mode):
            #    pred[i] = trajectories_no_modes[i,idx] 
            
        else:
            for i in range(5):
                #Model predicts relative_positions
                if self.model_type == 'vae_prior':
                    preds, mode_probs, KL_terms, z_sample  = self.model.inference(batched_graph, feats,e_w,snorm_n,snorm_e, maps) 
                    preds=preds.view(preds[:,:-1].shape[0],self.future_frames,-1)[:,:,:2]
                else:
                    preds = self.model.inference(batched_graph, feats,e_w,snorm_n,snorm_e,maps)
                    preds=preds.view(preds.shape[0],self.future_frames,-1)  
                
                if not hparams.local_frame:
                    #Convert prediction to absolute positions
                    for j in range(1,labels_pos.shape[1]):
                        preds[:,j,:] = torch.sum(preds[:,j-1:j+1,:],dim=-2) #6,2 
                    preds += last_loc
            
                # Provide predictions in global-coordinates
                pred_x = preds[:,:,0].cpu().numpy() + mean_xy[0][0]  # [N_agents, T]
                pred_y = preds[:,:,1].cpu().numpy() + mean_xy[0][1]
                
                prediction_all_agents.append(np.stack([pred_x, pred_y],axis=-1))
            prediction_all_agents = np.array(prediction_all_agents)  
              
        

        #VISUALIZE SEQUENCE
        #Get Scene from sample token ie current frame
        scene=nuscenes.get('scene', nuscenes.get('sample',sample_token)['scene_token'])
        scene_name = scene['name']
        scene_id = int(scene_name.replace('scene-', ''))
        if scene_id in scene_blacklist:
            print('Warning: %s is known to have a bad fit between ego pose and map.' % scene_name)
              
        log=nuscenes.get('log', scene['log_token'])
        location = log['location']
        nusc_map = NuScenesMap(dataroot=DATAROOT, map_name=location)

        #Render map with ego poses
        #sample_tokens = nuscenes.field2token('sample', 'scene_token', scene['token'])
        #ego_poses=[]
        #for sample_token in sample_tokens:
        
        sample_record = nuscenes.get('sample', sample_token)
            
        # Poses are associated with the sample_data. Here we use the lidar sample_data.
        sample_data_record = nuscenes.get('sample_data', sample_record['data']['LIDAR_TOP'])
        pose_record = nuscenes.get('ego_pose', sample_data_record['ego_pose_token'])

        # Calculate the pose on the map and append.
        ego_poses=np.array(pose_record['translation'][:2])
        # Check that ego poses aren't empty.
        #assert len(ego_poses) > 0, 'Error: Found 0 ego poses. Please check the inputs.'
        #ego_poses = np.vstack(ego_poses)[:, :2]
        
        # Render the map patch with the current ego poses.
        min_patch = np.floor(ego_poses - patch_margin)
        max_patch = np.ceil(ego_poses + patch_margin)
        diff_patch = max_patch - min_patch
        if any(diff_patch < min_diff_patch):
            center_patch = (min_patch + max_patch) / 2
            diff_patch = np.maximum(diff_patch, min_diff_patch)
            min_patch = center_patch - diff_patch / 2
            max_patch = center_patch + diff_patch / 2
        my_patch = (min_patch[0], min_patch[1], max_patch[0], max_patch[1])
        
        fig, ax = nusc_map.render_map_patch(my_patch, layers, figsize=(10, 10), alpha=0.3,
                                    render_egoposes_range=False,
                                    render_legend=True, bitmap=None)

        r_img = rotate(ego_car, quaternion_yaw(Quaternion(pose_record['rotation']))*180/math.pi,reshape=True)
        oi = OffsetImage(r_img, zoom=0.02, zorder=500)
        veh_box = AnnotationBbox(oi, (ego_poses[0], ego_poses[1]), frameon=False)
        veh_box.zorder = 500
        ax.add_artist(veh_box)
        #Print agents trajectories
        i = 0
        for token in tokens_eval[:-1]:
            idx = np.where(np.array(tokens_eval)== token[0])[0][0]  #idx ordered checked
            instance, sample_token = token
            annotation = helper.get_sample_annotation(instance, sample_token)
            category = annotation['category_name'].split('.')
            attribute = nuscenes.get('attribute', annotation['attribute_tokens'][0])['name']
            history = global_feats[idx,:self.history_frames,:2].cpu().numpy()
            

            if self.model_type == 'mtp':
                prediction = prediction_all_agents[idx, :]
                prediction = torch.tensor([ convert_local_coords_to_global(prediction[i].cpu().numpy(), annotation['translation'], annotation['rotation']) for i in range(prediction.shape[0])])
                probs = mode_probs[idx]
            else:
                prediction = prediction_all_agents[:,idx, :]
                if hparams.local_frame:
                    prediction = torch.tensor([convert_local_coords_to_global(prediction[i].cpu().numpy(), annotation['translation'], annotation['rotation']) for i in range(prediction.shape[0])])
                #probs = probs[idx]
            if self.scale_factor == 1:
                pass#history = history*12.4354+0.1579
            else:
                history = history*rescale_xy   
            
            #remove zero rows (no data in those frames) and rescale to obtain global coords.
            history = (history[history.all(axis=1)] + mean_xy[0])
            future = global_feats[idx, self.history_frames:, :2].cpu().numpy() #labels_pos[idx].cpu().numpy()
            future = future[future.all(axis=1)] + mean_xy[0]
            if history.shape[0] < 2:
                if history.shape[0] == 0:
                    history = np.array(annotation['translation'][:2])
                    if history.all() != mean_xy[0].all():
                        print(f'WATCH OUT!')

                history=np.vstack([history, history])
            if future.shape[0] == 1:
                future=np.vstack([future, future])
            
            # Plot predictions
            if category[0] != 'vehicle':
                if 'sitting_lying_down' not in attribute:
                    if self.model_type == 'scout':
                        ax.plot(prediction[0, :, 0], prediction[0, :, 1], 'bo-',
                                zorder=620,
                                markersize=2,
                                linewidth=1, alpha=0.7)
                    elif self.model_type == 'mtp' or self.model_type=='vae_prior':
                        for sample_num in range(prediction.shape[0]):
                            ax.plot(prediction[sample_num, :, 0], prediction[sample_num, :, 1], 'bo-',
                                    zorder=620,
                                    markersize=2,
                                    linestyle = '--',
                                    linewidth=1, alpha=0.7) #0.8+probs[sample_num]
                    else:
                        for t in range(prediction.shape[1]):
                            try:
                                sns.kdeplot(x=prediction[:,t,0], y=prediction[:,t,1],
                                    ax=ax, shade=True, thresh=0.05, 
                                    color='b', zorder=600, alpha=0.8)
                            except:
                                print('2-th leading minor of the array is not positive definite.',  sys.exc_info()[0], 'ocurred.' )
                                continue                    
            
            else:  
                if 'parked' not in attribute:
                    
                    if self.model_type == 'scout':
                        ax.plot(prediction[0, :, 0], prediction[0, :, 1], 'mo-',
                                zorder=620,
                                markersize=3,
                                linewidth=2, alpha=0.7)
                    elif self.model_type == 'mtp' or self.model_type=='vae_prior':
                        for sample_num in range(prediction.shape[0]):
                            #color_sample = [line_colors[i % len(line_colors)] if probs[sample_num] == max(probs) else 'b'][0]
                            ax.plot(prediction[sample_num, :, 0], prediction[sample_num, :, 1], 
                                    zorder=620,
                                    color='b',
                                    linestyle = '--',
                                    marker= 'o',
                                    markersize=2,#probs[sample_num]*3,
                                    linewidth=1, alpha=1) #0.8 + probs[sample_num]
                    else:
                        for t in range(prediction.shape[1]):
                            try:
                                sns.kdeplot(x=prediction[:,t,0], y=prediction[:,t,1],
                                    ax=ax, thresh=0.05, shade=True,
                                    color=line_colors[i % len(line_colors)], zorder=600, alpha=1)  #shade True
                            except:
                                print('2-th leading minor of the array is not positive definite')
                                continue
                    
            
            #Plot history
            ax.plot(history[:, 0], 
                    history[:, 1], 
                    'k--')

            #Plot ground truth
            if future.shape[0] > 0:
                ax.plot(future[:, 0],
                        future[:, 1],
                        'w--',
                        label='Ground Truth',
                        zorder=650,
                        path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])
                
            # Current Node Position
            node_circle_size=0.3
            circle_edge_width=0.5
            if category[1] == 'motorcycle' or category[1] == 'bicycle':
                circle = plt.Circle((history[-1, 0],
                                history[-1, 1]),
                                node_circle_size,
                                facecolor='y',
                                edgecolor='y',
                                lw=circle_edge_width,
                                zorder=3)
                ax.add_artist(circle)
            elif category[0] == 'vehicle': 
                r_img = rotate(cars[i % len(cars)], quaternion_yaw(Quaternion(annotation['rotation']))*180/math.pi,reshape=True)
                oi = OffsetImage(r_img, zoom=0.01, zorder=500)
                veh_box = AnnotationBbox(oi, (history[-1, 0], history[-1, 1]), frameon=False)
                veh_box.zorder = 500
                ax.add_artist(veh_box)
                i += 1
            else:
                circle = plt.Circle((history[-1, 0],
                                history[-1, 1]),
                                node_circle_size,
                                facecolor='c',
                                edgecolor='c',
                                lw=circle_edge_width,
                                zorder=3)
                ax.add_artist(circle)

            
        
        #ax.axis('off')
        run_name = self.ckpt.split('/')[5].split('-')[0]
        fig.savefig(os.path.join(base_path, 'visualizations' , scene_name + '_MTP_3s_' + run_name + str(self.cnt) + '_' + sample_token + '.jpg'), dpi=300, bbox_inches='tight')
        print('Image saved in: ', os.path.join(base_path, 'visualizations' , scene_name + '_MTP_' + str(self.cnt) + '_' + sample_token + '.jpg'))
        plt.clf()
        self.cnt += 1
   
def main(args: Namespace):
    print(args)

    test_dataset = nuscenes_Dataset(train_val_test='test', rel_types=args.ew_dims>1, history_frames=history_frames, future_frames=future_frames, 
                                    challenge_eval=True, local_frame = args.local_frame)  #230

    if args.model_type == 'vae_gated':
        model = VAE_GATED(input_dim_model, args.hidden_dims, z_dim=args.z_dims, output_dim=output_dim, fc=False, dropout=args.dropout,  ew_dims=args.ew_dims)
    elif  args.model_type == 'vae_gat':
        model = VAE_GNN(input_dim_model, args.hidden_dims//args.heads, args.z_dims, output_dim, fc=False, dropout=args.dropout, 
                        feat_drop=args.feat_drop, attn_drop=args.attn_drop, heads=args.heads, att_ew=args.att_ew, 
                        ew_dims=args.ew_dims, backbone=args.backbone)
    elif args.model_type == 'vae_prior':
        input_dim_model =  7 * (history_frames-1) if args.enc_type == 'emb' else 7
        model = VAE_GNN_prior(input_dim_model, args.hidden_dims, args.z_dims, output_dim, fc=False, dropout=args.dropout, feat_drop=args.feat_drop,
                        attn_drop=args.attn_drop, heads=args.heads, att_ew=args.att_ew, ew_dims=args.ew_dims, backbone=args.backbone, freeze=args.freeze,
                        bn=(args.norm=='bn'), gn=(args.norm=='gn'), encoding_type=args.enc_type)
    elif args.model_type == 'mtp':
        input_dim_model = 7 * (history_frames-1) if args.emb_type == 'emb' else 7
        model = SCOUT_MTP(input_dim=input_dim_model, hidden_dim=args.hidden_dims, emb_dim=args.emb_dims, output_dim=output_dim, heads=args.heads, dropout=args.dropout, 
                        feat_drop=args.feat_drop, attn_drop=args.attn_drop, att_ew=args.att_ew, ew_dims=args.ew_dims>1, backbone=args.backbone,
                        num_modes = NUM_MODES, history_frames=history_frames-1)
    else:
        model = SCOUT(input_dim=input_dim_model, hidden_dim=args.hidden_dims, output_dim=output_dim, heads=args.heads, dropout=args.dropout, 
                        feat_drop=args.feat_drop, attn_drop=args.attn_drop, att_ew=args.att_ew, ew_dims=args.ew_dims>1, backbone=args.backbone)
    

    LitGNN_sys = LitGNN(model=model,  model_type = args.model_type,history_frames=history_frames, future_frames= future_frames, train_dataset=None, val_dataset=None,
                 test_dataset=test_dataset, rel_types=args.ew_dims>1, scale_factor=args.scale_factor, scene_id=args.scene_id)
      
    trainer = pl.Trainer(gpus=1, deterministic=True) 
 
    LitGNN_sys = LitGNN.load_from_checkpoint(checkpoint_path=args.ckpt, model=LitGNN_sys.model, model_type = args.model_type, history_frames=history_frames, future_frames= future_frames,
                    train_dataset=None, val_dataset=None, test_dataset=test_dataset, rel_types=args.ew_dims>1, scale_factor=args.scale_factor, scene_id=args.scene_id, sample = args.sample,
                    ckpt = args.ckpt)


    trainer.test(LitGNN_sys)
   

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--scale_factor", type=int, default=1, help="Wether to scale x,y global positions (zero-centralized)")
    parser.add_argument("--ew_dims", type=int, default=2, choices=[1,2], help="Edge features: 1 for relative position, 2 for adding relationship type.")
    parser.add_argument("--z_dims", type=int, default=128, help="Dimensionality of the latent space")
    parser.add_argument("--hidden_dims", type=int, default=768)
    parser.add_argument("--model_type", type=str, default='mtp', help="Choose aggregation function between GAT or GATED",
                                        choices=['vae_gat', 'vae_gated', 'vae_prior', 'scout', 'mtp'])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--feat_drop", type=float, default=0.)
    parser.add_argument("--attn_drop", type=float, default=0.4)
    parser.add_argument("--heads", type=int, default=2, help='Attention heads (GAT)')
    parser.add_argument('--att_ew', type=str2bool, nargs='?', const=True, default=True, help="Add edge features in attention function (GAT)")
    parser.add_argument('--ckpt', type=str, default='/media/14TBDISK/sandra/logs/NuScenes VAE/dark-deluge-1630/epoch=22-step=3081.ckpt', help='ckpt path.')  
    parser.add_argument("--norm", type=str, default=None, help="Wether to apply BN (bn) or GroupNorm (gn).")
    parser.add_argument("--enc_type", type=str, default='emb', choices=['emb',  'gru'])
    parser.add_argument("--emb_dims", type=int, default=512)
    
    parser.add_argument('--maps', type=str2bool, nargs='?', const=True, default=True, help="Add HD Maps.")
    parser.add_argument('--local_frame',  type=str2bool, nargs='?', const=True, default=False, help='whether to use local or global features.')  
    parser.add_argument("--emb_type", type=str, default='emb', choices=['emb', 'pos_enc', 'gru'])
    
    parser.add_argument("--backbone", type=str, default='resnet18', help="Choose CNN backbone.",
                                        choices=['resnet_gray', 'mobilenet', 'resnet18','resnet50', 'map_encoder'])
    parser.add_argument("--scene_id", type=int, default=105, help="Scene id to visualize.")
    parser.add_argument("--sample", type=str, default=None, help="sample to visualize.")
    parser.add_argument('--nowandb', action='store_true', help='use this flag to DISABLE wandb logging')  
    
    hparams = parser.parse_args()

    main(hparams)




