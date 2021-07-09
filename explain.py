import dgl
import torch
from torch.utils.data import DataLoader
import os
import sys
sys.path.append('../../DBU_Graph')
os.environ['DGLBACKEND'] = 'pytorch'
import numpy as np
from NuScenes.nuscenes_Dataset import nuscenes_Dataset, collate_batch_test
from models.VAE_GNN import VAE_GNN
from models.VAE_PRIOR import VAE_GNN_prior
from models.scout import SCOUT
from models.scout_MTP import SCOUT_MTP
#from VAE_GATED import VAE_GATED
import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace
from utils import str2bool, compute_change_pos, MTPLoss
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from nuscenes.prediction import PredictHelper
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw, angle_diff
from utils import convert_local_coords_to_global
import efemarai as ef
from captum.attr import DeepLiftShap, DeepLift, FeaturePermutation, LayerDeepLift
import shap
from main_pylightning import LitGNN

FREQUENCY = 2
history = 3
future = 5
history_frames = history*FREQUENCY + 1
future_frames = future*FREQUENCY
input_dim_model = (history_frames-1)*7 #Input features to the model: x,y-global (zero-centralized), heading,vel, accel, heading_rate, type 
output_dim = future_frames*2 +1



def explain(model, dataloader,mtp_loss,args):
    dl = DeepLift(model)
    ldl = LayerDeepLift(model, model.embeddings['embedding_h'])
    feature_perm = FeaturePermutation(model)
    g = dgl.graph(([0, 0, 0, 1, 2, 1, 1], [0, 1, 2, 0, 1, 2, 0]))
    e_w = torch.rand(7, 2, requires_grad=True)
    feats = torch.rand(3, history_frames-1, 7, requires_grad=True)
    maps = torch.rand(3, 3, 112, 112, requires_grad=True)

    model.to('cuda:0').eval()
    with torch.no_grad():
        for batch in dataloader:
            g, output_masks,snorm_n, snorm_e, feats, labels, tokens,  scene, mean_xy, maps ,global_feats = batch
            if scene == args.scene_id:
                if feats[0][0].any():
                    #ef.inspect(maps, view=ef.View.Image, name=str(scene))
                    #ef.inspect(feats, name='Features')
                    e_w = g.edata['w']#.unsqueeze(1)
                    feats_vel, labels_vel = compute_change_pos(feats, labels[:,:,:2], hparams.local_frame)
                    last_loc = feats[:,-1:,:2]
                    if not hparams.local_frame:
                        feats = torch.cat([feats_vel, feats[:,:,2:]], dim=-1)[:,1:]
                    #ef.inspect(feats, name='Local Features')
                    out = model(feats.to('cuda:0'),e_w.to('cuda:0'), maps.to('cuda:0'), g.to('cuda:0'))
                    
                    pred = mtp_loss(out, feats[:,5,5], global_feats[:,history_frames:,:2].to('cuda:0').unsqueeze(1), last_loc.unsqueeze(1), output_masks.to('cuda:0').unsqueeze(1), 
                                    hparams.local_frame, tokens[:,-1], tokens[:,-2], global_feats[:,history_frames-1].to('cuda:0'))

                    pred = torch.argmax(out[:,-3:], dim=1)  #max prob
                    targets = 18 * (pred + 1) + pred*3  #18 39 60
                    attribution = dl.attribute((feats, e_w), additional_forward_args=(maps, g, True), target=targets) #target de los 63 outputs si 3 modes (20*3 + 3), 18 es x de mode 1 t=5s
                    layer_attr = ldl.attribute((feats, e_w), additional_forward_args=(maps, g, True), target=targets)  
                    feature_mask = torch.arange(6).repeat(7).view(1,7,6).transpose(1,2)
                    attr_feat_perm = feature_perm.attribute((feats), additional_forward_args=(e_w, maps, g), feature_mask=feature_mask, target=targets) 
                    
                    ef.inspect(layer_attr, name='attribution Layer deepLift')
                    ef.inspect(attribution[0], name='attribution FEATS deepLift')
                    ef.inspect(attribution[1], name='attribution EDGES deepLift')
                    ef.inspect(attr_feat_perm, name='attribution feature permutation', wait=True )
        '''
        out = model(feats, e_w,  maps, g, attr=False)
        trajectories_no_modes = out[:, :-3].clone().reshape(desired_shape)
        # we use the first 100 training examples as our background dataset to integrate over
        explainer = shap.DeepExplainer(model, dataloader[:100])

        # explain the first 10 predictions
        # explaining each prediction requires 2 * background dataset size runs
        shap_values = explainer.shap_values(dataloader[:10])
        # init the JS visualization code
        shap.initjs()
        '''

def main(args: Namespace):
    print(args)

    test_dataset = nuscenes_Dataset(train_val_test='train', rel_types=args.ew_dims>1, history_frames=history_frames, future_frames=future_frames, 
                                    challenge_eval=True, local_frame = args.local_frame)  #230
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,  collate_fn=collate_batch_test)
    input_dim_model =  7 * (history_frames-1)#Input features to the model: x,y-global (zero-centralized), heading,vel, accel, heading_rate, type 
    output_dim = future_frames*2 + 1 
    
    if  args.model_type == 'vae_gat':
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
                        feat_drop=args.feat_drop, attn_drop=args.attn_drop, att_ew=args.att_ew, ew_dims=args.ew_dims, backbone=args.backbone,
                        num_modes = args.num_modes, history_frames=history_frames-1)
    else:
        model = SCOUT(input_dim=input_dim_model, hidden_dim=args.hidden_dims, output_dim=output_dim, heads=args.heads, dropout=args.dropout, 
                        feat_drop=args.feat_drop, attn_drop=args.attn_drop, att_ew=args.att_ew, ew_dims=args.ew_dims>1, backbone=args.backbone)
    
    
    LitGNN_sys = LitGNN.load_from_checkpoint(checkpoint_path=args.ckpt, model=model, history_frames=history_frames, future_frames= future_frames,
                                                train_dataset=None, val_dataset=None, test_dataset=test_dataset, dataset='nuscenes', hparams=hparams)


    mtp_loss = MTPLoss(num_modes = args.num_modes, regression_loss_weight = 1, angle_threshold_degrees = 5.)
    model = LitGNN_sys.model.eval()

    explain(model, test_dataloader, mtp_loss, args)
   

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--scale_factor", type=int, default=1, help="Wether to scale x,y global positions (zero-centralized)")
    parser.add_argument("--ew_dims", type=int, default=2, choices=[1,2], help="Edge features: 1 for relative position, 2 for adding relationship type.")
    parser.add_argument("--z_dims", type=int, default=25, help="Dimensionality of the latent space")
    parser.add_argument("--hidden_dims", type=int, default=768)
    parser.add_argument("--model_type", type=str, default='mtp', help="Choose aggregation function between GAT or GATED",
                                        choices=['vae_gat', 'vae_gated', 'vae_prior', 'scout', 'mtp'])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--feat_drop", type=float, default=0.)
    parser.add_argument("--attn_drop", type=float, default=0.4)
    parser.add_argument("--heads", type=int, default=3, help='Attention heads (GAT)')
    parser.add_argument('--att_ew', type=str2bool, nargs='?', const=True, default=True, help="Add edge features in attention function (GAT)")
    parser.add_argument('--ckpt', type=str, default='/media/14TBDISK/sandra/logs/dainty-durian-12629/epoch=25-step=3769.ckpt', help='ckpt path.')  
    parser.add_argument("--norm", type=str, default=None, help="Wether to apply BN (bn) or GroupNorm (gn).")
    parser.add_argument("--enc_type", type=str, default='emb', choices=['emb',  'gru'])
    parser.add_argument("--emb_dims", type=int, default=512)
    
    parser.add_argument('--maps', type=str2bool, nargs='?', const=True, default=True, help="Add HD Maps.")
    parser.add_argument('--local_frame',  type=str2bool, nargs='?', const=True, default=False, help='whether to use local or global features.')  
    parser.add_argument("--emb_type", type=str, default='emb', choices=['emb', 'pos_enc', 'gru'])
    parser.add_argument('--num_modes', type=int, default=3, help="Number of decodings in training.")
    
    parser.add_argument("--backbone", type=str, default='resnet18', help="Choose CNN backbone.",
                                        choices=['resnet_gray', 'mobilenet', 'resnet18','resnet50', 'map_encoder'])
    parser.add_argument("--scene_id", type=int, default=651, help="Scene id to visualize.")
    parser.add_argument("--sample", type=str, default=None, help="sample to visualize.")
    parser.add_argument('--nowandb', action='store_true', help='use this flag to DISABLE wandb logging')  
    
    hparams = parser.parse_args()

    main(hparams)

