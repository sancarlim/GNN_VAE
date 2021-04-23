
import sys
sys.path.append('../../DBU_Graph')
import dgl
import torch
from torch.utils.data import DataLoader
import os
os.environ['DGLBACKEND'] = 'pytorch'
import numpy as np
from NuScenes.nuscenes_Dataset import nuscenes_Dataset
from models.VAE_GNN import VAE_GNN
from models.VAE_GATED import VAE_GATED
import wandb
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from argparse import ArgumentParser, Namespace
from utils import str2bool, compute_change_pos
from nuscenes.eval.prediction.data_classes import Prediction
import json
from NuScenes.nuscenes_visualize import collate_batch
from torchvision import transforms, utils

FREQUENCY = 2
dt = 1 / FREQUENCY
history = 2
future = 6
history_frames = history*FREQUENCY
future_frames = future*FREQUENCY
total_frames = history_frames + future_frames #2s of history + 6s of prediction
input_dim_model = (history_frames-1)*9 #Input features to the model: x,y-global (zero-centralized), heading,vel, accel, heading_rate, type 
output_dim = future_frames*2
base_path = '/home/sandra/PROGRAMAS/DBU_Graph/NuScenes'



class LitGNN(pl.LightningModule):
    def __init__(self, model,  train_dataset, val_dataset, test_dataset, history_frames: int=3, future_frames: int=3, lr: float = 1e-3, batch_size: int = 64, wd: float = 1e-1, beta: float = 0., delta: float = 1., rel_types: bool = False, scale_factor=1):
        super().__init__()
        self.model= model
        self.history_frames =history_frames
        self.future_frames = future_frames
        self.total_frames = history_frames + future_frames
        self.test_dataset = test_dataset
        self.rel_types = rel_types
        self.scale_factor = scale_factor
        self.challenge_predictions = []
        f = open('/media/14TBDISK/nuscenes/maps/prediction/prediction_scenes.json')
        self.prediction_scenes = json.load(f)  #Dict with keys "scene_id" : list("instances_samples")

        self.rescale_xy=torch.ones((1,1,2), device=self.device)*self.scale_factor
        
    
    def forward(self, graph, feats,e_w,snorm_n,snorm_e):
        pred = self.model(graph, feats,e_w,snorm_n,snorm_e)   
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
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos, tokens_eval, scene_id, mean_xy, maps = test_batch
        
        last_loc = feats[:,-1:,:2].detach().clone() 
        feats_vel, labels = compute_change_pos(feats,labels_pos, self.scale_factor)
        feats = torch.cat([feats_vel, feats[:,:,2:]], dim=-1)[:,1:]
        
        #last_loc = last_loc*self.rescale_xy       
        e_w = batched_graph.edata['w'].float()
        if not self.rel_types:
            e_w= e_w.unsqueeze(1)
        
        # Prediction: Prediction of model [num_modes, n_timesteps, state_dim] = [25, 12, 2]
        prediction_all_agents = []  # [num_agents, num_modes, n_timesteps, state_dim]
        for i in range(25):
            #Model predicts relative_positions
            preds = self.model.inference(batched_graph, feats,e_w,snorm_n,snorm_e, maps)  # [N_agents, 12, 2]
            preds=preds.view(preds.shape[0],self.future_frames,-1)  
            #Convert prediction to absolute positions
            for j in range(1,labels_pos.shape[1]):
                preds[:,j,:] = torch.sum(preds[:,j-1:j+1,:],dim=-2) #6,2 
            preds += last_loc

            # Provide predictions in global-coordinates
            pred_x = preds[:,:,0].cpu().numpy() + mean_xy[0][0]  # [N_agents, T]
            pred_y = preds[:,:,1].cpu().numpy() + mean_xy[0][1]
            
            prediction_all_agents.append(np.stack([pred_x, pred_y],axis=-1))

        prediction_all_agents = np.array(prediction_all_agents)
        for token in tokens_eval:
            if str(token[0]+'_'+token[1]) in self.prediction_scenes['scene-'+ str(scene_id).zfill(4)]:
                idx = np.where(np.array(tokens_eval)== token[0])[0][0]
                instance, sample = token
                pred = Prediction(str(instance), str(sample), prediction_all_agents[:,idx], np.ones(25)*1/25)  #need the pred to have 2d
                self.challenge_predictions.append(pred.serialize())
    
    def test_epoch_end(self, outputs):
        json.dump(self.challenge_predictions, open(os.path.join(base_path, 'challenge_inference.json'),'w'))

   
def main(args: Namespace):
    print(args)

    seed=seed_everything(0)

    test_dataset = nuscenes_Dataset(train_val_test='test', rel_types=args.ew_dims>1, history_frames=history_frames, 
                        future_frames=future_frames, challenge_eval=True)  #25 seq 2 scenes 103, 916

    if args.model_type == 'vae_gated':
        model = VAE_GATED(input_dim_model, args.hidden_dims, z_dim=args.z_dims, output_dim=output_dim, fc=False, dropout=args.dropout,  ew_dims=args.ew_dims)
    else:
        model = VAE_GNN(input_dim_model, args.hidden_dims//args.heads, args.z_dims, output_dim, fc=False, dropout=args.dropout, 
                        feat_drop=args.feat_drop, attn_drop=args.attn_drop, heads=args.heads, att_ew=args.att_ew, 
                        ew_dims=args.ew_dims, backbone=args.backbone)
    LitGNN_sys = LitGNN(model=model, history_frames=history_frames, future_frames= future_frames, train_dataset=None, val_dataset=None,
                 test_dataset=test_dataset, rel_types=args.ew_dims>1, scale_factor=args.scale_factor)
      
    trainer = pl.Trainer(gpus=args.gpus, deterministic=True, precision=16, profiler=True) 
 
    LitGNN_sys = LitGNN.load_from_checkpoint(checkpoint_path=args.ckpt, model=LitGNN_sys.model, history_frames=history_frames, future_frames= future_frames,
                    train_dataset=None, val_dataset=None, test_dataset=test_dataset, rel_types=args.ew_dims>1, scale_factor=args.scale_factor)

    
    trainer.test(LitGNN_sys)
   

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--scale_factor", type=int, default=1, help="Wether to scale x,y global positions (zero-centralized)")
    parser.add_argument("--ew_dims", type=int, default=2, choices=[1,2], help="Edge features: 1 for relative position, 2 for adding relationship type.")
    parser.add_argument("--z_dims", type=int, default=25, help="Dimensionality of the latent space")
    parser.add_argument("--hidden_dims", type=int, default=768)
    parser.add_argument("--model_type", type=str, default='vae_gat', help="Choose aggregation function between GAT or GATED",
                                        choices=['vae_gat', 'vae_gated'])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--feat_drop", type=float, default=0.)
    parser.add_argument("--attn_drop", type=float, default=0.4)
    parser.add_argument("--heads", type=int, default=2, help='Attention heads (GAT)')
    parser.add_argument('--att_ew', type=str2bool, nargs='?', const=True, default=True, help="Add edge features in attention function (GAT)")
    parser.add_argument('--ckpt', type=str, default=None, help='ckpt path.')   
    parser.add_argument('--nowandb', action='store_true', help='use this flag to DISABLE wandb logging')
    parser.add_argument("--backbone", type=str, default='resnet', help="Choose CNN backbone.",
                                        choices=['resnet_gray', 'mobilenet', 'resnet18', 'map_encoder'])
    
    
    hparams = parser.parse_args()

    main(hparams)




