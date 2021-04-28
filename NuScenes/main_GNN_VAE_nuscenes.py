import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
os.environ['DGLBACKEND'] = 'pytorch'
import sys
sys.path.append('..')
import numpy as np
from nuscenes_Dataset import nuscenes_Dataset, collate_batch
from models.VAE_PRIOR import VAE_GNN_prior
from models.VAE_GNN import VAE_GNN
from models.VAE_GATED import VAE_GATED
import random
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser, Namespace
import math
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from utils import str2bool, compute_change_pos, plot_grad_flow

FREQUENCY = 2
dt = 1 / FREQUENCY
history = 2
future = 6
history_frames = history*FREQUENCY
future_frames = future*FREQUENCY
total_frames = history_frames + future_frames #2s of history + 6s of prediction
input_dim_model = (history_frames-1)*8 #Input features to the model: x,y-global (zero-centralized), heading,vel, accel, heading_rate, type 
output_dim = future_frames*2


def collate_batch_test(samples):
    graphs, masks, feats, gt, tokens, scene_ids, mean_xy, maps = map(list, zip(*samples))  # samples is a list of pairs (graph, mask) mask es VxTx1
    masks = torch.vstack(masks)
    feats = torch.vstack(feats)
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
    return batched_graph, masks, snorm_n, snorm_e, feats, gt, tokens[0], scene_ids[0], mean_xy, maps


class LitGNN(pl.LightningModule):
    def __init__(self, model = None,  train_dataset = None, val_dataset = None, test_dataset = None, history_frames: int=3, future_frames: int=3, 
                    lr1: float = 1e-3, lr2: float = 1e-3, batch_size: int = 64, wd: float = 1e-1, delta: float = 1., 
                    rel_types: bool = False, scale_factor: int = 1, wandb : bool = True, decay_rate: float = 0.96, 
                    reconstruction_loss: str = 'huber',  gamma: float = 0.01, beta_period: float = 4, beta_ratio_annealing: float = 0.5,
                    beta_max_value: float = 1, model_type: str = 'vae_gat'):
        super().__init__()
        self.model= model
        self.lr1 = lr1
        self.lr2 = lr2
        self.batch_size = batch_size
        self.wd = wd
        self.history_frames =history_frames
        self.future_frames = future_frames
        self.total_frames = history_frames + future_frames
        self.gamma = gamma
        self.delta = delta
        self.overall_loss_time_list=[]
        self.overall_long_err_list=[]
        self.overall_lat_err_list=[]
        self.min_val_loss = 100
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.rel_types = rel_types
        self.scale_factor = scale_factor
        self.wandb = wandb
        self.decay_rate = decay_rate
        self.reconstruction_loss = reconstruction_loss
        self.beta = 0
        self.beta_period = beta_period
        self.beta_ratio_annealing = beta_ratio_annealing
        self.beta_max_value = beta_max_value
        self.model_type = model_type
        
    
    def forward(self, graph, feats,e_w,snorm_n,snorm_e):
        pred = self.model(graph, feats,e_w,snorm_n,snorm_e)   #inference
        return pred
    
    def configure_optimizers(self):
        #opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        opt = torch.optim.AdamW([
                {'params': self.model.base.parameters()},
                {'params': self.model.embedding_h.parameters(), 'lr': self.lr1},
                {'params': self.model.feature_extractor.parameters(), 'lr': self.lr1}], lr=self.lr2, weight_decay=self.wd)
        
        if self.decay_rate != 0:
            return {
                'optimizer': opt,
                'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, threshold=0.001, patience=3, verbose=True),
                'monitor': "Sweep/val_Reconstruction_Loss"
            }

        return opt
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, collate_fn=collate_batch)
    
    def val_dataloader(self):
        return  DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, collate_fn=collate_batch)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=16, shuffle=False, num_workers=8, collate_fn=collate_batch_test) 
    
    def compute_MSE(self,pred, gt, mask): 
        pred = pred*mask #B*V,T,C  (B n grafos en el batch)
        gt = gt*mask  # outputmask BV,T,C
        x2y2_error=torch.sum((pred-gt)**2,dim=-1) # x^2+y^2 BV,T  PROBABILISTIC -> gt[:,:,:2]
        overall_sum_time = (x2y2_error).sum(dim=-2)  #T - suma de los errores (x^2+y^2) de los BV agentes
        overall_num = mask.sum(dim=-1).type(torch.int)  #torch.Tensor[(BV,T)] - num de agentes (Y CON DATOS) en cada frame
        return overall_sum_time, overall_num, x2y2_error

    def huber_loss(self, pred, gt, mask, delta):
        pred = pred*mask #B*V,T,C  (B n grafos en el batch)
        gt = gt*mask  # outputmask BV,T,C
        #error = torch.sum(torch.where(torch.abs(gt-pred) < delta , 0.5*((gt-pred)**2)*(1/delta), torch.abs(gt - pred) - 0.5*delta), dim=-1)   # BV,T
        error = torch.sum(torch.where(torch.abs(gt-pred) < delta , (0.5*(gt-pred)**2), torch.abs(gt - pred)*delta - 0.5*(delta**2)), dim=-1)
        overall_sum_time = error.sum(dim=-2) #T - suma de los errores de los BV agentes
        overall_num = mask.sum(dim=-1).type(torch.int) 
        return overall_sum_time, overall_num

    def frange_cycle_linear(self, start=0.0, stop=1.0, ratio=0.7, period=500):
        #period = n_iter/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule
        
        if self.beta >= stop:
            if self.global_step % 500 == 0:
                self.beta = start
        else:
            self.beta += step
    
    
    def vae_loss_prior(self, pred, gt, mask, mu, log_var, mu_prior, log_var_prior, z0):
        ### Reconstruction Loss ###
        if self.reconstruction_loss == 'huber':
            overall_sum_time, overall_num = self.huber_loss(pred[:,:,:2], gt[:,:,:2], mask, self.delta)  #T
        else:
            overall_sum_time, overall_num,_ = self.compute_MSE(pred, gt, mask)  #T, (BV,T)

        recons_loss = torch.sum(overall_sum_time/overall_num.sum(dim=-2)) #T -> 1
        
        ### KL Loss ###
        std = torch.exp(log_var / 2)
        std_prior = torch.exp(log_var_prior / 2)
        kld_loss = kl_divergence(
            Normal(mu, std), Normal(torch.zeros_like(mu), torch.ones_like(std))
        ).sum(-1)

        ### KL PRIOR Loss ###
        kld_prior = kl_divergence(
            Normal(mu, std), Normal(mu_prior, std_prior)
        ).sum(-1)
        '''
        ### Supervise z0 as heading ###
        heading = (gt[:,:,2:]*mask).squeeze()[:,6]
        delta = 1
        error = torch.where(torch.abs(heading-z0) < delta , (0.5*(heading-z0)**2), torch.abs(heading-z0)*delta - 0.5*(delta**2))
        z0_loss =error.sum()/overall_num.sum(dim=0)[6] #BV -> 1
        '''
        '''
        z = self.model.reparameterize(mu_prior, log_var_prior)
        z0_loss =((heading[:,-1]-z[:,0])**2).sum()/overall_num.sum(dim=0)[-1] 
        for i in range(2):
            z =  self.model.reparameterize(mu_prior, log_var_prior)
            error =((heading[:,-1]-z[:,0])**2).sum()/overall_num.sum(dim=0)[-1] 
            if error < z0_loss:
                z0_loss = error
        '''
        if self.global_step > 1:
            self.frange_cycle_linear(stop=self.beta_max_value, ratio=self.beta_ratio_annealing, period=self.batch_size*self.beta_period)

        loss = recons_loss + self.beta * kld_loss.mean() + self.beta * kld_prior.mean() #+ self.gamma * z0_loss
        return loss, {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KL': kld_loss.mean(), 'KL_prior':  kld_prior.mean(),  'beta': self.beta} #'z0': self.gamma *z0_loss,}
  
      
    def vae_loss(self, pred, gt, mask, mu, log_var):
        #Train with Huber . Validate with MSE
        if self.reconstruction_loss == 'huber':
            overall_sum_time, overall_num = self.huber_loss(pred, gt[:,:,:2], mask, self.delta)  #T
        else:
            overall_sum_time, overall_num,_ = self.compute_RMSE(pred, gt[:,:,:2], mask)  #T, (BV,T)

        recons_loss = torch.sum(overall_sum_time/overall_num.sum(dim=-2)) #T -> 1
        #kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        std = torch.exp(log_var / 2)
        kld_loss = kl_divergence(
        Normal(mu, std), Normal(torch.zeros_like(mu), torch.ones_like(std))
        ).sum(-1)
        '''
        # Supervise z0 as heading
        heading = (gt[:,:,2:]*mask).squeeze()
        z_sample = torch.distributions.Normal(mu, std).sample_n(3)[:,:,0]
        z0_loss =((heading[:,-1]-z_sample[0])**2).sum()/overall_num.sum(dim=0)[-1] #BV -> 1
        for z in z_sample[1:]:
            diff = ((heading[:,-1]-z)**2).sum()/overall_num.sum(dim=0)[-1] 
            if diff < z0_loss:
                z0_loss = diff
        '''
        if self.global_step > 1:
            self.frange_cycle_linear(stop=self.beta_max_value, ratio=self.beta_ratio_annealing, period=self.batch_size*self.beta_period)

        loss = recons_loss + self.beta * kld_loss.mean() #+ self.gamma * z0_loss
        return loss, {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KL': kld_loss.mean(),  'beta': self.beta}#, 'z0': z0_loss}

    
    def training_step(self, train_batch, batch_idx):
        '''returns a loss from a single batch'''
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos, maps = train_batch
        feats_vel, labels_vel = compute_change_pos(feats,labels_pos[:,:,:2], self.scale_factor)
        feats = torch.cat([feats_vel, feats[:,:,2:]], dim=-1)[:,1:]
        #labels = torch.cat([labels_vel, labels_pos[:,:,2:]], dim=-1)
        e_w = batched_graph.edata['w'].float()
        if not self.rel_types:
            e_w= e_w.unsqueeze(1)
        if maps.shape[0] != feats.shape[0]:
            print('stop')
        
        if self.model_type == 'vae_prior':
            pred, mu, log_var, mu_prior, log_var_prior, z0 = self.model(batched_graph, feats,e_w,snorm_n,snorm_e, labels_vel, maps)
            pred=pred.view(feats.shape[0],self.future_frames,-1)
            total_loss, logs = self.vae_loss_prior(pred, labels_vel, output_masks, mu, log_var, mu_prior, log_var_prior,z0)
        else:
            pred, mu, log_var=self.model(batched_graph, feats,e_w,snorm_n,snorm_e, labels_vel, maps)
            pred=pred.view(feats.shape[0],self.future_frames,-1)
            total_loss, logs = self.vae_loss(pred,  labels_vel, output_masks, mu, log_var)

        self.log_dict({f"Sweep/train_{k}": v for k,v in logs.items()}, on_step=True, on_epoch=False)
        return total_loss

    #def training_epoch_end(self, output):
    #    plot_grad_flow(self.model.named_parameters)

        
    def validation_step(self, val_batch, batch_idx):
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos, maps = val_batch
        feats_vel, labels_vel = compute_change_pos(feats,labels_pos[:,:,:2], self.scale_factor)
        feats = torch.cat([feats_vel, feats[:,:,2:]], dim=-1)[:,1:]
        labels = torch.cat([labels_vel, labels_pos[:,:,2:]], dim=-1)
        
        e_w = batched_graph.edata['w']
        if not self.rel_types:
            e_w= e_w.unsqueeze(1)
        
        if self.model_type == 'vae_prior':
            pred, mu, log_var, mu_prior, log_var_prior, z0 = self.model(batched_graph, feats,e_w,snorm_n,snorm_e, labels_vel, maps)
            pred=pred.view(feats.shape[0],self.future_frames,-1)
            total_loss, logs = self.vae_loss_prior(pred, labels_vel, output_masks, mu, log_var, mu_prior, log_var_prior,z0)
        else:
            pred, mu, log_var=self.model(batched_graph, feats,e_w,snorm_n,snorm_e, labels_vel, maps)
            pred=pred.view(feats.shape[0],self.future_frames,-1)
            total_loss, logs = self.vae_loss(pred,  labels_vel, output_masks, mu, log_var)
        
        self.log_dict({f"Sweep/val_{k}": v for k,v in logs.items()})
        #self.log_dict({"Sweep/val_loss": logs['loss'], "Sweep/val_reconstruction": logs['Reconstruction_Loss']/self.future_frames, "Sweep/Val_KL": logs['KL']})
        return total_loss

    def validation_epoch_end(self, val_loss_over_batches):
        #log best val loss
        if self.wandb and torch.mean(torch.tensor(val_loss_over_batches,device=self.device)) < self.min_val_loss:            
            self.min_val_loss =  torch.mean(torch.tensor(val_loss_over_batches,device=self.device))
            self.logger.experiment.summary["best_val_loss"] = self.min_val_loss
        if self.current_epoch == 10:
            plot_grad_flow(self.model.named_parameters)
         
    def test_step(self, test_batch, batch_idx):
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos, tokens_eval, scene_id, mean_xy, maps  = test_batch
        last_loc = feats[:,-1:,:2].detach().clone() 
        rescale_xy=torch.ones((1,1,2), device=self.device)*self.scale_factor
        feats_vel, labels = compute_change_pos(feats,labels_pos[:,:,:2], self.scale_factor)
        feats = torch.cat([feats_vel, feats[:,:,2:]], dim=-1)[:,1:]
        
        if self.scale_factor == 1:
            pass#last_loc = last_loc*5.398+0.0013
        else:
            last_loc = last_loc*rescale_xy      
        e_w = batched_graph.edata['w'].float()
        if not self.rel_types:
            e_w= e_w.unsqueeze(1)
        
        ade = []
        fde = []         
        #En test batch=1 secuencia con n agentes
        #Para el most-likely coger el modo con pi mayor de los 3 y o bien coger muestra de la media 
        for i in range(5): # @top10 Saco el min ADE/FDE por escenario tomando 15 muestras (15 escenarios)
            #Model predicts relative_positions
            if self.model_type == 'vae_prior':
                preds, mu, logvar = self.model.inference(batched_graph, feats,e_w,snorm_n,snorm_e, maps)   
            else:
                preds = self.model.inference(batched_graph, feats,e_w,snorm_n,snorm_e, maps) 
            preds=preds.view(preds.shape[0],self.future_frames,-1)[:,:,:2]
            #Convert prediction to absolute positions
            for j in range(1,labels_pos.shape[1]):
                preds[:,j,:] = torch.sum(preds[:,j-1:j+1,:],dim=-2) #6,2 
            preds += last_loc
            #Compute error for this sample
            _ , overall_num, x2y2_error = self.compute_MSE(preds, labels_pos[:,:,:2], output_masks)
            ade_ts = torch.sum((x2y2_error**0.5), dim=0) / torch.sum(overall_num, dim=0)   
            ade_s = torch.sum(ade_ts)/ self.future_frames  #T ->1
            fde_s = torch.sum((x2y2_error**0.5), dim=0)[-1] / torch.sum(overall_num, dim=0)[-1]
            if torch.isnan(fde_s):  #visible pero no tiene datos para los siguientes 6 frames
                print('stop')
                fde_s[torch.isnan(fde_s)]=0
                ade_ts[torch.isnan(ade_ts)]=0
                for j in range(self.future_frames-2,-1,-1):
                    if ade_ts[j] != 0:
                            fde_s =  torch.sum((x2y2_error**0.5), dim=0)[j] / torch.sum(overall_num, dim=0)[j]  #compute FDE with the last frame with data
                            ade_s = torch.sum(ade_ts)/ (j+1) #compute ADE dividing by number of frames with data
                            break
            ade.append(ade_s) #S samples
            fde.append(fde_s)
        self.log_dict({'test/ade': min(ade), "test/fde": fde[ade.index(min(ade))]})
        
   
def main(args: Namespace):
    seed=seed_everything(0)

    train_dataset = nuscenes_Dataset(train_val_test='train',  rel_types=args.ew_dims>1, history_frames=history_frames, future_frames=future_frames) #3447
    val_dataset = nuscenes_Dataset(train_val_test='val',  rel_types=args.ew_dims>1, history_frames=history_frames, future_frames=future_frames)  #919
    test_dataset = nuscenes_Dataset(train_val_test='test', rel_types=args.ew_dims>1, history_frames=history_frames, future_frames=future_frames, challenge_eval=True)  #230

    '''
    if args.ckpt is not None:
        hparams = torch.load(args.ckpt)['hyper_parameters']
        ckpt = args.ckpt
        args = hparams['hparams']
        test = True
    LitGNN_sys = LitGNN.load_from_checkpoint(checkpoint_path=ckpt, hparams = args, model=model, test_dataset = hparams['test_dataset'], 
                        history_frames = hparams['history_frames'], future_frames = hparams['future_frames'])
    '''

    if args.model_type == 'vae_gated':
        model = VAE_GATED(input_dim_model, args.hidden_dims, z_dim=args.z_dims, output_dim=output_dim, fc=False, dropout=args.dropout, 
                            ew_dims=args.ew_dims, backbone=args.backbone, freeze=args.freeze)
    elif args.model_type == 'vae_prior':
        model = VAE_GNN_prior(input_dim_model, args.hidden_dims//args.heads, args.z_dims, output_dim, fc=False, dropout=args.dropout, feat_drop=args.feat_drop,
                        attn_drop=args.attn_drop, heads=args.heads, att_ew=args.att_ew, ew_dims=args.ew_dims, backbone=args.backbone, freeze=args.freeze,
                        bn=(args.norm=='bn'), gn=(args.norm=='gn'))
    else:
        model = VAE_GNN(input_dim_model, args.hidden_dims//args.heads, args.z_dims, output_dim, fc=False, dropout=args.dropout, feat_drop=args.feat_drop,
                        attn_drop=args.attn_drop, heads=args.heads, att_ew=args.att_ew, ew_dims=args.ew_dims, backbone=args.backbone, freeze=args.freeze,
                        bn=(args.norm=='bn'), gn=(args.norm=='gn'))
    

    
    
    LitGNN_sys = LitGNN(model=model, lr1 = args.lr1, lr2 = args.lr2,  wd = args.wd, history_frames = history_frames, future_frames = future_frames, delta = args.delta,
    train_dataset = train_dataset, val_dataset = val_dataset, test_dataset = test_dataset, rel_types = args.ew_dims>1, scale_factor = args.scale_factor, wandb = not args.nowandb,
    decay_rate = args.decay_rate, reconstruction_loss = args.reconstruction_loss, gamma = args.gamma, batch_size = args.batch_size, beta_period = args.beta_period, 
    beta_ratio_annealing = args.beta_ratio_annealing, beta_max_value = args.beta_max_value, model_type = args.model_type)
    
    
    early_stop_callback = EarlyStopping('Sweep/val_Reconstruction_Loss', patience=10)

    if not args.nowandb:
        run=wandb.init(job_type="training", entity='sandracl72', project='nuscenes')  
        wandb_logger = pl_loggers.WandbLogger() 
        wandb_logger.experiment.log({'seed': seed}) 
        wandb_logger.watch(LitGNN_sys.model, log='gradients')  #log='all' for params & grads
        if os.environ.get('WANDB_SWEEP_ID') is not None: 
            ckpt_folder = os.path.join(os.environ.get('WANDB_SWEEP_ID'), run.name)
        else:
            ckpt_folder = run.name
        checkpoint_callback = ModelCheckpoint(monitor='Sweep/val_loss', mode='min', dirpath=os.path.join('/media/14TBDISK/sandra/logs/', ckpt_folder))
        trainer = pl.Trainer( weights_summary='full', gpus=args.gpus, deterministic=True, precision=16, log_every_n_steps=5, flush_logs_every_n_steps=10 ,logger=wandb_logger, callbacks=[early_stop_callback,checkpoint_callback], profiler=True)  # resume_from_checkpoint=config.path, precision=16, limit_train_batches=0.5, progress_bar_refresh_rate=20,
    else:
        checkpoint_callback = ModelCheckpoint(monitor='Sweep/val_loss', mode='min', dirpath='/media/14TBDISK/sandra/logs/',filename='nowandb-{epoch:02d}')
        trainer = pl.Trainer( weights_summary='full', gpus=args.gpus, deterministic=True, precision=16, callbacks=[early_stop_callback,checkpoint_callback], profiler=True) 

    if args.ckpt is not None:
        LitGNN_sys = LitGNN.load_from_checkpoint(checkpoint_path=args.ckpt, model=LitGNN_sys.model, history_frames=history_frames, future_frames= future_frames,
                    train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, rel_types=args.ew_dims>1, scale_factor=args.scale_factor, 
                    model_type=args.model_type)

        print('############ TEST  ##############')
        trainer = pl.Trainer(gpus=1, profiler=True)
        trainer.test(LitGNN_sys)
    else:        
        print('GPU Nº: ', device)
        print("############### TRAIN ####################")
        trainer.fit(LitGNN_sys)
        print('Model checkpoint path:',trainer.checkpoint_callback.best_model_path)
        
        print("############### TEST ####################")
        trainer.test(ckpt_path='best')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--scale_factor", type=int, default=1, help="Wether to scale x,y global positions (zero-centralized)")
    parser.add_argument("--ew_dims", type=int, default=1, choices=[1,2], help="Edge features: 1 for relative position, 2 for adding relationship type.")
    parser.add_argument("--lr1", type=float, default=1e-4, help="Adam: Embedding learning rate")
    parser.add_argument("--lr2", type=float, default=1e-3, help="Adam: Base learning rate")
    parser.add_argument("--wd", type=float, default=0.01, help="Adam: weight decay")
    parser.add_argument("--batch_size", type=int, default=128, help="Size of the batches")
    parser.add_argument("--z_dims", type=int, default=25, help="Dimensionality of the latent space")
    parser.add_argument("--hidden_dims", type=int, default=256)
    parser.add_argument("--model_type", type=str, default='vae_prior', help="Choose aggregation function between GAT or GATED",
                                        choices=['vae_gat', 'vae_gated', 'vae_prior'])
    parser.add_argument("--reconstruction_loss", type=str, default='huber', help="Choose reconstruction loss.",
                                        choices=['huber', 'mse'])
    parser.add_argument("--backbone", type=str, default='resnet', help="Choose CNN backbone.",
                                        choices=['resnet_gray', 'resnet', 'map_encoder'])
    parser.add_argument("--norm", type=str, default=None, help="Wether to apply BN (bn) or GroupNorm (gn).")
    parser.add_argument('--freeze', type=int, default=7, help="Layers to freeze in resnet18.")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--feat_drop", type=float, default=0.)
    parser.add_argument("--attn_drop", type=float, default=0.25)
    parser.add_argument("--heads", type=int, default=2, help='Attention heads (GAT)')
    parser.add_argument("--beta_period", type=float, default=4, help='Period of the cyclical annealing: period = batch_size*args.period')
    parser.add_argument("--beta_ratio_annealing", type=float, default=0.7)
    parser.add_argument("--beta_max_value", type=float, default=1, help='Max value of beta.')
    #parser.add_argument("--beta", type=float, default=1, help='Weighting factor of the KL divergence loss term')
    #parser.add_argument("--beta_p", type=float, default=1, help='Weighting factor of the KL divergence loss term for the prior')
    parser.add_argument("--gamma", type=float, default=0.01, help='Weighting factor of the z0 supervision loss')
    parser.add_argument("--delta", type=float, default=.001, help='Delta factor in Huber Loss (Reconstruction Loss)')
    #parser.add_argument('--att_ew', action='store_true', help='use this flag to add edge features in attention function (GAT)')    
    parser.add_argument('--att_ew', type=str2bool, nargs='?', const=True, default=True, help="Add edge features in attention function (GAT)")
    parser.add_argument("--decay_rate", type=float, default=1., help='wether to apply lr_scheduling. If != 0 apply ReduceLROnPlateau')
    parser.add_argument('--maps', type=str2bool, nargs='?', const=True, default=True, help="Add HD Maps.")
    
    parser.add_argument('--nowandb', action='store_true', help='use this flag to DISABLE wandb logging')  
    parser.add_argument('--ckpt', type=str, default=None, help='ckpt path for only testing.')   

    
    device=os.environ.get('CUDA_VISIBLE_DEVICES')
    hparams = parser.parse_args()

    main(hparams)

