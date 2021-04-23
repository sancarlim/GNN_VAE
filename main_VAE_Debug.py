import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
os.environ['DGLBACKEND'] = 'pytorch'
import numpy as np
from ApolloScape_Dataset import ApolloScape_DGLDataset
from inD_Dataset import inD_DGLDataset
from roundD_Dataset import roundD_DGLDataset
from models.VAE_GNN import VAE_GNN
from models.VAE_GATED import VAE_GATED
import random
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import math
from torch.distributions.kl import kl_divergence



def collate_batch(samples):
    graphs, masks, feats, gt = map(list, zip(*samples))  # samples is a list of pairs (graph, mask) mask es VxTx1
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
    return batched_graph, masks, snorm_n, snorm_e, feats, gt



class LitGNN(pl.LightningModule):
    def __init__(self, model,  train_dataset, val_dataset, test_dataset, dataset,  history_frames: int=3, future_frames: int=3, input_dim: int=2, lr: float = 1e-3, batch_size: int = 64, wd: float = 1e-1, alfa: float = 2, beta: float = 0., delta: float = 1., rel_types: bool = False, scale_factor=1):
        super().__init__()
        self.model= model
        self.lr = lr
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.wd = wd
        self.history_frames =history_frames
        self.future_frames = future_frames
        self.total_frames = history_frames + future_frames
        self.alfa = alfa
        self.beta = beta
        self.delta = delta
        self.overall_loss_time_list=[]
        self.overall_long_err_list=[]
        self.overall_lat_err_list=[]
        self.min_val_loss = 100
        self.dataset = dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.rel_types = rel_types
        self.scale_factor = scale_factor
        
    
    def forward(self, graph, feats,e_w,snorm_n,snorm_e):
        pred = self.model(graph, feats,e_w,snorm_n,snorm_e)   #inference
        return pred
    
    def configure_optimizers(self):
        #opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return opt
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=12, collate_fn=collate_batch)
    
    def val_dataloader(self):
        return  DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=12, collate_fn=collate_batch)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=12, collate_fn=collate_batch) 
    
    def compute_RMSE(self,pred, gt, mask): 
        pred = pred*mask #B*V,T,C  (B n grafos en el batch)
        gt = gt*mask  # outputmask BV,T,C
        x2y2_error=torch.sum((pred-gt)**2,dim=-1) # x^2+y^2 BV,T  PROBABILISTIC -> gt[:,:,:2]
        overall_sum_time = x2y2_error.sum(dim=-2)  #T - suma de los errores (x^2+y^2) de los BV agentes
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

    
    def vae_loss(self, pred, gt, mask, mu, log_var, beta=1, reconstruction_loss='huber'):
        #overall_sum_time, overall_num, _ = self.compute_RMSE(pred, gt, mask) #T
        if reconstruction_loss == 'huber':
            overall_sum_time, overall_num = self.huber_loss(pred, gt, mask, self.delta)  #T
        else:
            overall_sum_time, overall_num,_ = self.compute_RMSE(pred, gt, mask)  #T

        recons_loss = torch.sum(overall_sum_time)/torch.sum(overall_num.sum(dim=-2)) #T -> 1
        #kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        std = torch.exp(log_var / 2)
        kld_loss = kl_divergence(
        torch.distributions.Normal(mu, std), torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(log_var))
        ).sum(-1)
        loss = recons_loss + beta * kld_loss.mean()
        return loss, {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KL':kld_loss}
  

    def check_overlap(self, preds):
        intersect=[]
        #y_intersect=[np.argwhere(np.diff(np.sign(preds[i,:,1].detach().cpu().numpy()-preds[j,:,1].detach().cpu().numpy()))).size > 0  for i in range(len(preds)-1) for j in range(i+1,len(preds))]
        #x_intersect=[np.argwhere(np.diff(np.sign(preds[i,:,0].detach().cpu().numpy()-preds[j,:,0].detach().cpu().numpy()[::-1]))).size > 0 for i in range(len(preds)-1)  for j in range(i+1,len(preds))]
        #intersect = [True if y and x else False for y,x in zip(y_intersect,x_intersect)]
        '''
        for i in range(len(preds)-1):
            for j in range(i+1,len(preds)):
                y_intersect=(torch.sign(preds[i,:,1]-preds[j,:,1])-torch.sign(preds[i,:,1]-preds[j,:,1])[0]).bool().any()  #True if non all-zero
                x_intersect=(torch.sign(preds[i,:,0]-reversed(preds[j,:,0]))-torch.sign(preds[i,:,0]-reversed(preds[j,:,0]))[0]).bool().any()
                intersect.append(True if y_intersect and x_intersect else False)
        '''
        y_sub = torch.cat([torch.sign(preds[i:-1,:,1]-preds[i+1:,:,1]) for i in range(len(preds)-1)])  #N(all combinations),6
        y_intersect=( y_sub - y_sub[:,0].view(len(y_sub),-1)).bool().any(dim=1) #True if non all-zero (change sign)
        x_sub = torch.cat([torch.sign(preds[i:-1,:,0]-reversed(preds[i+1:,:,0])) for i in range(len(preds)-1)])
        x_intersect = (x_sub -x_sub[:,0].view(len(x_sub),-1)).bool().any(dim=1)
        #x_intersect=torch.cat([(torch.sign(preds[i:-1,:,0]-reversed(preds[i+1:,:,0]))-torch.sign(preds[i:-1,:,0]-reversed(preds[i+1:,:,0]))[0]).bool().any(dim=1) for i in range(len(preds)-1)])
        intersect = torch.logical_and(y_intersect,x_intersect) #[torch.count_nonzero(torch.logical_and(y,x))/len(x) for y,x in zip(y_intersect,x_intersect)] #to intersect, both True
        return torch.count_nonzero(intersect)/len(intersect) #percentage of intersections between all combinations
        #y_intersect=[np.argwhere(np.diff(np.sign(preds[i,:,1].cpu()-preds[j,:,1].cpu()))).size > 0 for j in range(i+1,len(preds)) for i in range(len(preds)-1)]

    def compute_change_pos(self, feats,gt):
        gt_vel = gt.clone()  #.detach().clone()
        feats_vel = feats[:,:,:2].clone()
        new_mask_feats = (feats_vel[:, 1:]!=0) * (feats_vel[:, :-1]!=0) 
        new_mask_gt = (gt_vel[:, 1:]!=0) * (gt_vel[:, :-1]!=0) 

        rescale_xy=torch.ones((1,1,2), device=self.device)*self.scale_factor

        gt_vel[:, 1:] = (gt_vel[:, 1:] - gt_vel[:, :-1]) * new_mask_gt
        gt_vel[:, :1] = (gt_vel[:, :1] - feats_vel[:, -1:]*rescale_xy) * new_mask_gt[:,0:1]
        feats_vel[:, 1:] = (feats_vel[:, 1:] - feats_vel[:, :-1]) * new_mask_feats
        feats_vel[:, 0] = 0
        
        return feats_vel, gt_vel

    def compute_long_lat_error(self,pred,gt,mask):
        pred = pred*mask #B*V,T,C  (B n grafos en el batch)
        gt = gt*mask  # outputmask BV,T,C
        lateral_error = pred[:,:,0]-gt[:,:,0]
        long_error = pred[:,:,1] - gt[:,:,1]  #BV,T
        overall_num = mask.sum(dim=-1).type(torch.int)  #torch.Tensor[(BV,T)] - num de agentes (Y CON DATOS) en cada frame
        return lateral_error, long_error, overall_num

    
    
    def training_step(self, train_batch, batch_idx):
        '''returns a loss from a single batch'''
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels = train_batch
        if self.dataset  == 'apollo':
            #USE CHANGE IN POS AS INPUT
            feats_vel, labels = self.compute_change_pos(feats,labels)
            #Input pos + heading + vel
            feats = torch.cat([feats_vel, feats[:,:,2:]], dim=-1)[:,1:,:] # torch.cat([feats[:,:,:self.input_dim], feats_vel], dim=-1)
        else:
            _, labels = self.compute_change_pos(feats,labels)

        e_w = batched_graph.edata['w'].float()
        if not self.rel_types:
            e_w= e_w.unsqueeze(1)

        pred, mu, log_var = self.model(batched_graph, feats,e_w,snorm_n,snorm_e, labels)
        pred=pred.view(labels.shape[0],self.future_frames,-1)
        total_loss, logs = self.vae_loss(pred, labels, output_masks, mu, log_var, beta=1)

        self.log_dict({f"Sweep/train_{k}": v for k,v in logs.items()}, on_step=False, on_epoch=True)
        return total_loss


    def validation_step(self, val_batch, batch_idx):
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos = val_batch
        last_loc = feats[:,-1:,:2]
        if self.dataset == 'apollo':
            #USE CHANGE IN POS AS INPUT
            feats_vel,labels = self.compute_change_pos(feats,labels_pos)
            #Input pos + heading + vel
            feats = torch.cat([feats_vel, feats[:,:,2:self.input_dim]], dim=-1)[:,1:,:] #torch.cat([feats[:,:,:self.input_dim], feats_vel], dim=-1)
        else:
            _, labels = self.compute_change_pos(feats,labels_pos)

        e_w = batched_graph.edata['w']
        if not self.rel_types:
            e_w= e_w.unsqueeze(1)
        pred, mu, log_var = self.model(batched_graph, feats,e_w,snorm_n,snorm_e,labels)
        pred=pred.view(labels.shape[0],self.future_frames,-1)
        total_loss, logs = self.vae_loss(pred, labels, output_masks, mu, log_var, beta=1)#, reconstruction_loss='mse')

        self.log_dict({"Sweep/val_loss": logs['loss'], "Sweep/val_recons_loss": logs['Reconstruction_Loss'], "Sweep/Val_KL": logs['KL']})
        return total_loss

    def validation_epoch_end(self, val_loss_over_batches):
        #log best val loss
        if torch.mean(torch.tensor(val_loss_over_batches,device=self.device)) < self.min_val_loss:            
            self.min_val_loss =  torch.mean(torch.tensor(val_loss_over_batches,device=self.device))
            self.logger.experiment.summary["best_val_loss"] = self.min_val_loss
    
         
    def test_step(self, test_batch, batch_idx):
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos = test_batch
        rescale_xy=torch.ones((1,1,2), device=self.device)*10
        last_loc = feats[:,-1:,:2].detach().clone() 
        last_loc = last_loc*rescale_xy            
        e_w = batched_graph.edata['w'].float()
        if not self.rel_types:
            e_w= e_w.unsqueeze(1)
        ade = []
        fde = []         
        #En test batch=1 secuencia con n agentes
        #Para el most-likely coger el modo con pi mayor de los 3 y o bien coger muestra de la media 
        for i in range(10): # @top10 Saco el min ADE/FDE por escenario tomando 15 muestras (15 escenarios)
            #Model predicts relative_positions
            preds = self.model.inference(batched_graph, feats,e_w,snorm_n,snorm_e)
            preds=preds.view(preds.shape[0],self.future_frames,-1)
            #Convert prediction to absolute positions
            for j in range(1,labels_pos.shape[1]):
                preds[:,j,:] = torch.sum(preds[:,j-1:j+1,:],dim=-2) #6,2 
            preds += last_loc
            #Compute error for this sample
            _ , overall_num, x2y2_error = self.compute_RMSE(preds[:,:self.future_frames,:], labels_pos[:,:self.future_frames,:], output_masks)
            ade_ts = torch.sum((x2y2_error**0.5), dim=0) / torch.sum(overall_num, dim=0)   
            ade_s = torch.sum(ade_ts)/ self.future_frames  #T ->1
            fde_s = torch.sum((x2y2_error**0.5), dim=0)[-1] / torch.sum(overall_num, dim=0)[-1]
            if torch.isnan(fde_s):  #visible pero no tiene datos para los siguientes 6 frames
                print('stop')
                fde_s[np.isnan(fde_s)]=0
                ade_ts[np.isnan(ade_ts)]=0
                for j in range(self.future_frames-2,-1,-1):
                    if ade_ts[j] != 0:
                            fde_s =  torch.sum((x2y2_error**0.5), dim=0)[j] / torch.sum(overall_num, dim=0)[j]  #compute FDE with the last frame with data
                            ade_s = torch.sum(ade_ts)/ (j+1) #compute ADE dividing by number of frames with data
                            break
            ade.append(ade_s) #S samples
            fde.append(fde_s)
    
        self.log_dict({'test/ade': min(ade), "test/fde": fde[ade.index(min(ade))]})
        
   
def sweep_train():

    seed=seed_everything(np.random.randint(1000000))
    #run=wandb.init()  #for sweep
    run=wandb.init(project="dbu_graph", config=default_config) #for single run
    config = wandb.config

    history_frames = config.history_frames
    future_frames = config.future_frames

    if config.dataset == 'apollo':
        train_dataset = ApolloScape_DGLDataset(train_val='train', test=False, rel_types=config.ew_dims>1) #3447
        val_dataset = ApolloScape_DGLDataset(train_val='val', test=False, rel_types=config.ew_dims>1)  #919
        test_dataset = ApolloScape_DGLDataset(train_val='test', test=False, rel_types=config.ew_dims>1)  #230
        print(len(train_dataset), len(val_dataset))
        input_dim = 5
    elif config.dataset == 'ind':
        train_dataset = inD_DGLDataset(train_val='train', history_frames=history_frames, future_frames=future_frames, classes=(1,2,3,4), rel_types=config.ew_dims>1) #12281
        val_dataset = inD_DGLDataset(train_val='val', history_frames=history_frames, future_frames=future_frames,  classes=(1,2,3,4), rel_types=config.ew_dims>1)  #3509
        test_dataset = inD_DGLDataset(train_val='test', history_frames=history_frames, future_frames=future_frames,  classes=(1,2,3,4), rel_types=config.ew_dims>1)  #1754
        print(len(train_dataset), len(val_dataset), len(test_dataset))
        input_dim = 6
    wandb_logger = pl_loggers.WandbLogger() 
    wandb_logger.experiment.log({'seed': seed}) 
    
    input_dim_model = input_dim*(history_frames-1) if config.dataset=='apollo' else input_dim*history_frames
    output_dim = 2*future_frames #if config.probabilistic == False else 5*future_frames

    if config.model == 'vae_gated':
        model = VAE_GATED(input_dim_model, config.hidden_dims, z_dim=config.z_dims, output_dim=output_dim, fc=False, dropout=config.dropout,  ew_dims=config.ew_dims)
    else:
        model = VAE_GNN(input_dim_model, config.hidden_dims//config.heads, config.z_dims, output_dim, fc=False, dropout=config.dropout, feat_drop=config.feat_drop, attn_drop=config.attn_drop, heads=config.heads, att_ew=config.att_ew, ew_dims=config.ew_dims)

    LitGNN_sys = LitGNN(model=model, input_dim=input_dim, lr=config.learning_rate,  wd=config.wd, history_frames=config.history_frames, future_frames= config.future_frames, alfa= config.alfa, beta = config.beta, delta=config.delta,
                        dataset=config.dataset, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, rel_types=config.ew_dims>1, scale_factor=config.scale_factor)
    #wandb_logger.watch(LitGNN_sys.model)  #log='all' for params & grads
    '''
    if config.dataset == 'ind':
        path = '/media/14TBDISK/sandra/logs/VAE/whole-sweep-72/epoch=27-step=10107.ckpt' #DGX/APOLLO/helpful-sweep-17/epoch=48-step=3282.ckpt'
        LitGNN_sys = LitGNN.load_from_checkpoint(checkpoint_path=path,model=LitGNN_sys.model, input_dim=input_dim, lr=config.learning_rate,  wd=config.wd, history_frames=config.history_frames, future_frames= config.future_frames, alfa= config.alfa, beta = config.beta, delta=config.delta,
                    dataset=config.dataset, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, rel_types=config.ew_dims>1)
        print('############ TEST  ##############')
        trainer = pl.Trainer(gpus=1, profiler=True)
        trainer.test(LitGNN_sys)
    '''
    
    checkpoint_callback = ModelCheckpoint(monitor='Sweep/val_loss', mode='min', dirpath='/media/14TBDISK/sandra/logs/'+'DEBUG/'+run.name)
    early_stop_callback = EarlyStopping('Sweep/val_loss', patience=3)
    trainer = pl.Trainer( weights_summary='full', gpus=1, deterministic=True, precision=16, logger=wandb_logger, callbacks=[early_stop_callback,checkpoint_callback], profiler=True)  # resume_from_checkpoint=config.path, precision=16, limit_train_batches=0.5, progress_bar_refresh_rate=20,
    #resume_from_checkpoint=path, 
    
    print('Best lr: ', LitGNN_sys.lr)
    print('GPU Nº: ', device)
    print("############### TRAIN ####################")
    trainer.fit(LitGNN_sys)
    print('Model checkpoint path:',trainer.checkpoint_callback.best_model_path)
    
    print("############### TEST ####################")
    if config.dataset !='apollo':
        trainer.test(ckpt_path='best')
    
    
default_config = {
            "ew_dims":2,
            "model": 'vae_gated',
            "scale_factor": 10,
            "input_dim": 6,
            "dataset":'apollo',
            "history_frames":6,
            "future_frames":6,
            "learning_rate":1e-6,
            "batch_size": 1,
            "hidden_dims": 1024,
            "z_dims": 100,
            "dropout": 0.1,
            "alfa": 0,
            "beta": 1,
            "delta": 1,
            "feat_drop": 0.,
            "attn_drop":0.2,
            "bn":False,
            "wd": 0.01,
            "heads": 2,
            "att_ew": True,               
            "gcn_drop": 0.,
            "gcn_bn": True,
            'embedding':True
        }
device=os.environ.get('CUDA_VISIBLE_DEVICES')
sweep_train()    
