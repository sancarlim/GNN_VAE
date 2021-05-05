import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import os
os.environ['DGLBACKEND'] = 'pytorch'
import numpy as np
from ApolloScape_Dataset import ApolloScape_DGLDataset
from inD_Dataset import inD_DGLDataset
from roundD_Dataset import roundD_DGLDataset
from NuScenes.nuscenes_Dataset import nuscenes_Dataset, collate_batch
from models.GCN import GCN 
from models.scout import SCOUT
from models.SCOUT_MDN import SCOUT_MDN
from models.Gated_MDN import Gated_MDN
from models.Gated_GCN import GatedGCN
from models.RGCN import RGCN
import random
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser, Namespace
import math
from utils import str2bool, compute_change_pos, compute_long_lat_error, check_overlap, MTPLoss



ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)
NUM_MODES = 3

class LitGNN(pl.LightningModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, dataset, history_frames: int=3, future_frames: int=3, 
                        input_dim: int=2, model: nn.Module = GCN, lr1: float = 1e-3, lr2: float = 1e-3, batch_size: int = 64, model_type: str = 'gcn', 
                        wd: float = 1e-1, alfa: float = 2, beta: float = 0., delta: float = 1., prob: bool = False, 
                        mask: bool = False, rel_types: bool = False, scale_factor: int = 1, wandb: bool = True):
        super().__init__()
        self.model= model
        self.lr1 = lr1
        self.lr2 = lr2
        self.input_dim = input_dim
        self.model_type = model_type
        self.batch_size = batch_size
        self.wd = wd
        self.history_frames =history_frames
        self.future_frames = future_frames
        self.total_frames = history_frames + future_frames
        self.alfa = alfa
        self.beta = beta
        self.delta = delta
        self.probabilistic = prob
        self.overall_loss_time_list=[]
        self.overall_long_err_list=[]
        self.overall_lat_err_list=[]
        self.min_val_loss = 100
        self.dataset = dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.mask = mask
        self.rel_types = rel_types
        self.scale_factor = scale_factor
        self.wandb = wandb
        
        self.mtp_loss = MTPLoss(num_modes = 3, regression_loss_weight = 2., angle_threshold_degrees = 5.)

    def forward(self, graph, feats,e_w,snorm_n,snorm_e):
        pred = self.model(graph, feats,e_w,snorm_n,snorm_e)   #inference
        return pred

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr1, weight_decay=self.wd)
        #opt = torch.optim.AdamW([
        #        {'params': self.model.base.parameters()},
        #        {'params': self.model.embedding_h.parameters(), 'lr': self.lr1},
        #        {'params': self.model.feature_extractor.parameters(), 'lr': self.lr1}          ], lr=self.lr2, weight_decay=self.wd)
        
        return {
            'optimizer': opt,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, threshold=0.001, patience=3, verbose=True),
            'monitor': "Sweep/val_loss"
        }


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,num_workers=8, shuffle=True,  collate_fn=collate_batch)

    def val_dataloader(self):
        return  DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8,collate_fn=collate_batch)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False,num_workers=8, collate_fn=collate_batch) # 

    def gaussian_probability(self,sigma, mu, target):
        """Returns the probability of `target` given MoG parameters `sigma` and `mu`.
        
        Arguments:
            sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
                size, G is the number of Gaussians, and O is the number of
                dimensions per Gaussian.
            mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
                number of Gaussians, and O is the number of dimensions per Gaussian.
            target (BxI): A batch of target. B is the batch size and I is the number of
                input dimensions.
        Returns:
            probabilities (BxG): The probability of each point in the probability
                of the distribution in the corresponding sigma/mu index.
        """
        target = target.unsqueeze(1).expand_as(sigma)
        ret = ONEOVERSQRT2PI * torch.clamp(torch.exp(-0.5 * ((target - mu) / sigma)**2), min=1e-10) / sigma  #assume diagonal covariance
        return torch.clamp(torch.prod(ret, 2),min=1e-30, max=1e10)  #La prob total es el producto de las prob marginales unidimensionales



    def mdn_loss(self,pred, target, mask):
        """Calculates the error, given the MoG parameters and the target
        The loss is the negative log likelihood of the data given the MoG
        parameters.
            mask (B,G,12)  -  indicates wether there is data in that frame for each agent
        """
        pi, sigma, mu = pred  #B G 12
        if self.mask:    
            mu = mu*mask  #Don't penalize no-data frames
        target = target.contiguous().view(target.shape[0],-1)  
        prob = pi * self.gaussian_probability(sigma, mu, target)
        nll = -torch.log(torch.sum(prob, dim=1))
        return torch.mean(nll)


    def sample(self,pred):
        """Draw samples from a MoG.
        """
        pi, sigma, mu = pred
        categorical = Categorical(pi)
        pis = list(categorical.sample().data)
        sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_())
        for i, idx in enumerate(pis):
            sample[i] = sample[i].mul(sigma[i,idx]).add(mu[i,idx])
        return sample

    def compute_RMSE_batch(self,pred, gt, mask): 
        pred = pred*mask #B*V,T,C - B grafos en el batch con V_i (variable) nodos cada uno
        gt = gt*mask  # outputmask BV,T,C
        #x2y2_error= torch.sum(torch.where(torch.abs(gt-pred) > 1 , (gt-pred)**2, torch.abs(gt - pred)), dim=-1) 
        x2y2_error=torch.sum((pred-gt)**2,dim=-1) # x^2+y^2 BV,T  PROBABILISTIC -> gt[:,:,:2]
        overall_sum_time = (x2y2_error**0.5).sum(dim=-2)  #T - suma de los errores (x^2+y^2) de los BV agentes
        overall_num = mask.sum(dim=-1).type(torch.int)  #torch.Tensor[(BV,T)] - num de agentes (Y CON DATOS) en cada frame

        return overall_sum_time, overall_num, x2y2_error

    def huber_loss(self, pred, gt, mask, delta):
        pred = pred*mask #B*V,T,C 
        gt = gt*mask  # outputmask BV,T,C
        #error = torch.sum(torch.where(torch.abs(gt-pred) < delta , 0.5*((gt-pred)**2)*(1/delta), torch.abs(gt - pred) - 0.5*delta), dim=-1)   # BV,T
        error = torch.sum(torch.where(torch.abs(gt-pred) < delta , (0.5*(gt-pred)**2), torch.abs(gt - pred)*delta - 0.5*(delta**2)), dim=-1)
        overall_sum_time = error.sum(dim=-2) #T - suma de los errores de los BV agentes
        overall_num = mask.sum(dim=-1).type(torch.int) 
        return overall_sum_time, overall_num



    def training_step(self, train_batch, batch_idx):
        '''needs to return a loss from a single batch'''
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos, maps, scene_id = train_batch
        #last_loc = feats[:,-1:,:2].detach().clone() 
        feats_vel, labels_vel = compute_change_pos(feats,labels_pos[:,:,:2], self.scale_factor)
        feats = torch.cat([feats_vel, feats[:,:,2:]], dim=-1)[:,1:]
        e_w = batched_graph.edata['w'].float()
        if self.model_type != 'gcn' and not self.rel_types:
            e_w= e_w.unsqueeze(1)

        if self.model_type == 'rgcn':
            rel_type = batched_graph.edata['rel_type'].long()
            norm = batched_graph.edata['norm']
            pred = self.model(batched_graph, feats,e_w, rel_type,norm)
        else:
            pred = self.model(batched_graph, feats,e_w,snorm_n,snorm_e, maps)  


        #Probabilistic vs. Deterministic output
        if self.probabilistic:
            #total_loss = self.bivariate_loss(pred, labels, output_masks[:,self.history_frames:self.total_frames,:])
            mask = output_masks.expand(output_masks.shape[0],self.future_frames, 2)  #expand mask (B,Tpred,1) -> (B,T_pred,2)
            total_loss = self.mdn_loss(pred, labels,mask.contiguous().view(mask.shape[0],-1).unsqueeze(1).expand_as(pred[1]))  
        else:
            pred=pred.view(feats.shape[0],self.future_frames,-1)
            #Socially consistent
            #perc_overlap = check_overlap(pred*output_masks) if self.alfa !=0 else 0
            overall_sum_time, overall_num = self.huber_loss(pred, labels_vel, output_masks, self.delta)  #(B,6)
            #overall_sum_time , overall_num, _ = self.compute_RMSE_batch(pred[:,:self.future_frames,:], labels[:,:self.future_frames,:], output_masks[:,self.history_frames:self.total_frames,:])
            total_loss = torch.sum(overall_sum_time)/torch.sum(overall_num.sum(dim=-2))#*(1+self.alfa*perc_overlap) + self.beta*(overall_sum_time[-1]/overall_num.sum(dim=-2)[-1])
            #if self.dataset == 'apollo':
            #total_loss = self.mtp_loss(pred, labels_pos.unsqueeze(1), last_loc.unsqueeze(1), output_masks.unsqueeze(1))

        # Log metrics
        self.log("Sweep/train_loss",  total_loss, on_step=True, on_epoch=False)
        return total_loss
    
    def validation_step(self, val_batch, batch_idx):
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos, maps, scene_id = val_batch
        last_loc = feats[:,-1:,:2].detach().clone() 
        #Rescale last_loc to compare with labels_pos
        if self.scale_factor == 1:
            pass#last_loc = last_loc*12.4354+0.15797
        else:
            rescale_xy=torch.ones((1,1,2), device=self.device)*self.scale_factor
            last_loc = last_loc*rescale_xy
        '''
        if self.dataset == 'apollo':
            #USE CHANGE IN POS AS INPUT
            feats_vel,labels = compute_change_pos(feats,labels_pos)
            #Input pos + heading + vel
            feats = torch.cat([feats_vel, feats[:,:,2:]], dim=-1)[:,1:,:] #torch.cat([feats[:,:,:self.input_dim], feats_vel], dim=-1)
        else:
        '''
        feats_vel, labels_vel = compute_change_pos(feats,labels_pos[:,:,:2], self.scale_factor)
        feats = torch.cat([feats_vel, feats[:,:,2:]], dim=-1)[:,1:]

        e_w = batched_graph.edata['w']
        if self.model_type != 'gcn' and not self.rel_types:
            e_w= e_w.unsqueeze(1)

        if self.model_type == 'rgcn':
            rel_type = batched_graph.edata['rel_type'].long()
            norm = batched_graph.edata['norm']
            pred = self.model(batched_graph, feats, e_w, rel_type,norm)
        else:
            pred = self.model(batched_graph, feats,e_w,snorm_n,snorm_e, maps)
            '''
            mode_prob = model_prediction[:, -NUM_MODES:].clone()
            desired_shape = (model_prediction.shape[0], NUM_MODES, -1, 2)
            trajectories_no_modes = model_prediction[:, :-NUM_MODES].clone().reshape(desired_shape)
            best_mode = np.argmax(mode_prob.detach().cpu().numpy(), axis = 1)
            pred = torch.zeros_like(labels_pos)
            for i, idx in enumerate(best_mode):
                pred[i] = trajectories_no_modes[i,idx]        
            '''
        if self.probabilistic:
            mask = output_masks.expand(output_masks.shape[0],self.future_frames, 2)  #expand mask (B,Tpred,1) -> (B,T_pred,2)
            total_loss = self.mdn_loss(pred, labels,mask.contiguous().view(mask.shape[0],-1).unsqueeze(1).expand_as(pred[1]))  
        else:
            pred=pred.view(feats.shape[0],self.future_frames,-1)
            '''
            #for debugging purposees - comparing directly with training_loss
            overall_sum_time, overall_num = self.huber_loss(pred, labels_vel, output_masks, self.delta)  #(B,6)
            huber_loss = torch.sum(overall_sum_time)/torch.sum(overall_num.sum(dim=-2))
            '''
            for i in range(1,labels_pos.shape[-2]):
                pred[:,i,:] = torch.sum(pred[:,i-1:i+1,:],dim=-2) #BV,6,2 
            pred += last_loc
            
            _ , overall_num, x2y2_error = self.compute_RMSE_batch(pred, labels_pos, output_masks)
            rmse_loss = torch.sum(torch.sum((x2y2_error**0.5), dim=0) / torch.sum(overall_num, dim=0)) / self.future_frames #T->1
            
        #self.log_dict({"Sweep/val_huber_loss": huber_loss, "Sweep/val_rmse_loss": rmse_loss})
        self.log( "Sweep/val_loss", rmse_loss)   
        return rmse_loss
    
    def validation_epoch_end(self, val_loss_over_batches):
        #log best val loss
        if self.wandb and torch.mean(torch.tensor(val_loss_over_batches,device=self.device)) < self.min_val_loss:            
                self.min_val_loss =  torch.mean(torch.tensor(val_loss_over_batches,device=self.device))
                self.logger.experiment.summary["best_val_loss"] = self.min_val_loss


    def test_step(self, test_batch, batch_idx):
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos, maps,scene_id = test_batch
        rescale_xy=torch.ones((1,1,2), device=self.device)*self.scale_factor
        last_loc = feats[:,-1:,:2].detach().clone() 
        feats_vel, _ = compute_change_pos(feats,labels_pos[:,:,:2], self.scale_factor)
        feats = torch.cat([feats_vel, feats[:,:,2:]], dim=-1)[:,1:]
        #Rescale last_loc to compare with labels_pos
        if self.scale_factor == 1:
            pass#last_loc = last_loc*12.4354+0.1579
        else:
            last_loc = last_loc*rescale_xy
        '''
        if self.dataset == 'apollo':
            #USE CHANGE IN POS AS INPUT
            feats_vel,_ = compute_change_pos(feats,labels_pos)
            #Input pos + heading + vel
            feats = torch.cat([feats_vel, feats[:,:,2:]], dim=-1)[:,1:,:] #torch.cat([feats[:,:,:self.input_dim], feats_vel], dim=-1)
        '''
        #feats_vel,_ = compute_change_pos(feats,labels_pos, self.scale_factor)
        #feats = torch.cat([feats_vel, feats[:,:,2:]], dim=-1)[:,1:]

        e_w = batched_graph.edata['w'].float()
        if self.model_type != 'gcn' and not self.rel_types:
            e_w= e_w.unsqueeze(1)

        if self.model_type == 'rgcn':
            rel_type = batched_graph.edata['rel_type'].long()
            norm = batched_graph.edata['norm']
            pred = self.model(batched_graph, feats,e_w, rel_type,norm)
        else:
            pred = self.model(batched_graph, feats,e_w,snorm_n,snorm_e, maps)
            pred=pred.view(feats.shape[0],self.future_frames,-1)
            #mode_prob = model_prediction[:, -NUM_MODES:].clone()
            #desired_shape = (model_prediction.shape[0], NUM_MODES, -1, 2)
            #trajectories_no_modes = model_prediction[:, :-NUM_MODES].clone().reshape(desired_shape)
            #best_mode = np.argmax(mode_prob.detach().cpu().numpy(), axis = 1)
            #pred = torch.zeros_like(labels_pos)
            #for i, idx in enumerate(best_mode):
            #    pred[i] = trajectories_no_modes[i,idx] 


        if self.probabilistic:
            ade = []
            fde = []         
            #En test batch=1 secuencia con n agentes
            #Para el most-likely coger el modo con pi mayor de los 3 y o bien coger muestra de la media 
            for i in range(10): # @top10 Saco el min ADE/FDE por escenario tomando 15 muestras (15 escenarios)
                preds=self.sample(pred)  #N,12
                preds=preds.view(preds.shape[0],self.future_frames,-1)
                #if self.dataset == 'apollo':
                
                for j in range(1,labels_pos.shape[1]):
                    preds[:,j,:] = torch.sum(preds[:,j-1:j+1,:],dim=-2) #6,2 
                preds += last_loc
                
                _ , overall_num, x2y2_error = self.compute_RMSE_batch(preds, labels_pos, output_masks)
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
        
            self.log_dict({'test/ade': min(ade), "test/fde": fde[ade.index(min(ade))]}) #, sync_dist=True
        
        else:
            for i in range(1,labels_pos.shape[-2]):
                pred[:,i,:] = torch.sum(pred[:,i-1:i+1,:],dim=-2) #BV,6,2 
            pred += last_loc
           
            # Compute predicted trajs.                
            _, overall_num, x2y2_error = self.compute_RMSE_batch(pred, labels_pos, output_masks)
            long_err, lat_err, _ = compute_long_lat_error(pred, labels_pos, output_masks)
            overall_loss_time = torch.sum((x2y2_error**0.5),dim=0) / torch.sum(overall_num, dim=0) #T
            overall_loss_time[torch.isnan(overall_loss_time)]=0
            overall_long_err = torch.sum(long_err.detach(),dim=0) / torch.sum(overall_num, dim=0) #T
            overall_lat_err = torch.sum(lat_err.detach(),dim=0) / torch.sum(overall_num, dim=0) #T
            overall_long_err[torch.isnan(overall_long_err)]=0
            overall_lat_err[torch.isnan(overall_lat_err)]=0
            self.overall_loss_time_list.append(overall_loss_time.detach().cpu().numpy())
            self.overall_long_err_list.append(overall_long_err.detach().cpu().numpy())
            self.overall_lat_err_list.append(overall_lat_err.detach().cpu().numpy())
            
            if self.future_frames == 8:
                self.log_dict({'Sweep/test_loss': torch.sum(overall_loss_time), "test/loss_1": overall_loss_time[1:2], "test/loss_2": overall_loss_time[4:5], "test/loss_2.5": overall_loss_time[6:7], "test/loss_3.2": overall_loss_time[-1:] })
            else:
                self.log_dict({'Sweep/test_loss': torch.sum(overall_loss_time)/self.future_frames, "test/loss_1": overall_loss_time[1:2], "test/loss_2": overall_loss_time[3:4], "test/loss_3": overall_loss_time[5:6], "test/loss_4": overall_loss_time[7:8], "test/loss_5": overall_loss_time[9:10], "test/loss_6": overall_loss_time[11:] }) #, sync_dist=True
            
    def on_test_epoch_end(self):
        #wandb_logger.experiment.save(run.name + '.ckpt')
        print(self.lr1, self.lr2)
        if not self.probabilistic:
            overall_loss_time = np.array(self.overall_loss_time_list)
            avg = [sum(overall_loss_time[:,i])/overall_loss_time.shape[0] for i in range(len(overall_loss_time[0]))]
            var = [sum((overall_loss_time[:,i]-avg[i])**2)/overall_loss_time.shape[0] for i in range(len(overall_loss_time[0]))]
            print('Loss variance: ' , var)

            overall_long_err = np.array(self.overall_long_err_list)
            avg_long = [sum(overall_long_err[:,i])/overall_long_err.shape[0] for i in range(len(overall_long_err[0]))]
            var_long = [sum((overall_long_err[:,i]-avg[i])**2)/overall_long_err.shape[0] for i in range(len(overall_long_err[0]))]
            
            overall_lat_err = np.array(self.overall_lat_err_list)
            avg_lat = [sum(overall_lat_err[:,i])/overall_lat_err.shape[0] for i in range(len(overall_lat_err[0]))]
            var_lat = [sum((overall_lat_err[:,i]-avg[i])**2)/overall_lat_err.shape[0] for i in range(len(overall_lat_err[0]))]
            print('\n'.join('Long avg error in frame {}: {:.2f}, var: {:.2f}'.format(i+1, avg, var) for i,(avg,var) in enumerate(zip(avg_long, var_long))))
            print('\n'.join('Lat avg error in frame {}: {:.2f}, var: {:.2f}'.format(i+1, avg, var) for i,(avg,var) in enumerate(zip(avg_lat, var_lat))))

            self.log_dict({ "test/var_s1": torch.tensor(var[1]), "test/var_s2": torch.tensor(var[3]),"test/var_s3": torch.tensor(var[5])})
            if self.future_frames == 5:
                self.log_dict({"var_s4": torch.tensor(var[3]), "var_s5": torch.tensor(var[-1])})
            elif self.future_frames == 12:
                self.log_dict({"test/var_4": torch.tensor(var[7]), "test/var_5": torch.tensor(var[9]), "test/var_6": torch.tensor(var[-1:]) }) #, sync_dist=True

def main(args: Namespace):
    seed=seed_everything(121958)

    if args.dataset == 'apollo':
        history_frames = 6
        future_frames = 6
        train_dataset = ApolloScape_DGLDataset(train_val='train', test=False, rel_types=args.ew_dims>1) #3447
        val_dataset = ApolloScape_DGLDataset(train_val='val', test=False, rel_types=args.ew_dims>1)  #919
        test_dataset = ApolloScape_DGLDataset(train_val='test', test=False, rel_types=args.ew_dims>1)  #230
        print(len(train_dataset), len(val_dataset))
        input_dim = 5
    elif args.dataset == 'ind':
        history_frames = 8
        future_frames = 12
        train_dataset = inD_DGLDataset(train_val='train', history_frames=history_frames, future_frames=future_frames, model_type=args.model_type, classes=(1,2,3,4), rel_types=config.ew_dims>1) #12281
        val_dataset = inD_DGLDataset(train_val='val', history_frames=history_frames, future_frames=future_frames, model_type=args.model_type, classes=(1,2,3,4), rel_types=config.ew_dims>1)  #3509
        test_dataset = inD_DGLDataset(train_val='test', history_frames=history_frames, future_frames=future_frames, model_type=args.model_type, classes=(1,2,3,4), rel_types=config.ew_dims>1)  #1754
        print(len(train_dataset), len(val_dataset), len(test_dataset))
        input_dim = 6
    else:
        history_frames = 4
        future_frames = 12
        train_dataset = nuscenes_Dataset( train_val_test='train',  rel_types=args.ew_dims>1, history_frames=history_frames, future_frames=future_frames) #3447
        val_dataset = nuscenes_Dataset(train_val_test='val',  rel_types=args.ew_dims>1, history_frames=history_frames, future_frames=future_frames)  #919
        test_dataset = nuscenes_Dataset(train_val_test='test', rel_types=args.ew_dims>1, history_frames=history_frames, future_frames=future_frames)  #230
        input_dim = 8


    input_dim_model = input_dim*(history_frames - 1)  #input_dim*(history_frames-1) if config.dataset=='apollo' else input_dim*history_frames
    output_dim = 2*future_frames#3 * (2*future_frames + 1) #if config.probabilistic == False else 5*future_frames

    if args.model_type == 'gat_mdn':
        hidden_dims = args.hidden_dims // args.heads
        model = SCOUT_MDN(input_dim=input_dim_model, hidden_dim=hidden_dims, output_dim=output_dim, heads=args.heads, dropout=args.dropout, bn=False, feat_drop=args.feat_drop, attn_drop=args.attn_drop, att_ew=args.att_ew, ew_type=args.ew_dims>1)
    elif args.model_type == 'gat':
        hidden_dims = args.hidden_dims // args.heads
        model = SCOUT(input_dim=input_dim_model, hidden_dim=hidden_dims, output_dim=output_dim, heads=args.heads, dropout=args.dropout, bn=(args.norm=='bn'), gn=(args.norm=='gn'),
                        feat_drop=args.feat_drop, attn_drop=args.attn_drop, att_ew=args.att_ew, ew_dims=args.ew_dims>1, backbone=args.backbone, freeze=args.freeze)
    elif args.model_type == 'gcn':
        model = model = GCN(in_feats=input_dim_model, hid_feats=args.hidden_dims, out_feats=output_dim, dropout=config.dropout, gcn_drop=config.gcn_drop, bn=config.bn, gcn_bn=config.gcn_bn, embedding=config.embedding)
    elif args.model_type == 'gated':
        model = GatedGCN(input_dim=input_dim_model, hidden_dim=args.hidden_dims, output_dim=output_dim, dropout=args.dropout, bn=args.bn)
    elif args.model_type == 'gated_mdn':
        model = Gated_MDN(input_dim=input_dim_model, hidden_dim=args.hidden_dims, output_dim=output_dim, dropout=args.dropout, bn=args.bn, ew_dims=args.ew_dims>1)
    elif args.model_type == 'baseline':
        model = RNN_baseline(input_dim=5, hidden_dim=args.hidden_dims, output_dim=output_dim, pred_length=config.future_frames, dropout=config.dropout, bn=config.bn)
    elif args.model_type == 'rgcn':
        model = RGCN(in_dim=input_dim_model, h_dim=args.hidden_dims, out_dim=output_dim, num_rels=3, num_bases=-1, num_hidden_layers=2, embedding=True, bn=config.bn, dropout=config.dropout)
    

    LitGNN_sys = LitGNN(model=model, input_dim=input_dim, lr1=args.lr1, lr2=args.lr2, model_type= args.model_type, wd=args.wd, history_frames=history_frames, future_frames= future_frames, alfa= args.alfa,
                        beta = args.beta, delta=args.delta, prob=args.probabilistic, dataset=args.dataset, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, 
                        mask=args.mask, rel_types=args.ew_dims>1, scale_factor=args.scale_factor, wandb = not args.nowandb)  

    early_stop_callback = EarlyStopping('Sweep/val_loss', patience=6)
    
    if not args.nowandb:
        run=wandb.init(job_type="training", entity='sandracl72', project='nuscenes', sync_tensorboard=True)  
        wandb_logger = pl_loggers.WandbLogger() 
        wandb_logger.experiment.log({'seed': seed}) 
        wandb_logger.watch(LitGNN_sys.model, log='all')  #log='all' for params & grads
        if os.environ.get('WANDB_SWEEP_ID') is not None: 
            ckpt_folder = os.path.join(os.environ.get('WANDB_SWEEP_ID'), run.name)
        else:
            ckpt_folder = run.name
        checkpoint_callback = ModelCheckpoint(monitor='Sweep/val_loss', mode='min', dirpath=os.path.join('/media/14TBDISK/sandra/logs/', ckpt_folder))
        trainer = pl.Trainer( weights_summary='full', gpus=args.gpus, deterministic=False, precision=16, logger=wandb_logger, callbacks=[early_stop_callback,checkpoint_callback], profiler=True)  # resume_from_checkpoint=config.path, precision=16, limit_train_batches=0.5, progress_bar_refresh_rate=20,
    else:
        checkpoint_callback = ModelCheckpoint(monitor='Sweep/val_loss', mode='min', dirpath='/media/14TBDISK/sandra/logs/',filename='nowandb-{epoch:02d}.ckpt')
        trainer = pl.Trainer( weights_summary='full', gpus=args.gpus, deterministic=False, precision=16, callbacks=[early_stop_callback,checkpoint_callback], profiler=True) 

    
    if args.ckpt is not None:
        LitGNN_sys = LitGNN.load_from_checkpoint(checkpoint_path=args.ckpt, model=LitGNN_sys.model, history_frames=history_frames, future_frames= future_frames,
                     train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, rel_types=args.ew_dims>1, scale_factor=args.scale_factor)
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
    parser.add_argument("--lr2", type=float, default=1e-4, help="Adam: Base learning rate")
    parser.add_argument("--wd", type=float, default=0.01, help="Adam: weight decay")
    parser.add_argument("--batch_size", type=int, default=512, help="Size of the batches")
    parser.add_argument("--hidden_dims", type=int, default=768)
    parser.add_argument("--model_type", type=str, default='gat', help="Choose model type / aggregation function.")
    parser.add_argument("--dataset", type=str, default='nuscenes', help="Choose dataset.",
                                        choices=['nuscenes', 'ind', 'apollo'])
    parser.add_argument("--norm", type=str, default=None, help="Wether to apply BN or GroupNorm.")
    parser.add_argument("--backbone", type=str, default='resnet18', help="Choose CNN backbone.",
                                        choices=['resnet_gray', 'resnet18', 'resnet50', 'map_encoder', 'None'])
    parser.add_argument('--freeze', type=int, default=7, help="Layers to freeze in resnet18.")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--feat_drop", type=float, default=0.2)
    parser.add_argument("--attn_drop", type=float, default=0.2)
    parser.add_argument("--heads", type=int, default=2, help='Attention heads (GAT)')
    parser.add_argument("--alfa", type=float, default=0, help='Weighting factor of the overlap loss term')
    parser.add_argument("--beta", type=float, default=0, help='Weighting factor of the FDE loss term')
    parser.add_argument("--delta", type=float, default=.1, help='Delta factor in Huber Loss')
    parser.add_argument('--mask', type=str2bool, nargs='?', const=True, default=False, help='use the mask to not taking into account unexisting frames in loss function')  
    parser.add_argument('--probabilistic', action='store_true', help='use probabilistic loss function (MDN)')  
    #parser.add_argument('--att_ew', action='store_true', help='use this flag to add edge features in attention function (GAT)')    
    parser.add_argument('--att_ew', type=str2bool, nargs='?', const=True, default=True, help="Add edge features in attention function (GAT)")

    parser.add_argument('--nowandb', action='store_true', help='use this flag to DISABLE wandb logging')  
    parser.add_argument('--ckpt', type=str, default=None, help='ckpt path for only testing.')   

    
    device=os.environ.get('CUDA_VISIBLE_DEVICES')
    hparams = parser.parse_args()

    main(hparams)
