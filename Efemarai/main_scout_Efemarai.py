import torch
import torch.nn as nn
import os
import sys
import dgl
sys.path.append('../..')
os.environ['DGLBACKEND'] = 'pytorch'
from models.GCN import GCN 
from scout_MTP_Ef import SCOUT_MTP
from torch.utils.data import DataLoader
from ns_dataset_ef import nuscenes_Dataset_Ef, collate_batch_ef
from NuScenes.nuscenes_Dataset import nuscenes_Dataset, collate_batch
import wandb
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from argparse import ArgumentParser, Namespace
import math
import efemarai as ef
from utils import str2bool, compute_change_pos, MTPLoss


ef.register_assertion(ef.assertions.NoDeadReLULayers(threshold=0.5))
ef.register_assertion(ef.assertions.ValidInputsDiscreteNLLLoss())
ef.register_assertion(ef.assertions.ValidInputsDiscreteKLDivLoss())
ef.register_assertion(ef.assertions.NoVanishingTensors(threshold=1e-5))
ef.register_assertion(ef.assertions.NoZeroGradientsAssertion())
print(f'registered assertions: {ef.get_registered_assertions()}')


ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)

class LitGNN(pl.LightningModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, dataset, history_frames: int=3, future_frames: int=3, 
                        model: nn.Module = GCN, prob: bool = False, wandb: bool = True, reg_loss_w: float = 1.):
        super().__init__()
        self.model= model
        self.history_frames =history_frames
        self.future_frames = future_frames
        self.total_frames = history_frames + future_frames
        self.min_val_loss = 100
        self.dataset = dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.wandb = wandb
        
        self.automatic_optimization = False
        self.mtp_loss = MTPLoss(num_modes = hparams.num_modes, regression_loss_weight = reg_loss_w, angle_threshold_degrees = 5.)

    def forward(self, graph, feats,e_w,snorm_n,snorm_e, maps):
        pred = self.model(graph, feats,e_w,snorm_n,snorm_e, maps)
        return pred

    def configure_optimizers(self):
        opt = torch.optim.AdamW([
                {'params': self.model.base.parameters()},
                {'params': self.model.embeddings.parameters(), 'lr': hparams.lr1}], lr=hparams.lr2, weight_decay=hparams.wd)
    
        return {
            'optimizer': opt,
            'lr_scheduler':  torch.optim.lr_scheduler.CosineAnnealingLR(opt, 100), #   ##torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, threshold=0.001, patience=4, verbose=True),
            'monitor': "Sweep/val_rmse_loss"
        }


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1,num_workers=1, shuffle=True,  collate_fn=collate_batch)

    def val_dataloader(self):
        return  DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=8,collate_fn=collate_batch)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False,num_workers=1, collate_fn=collate_batch) 

    def compute_RMSE_batch(self,pred, gt, mask): 
        pred = pred*mask 
        gt = gt*mask  
        x2y2_error = torch.sum((pred-gt)**2,dim=-1)

        with torch.no_grad():
            index = torch.tensor( [ mask_i.nonzero()[-1][0] if mask_i.any() else 0 for mask_i in mask.squeeze(-1) ], device=pred.device)
        
        fde_error = (x2y2_error[torch.arange(gt.shape[0]),index]**0.5).unsqueeze(1).sum(dim=-2) 
        ade_error = (x2y2_error**0.5).sum(dim=-2) 

        overall_num = mask.sum(dim = -1) 
        fde_num = 0

        return ade_error, fde_error, overall_num, fde_num, x2y2_error


    def huber_loss(self, pred, gt, mask, delta):
        pred = pred*mask 
        gt = gt*mask 
        huber_error = torch.sum(torch.where(torch.abs(gt-pred) < delta , (0.5*(gt-pred)**2), torch.abs(gt - pred)*delta - 0.5*(delta**2)), dim=-1) 
        
        with torch.no_grad():
            index = torch.tensor( [ mask_i.nonzero()[-1][0] if mask_i.any() else 0 for mask_i in mask.squeeze() ], device=huber_error.device)
        
        fde_error =  huber_error[torch.arange(gt.shape[0]),index].unsqueeze(1).sum(dim=-2)
        ade_error = huber_error.sum(dim=-2) 

        overall_num = mask.sum(dim = -1)
        fde_num = mask[torch.arange(gt.shape[0]),index]
        return ade_error, fde_error, overall_num, fde_num



    def training_step(self, train_batch, batch_idx):
        '''needs to return a loss from a single batch'''
        #batched_graph, output_masks, snorm_n, snorm_e, feats, labels, maps, e_w = train_batch
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels, maps, scenes = train_batch
        e_w = batched_graph.edata['w'].float()
        if hparams.model_type != 'gcn' and not hparams.ew_dims > 1:
            e_w= e_w.unsqueeze(1)

        last_loc =  torch.rand(6, 1,1,2)

        opt = self.optimizers(use_pl_optimizer=True)
        #ef.add_view(maps, view=ef.View.Image)
        #ef.inspect(maps, name=f'maps', view=ef.View.Image)
        with ef.scan(wait=True):
            pred = self.model(batched_graph, feats,e_w,snorm_n,snorm_e, maps)  
            opt.zero_grad()
            total_loss, regression_loss, class_loss,_ = self.mtp_loss(pred, labels, last_loc, output_masks)
            self.manual_backward(total_loss)

        opt.step()
        sch = self.lr_schedulers()
        sch.step()
        
        self.log_dict({"Sweep/train_loss":  total_loss,  "Sweep/train_huber_loss": regression_loss, "Sweep/train_class_loss": class_loss})


    def validation_step(self, val_batch, batch_idx):
        #batched_graph, output_masks, snorm_n, snorm_e, feats, labels, maps, e_w = val_batch
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels, maps, scenes = val_batch
        e_w = batched_graph.edata['w'].float()
        if hparams.model_type != 'gcn' and not hparams.ew_dims > 1:
            e_w= e_w.unsqueeze(1)
        last_loc =  torch.rand(6, 1,1,2)

        pred = self.model(batched_graph, feats,e_w,snorm_n,snorm_e, maps)
        
        total_loss, regression_loss, class_loss,_ = self.mtp_loss(pred, labels, last_loc, output_masks)

        self.log_dict({"Sweep/val_class_loss": class_loss, "Sweep/val_rmse_loss": regression_loss})
        return regression_loss


    def test_step(self, test_batch, batch_idx):
        batched_graph, output_masks, snorm_n, snorm_e, feats, labels, maps, e_w = test_batch
        last_loc =  torch.rand(6, 1,1,2)

        pred = self.model(batched_graph, feats,e_w,snorm_n,snorm_e, maps)
        
        avg_loss, regression_loss, class_loss, pred = self.mtp_loss(pred, labels_vel.unsqueeze(1), last_loc.unsqueeze(1), output_masks.unsqueeze(1))

        for j in range(1,labels.shape[1]):
            pred[:,j,:] = torch.sum(pred[:,j-1:j+1,:],dim=-2) 
        
        ade_error, fde_error, overall_num, fde_num, x2y2_error = self.compute_RMSE_batch(pred, labels, output_masks)
        overall_loss_time = ade_error / torch.sum(overall_num, dim=0)#T
        overall_loss_time[torch.isnan(overall_loss_time)]=0

        self.log_dict({'Sweep/test_loss': torch.sum(overall_loss_time)/self.future_frames, "test/loss_1": overall_loss_time[1:2], "test/loss_2": overall_loss_time[3:4], "test/loss_3": overall_loss_time[5:6], "test/loss_4": overall_loss_time[7:8], "test/loss_5": overall_loss_time[9:10], "test/loss_6": overall_loss_time[11:] }) #, sync_dist=True
        

        

def main(args: Namespace):
    seed=seed_everything(121958)

    history_frames = 7
    future_frames = 10
    #dataset = nuscenes_Dataset_ef(future_frames, history_frames)  #230
    dataset = nuscenes_Dataset( train_val_test='train',  rel_types=args.ew_dims>1, history_frames=history_frames, future_frames=future_frames, local_frame = args.local_frame) #3447
        
    output_dim = (2*future_frames+ 1)  

    input_dim_model = 7 * (history_frames-1) if args.emb_type == 'emb' else 7
    model = SCOUT_MTP(input_dim=input_dim_model, hidden_dim=args.hidden_dims, emb_dim=args.emb_dims, emb_type=args.emb_type, 
            output_dim=output_dim, heads=args.heads, dropout=args.dropout, bn=(args.norm=='bn'), gn=(args.norm=='gn'), num_modes=args.num_modes,
            feat_drop=args.feat_drop, attn_drop=args.attn_drop, att_ew=args.att_ew, ew_dims=args.ew_dims>1, backbone=args.backbone, freeze=args.freeze)
    

    LitGNN_sys = LitGNN(model=model, history_frames=history_frames, future_frames= future_frames, dataset=args.dataset, 
                        train_dataset=dataset, val_dataset=dataset, test_dataset=dataset, wandb = not args.nowandb)  

    early_stop_callback = EarlyStopping('Sweep/val_rmse_loss', patience=10)

    if not args.nowandb: #or args.ckpt is not None:
        run=wandb.init(job_type="training", entity='sandracl72', project='nuscenes', sync_tensorboard=True)  
        wandb_logger = pl_loggers.WandbLogger() 
        wandb_logger.experiment.log({'seed': seed}) 
        #wandb_logger.watch(LitGNN_sys.model, log='')  #log='all' for params & grads
        if os.environ.get('WANDB_SWEEP_ID') is not None: 
            ckpt_folder = os.path.join(os.environ.get('WANDB_SWEEP_ID'), run.name)
        else:
            ckpt_folder = run.name
        checkpoint_callback = ModelCheckpoint(monitor='Sweep/val_rmse_loss', mode='min',  save_last=True, dirpath=os.path.join('/media/14TBDISK/sandra/logs/', ckpt_folder))
        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer = pl.Trainer(track_grad_norm=2,stochastic_weight_avg=True, weights_summary='full', gpus=args.gpus, deterministic=False, precision=16, logger=wandb_logger, callbacks=[early_stop_callback,checkpoint_callback, lr_monitor], profiler='simple')  # track_grad_norm=2,  resume_from_checkpoint=config.path, precision=16, limit_train_batches=0.5, progress_bar_refresh_rate=20,
    else:
        checkpoint_callback = ModelCheckpoint(monitor='Sweep/val_rmse_loss', mode='min', save_last=True,dirpath='/media/14TBDISK/sandra/logs/',filename='nowandb-{epoch:02d}.ckpt')
        trainer = pl.Trainer(track_grad_norm=2,stochastic_weight_avg=True, weights_summary='full', gpus=args.gpus, deterministic=True, precision=16, callbacks=[early_stop_callback,checkpoint_callback], profiler='simple') 

        print("############### TRAIN ####################")
        trainer.fit(LitGNN_sys)
        print('Model checkpoint path:',trainer.checkpoint_callback.best_model_path)

        print("############### TEST ####################")
        trainer.test(ckpt_path='best')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--scale_factor", type=int, default=1, help="Wether to scale x,y global positions (zero-centralized)")
    parser.add_argument("--ew_dims", type=int, default=2, choices=[1,2], help="Edge features: 1 for relative position, 2 for adding relationship type.")
    parser.add_argument("--lr1", type=float, default=1e-4, help="Adam: Embedding learning rate")
    parser.add_argument("--lr2", type=float, default=1e-3, help="Adam: Base learning rate")
    parser.add_argument("--wd", type=float, default=0.01, help="Adam: weight decay")
    parser.add_argument("--batch_size", type=int, default=128, help="Size of the batches")
    parser.add_argument("--hidden_dims", type=int, default=512)
    parser.add_argument("--emb_dims", type=int, default=512)
    parser.add_argument("--emb_type", type=str, default='emb', choices=['emb', 'pos_enc', 'gru'])
    parser.add_argument("--model_type", type=str, default='mtp', help="Choose model type / aggregation function.")
    parser.add_argument("--dataset", type=str, default='nuscenes', help="Choose dataset.",
                                        choices=['nuscenes', 'ind', 'apollo'])
    parser.add_argument('--local_frame',  type=str2bool, nargs='?', const=True, default=True, help='whether to use local or global features.')   
    parser.add_argument("--norm", type=str, default=None, help="Wether to apply BN or GroupNorm.")
    parser.add_argument("--backbone", type=str, default='resnet18', help="Choose CNN backbone.",
                                        choices=['resnet_gray', 'None', 'resnet18', 'resnet50', 'mobilenet', 'map_encoder'])
    parser.add_argument('--freeze', type=int, default=100, help="Layers to freeze in resnet18.")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--feat_drop", type=float, default=0.)
    parser.add_argument("--attn_drop", type=float, default=0.4)
    parser.add_argument("--heads", type=int, default=1, help='Attention heads (GAT)')
    parser.add_argument("--num_modes", type=int, default=3)
    parser.add_argument("--alfa", type=float, default=0, help='Weighting factor of the overlap loss term')
    parser.add_argument("--beta", type=float, default=0, help='Weighting factor of the FDE loss term')
    parser.add_argument("--delta", type=float, default=.001, help='Delta factor in Huber Loss')
    parser.add_argument('--mask', type=str2bool, nargs='?', const=True, default=False, help='use the mask to not taking into account unexisting frames in loss function')  
    parser.add_argument('--probabilistic', action='store_true', help='use probabilistic loss function (MDN)')  
    #parser.add_argument('--att_ew', action='store_true', help='use this flag to add edge features in attention function (GAT)')    
    parser.add_argument('--att_ew', type=str2bool, nargs='?', const=True, default=True, help="Add edge features in attention function (GAT)")
    parser.add_argument('--nowandb', action='store_true', help='use this flag to DISABLE wandb logging')  
    parser.add_argument('--ckpt', type=str, default=None, help='ckpt path for only testing.')     
    parser.add_argument('--reg_loss_w', type=float, default=1)   

    


    device=os.environ.get('CUDA_VISIBLE_DEVICES')
    hparams = parser.parse_args()

    main(hparams)