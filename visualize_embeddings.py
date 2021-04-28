import torch
import numpy as np
import os
import cv2
import sys
sys.path.append('./NuScenes')
from tensorboardX import SummaryWriter
from NuScenes.main_GNN_VAE_nuscenes import LitGNN
from argparse import ArgumentParser, Namespace
from models.VAE_GNN import VAE_GNN
from models.VAE_PRIOR import VAE_GNN_prior
from NuScenes.nuscenes_Dataset import nuscenes_Dataset
from torch.utils.data import DataLoader


IMGS_FOLDER = './NuScenes/Outputs_VisEmb/Images'
EMBS_FOLDER = './NuScenes/Outputs_VisEmb/Embeddings'
TB_FOLDER = './NuScenes/Outputs_VisEmb/Tensorboard'

def collate_batch(samples):
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



class DeepFeatures(torch.nn.Module):
    '''
    This class extracts, reads, and writes data embeddings using a pretrained deep neural network. Meant to work with 
    Tensorboard's Embedding Viewer (https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin).
    When using with a 3 channel image input and a pretrained model from torchvision.models please use the 
    following pre-processing pipeline:
    
    transforms.Compose([transforms.Resize(imsize), 
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]) ## As per torchvision docs
    
    Args:
        model (nn.Module): A Pytorch model that returns an (B,1) embedding for a length B batched input
        imgs_folder (str): The folder path where the input data elements should be written to
        embs_folder (str): The folder path where the output embeddings should be written to
        tensorboard_folder (str): The folder path where the resulting Tensorboard log should be written to
        experiment_name (str): The name of the experiment to use as the log name
    '''

    def __init__(self, model,
                 imgs_folder,
                 embs_folder, 
                 tensorboard_folder,
                 experiment_name=None):
        
        super(DeepFeatures, self).__init__()
        
        self.model = model
        self.model.eval()
        
        self.imgs_folder = imgs_folder
        self.embs_folder = embs_folder
        self.tensorboard_folder = tensorboard_folder
        
        self.name = experiment_name
        
        self.writer = None
             
    
    def generate_embeddings(self,  g, feats, e_w, snorm_n,snorm_e, maps):
        '''
        Generate embeddings for an input batched tensor
        
        Args:
            x (torch.Tensor) : A batched pytorch tensor
            
        Returns:
            (torch.Tensor): The output of self.model against x
        '''
        y, mu, logvar = self.model.inference(g, feats, e_w, snorm_n,snorm_e, maps)
        return mu   #puede ser el encoder/prior para sacar las medias de z o salida del backbone
    
    
    def write_embeddings(self, g, feats, e_w, snorm_n,snorm_e, maps, outsize=(50,50)):
        '''
        Generate embeddings for an input batched tensor and write inputs and 
        embeddings to self.imgs_folder and self.embs_folder respectively. 
        
        Inputs and outputs will be stored in .npy format with randomly generated
        matching filenames for retrieval
        
        Args:
            x (torch.Tensor) : An input batched tensor that can be consumed by self.model
            outsize (tuple(int, int)) : A tuple indicating the size that input data arrays should be
            written out to
            
        Returns: 
            (bool) : True if writing was succesful
        
        '''
        
        assert len(os.listdir(self.imgs_folder))==0, "Images folder must be empty"
        assert len(os.listdir(self.embs_folder))==0, "Embeddings folder must be empty"
        
        # Generate embeddings ( Z means Nx25)
        embs = self.generate_embeddings(g, feats, e_w, snorm_n,snorm_e, maps)
        
        # Detach from graph
        embs = embs.detach().cpu().numpy()
            
        # Start writing to output folders
        for i in range(len(embs)):
            key = str(np.random.random())[-7:]
            np.save(self.imgs_folder + r"/" + key + '.npy', tensor2np(maps[i], outsize))  #aunque lo haga con Z, la img me sirve como label
            np.save(self.embs_folder + r"/" + key + '.npy', embs[i])
        return(True)
    
    
    def _create_writer(self, name):
        '''
        Create a TensorboardX writer object given an experiment name and assigns it to self.writer
        
        Args:
            name (str): Optional, an experiment name for the writer, defaults to self.name
        
        Returns:
            (bool): True if writer was created succesfully
        
        '''
        
        if self.name is None:
            name = 'Experiment_' + str(np.random.random())
        else:
            name = self.name
        
        dir_name = os.path.join(self.tensorboard_folder, 
                                name)
        
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        
        else:
            print("Warning: logfile already exists")
            print("logging directory: " + str(dir_name))
        
        logdir = dir_name
        self.writer = SummaryWriter(logdir=logdir)
        return(True)

    
    
    def create_tensorboard_log(self):        
        '''
        Write all images and embeddings from imgs_folder and embs_folder into a tensorboard log
        '''
        
        if self.writer is None:
            self._create_writer(self.name)
        
        
        ## Read in
        all_embeddings = [np.load(os.path.join(self.embs_folder, p)) for p in os.listdir(self.embs_folder) if p.endswith('.npy')]
        all_images = [np.load(os.path.join(self.imgs_folder, p)) for p in os.listdir(self.imgs_folder) if p.endswith('.npy')]
        all_images = [np.moveaxis(a, 2, 0) for a in all_images] # (HWC) -> (CHW)

        ## Stack into tensors
        all_embeddings = torch.Tensor(all_embeddings)
        all_images = torch.Tensor(all_images)

        print(all_embeddings.shape)
        print(all_images.shape)

        self.writer.add_embedding(all_embeddings, label_img = all_images)

        

def tensor2np(tensor, resize_to=None):
    '''
    Convert an image tensor to a numpy image array and resize
    
    Args:
        tensor (torch.Tensor): The input tensor that should be converted
        resize_to (tuple(int, int)): The desired output size of the array
        
    Returns:
        (np.ndarray): The input tensor converted to a channel last resized array
    '''
    
    out_array = tensor.detach().cpu().numpy()
    out_array = np.moveaxis(out_array, 0, 2) # (CHW) -> (HWC)
    
    if resize_to is not None:
        out_array = cv2.resize(out_array, dsize=resize_to, interpolation=cv2.INTER_CUBIC)
    
    return(out_array)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--model_type", type = str, default='vae_prior', choices = ['vae_gat', 'vae_gated', 'vae_prior'])
    parser.add_argument('--ckpt', type=str, default=None, help='ckpt path.')   
    parser.add_argument("--ew_dims", type=int, default=1, choices=[1,2], help="Edge features: 1 for relative position, 2 for adding relationship type.")
    parser.add_argument('--freeze', type=int, default=7, help="Layers to freeze in resnet18.")
    parser.add_argument("--z_dims", type=int, default=25, help="Dimensionality of the latent space")
    parser.add_argument("--hidden_dims", type=int, default=256)
    parser.add_argument("--backbone", type=str, default='resnet', help="Choose CNN backbone.",
                                        choices=['resnet_gray', 'resnet', 'map_encoder'])
    parser.add_argument("--norm", type=str, default=None, help="Wether to apply BN (bn) or GroupNorm (gn).")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--feat_drop", type=float, default=0.)
    parser.add_argument("--attn_drop", type=float, default=0.25)
    parser.add_argument("--heads", type=int, default=2, help='Attention heads (GAT)')
    
    args = parser.parse_args()

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
    


    test_dataset = nuscenes_Dataset(train_val_test = 'test', rel_types = LitGNN_sys.rel_types, history_frames=LitGNN_sys.history_frames, future_frames=LitGNN_sys.future_frames, challenge_eval=True)  
    test_dataloader = DataLoader(test_dataset, batch_size=3, shuffle=False, collate_fn=collate_batch)

    LitGNN_sys = LitGNN.load_from_checkpoint(checkpoint_path=args.ckpt, model=model, history_frames=history_frames, future_frames= future_frames,
                    test_dataset=test_dataset, rel_types=args.ew_dims>1, model_type=args.model_type)

    DF = DeepFeatures(model = LitGNN_sys.model, imgs_folder = IMGS_FOLDER, embs_folder = EMBS_FOLDER, tensorboard_folder = TB_FOLDER)
    
    for batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos, tokens_eval, scene_id, mean_xy, maps in test_batch:
        if scene_id != self.scene_id:
            return 

    DF.write_embeddings(g.to('cuda'), feats.to('cuda'), e_w.to('cuda'), snorm_n,snorm_e, maps.to('cuda'))
    DF.create_tensorboard_log()
