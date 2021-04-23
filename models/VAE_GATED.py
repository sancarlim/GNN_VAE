import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['DGLBACKEND'] = 'pytorch'
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from NuScenes.nuscenes_Dataset import nuscenes_Dataset, collate_batch
from torch.utils.data import DataLoader
from models.VAE_GNN import MLP_Dec, MLP_Enc



class GatedGCN_layer(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.A = nn.Linear(input_dim, output_dim)
        self.B = nn.Linear(input_dim, output_dim)
        self.C = nn.Linear(input_dim, output_dim)
        self.D = nn.Linear(input_dim, output_dim)
        self.E = nn.Linear(input_dim, output_dim)
        self.bn_node_h = nn.BatchNorm1d(output_dim)  
        self.bn_node_e = nn.BatchNorm1d(output_dim) #nn.GroupNorm(32, output_dim) 

        self.reset_parameters()
    
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.A.weight,gain=gain)
        nn.init.xavier_normal_(self.B.weight, gain=gain)
        nn.init.xavier_normal_(self.C.weight, gain=gain) #sigmoid -> relu
        nn.init.xavier_normal_(self.D.weight, gain=gain)
        nn.init.xavier_normal_(self.E.weight, gain=gain)


    def message_func(self, edges):
        Bh_j = edges.src['Bh'] #n_e,256
        # e_ij = Ce_ij + Dhi + Ehj   N*B,256
        e_ij = edges.data['Ce'] + edges.src['Dh'] + edges.dst['Eh'] #n_e,256
        edges.data['e'] = e_ij
        return {'Bh_j' : Bh_j, 'e_ij' : e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        e = nodes.mailbox['e_ij']
        #torch.clamp(e.sigmoid_(), min=1e-4, max=1-1e-4) 
        sigma_ij = torch.clamp(torch.sigmoid(e), min=1e-4, max=1-1e-4) 
        # hi = Ahi + sum_j eta_ij * Bhj   
        h = Ah_i + torch.sum(sigma_ij * Bh_j, dim=1) / torch.sum(sigma_ij, dim=1)  #shape n_nodes*256
        
        return {'h' : h}
    
    def forward(self, g, h, e, snorm_n, snorm_e):
        with g.local_scope():
            h_in = h # residual connection
            e_in = e # residual connection
            
            
            g.ndata['h']  = h
            g.ndata['Ah'] = self.A(h) 
            g.ndata['Bh'] = self.B(h) 
            g.ndata['Dh'] = self.D(h)
            g.ndata['Eh'] = self.E(h) 
            g.edata['e']  = e 
            g.edata['Ce'] = self.C(e)
            
            g.update_all(self.message_func, self.reduce_func)
            
            h = g.ndata['h'] # result of graph convolution
            e = g.edata['e'] # result of graph convolution

            h = h * snorm_n # normalize activation w.r.t. graph node size
            e = e * snorm_e # normalize activation w.r.t. graph edge size
            
            h = self.bn_node_h(h) # batch normalization  
            e = self.bn_node_e(e) # batch normalization  
            
            h = torch.relu(h) # non-linear activation
            e = torch.relu(e) # non-linear activation
            
            h = h_in + h # residual connection
            e = e_in + e # residual connection


            return h, e

    
class GATED_VAE(nn.Module):
    def __init__(self, hidden_dim, fc=False, dropout=0.2):
        super().__init__()
        self.fc = fc
        
        self.GatedGCN1 = GatedGCN_layer(hidden_dim, hidden_dim)
        self.GatedGCN2 = GatedGCN_layer(hidden_dim, hidden_dim)

        if fc:
            self.dropout_l = nn.Dropout(dropout)
            self.linear = nn.Linear(hidden_dim, hidden_dim//2)
            nn.init.kaiming_normal_(self.linear.weight, a=0.01, nonlinearity='leaky_relu') 
        
    
    def forward(self, g, h, e, snorm_n, snorm_e):
        h, e = self.GatedGCN1(g, h, e, snorm_n, snorm_e)
        h, e = self.GatedGCN2(g, h, e, snorm_n, snorm_e)
        if self.fc:
            h = self.dropout_l(h)
            h=F.leaky_relu(self.linear(h))
        return h, e
    

class VAE_GATED(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, output_dim, fc=False, dropout=0.2,  ew_dims=1,  backbone='map_encoder', freeze=6,
                    bn=False, gn=False):
        super().__init__()
        self.fc = fc
        self.z_dim = z_dim
        self.bn = bn
        self.gn = gn

        ###############
        # Map Encoder #
        ###############
        if backbone == 'map_encoder':
            self.feature_extractor = My_MapEncoder(input_channels = 1, input_size=112, 
                                                    hidden_channels = [10,32,64,128,256], output_size = hidden_dim, 
                                                    kernels = [5,5,3,3,3], strides = [1,2,2,2,2])
            enc_dims = hidden_dim*2+output_dim    
            dec_dims = z_dim + hidden_dim*2
        
        elif backbone == 'resnet':       
            model_ft = resnet18(pretrained=True)
            modules = list(model_ft.children())[:-3]
            modules.append(torch.nn.AdaptiveAvgPool2d((1, 1))) 
            self.feature_extractor = torch.nn.Sequential(*modules) 
            ct=0
            for child in self.feature_extractor.children():
                ct+=1
                if ct < freeze:  #freeze 2 BasicBlocks , train last one 128 -> 256
                    for param in child.parameters():
                        param.requires_grad = False
            enc_dims = hidden_dim + output_dim + 256
            dec_dims = z_dim + hidden_dim + 256
        
        elif backbone == 'resnet_gray':
            resnet = resnet18(pretrained=False)
            modules = list(resnet.children())[:-3]
            modules.append(torch.nn.AdaptiveAvgPool2d((1, 1))) 
            modules[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)  #stride=1 if list[:-1]
            nn.init.kaiming_normal_(modules[0].weight, mode='fan_out', nonlinearity='relu')
            self.feature_extractor=torch.nn.Sequential(*modules)   
            enc_dims = hidden_dim + output_dim + 256
            dec_dims = z_dim + hidden_dim + 256


        ############################
        # Input Features Embedding #
        ############################
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_e = nn.Linear(ew_dims, enc_dims)
        
        #############
        #  ENCODER  #
        #############
        self.GNN_enc = GATED_VAE(enc_dims, fc=fc, dropout=dropout)
        self.MLP_encoder = MLP_Enc(enc_dims, z_dim, dropout=dropout)

        #############
        #  DECODER  #
        ############# 
        self.embedding_e_dec = nn.Linear(ew_dims, dec_dims)
        self.GNN_decoder = GATED_VAE(dec_dims, fc=fc, dropout=dropout) 
        self.MLP_decoder = MLP_Dec(dec_dims+z_dim, dec_dims, output_dim, dropout)

        self.base = nn.ModuleList([
            self.embedding_e,
            self.embedding_e_dec,
            self.GNN_enc,
            self.MLP_encoder,
            self.GNN_decoder,
            self.MLP_decoder
        ])

        if self.bn:
            self.bn_enc = nn.BatchNorm1d(enc_dims) 
            self.bn_dec = nn.BatchNorm1d(dec_dims) 
            self.base.append(self.bn_enc)
            self.base.append(self.bn_dec)
        elif self.gn:
            self.gn_enc = nn.GroupNorm(32, enc_dims)
            self.gn_dec = nn.GroupNorm(32, dec_dims)
            self.base.append(self.gn_enc)
            self.base.append(self.gn_dec)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.bn:
            nn.init.constant_(self.bn_enc.weight, 1)
            nn.init.constant_(self.bn_enc.bias, 0)
            nn.init.constant_(self.bn_dec.weight, 1)
            nn.init.constant_(self.bn_dec.bias, 0)
        elif self.gn:
            nn.init.constant_(self.gn_enc.weight, 1)
            nn.init.constant_(self.gn_enc.bias, 0)
            nn.init.constant_(self.gn_dec.weight, 1)
            nn.init.constant_(self.gn_dec.bias, 0)
        nn.init.xavier_normal_(self.embedding_h.weight)
        nn.init.xavier_normal_(self.embedding_e.weight)          
        nn.init.xavier_normal_(self.embedding_e_dec.weight) 
    
    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def inference(self, g, feats, e_w, snorm_n,snorm_e, maps):
        """
        Samples from a normal distribution and decodes conditioned to the GNN outputs.   
        """
        # Reshape from (B*V,T,C) to (B*V,T*C) 
        feats = feats.contiguous().view(feats.shape[0],-1)
        
        # Input embedding
        h_emb = self.embedding_h(feats)  #input (N, 24)- (N,hid)
        e_emb = self.embedding_e_dec(e_w)
        g.edata['w']=e_emb

        # Maps feature extraction
        maps_emb = self.feature_extractor(maps)

        #Sample from gaussian distribution (BV, Z_dim)
        z_sample = torch.distributions.Normal(torch.zeros((h_emb.shape[0],self.z_dim), dtype=h_emb.dtype, device=h_emb.device), 
                                              torch.ones((h_emb.shape[0],self.z_dim), dtype=h_emb.dtype, device=h_emb.device)).sample()
        
        #DECODE 
        h_dec = torch.cat([maps_emb.flatten(start_dim=1), h_emb, z_sample],dim=-1)
        if self.bn:
            h_dec = self.bn_dec(h_dec)
        elif self.gn:
            h_dec = self.gn_dec(h_dec)
        h, _ = self.GNN_decoder(g,h_dec,e_emb,snorm_n, snorm_e)
        h = torch.cat([h, z_sample],dim=-1)
        recon_y = self.MLP_decoder(h)
        return recon_y
    
    def forward(self, g, feats, e_w, snorm_n, snorm_e, gt, maps):
        # Reshape from (B*V,T,C) to (B*V,T*C) 
        feats = feats.contiguous().view(feats.shape[0],-1)
        gt = gt.contiguous().view(gt.shape[0],-1)

        # Input embedding
        h_emb = self.embedding_h(feats)  #input (N, 24)- (N,hid)
        e_emb = self.embedding_e(e_w)
        g.edata['w']=e_emb

        # Maps feature extraction
        maps_emb = self.feature_extractor(maps)

        # Embeddings concatenation
        h = torch.cat([maps_emb.flatten(start_dim=1), h_emb, gt], dim=-1)
        if self.bn:
            h = self.bn_enc(h)
        elif self.gn:
            h = self.gn_enc(h)

        #### ENCODE ####
        # Embeddings concatenation
        h = torch.cat([maps_emb.flatten(start_dim=1), h_emb, gt], dim=-1)
        h, e = self.GNN_enc(g, h, e_emb, snorm_n, snorm_e)
        mu, log_var = self.MLP_encoder(h) 
        
        #### Sample from the latent distribution ###
        z_sample = self.reparameterize(mu, log_var)
        
        #### DECODE #### 
        h_dec = torch.cat([maps_emb.flatten(start_dim=1), h_emb, z_sample],dim=-1)
        if self.bn:
            h = self.bn_dec(h_dec)
        elif self.gn:
            h = self.gn_dec(h_dec)
        
        #Embedding for having dimmensions of edge feats = dimmensions of node feats
        e_dec = self.embedding_e_dec(e_w)
        h, _ = self.GNN_decoder(g,h_dec,e_dec,snorm_n, snorm_e)
        h = torch.cat([h, z_sample],dim=-1)
        recon_y = self.MLP_decoder(h)
        return recon_y, mu, log_var

if __name__ == '__main__':
    history_frames = 4
    future_frames = 12
    hidden_dims = 768
    heads = 2

    input_dim = 9*history_frames
    output_dim = 2*future_frames 

    model = VAE_GATED(input_dim, hidden_dims, 16, output_dim, bn=True,fc=False, dropout=0.2, ew_dims=2, backbone='resnet')
    print(model)

    test_dataset = nuscenes_Dataset(train_val_test='val', rel_types=True, history_frames=history_frames, future_frames=future_frames) 
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_batch)

    for batch in test_dataloader:
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos, maps = batch
        e_w = batched_graph.edata['w']
        y, mu, log_var = model(batched_graph, feats,e_w,snorm_n,snorm_e, labels_pos, maps)
        print(y.shape)
    
    