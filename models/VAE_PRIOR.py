import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['DGLBACKEND'] = 'pytorch'
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet18, resnet50
from NuScenes.nuscenes_Dataset import nuscenes_Dataset, collate_batch
from torch.utils.data import DataLoader
from models.MapEncoder import My_MapEncoder, ResNet50, ResNet18
from models.VAE_GNN import GAT_VAE
from torchsummary import summary

from models.backbone import MobileNetBackbone, ResNetBackbone, calculate_backbone_feature_dim

class MLP_Enc(nn.Module):
    "Encoder: MLP that takes GNN output as input and returns mu and log variance of the latent distribution."
    "The stddev of the distribution is treated as the log of the variance of the normal distribution for numerical stability."
    def __init__(self, in_dim, z_dim, dropout=0.2):
        super(MLP_Enc, self).__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.linear = nn.Linear(in_dim, in_dim//2)
        self.log_var = nn.Linear(in_dim//2, z_dim)
        self.mu = nn.Linear(in_dim//2, z_dim)
        self.dropout_l = nn.Dropout(dropout)

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.normal_(self.log_var.weight, 0, sqrt(1. / self.sigma.in_dim))
        nn.init.xavier_normal_(self.mu.weight)
        nn.init.kaiming_normal_(self.linear.weight, a=0.01, nonlinearity='leaky_relu')

    def forward(self, h):
        h = self.dropout_l(h)
        h = F.leaky_relu(self.linear(h))
        h = self.dropout_l(h)
        log_var = self.log_var(h) 
        mu = self.mu(h)
        return mu, log_var


class MLP_Dec(nn.Module):
    def __init__(self, in_dim, hid_dim, output_dim, dropout=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.dropout = nn.Dropout(dropout)
        self.linear0 = nn.Linear(in_dim, hid_dim)
        self.linear1 = nn.Linear(hid_dim, output_dim) 

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.kaiming_normal_(self.linear0.weight, a=0.02, nonlinearity='leaky_relu')

    def forward(self, h):
        h = self.dropout(h)
        h = F.leaky_relu(self.linear0(h), negative_slope=0.02)
        h = self.dropout(h) 
        y = self.linear1(h)
        return y


class VAE_GNN_prior(nn.Module):
    def __init__(self, input_dim = 24, hidden_dim = 128, z_dim = 25, output_dim = 12, fc = False, dropout = 0.2, feat_drop = 0., 
                    attn_drop = 0., heads = 1, att_ew = True, ew_dims = 1, backbone = 'map_encoder', freeze = 6,
                    bn=False, gn=False):
        super().__init__()
        self.heads = heads
        self.fc = fc
        self.z_dim = z_dim
        self.bn = bn
        self.gn = gn

        ###############
        # Map Encoder #
        ###############
        if backbone == 'map_encoder':
            self.feature_extractor = My_MapEncoder(input_channels = 3, input_size=128, 
                                                    hidden_channels = [16,32,32,64], output_size = hidden_dim, 
                                                    kernels = [5,5,3,3], strides = [1,2,2,2])
            enc_dims = hidden_dim*2+(output_dim-1)    
            dec_dims = z_dim + hidden_dim*2
        
        elif backbone == 'resnet18':       
            self.feature_extractor = ResNet18(hidden_dim, freeze)
            enc_dims = 2*hidden_dim + (output_dim-1) 
            dec_dims = z_dim + hidden_dim*2
        elif backbone == 'resnet50':       
            self.feature_extractor = ResNetBackbone('resnet50', freeze = freeze) #ResNet50(hidden_dim, freeze)  #[n,2048]   #self.feature_extractor = ResNet50(hidden_dim, freeze)
            enc_dims = 2048 + hidden_dim + (output_dim-1)
            dec_dims = z_dim + hidden_dim + 2048
        
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

        ################# NO MAPS
        #enc_dims = hidden_dim + output_dim
        #dec_dims = hidden_dim + z_dim
        ###############
        self.embedding_h = nn.Linear(input_dim, hidden_dim)     #self.embedding_h = nn.Linear(input_dim+64, hidden_dim)        
        
        #############
        #  ENCODER  #
        #############
        self.GNN_enc = GAT_VAE(enc_dims, layers=2, dropout=dropout, feat_drop=feat_drop, attn_drop=attn_drop, heads=heads, att_ew=att_ew, ew_dims=ew_dims)
        encoder_dims = enc_dims*heads
        self.MLP_encoder = MLP_Enc(encoder_dims+(output_dim-1), z_dim, dropout=dropout)
        
        #############
        #   PRIOR   #
        #############
        self.GNN_prior = GAT_VAE(enc_dims-(output_dim-1),layers=2, dropout=dropout, feat_drop=feat_drop, attn_drop=attn_drop, heads=heads, att_ew=att_ew, ew_dims=ew_dims)
        encoder_dims = (enc_dims-(output_dim-1))*heads
        self.MLP_prior = MLP_Enc(encoder_dims, z_dim, dropout=dropout)
        

        #############
        #  DECODER  #
        ############# 
        #self.embedding_z = nn.Linear(z_dim, hidden_dim)
        #dec_dims = hidden_dim 
        self.GNN_decoder = GAT_VAE(dec_dims, dropout=dropout, feat_drop=feat_drop, attn_drop=attn_drop, heads=heads, att_ew=False, ew_dims=ew_dims) #If att_ew --> embedding_e_dec
        self.MLP_decoder = MLP_Dec(dec_dims*heads+z_dim, dec_dims, output_dim, dropout)

        self.base = nn.ModuleList([
            self.GNN_enc,
            self.MLP_encoder,
            self.GNN_prior,
            self.MLP_prior,
            #self.embedding_z,
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

        ####  EMBEDDINGS  ####
        # Input embedding
        h_emb = self.embedding_h(feats)  #input (N, 24)- (N,hid)
        # Maps feature extraction
        maps_emb = self.feature_extractor(maps)

        #### PRIOR ####
        # Embeddings concatenation
        h_prior = torch.cat([maps_emb.flatten(start_dim=1), h_emb], dim=-1)

        #h = self.linear_cat(h)
        if self.bn:
            h = self.bn_enc(h_prior)
        elif self.gn:
            h = self.gn_enc(h_prior)

        h_prior = self.GNN_prior(g, h_prior, e_w, snorm_n)    

        mu_prior, log_var_prior = self.MLP_prior(h_prior)   # Latent distribution

        #### Sample from the latent distribution ###
        z_sample = self.reparameterize(mu_prior, log_var_prior)
        
        #### DECODE ####      
        #z_dec = self.embedding_z(z_sample)      
        h_dec = torch.cat([maps_emb.flatten(start_dim=1), h_emb, z_sample],dim=-1)

        if self.bn:
            h_dec = self.bn_dec(h_dec)
        elif self.gn:
            h_dec = self.gn_dec(h_dec) 

        h_dec = self.GNN_decoder(g,h_dec,e_w,snorm_n)

        h_dec = torch.cat([h_dec, z_sample],dim=-1)

        recon_y = self.MLP_decoder(h_dec)

        return recon_y, mu_prior, log_var_prior
 
    def forward(self, g, feats, e_w, snorm_n, snorm_e, gt, maps):
        # Reshape from (B*V,T,C) to (B*V,T*C) 
        feats = feats.contiguous().view(feats.shape[0],-1)
        gt = gt.contiguous().view(gt.shape[0],-1)

        ####  EMBEDDINGS  ####
        # Map encoding
        maps_emb = self.feature_extractor(maps)
        # Input embedding
        #h = torch.cat([maps_emb.flatten(start_dim=1),feats], dim=-1)
        h_emb = self.embedding_h(feats) 

        #### ENCODER ####
        # Embeddings concatenation
        h = torch.cat([maps_emb, h_emb, gt], dim=-1)
        #h = self.linear_cat(h)
        if self.bn:
            h = self.bn_enc(h)
        elif self.gn:
            h = self.gn_enc(h)
        h = self.GNN_enc(g, h, e_w, snorm_n)
        h = torch.cat([h, gt], dim=-1)            
        mu, log_var = self.MLP_encoder(h)   # Latent distribution

        #### PRIOR ####
        # Embeddings concatenation
        h_prior = torch.cat([maps_emb, h_emb], dim=-1)
        #h = self.linear_cat(h)
        if self.bn:
            h = self.bn_enc(h_prior)
        elif self.gn:
            h = self.gn_enc(h_prior)
        h_prior = self.GNN_prior(g, h_prior, e_w, snorm_n)    
        mu_prior, log_var_prior = self.MLP_prior(h_prior)   # Latent distribution

        #### Sample from the latent distribution ###
        z_sample = self.reparameterize(mu_prior, log_var_prior)
        
        #### DECODE ####      
        #h_dec = self.embedding_z(z_sample)       
        h_dec = torch.cat([maps_emb.flatten(start_dim=1), h_emb, z_sample],dim=-1)
        if self.bn:
            h = self.bn_dec(h_dec)
        elif self.gn:
            h = self.gn_dec(h_dec)
        h_dec = self.GNN_decoder(g,h_dec,e_w,snorm_n)
        h_dec = torch.cat([h_dec, z_sample],dim=-1)
        recon_y = self.MLP_decoder(h_dec)
        return recon_y,  [mu, log_var, mu_prior, log_var_prior], z_sample

if __name__ == '__main__':
    history_frames = 5
    future_frames = 12
    hidden_dims = 128
    heads = 1

    input_dim = 6*history_frames
    output_dim = 2*future_frames + 1

    model = VAE_GNN_prior(input_dim, hidden_dims, 25, output_dim, bn=False,fc=False, dropout=0.2,feat_drop=0., attn_drop=0., heads=2,att_ew=True, ew_dims=2, backbone='resnet18')
    #summary(model.feature_extractor, input_size=(3,224,224), device='cpu')
    test_dataset = nuscenes_Dataset(train_val_test='train', rel_types=True, history_frames=history_frames, future_frames=future_frames) 
    test_dataloader = DataLoader(test_dataset, batch_size=3, shuffle=False, collate_fn=collate_batch)

    for batch in test_dataloader:
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos, maps, scene = batch
        e_w = batched_graph.edata['w']
        #e_w= e_w.unsqueeze(1)
        y, _,_ = model(batched_graph, feats, e_w,snorm_n,snorm_e, labels_pos[:,:,:2],  maps)
        print(y.shape)

    