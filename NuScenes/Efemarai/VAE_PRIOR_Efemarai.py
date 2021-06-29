
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['DGLBACKEND'] = 'pytorch'
import numpy as np
from torchvision.models import resnet18, resnet50
from NuScenes.nuscenes_Dataset import nuscenes_Dataset, collate_batch
from torch.utils.data import DataLoader
from models.VAE_GNN import GAT_VAE


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
        nn.init.kaiming_normal_(self.linear.weight, a=0.2, nonlinearity='leaky_relu')

    def forward(self, h):
        h = self.dropout_l(h)
        h = F.leaky_relu(self.linear(h), negative_slope=0.2)
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
        nn.init.kaiming_normal_(self.linear0.weight, a=0.2, nonlinearity='leaky_relu')

    def forward(self, h):
        h = self.dropout(h)
        h = F.leaky_relu(self.linear0(h), negative_slope=0.2)
        h = self.dropout(h) 
        y = self.linear1(h)
        return y


class VAE_GNN_prior(nn.Module):
    def __init__(self, input_dim = 24, hidden_dim = 128, z_dim = 25, output_dim = 12, fc = False, dropout = 0.2, feat_drop = 0., 
                    attn_drop = 0., heads = 1, att_ew = True, ew_dims = 1, backbone = 'map_encoder', freeze = 6,
                    bn=False, gn=False, encoding_type = 'emb', num_modes = 3):
        super().__init__()
        self.heads = heads
        self.fc = fc
        self.z_dim = z_dim
        self.bn = bn
        self.gn = gn
        self.encoding_type = encoding_type

        self.num_modes =  num_modes

        ###############
        # Map Encoder #
        ###############
        enc_dims = 512 + hidden_dim + (output_dim-1)  #2*hidden_dim + (output_dim-1) 
        dec_dims = z_dim + hidden_dim + 512 #hidden_dim*2
       
        ############################
        # Input Features Embedding #
        ############################
        if encoding_type == 'emb':
            self.embedding_h = nn.Linear(input_dim, hidden_dim)  
        else:
            self.embedding_h = nn.Linear(input_dim, hidden_dim//2)  
            self.encode_h = nn.GRU(hidden_dim//2, hidden_dim, batch_first=True)

        #############
        #  ENCODER  #
        #############
        '''
        GNN_enc = GAT_VAE(hidden_dim+(output_dim-1), layers=2, dropout=dropout, feat_drop=feat_drop, attn_drop=attn_drop, heads=heads, att_ew=att_ew, ew_dims=ew_dims)
        encoder_dims = hidden_dim+(output_dim-1)*2 # hidden_dim+(output_dim-1)*2 #enc_dims#*heads
        MLP_encoder = MLP_Enc(encoder_dims, z_dim, dropout=dropout)
        '''
        GNN_enc = GAT_VAE(enc_dims, layers=2, dropout=dropout, feat_drop=feat_drop, attn_drop=attn_drop, heads=heads, att_ew=att_ew, ew_dims=ew_dims)
        encoder_dims = enc_dims#*heads
        MLP_encoder = MLP_Enc(encoder_dims+(output_dim-1), z_dim, dropout=dropout)
        
        #############
        #   PRIOR   #
        #############
        GNN_prior = GAT_VAE(enc_dims-(output_dim-1),layers=2, dropout=dropout, feat_drop=feat_drop, attn_drop=attn_drop, heads=heads, att_ew=att_ew, ew_dims=ew_dims)
        encoder_dims = (enc_dims-(output_dim-1))#*heads
        MLP_prior = MLP_Enc(encoder_dims, z_dim, dropout=dropout)
        

        #############
        #  DECODER  #
        ############# 
        #self.embedding_z = nn.Linear(z_dim, hidden_dim)
        #dec_dims = hidden_dim 
        GNN_decoder = GAT_VAE(dec_dims, dropout=dropout, feat_drop=feat_drop, attn_drop=attn_drop, heads=heads, att_ew=False, ew_dims=ew_dims) #If att_ew --> embedding_e_dec
        MLP_decoder = MLP_Dec(dec_dims+z_dim, dec_dims, output_dim, dropout) #dec_dims*heads
        
        self.base = nn.ModuleDict({
            'GNN_enc':      GNN_enc,
            'MLP_encoder':  MLP_encoder,
            'GNN_prior':    GNN_prior,
            'MLP_prior':    MLP_prior,
            'GNN_decoder':  GNN_decoder,
            'MLP_decoder':  MLP_decoder
        })

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
        
        self.leaky_relu = nn.LeakyReLU(0.1)

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
        std = F.elu(0.5 * logvar) + 1  + 1e-5 #torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    
    def encode(self, g, h_emb, e_w, snorm_n, maps_emb, gt):
        # Embeddings concatenation
        h = torch.cat([maps_emb, h_emb, gt], dim=-1)
        #h = self.linear_cat(h)

        if self.bn:
            h = self.bn_enc(h)
        elif self.gn:
            h = self.gn_enc(h)

        h = self.base['GNN_enc'](g, h, e_w, snorm_n)
        h = torch.cat([h, gt], dim=-1)            
        mu, log_var = self.base['MLP_encoder'](h)   # Latent distribution
        
        return mu, log_var
    
    def prior(self, g, h_emb, e_w, snorm_n, maps_emb):
        # Embeddings concatenation
        h_prior = torch.cat([maps_emb, h_emb], dim=-1)
        #h = self.linear_cat(h)

        if self.bn:
            h = self.bn_enc(h_prior)
        elif self.gn:
            h = self.gn_enc(h_prior)

        h_prior = self.base['GNN_prior'](g, h_prior, e_w, snorm_n)    

        mu_prior, log_var_prior = self.base['MLP_prior'](h_prior)   # Latent distribution

        return mu_prior, log_var_prior
        
    
    def decode(self, g, h_emb, e_w, snorm_n, maps_emb, z_sample):   
        #h_dec = self.embedding_z(z_sample)       
        h_dec = torch.cat([maps_emb, h_emb, z_sample],dim=-1)
        
        if self.bn:
            h = self.bn_dec(h_dec)
        elif self.gn:
            h = self.gn_dec(h_dec)
            
        h_dec = self.base['GNN_decoder'](g,h_dec,e_w,snorm_n)
        h_dec = torch.cat([h_dec, z_sample],dim=-1)

        return self.base['MLP_decoder'](h_dec) 

    
    def inference(self, g, features, e_w, snorm_n,snorm_e):
        """
        Samples from a normal distribution and decodes conditioned to the GNN outputs.   
        """
        ####  EMBEDDINGS  ####
        if self.encoding_type == 'emb':
            # Reshape from (B*V,T,C) to (B*V,T*C) 
            feats = features.contiguous().view(features.shape[0],-1)
            # Input embedding
            h_emb = self.embedding_h(feats)  #input (N, 24)- (N,hid)
        else:
            h_emb = self.encode_h(self.leaky_relu(self.embedding_h(features)))[1].squeeze() 
        # Maps feature extraction
        maps_emb = torch.rand(features.shape[0], 512).to('cuda:0')

        #### PRIOR ####
        mu_prior, log_var_prior = self.prior(g, h_emb, e_w, snorm_n, maps_emb)
        z_sample = self.reparameterize(mu_prior, log_var_prior)
        
        #### DECODE ####      
        pred = self.decode(g, h_emb, e_w, snorm_n, maps_emb, z_sample)

        return pred[:,:-1], pred[:,-1], [mu_prior, log_var_prior], z_sample

 
    def forward(self, g, features, e_w, snorm_n, snorm_e, labels):
        # Reshape from (B*V,T,C) to (B*V,T*C) 
        gt = labels.contiguous().view(labels.shape[0],-1)

        ####  EMBEDDINGS  ####
        # Map encoding
        maps_emb = torch.rand(features.shape[0], 512).to('cuda:0')
        ####  EMBEDDINGS  ####
        if self.encoding_type == 'emb':
            # Reshape from (B*V,T,C) to (B*V,T*C) 
            feats = features.contiguous().view(features.shape[0],-1)
            # Input embedding
            h_emb = self.embedding_h(feats)  #input (N, 24)- (N,hid)
        else:
            h_emb = self.encode_h(self.leaky_relu(self.embedding_h(features)))[1].squeeze() 
            
        #### ENCODER ####
        mu, log_var = self.encode(g, h_emb, e_w, snorm_n, maps_emb, gt)

        #### PRIOR ####
        mu_prior = log_var_prior = 0.1 #mu_prior, log_var_prior = self.prior(g, h_emb, e_w, snorm_n, maps_emb)

        #### DECODE #### 
        
        pred = [] #torch.Tensor().requires_grad_(True).to(feats.device)
        for i in range(self.num_modes):
            #### Sample from the latent distribution ###
            z_sample = self.reparameterize(mu, log_var)
            pred.append( self.decode(g, h_emb, e_w, snorm_n, maps_emb, z_sample) )#torch.cat( ,  dim = 0) # 3, N, 25
        
        pred = torch.stack(pred,dim=0)
        return pred[:,:,:-1], pred[:,:,-1], [mu, log_var, mu_prior, log_var_prior], z_sample
        '''
        z_sample = self.reparameterize(mu_prior, log_var_prior)
        pred = self.decode(g, h_emb, e_w, snorm_n, maps_emb, z_sample)
        
        return pred[:,:-1], pred[:,-1], [mu, log_var, mu_prior, log_var_prior], z_sample
        '''

if __name__ == '__main__':
    history_frames = 7
    future_frames = 10
    hidden_dims = 128
    heads = 2

    input_dim = 7
    output_dim = 2*future_frames + 1

    model = VAE_GNN_prior(input_dim, hidden_dims, 25, output_dim, bn=False,fc=False, dropout=0.2,feat_drop=0., attn_drop=0., 
                    heads=2,att_ew=True, ew_dims=2, backbone='resnet18', encoding_type='gru')
    #summary(model.feature_extractor, input_size=(3,224,224), device='cpu')
    test_dataset = nuscenes_Dataset(train_val_test='val', rel_types=True, history_frames=history_frames, future_frames=future_frames, local_frame=False) 
    test_dataloader = DataLoader(test_dataset, batch_size=3, shuffle=False, collate_fn=collate_batch)


    for batch in test_dataloader:
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos, maps, scene = batch
        e_w = batched_graph.edata['w']
        #e_w= e_w.unsqueeze(1)
        #y = model.inference(batched_graph, feats, e_w,snorm_n,snorm_e, maps)
        
        y, prob,_,_ = model(batched_graph, feats, e_w,snorm_n,snorm_e, labels_pos[:,:,:2],  maps)
        print(y.shape)

    