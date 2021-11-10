import dgl
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['DGLBACKEND'] = 'pytorch'
from dgl import DGLGraph
import numpy as np
import dgl.function as fn
from dgl.nn.pytorch.conv.gatconv import edge_softmax, Identity, expand_as_pair
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader
import torchvision
from torchsummary import summary
from NuScenes.nuscenes_Dataset import nuscenes_Dataset, collate_batch_ns


class MDN(nn.Module):
    """A mixture density network layer
    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.
    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions
    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.
    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """
    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Linear(in_features, num_gaussians)
        self.sigma = nn.Linear(in_features, out_features*num_gaussians)
        self.mu = nn.Linear(in_features, out_features*num_gaussians)

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.normal_(self.sigma.weight, 0, sqrt(1. / self.sigma.in_features))
        nn.init.xavier_normal_(self.mu.weight)
        nn.init.xavier_normal_(self.pi.weight)

    def forward(self, h):
        pi = F.softmax(self.pi(h), dim=1)
        sigma = F.elu(self.sigma(h)) + 1  + 1e-5 #torch.exp(self.sigma(h) 
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(h)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu



class My_GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats,  relu=True, feat_drop=0., attn_drop=0., att_ew=False, res_weight=True, res_connection=True):
        super(My_GATLayer, self).__init__()
        self.linear_self = nn.Linear(in_feats, out_feats, bias=False)
        self.linear_func = nn.Linear(in_feats, out_feats, bias=False)
        self.att_ew=att_ew
        self.relu = relu
        if att_ew:
            self.attention_func = nn.Linear(3 * out_feats, 1, bias=False)
        else:
            self.attention_func = nn.Linear(2 * out_feats, 1, bias=False)
        self.feat_drop_l = nn.Dropout(feat_drop)
        self.attn_drop_l = nn.Dropout(attn_drop)   
        self.res_con = res_connection
        self.reset_parameters()
      
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.kaiming_normal_(self.linear_self.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear_func.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.attention_func.weight, a=0.01, nonlinearity='leaky_relu')
    
    def edge_attention(self, edges):
        concat_z = torch.cat([edges.src['z'], edges.dst['z']], dim=-1) #(n_edg,hid)||(n_edg,hid) -> (n_edg,2*hid) 
        
        if self.att_ew:
           concat_z = torch.cat([edges.src['z'], edges.dst['z'], edges.data['w']], dim=-1) 
        
        src_e = self.attention_func(concat_z)  #(n_edg, 1) att logit
        src_e = F.leaky_relu(src_e)
        return {'e': src_e}
    
    def message_func(self, edges):
        return {'z': edges.src['z'], 'e':edges.data['e']}
        
    def reduce_func(self, nodes):
        h_s = nodes.data['h_s']      
        #Attention score
        a = self.attn_drop_l(   F.softmax(nodes.mailbox['e'], dim=1)  )  #attention score between nodes i and j
        h = h_s + torch.sum(a * nodes.mailbox['z'], dim=1)
        return {'h': h}
                               
    def forward(self, g, h,snorm_n):
        with g.local_scope():
            h_in = h.clone()
            g.ndata['h']  = h 
            #feat dropout
            h=self.feat_drop_l(h)
            g.ndata['h_s'] = self.linear_self(h) 
            g.ndata['z'] = self.linear_func(h) 
            g.apply_edges(self.edge_attention)
            g.update_all(self.message_func, self.reduce_func)
            h =  g.ndata['h'] #+g.ndata['h_s'] 
            #h = h * snorm_n # normalize activation w.r.t. graph node size
            if self.relu:
                h = torch.relu(h)            
            if self.res_con:
                h = h_in + h # residual connection           
            return h #graph.ndata.pop('h') - another option to g.local_scope()


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, relu=True, merge='cat',  feat_drop=0., attn_drop=0., att_ew=False, res_weight=True, res_connection=True):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(My_GATLayer(in_feats, out_feats,feat_drop=feat_drop, attn_drop=attn_drop, att_ew=att_ew, res_weight=res_weight, res_connection=res_connection))
        self.merge = merge

    def forward(self, g, h, snorm_n):
        head_outs = [attn_head(g, h,snorm_n) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1), for intermediate layers
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average, for final layer
            return torch.mean(torch.stack(head_outs))

    
class SCOUT_MDN(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2, bn=True, feat_drop=0., attn_drop=0., heads=1,att_ew=False, ew_type=False, map_encoding=False):
        super().__init__()

        self.map_encoding = map_encoding
        self.heads = heads
        
        if self.map_encoding:
            model_ft = torchvision.models.resnet18(pretrained=True)
            self.feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1])
            ct=0
            for child in self.feature_extractor.children():
                ct+=1
                if ct < 7:
                    for param in child.parameters():
                        param.requires_grad = False
            
            self.linear_cat = nn.Linear(hidden_dim + 512, hidden_dim) 

        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_e = nn.Linear(2, hidden_dim) if  ew_type else nn.Linear(1, hidden_dim)

        if heads == 1:
            self.gat_1 = My_GATLayer(hidden_dim, hidden_dim, feat_drop, attn_drop,att_ew, res_weight=res_weight, res_connection=res_connection ) #GATConv(hidden_dim, hidden_dim, 1,feat_drop, attn_drop,residual=True, activation=torch.relu) 
            self.gat_2 = My_GATLayer(hidden_dim, hidden_dim, 0., 0.,att_ew, res_weight=res_weight, res_connection=res_connection )  #GATConv(hidden_dim, hidden_dim, 1,feat_drop, attn_drop,residual=True, activation=torch.relu)
            self.linear1 = nn.Linear(hidden_dim, hidden_dim//2)          
            self.mdn = MDN(hidden_dim//2, output_dim, 3)
        else:
            self.gat_1 = MultiHeadGATLayer(hidden_dim, hidden_dim, res_weight=True, res_connection=True , num_heads=heads,feat_drop=feat_drop, attn_drop=attn_drop, att_ew=att_ew)            
            self.embedding_e2 = nn.Linear(2, hidden_dim*heads) if ew_type else nn.Linear(1, hidden_dim*heads)

            self.gat_2 = MultiHeadGATLayer(hidden_dim*heads,hidden_dim*heads, res_weight=True , res_connection=True ,num_heads=1, feat_drop=0., attn_drop=0., att_ew=att_ew) #GATConv(hidden_dim*heads, hidden_dim*heads, heads,feat_drop, attn_drop,residual=True, activation='relu')

            self.linear1 = nn.Linear(hidden_dim*heads, 512)
            #self.linear2 = nn.Linear(512, 256)
            self.mdn = MDN(256, output_dim, 3)

        self.dropout_l = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.xavier_normal_(self.embedding_h.weight)
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='tanh')
        #nn.init.xavier_normal_(self.linear2.weight, nn.init.calculate_gain('tanh'))  
        nn.init.xavier_normal_(self.embedding_e.weight)       
        if self.heads > 1:
            nn.init.xavier_normal_(self.embedding_e2.weight)
    
    def forward(self, g, feats,e_w,snorm_n,snorm_e, maps):
        #reshape to have shape (B*V,T*C) [c1,c2,...,c6]
        feats = feats.contiguous().view(feats.shape[0],-1)

        # Input Features embedding
        h = self.embedding_h(feats)  
        e = self.embedding_e(e_w)

        if self.map_encoding:
            # Maps feature extraction
            maps_embedding = self.feature_extractor(maps)

            # Embeddings concatenation
            h = torch.cat([maps_embedding, h], dim=-1)
            h = self.linear_cat(h)

        # GAT Layers
        g.edata['w']=e
        h = self.gat_1(g, h,snorm_n) 
        if self.heads > 1:
            e = self.embedding_e2(e_w)
            g.edata['w']=e
        h = self.gat_2(g, h, snorm_n)  #BN Y RELU DENTRO DE LA GAT_LAYER
        h = self.dropout_l(h)
        #h = F.relu(self.linear1(h))
        h = F.tanh(self.linear1(h))
        pi, sigma, mu = self.mdn(h)   
        return pi, sigma, mu
    
if __name__ == '__main__':

    history_frames = 4
    future_frames = 12
    hidden_dims = 768
    heads = 2

    input_dim = 9*history_frames
    output_dim = 2*future_frames 

    hidden_dims = round(hidden_dims / heads) 
    model = SCOUT_MDN(input_dim=input_dim, hidden_dim=hidden_dims, output_dim=output_dim, heads=heads, dropout=0.1, bn=True, feat_drop=0., attn_drop=0., att_ew=True, ew_type=True, map_encoding=True)
    summary(model.feature_extractor, input_size=(3,112,112))

    test_dataset = nuscenes_Dataset(train_val_test='test', rel_types=args.ew_dims>1, history_frames=history_frames, future_frames=future_frames, map_encodding=True) 
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)

    for batch in test_dataloader:
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos, maps = batch
        out = model(batched_graph, feats,e_w,snorm_n,snorm_e, maps)
        print(out.shape)