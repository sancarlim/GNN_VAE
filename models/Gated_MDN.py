
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['DGLBACKEND'] = 'pytorch'
from dgl import DGLGraph
import numpy as np
import matplotlib.pyplot as plt
import dgl.function as fn

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

    def forward(self, minibatch):
        pi = F.softmax(self.pi(minibatch), dim=1)
        sigma = F.elu(self.sigma(minibatch)) + 1 + 1e-5   #torch.exp(self.sigma(minibatch) max 12.5 min 0.8
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu



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


class Gated_MDN(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, bn, ew_type=False):
        super().__init__()
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_e = nn.Linear(2, hidden_dim) if ew_type else nn.Linear(1, hidden_dim)
        self.GatedGCN1 = GatedGCN_layer(hidden_dim, hidden_dim)
        self.GatedGCN2 = GatedGCN_layer(hidden_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.mdn = MDN(hidden_dim//2, output_dim, 3)

        if dropout:
            self.linear_dropout = nn.Dropout(dropout)
        else:
            self.linear_dropout =  nn.Dropout(0.)

        self.batch_norm = nn.BatchNorm1d(hidden_dim)#nn.GroupNorm(32, hidden_dim) 
        self.bn = bn
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.xavier_normal_(self.embedding_h.weight)
        nn.init.xavier_normal_(self.linear1.weight, nn.init.calculate_gain('tanh'))
        nn.init.xavier_normal_(self.embedding_e.weight)

    def forward(self, g, inputs, e, snorm_n, snorm_e):
        #reshape to have shape (B*V,T*C) [c1,c2,...,c6]
        inputs = inputs.contiguous().view(inputs.shape[0],-1)
        # input embedding
        h = self.embedding_h(inputs)
        e = self.embedding_e(e)
        # graph convnet layers
        h, e = self.GatedGCN1(g, h, e, snorm_n, snorm_e)
        h, e = self.GatedGCN2(g, h, e, snorm_n, snorm_e)
        # MLP 
        h = self.linear_dropout(h)
        h = torch.tanh(self.linear1(h))
        pi, sigma, mu = self.mdn(h)   
        return pi, sigma, mu