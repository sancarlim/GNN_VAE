import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['DGLBACKEND'] = 'pytorch'
from dgl import DGLGraph
import numpy as np
import dgl.function as fn
import matplotlib.pyplot as plt

class My_GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, bn=True, feat_drop=0., attn_drop=0., att_ew=False):
        super(My_GATLayer, self).__init__()
        self.linear_self = nn.Linear(in_feats, out_feats, bias=False)
        self.linear_func = nn.Linear(in_feats, out_feats, bias=False)
        self.att_ew=att_ew
        if att_ew:
            self.attention_func = nn.Linear(3 * out_feats, 1, bias=False)
        else:
            self.attention_func = nn.Linear(2 * out_feats, 1, bias=False)

        self.feat_drop_l = nn.Dropout(feat_drop)
        self.attn_drop_l = nn.Dropout(attn_drop)
        #self.bn = bn
        
        #self.bn_node_h = nn.GroupNorm(32, out_feats) #nn.BatchNorm1d(out_feats)
        
        self.reset_parameters()
    
    
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear_self.weight, gain=gain)
        nn.init.xavier_normal_(self.linear_func.weight, gain=gain)
        nn.init.xavier_normal_(self.attention_func.weight, gain=gain)
    
    def edge_attention(self, edges):
        concat_z = torch.cat([edges.src['z'], edges.dst['z']], dim=-1) #(n_edg,hid)||(n_edg,hid) -> (n_edg,2*hid) 
        
        if self.att_ew:
           concat_z = torch.cat([edges.src['z'], edges.dst['z'], edges.data['w']], dim=-1) 
        
        src_e = self.attention_func(concat_z)  #(n_edg, 1) att logit
        src_e = F.leaky_relu(src_e)

        #VISUALIZE
        att_score = F.softmax(src_e, dim=0) 
        
        return {'e': src_e, 'a': att_score}
    
    def message_func(self, edges):
        return {'z': edges.src['z'], 'e':edges.data['e']}
        
    def reduce_func(self, nodes):
        h_s = nodes.data['h_s']
        
        #ATTN DROPOUT OP A
        a = self.attn_drop_l(   F.softmax(nodes.mailbox['e'], dim=1)  )  #attention score between nodes i and j
        
        #h = torch.sum(a * nodes.mailbox['z'], dim=1) 
        #OPCION A
        h = h_s + torch.sum(a * nodes.mailbox['z'], dim=1)
        #OPCION B
        #h = h_s + torch.sum(nodes.mailbox['a'] * nodes.mailbox['z'], dim=1)
        return {'h': h}
                               
    def forward(self, g, h,snorm_n):
        with g.local_scope():

            #feat = h.detach().cpu().numpy().astype('uint8')
            #feat=(feat*255/np.max(feat))

            #feat dropout
            h=self.feat_drop_l(h)
            
            h_in = h
            g.ndata['h']  = h 
            g.ndata['h_s'] = self.linear_self(h) 
            g.ndata['z'] = self.linear_func(h) 
            g.apply_edges(self.edge_attention)
            g.update_all(self.message_func, self.reduce_func)
            #M = g.ndata['h'].detach().cpu().numpy().astype('uint8')-feat
            #M=(M*255/np.max(M))
            h =  g.ndata['h'] #+g.ndata['h_s'] 
            #h = h * snorm_n # normalize activation w.r.t. graph node size

            #VISUALIZE
            '''
            A = g.adjacency_matrix(scipy_fmt='coo').toarray().astype('uint8')
            A=(A*255/np.max(A))
            plt.imshow(A,cmap='hot')
            plt.show()
            
            fig,ax=plt.subplots(1,2)
            im1=ax[0].imshow(feat,cmap='hot',aspect='auto')
            ax[0].set_title('X',fontsize=8)
            im4=ax[1].imshow(M,cmap='hot',aspect='auto')
            ax[1].set_title('M-X',fontsize=8)
            plt.show()
            '''
            h = torch.relu(h) # non-linear activation
            h = h_in + h # residual connection
            
            return h, g.edata['a'] #graph.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, merge='cat', bn=True, feat_drop=0., attn_drop=0., att_ew=False):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(My_GATLayer(in_feats, out_feats, bn=bn, feat_drop=feat_drop, attn_drop=attn_drop, att_ew=att_ew))
        self.merge = merge

    def forward(self, g, h, snorm_n):
        head_outs = [attn_head(g, h,snorm_n) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1), for intermediate layers
            outs = [head_outs[i][0] for i in range(0,len(head_outs))]
            atts = [head_outs[i][1] for i in range(0,len(head_outs))]
            return torch.cat(outs, dim=1), atts
        else:
            outs = [head_outs[i][0] for i in range(0,len(head_outs))]
            atts = [head_outs[i][1] for i in range(0,len(head_outs))]
            # merge using average, for final layer
            return torch.mean(torch.stack(outs)), atts

    
class My_GAT_vis(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2, bn=True, bn_gat=True, feat_drop=0., attn_drop=0., heads=1,att_ew=False):
        super().__init__()
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_e = nn.Linear(1, hidden_dim)
        self.heads = heads
        if heads == 1:
            self.gat_1 = My_GATLayer(hidden_dim, hidden_dim, feat_drop, attn_drop, bn_gat,att_ew)
            self.gat_2 = My_GATLayer(hidden_dim, hidden_dim, 0., 0., False,att_ew)
            self.linear1 = nn.Linear(hidden_dim, output_dim)
            self.batch_norm = nn.BatchNorm1d(hidden_dim)
            #self.group_norm = nn.GroupNorm(32, hidden_dim)
        else:
            self.gat_1 = MultiHeadGATLayer(hidden_dim, hidden_dim, num_heads=heads, bn=bn_gat, feat_drop=feat_drop, attn_drop=attn_drop, att_ew=att_ew)
            self.embedding_e2 = nn.Linear(1, hidden_dim*heads)
            self.gat_2 = MultiHeadGATLayer(hidden_dim*heads, hidden_dim*heads, num_heads=1, bn=False, feat_drop=0., attn_drop=0., att_ew=att_ew)
            self.batch_norm = nn.BatchNorm1d(hidden_dim*heads)
            #self.group_norm = nn.GroupNorm(32, hidden_dim*heads)
            self.linear1 = nn.Linear(hidden_dim*heads, output_dim)
            
        #self.linear2 = nn.Linear( int(hidden_dim/2),  output_dim)
        
        if dropout:
            self.dropout_l = nn.Dropout(dropout)
        else:
            self.dropout_l = nn.Dropout(0.)
        self.bn = bn

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.embedding_h.weight)
        nn.init.xavier_normal_(self.linear1.weight, gain=gain)
        nn.init.xavier_normal_(self.embedding_e.weight)
        
        if self.heads == 3:
            nn.init.xavier_normal_(self.embedding_e2.weight, gain=gain)
    
        
    def forward(self, g, feats,e_w,snorm_n,snorm_e):
        

        #reshape to have shape (B*V,T*C) [c1,c2,...,c6]
        feats = feats.view(feats.shape[0],-1)

        # input embedding
        h = self.embedding_h(feats)  #input (N, 24)- (N,hid)
        e = self.embedding_e(e_w)
        g.edata['w']=e

        # gat layers
        h, att1 = self.gat_1(g, h,snorm_n)
        if self.heads > 1:
            e = self.embedding_e2(e_w)
            g.edata['w']=e
        h, att2 = self.gat_2(g, h,snorm_n)  #BN Y RELU DENTRO DE LA GAT_LAYER
        
        h = self.dropout_l(h)
        #Last linear layer
        y = self.linear1(h) 
        #y = self.linear2(torch.relu(y))
        return y, att1, att2 