import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['DGLBACKEND'] = 'pytorch'
from dgl import DGLGraph
import numpy as np
import dgl.function as fn
from models.My_GAT import My_GATLayer, MultiHeadGATLayer
from models.seq2seq import Seq2Seq
    
class Model_GNN_RNN(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, pred_length=3,dropout=0.2, bn=True, bn_gat=True, feat_drop=0., attn_drop=0., heads=1,att_ew=False):
        super().__init__()
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_e = nn.Linear(1, hidden_dim)
        self.heads = heads
        self.pred_length = pred_length
        if heads == 1:
            self.gat_1 = My_GATLayer(hidden_dim, hidden_dim, feat_drop, attn_drop, bn_gat,att_ew)
            self.gat_2 = My_GATLayer(hidden_dim, hidden_dim, 0., 0., False,att_ew)
            self.batch_norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.gat_1 = MultiHeadGATLayer(hidden_dim, hidden_dim, num_heads=heads, bn=bn_gat, feat_drop=feat_drop, attn_drop=attn_drop, att_ew=att_ew)
            self.embedding_e2 = nn.Linear(1, hidden_dim*heads)
            self.gat_2 = MultiHeadGATLayer(hidden_dim*heads, hidden_dim*heads, num_heads=1, bn=False, feat_drop=0., attn_drop=0., att_ew=att_ew)
            self.batch_norm = nn.BatchNorm1d(hidden_dim*heads)   #OJO, SI USO HEADS TENGO QUE CONVERTIRLO A HIDDEN_DIMS
        
        if dropout:
            self.dropout_l = nn.Dropout(dropout)
        else:
            self.dropout_l = nn.Dropout(0.)
        self.bn = bn

        #RNN for prediction
        self.seq2seq = Seq2Seq(input_size=hidden_dim, hidden_size=2, num_layers=2, dropout=0.5)

        
    def forward(self, g, inputs, e_w, snorm_n, snorm_e):
        
        # input embedding
        h = self.embedding_h(inputs)  #input (BV, 6, 4)- (BV, 6, hid)
        e = self.embedding_e(e_w)
        g.edata['w']=e

        # gat layers
        h = self.gat_1(g, h,snorm_n)
        if self.heads > 1:
            e = self.embedding_e2(e_w)
            g.edata['w']=e
        h = self.gat_2(g, h,snorm_n)  #BV,6,hid
        '''
        h = self.dropout_l(h)
        if self.bn:
            h = self.batch_norm(h)
        '''
        y = self.seq2seq(in_data=h, last_location=inputs[:,-1:,:2], pred_length=self.pred_length)  # (BV,6,hid) -> (BV,6,2)
        #y = self.linear2(torch.relu(y))
        return y
    