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
    
class RNN_baseline(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, pred_length=3,dropout=0.2, bn=True):
        super().__init__()
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.pred_length = pred_length
        
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
        
        '''
        h = self.dropout_l(h)
        if self.bn:
            h = self.batch_norm(h)
        '''
        y = self.seq2seq(in_data=h, last_location=inputs[:,-1:,:2], pred_length=self.pred_length)  # (BV,6,hid) -> (BV,6,2)
        #y = self.linear2(torch.relu(y))
        return y
    