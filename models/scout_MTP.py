import sys
sys.path.append('..')
import dgl
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import OrderedDict
os.environ['DGLBACKEND'] = 'pytorch'
import dgl.function as fn
from dgl.nn.pytorch.conv.gatconv import edge_softmax, Identity, expand_as_pair
import math
from torch.utils.data import DataLoader
from NuScenes.nuscenes_Dataset import nuscenes_Dataset, collate_batch
from torchvision.models import resnet18
from torchsummary import summary
from models.MapEncoder import My_MapEncoder, ResNet18, ResNet50
from models.backbone import MobileNetBackbone, ResNetBackbone, calculate_backbone_feature_dim


class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                                'output for those nodes will be invalid. '
                                'This is harmful for some applications, '
                                'causing silent performance regression. '
                                'Adding self-loop on the input graph by '
                                'calling `g = dgl.add_self_loop(g)` will resolve '
                                'the issue. Setting ``allow_zero_in_degree`` '
                                'to be `True` when constructing this module will '
                                'suppress the check and let the code run.')

            
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=7):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        '''
        pe = torch.zeros(max_len, d_model)   #T,512
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  #T,1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  #256
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#.transpose(0, 1)   #1,T,512
        self.register_buffer('pe', pe)
        '''
        self.pe = nn.Parameter(torch.randn(1,max_len, d_model))

    def forward(self, x):
        x = x + self.pe[:,:x.size(1), :]  #x is N,T,512 + (1,T,512) 
        return self.dropout(x)


class My_GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, e_dims, relu=True, feat_drop=0., attn_drop=0., att_ew=False, res_weight=True, res_connection=True):
        super(My_GATLayer, self).__init__()
        self.linear_self = nn.Linear(in_feats, out_feats, bias=False)
        self.linear_func = nn.Linear(in_feats, out_feats, bias=False)
        self.att_ew=att_ew
        self.relu = relu
        if att_ew:
            self.attention_func = nn.Linear(2 * out_feats + e_dims, 1, bias=False)
        else:
            self.attention_func = nn.Linear(2 * out_feats, 1, bias=False)
        self.feat_drop_l = nn.Dropout(feat_drop)
        self.attn_drop_l = nn.Dropout(attn_drop)   
        self.res_con = res_connection
        self.reset_parameters()
      
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = torch.nn.init.calculate_gain('selu', param=None)
        nn.init.xavier_normal_(self.linear_self.weight, gain)
        nn.init.xavier_normal_(self.linear_func.weight, gain)
        ##nn.init.kaiming_normal_(self.linear_self.weight, nonlinearity='relu')
        ##nn.init.kaiming_normal_(self.linear_func.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.attention_func.weight, a=0.02, nonlinearity='leaky_relu')
        
    
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
        a = self.attn_drop_l(   F.softmax(nodes.mailbox['e'], dim=-1)  )  #attention score between nodes i and j
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
                h = torch.relu(h) # non-linear activation
            if self.res_con:
                h = h_in + h # residual connection           
            return h #graph.ndata.pop('h') - another option to g.local_scope()


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, e_dims, relu=True, merge='cat',  feat_drop=0., attn_drop=0., att_ew=False, res_weight=True, res_connection=True):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(My_GATLayer(in_feats, out_feats, e_dims, feat_drop=feat_drop, attn_drop=attn_drop, att_ew=att_ew, res_weight=res_weight, res_connection=res_connection))
        self.merge = merge

    def forward(self, g, h, snorm_n):
        if isinstance(h, list):
            head_outs = [attn_head(g, h_mode,snorm_n) for attn_head, h_mode in zip(self.heads, h)]
        else:
            head_outs = [attn_head(g, h,snorm_n) for attn_head in self.heads]
            
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1), for intermediate layers
            return torch.cat(head_outs, dim=-1)
        elif self.merge == 'list':
            return head_outs
        else:
            # merge using average, for final layer
            return torch.mean(torch.stack(head_outs, dim=1),dim=-1)

    
class SCOUT_MTP(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, emb_dim, output_dim, dropout=0.2, bn=False, gn=False, 
                feat_drop=0., attn_drop=0., heads=1,att_ew=False, res_weight=True, emb_type = 'emb',
                res_connection=True, ew_dims=False,  backbone='mobilenet', freeze=0, num_modes=3):
        super().__init__()

        self.heads = heads
        self.bn = bn
        self.gn = gn
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.emb_type = emb_type
        self.output_dim = output_dim
        self.ew_dims = ew_dims
        self.backbone = backbone
        self.num_modes = num_modes
        

        ###############
        # Map Encoder #
        ###############
        
        if backbone == 'map_encoder':            
            self.feature_extractor = My_MapEncoder(input_channels = 1, input_size=112, 
                                                    hidden_channels = [10,32,64,128,256], output_size = hidden_dim, 
                                                    kernels = [5,5,3,3,3], strides = [1,2,2,2,2])
            hidden_dims = hidden_dim*2
            '''
            model_ft = resnet18(pretrained=False)
            model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
            nn.init.kaiming_normal_(model_ft.conv1.weight, mode='fan_out', nonlinearity='relu')
            self.feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1]) 
            hidden_dims = hidden_dim+512
            '''

        elif backbone == 'mobilenet':       
            self.feature_extractor = MobileNetBackbone('mobilenet_v2', freeze = freeze)  #returns [n,1280] # 18 layers
            #self.hidden_dim = hidden_dim + 1280

        elif backbone == 'resnet18':       
            feature_extractor = ResNetBackbone('resnet18', freeze = freeze) #ResNet18(hidden_dim, freeze)  [n, 512] #9 layers (con avgpool) - if freeze=8 train last conv block
            #self.hidden_dim = hidden_dim + hidden_dim * 2

        elif backbone == 'resnet50':       
            feature_extractor = ResNetBackbone('resnet50', freeze = freeze) #ResNet50(hidden_dim, freeze)  #[n,2048] #9 layers
            #self.hidden_dim = hidden_dim + hidden_dim * 2

        elif backbone == 'resnet_gray':
            resnet = resnet18(pretrained=False)
            modules = list(resnet.children())[:-3]
            modules.append(torch.nn.AdaptiveAvgPool2d((1, 1))) 
            modules[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
            nn.init.kaiming_normal_(modules[0].weight, mode='fan_out', nonlinearity='relu')
            feature_extractor=torch.nn.Sequential(*modules)   
            self.hidden_dim = hidden_dim + 256

        else:
            feature_extractor = None

        backbone_feature_dim = calculate_backbone_feature_dim(feature_extractor, input_shape = (3,224,224)) if backbone != 'None' else 0
        '''
        embedding_h = nn.Linear(input_dim, backbone_feature_dim//4)###//2)
        self.hidden_dim = backbone_feature_dim//8 + backbone_feature_dim # hidden_dim + backbone_feature_dim
        encode_h = nn.GRU( backbone_feature_dim//4,  backbone_feature_dim//8, batch_first=True)
        linear_cat = nn.Linear(self.hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        '''
        if emb_type == 'gru':
            embedding_h = nn.Linear(input_dim, emb_dim//2)     #self.embedding_h = nn.Linear(input_dim+64, hidden_dim)        
            encode_h = nn.GRU(emb_dim//2, emb_dim, batch_first=True)
        elif emb_type == 'pos_enc':
            embedding_h = nn.Linear(input_dim, emb_dim)
            encode_h = PositionalEncoding(emb_dim, dropout)
        else:
            embedding_h = nn.Linear(input_dim, emb_dim)

        linear_cat = nn.Linear(emb_dim + backbone_feature_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        
        #self.embedding_e = nn.Linear(2, hidden_dims) if  ew_type else nn.Linear(1, hidden_dims)
        resize_e = nn.ReplicationPad1d(self.hidden_dim//2-1)
        resize_e2 = nn.ReplicationPad1d(self.hidden_dim * heads//2-1)

        if bn:
            self.batch_norm = nn.BatchNorm1d(hidden_dim)
        elif gn:
            self.group_norm = nn.GroupNorm(32, hidden_dim) 

        if heads == 1:
            gat_1 = My_GATLayer(self.hidden_dim, self.hidden_dim, e_dims = self.hidden_dim//2*2, feat_drop=feat_drop, attn_drop=attn_drop, att_ew=att_ew, res_weight=res_weight, res_connection=res_connection) #GATConv(hidden_dim, hidden_dim, 1,feat_drop, attn_drop,residual=True, activation=torch.relu) 
            gat_2 = My_GATLayer(self.hidden_dim, self.hidden_dim,  e_dims = self.hidden_dim//2*2, feat_drop=feat_drop, attn_drop=attn_drop, att_ew=att_ew, res_weight=res_weight, res_connection=res_connection)  #GATConv(hidden_dim, hidden_dim, 1,feat_drop, attn_drop,residual=True, activation=torch.relu)
            linear1 = nn.Linear(self.hidden_dim, output_dim * self.num_modes)
        else:
            gat_1 = MultiHeadGATLayer(self.hidden_dim, self.hidden_dim, e_dims=self.hidden_dim//2*2,res_weight=res_weight, merge='cat', res_connection=res_connection , num_heads=heads,feat_drop=feat_drop, attn_drop=attn_drop, att_ew=att_ew) #GATConv(hidden_dim, hidden_dim, heads,feat_drop, attn_drop,residual=True, activation='relu')
            #self.embedding_e2 = nn.Linear(2, hidden_dims*heads) if ew_type else nn.Linear(1, hidden_dims*heads)
            gat_2 = MultiHeadGATLayer(self.hidden_dim*heads, self.hidden_dim*heads,e_dims=self.hidden_dim*heads//2*2, res_weight=res_weight, merge='cat', res_connection=res_connection ,num_heads=self.num_modes, feat_drop=0., attn_drop=0., att_ew=att_ew) #GATConv(hidden_dim*heads, hidden_dim*heads, heads,feat_drop, attn_drop,residual=True, activation='relu')
        
            #linear1 = nn.Linear(self.hidden_dim * heads * self.num_modes, 2 * self.num_modes ) if self.emb_type=='pos_enc' else nn.Linear(self.hidden_dim * heads * self.num_modes, output_dim * self.num_modes) #nn.ModuleList()
            '''
            for i in range(NUM_MODES):
                self.linear1.append( nn.Linear(self.hidden_dim, output_dim) )
            '''
            
        if dropout:
            self.dropout_l = nn.Dropout(dropout, inplace=False)
        else:
            self.dropout_l = nn.Dropout(0.)
        
        self.leaky_relu = nn.LeakyReLU(0.1) 
        self.embeddings = nn.ModuleDict({
            'embedding_h': embedding_h,
            'map_encoder': feature_extractor,
            'linear_cat': linear_cat
        })
        if self.emb_type != 'emb':
            self.embeddings['encode_h'] = encode_h
        
        self.base = nn.ModuleDict({
          'resize_e': resize_e,
          'gat1': gat_1,
          'resize_e2': resize_e2,
          'gat2': gat_2
        })

        ###############
        # DECODER_GRU #
        ###############
        # OPTION 1  
        dec_gru = nn.GRUCell(self.hidden_dim * heads * self.num_modes, emb_dim * num_modes)
        # Once we decode over T frames we output final traj + prob
        linear1 = nn.Linear(5 * emb_dim * self.num_modes, self.output_dim * self.num_modes ) 
        # OPTION 2  ,  gat_2
        ##self.dec_gru = nn.GRUCell(self.hidden_dim * heads, emb_dim)
        # Once we decode over T frames we output final traj + prob
        ##linear1 = nn.Linear(5 * emb_dim, self.output_dim) 
        self.base['dec'] = dec_gru
        self.base['linear1'] = linear1

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.bn:
            nn.init.constant_(self.batch_norm.weight, 1)
            nn.init.constant_(self.batch_norm.bias, 0)
        elif self.gn:
            nn.init.constant_(self.group_norm.weight, 1)
            nn.init.constant_(self.group_norm.bias, 0)
        #nn.init.kaiming_normal_(self.embedding_h[0].weight, nonlinearity='leaky_relu', a=0.2)
        nn.init.xavier_normal_(self.embeddings['embedding_h'].weight, torch.nn.init.calculate_gain('selu'))
        ''' 
        OTRA OPCION
        fan_in = self.embeddings['embedding_h'].in_features
        nn.init.normal(self.embeddings['embedding_h'].weight, 0, sqrt(1. / fan_in))
        '''
        nn.init.kaiming_normal_(self.base['linear1'].weight, nonlinearity='relu')
        nn.init.xavier_normal_(self.embeddings['linear_cat'].weight)       
        #if self.heads > 1:
        #    nn.init.xavier_normal_(self.embedding_e2.weight)
    
    def inference(self, g, feats, e_w,snorm_n,snorm_e, maps):
        y=self.forward(g, feats, e_w, snorm_n, snorm_e, maps)
        return y

    def forward(self, g, feats,e_w,snorm_n,snorm_e, maps):
        
        # Input embedding
        if self.emb_type =='gru':
            h_enc = self.embeddings['encode_h'](F.selu(self.embeddings['embedding_h'](feats)))[1].squeeze(dim=0)
        elif self.emb_type == 'pos_enc':
            h_enc = self.embeddings['encode_h'](F.selu(self.embeddings['embedding_h'](feats)))
        else:
            #reshape to have shape (B*V,T*C) [c1,c2,...,c6]
            feats = feats.contiguous().view(feats.shape[0],-1)
            h_enc = self.embeddings['embedding_h'](feats)  #[N,hidds]   


        if self.backbone != 'None':
            # Maps feature extraction
            maps_embedding = self.embeddings['map_encoder'](maps)  
            if self.emb_type == 'pos_enc':
                maps_embedding = maps_embedding.unsqueeze(1).repeat(1,h_enc.shape[1],1)
            # Embeddings concatenation
            h = torch.cat([maps_embedding, h_enc], dim=-1)
            h = self.embeddings['linear_cat'](h)
        
        #h = F.relu(h)
        if self.bn:
            h = self.batch_norm(h)
            h = F.relu(h)
        elif self.gn:
            h = self.group_norm(h)

        # GAT Layers
        if self.ew_dims:
            e = self.base['resize_e'](torch.unsqueeze(e_w,dim=1)).flatten(start_dim=1) #self.embedding_e(e_w)
        else:
            e = torch.ones((1, self.hidden_dim), device=h.device) * e_w
        g.edata['w'] = e 

        h = self.base['gat1'](g, h,snorm_n) 
        
        if self.heads > 1:
            if self.ew_dims:
                e = self.base['resize_e2'](torch.unsqueeze(e_w,dim=1)).flatten(start_dim=1) #self.embedding_e2(e_w)
            else:
                e = torch.ones((1, self.hidden_dim*self.heads), device=h.device) * e_w
            g.edata['w'] = e 
        
        h_modes = self.base['gat2'](g, h, snorm_n)  #BN Y RELU DENTRO DE LA GAT_LAYER

        y = torch.zeros(5, feats.shape[0], self.emb_dim * self.num_modes).float().to(h_modes.device)
        h = h_enc.repeat(1,3)
        #for i, mode in enumerate(h_modes):
        for t in range(5):
            h = self.base['dec'](h_modes, h)
            y[t] = h
        
        y = self.dropout_l(y)
        y = self.base['linear1'](y.view(feats.shape[0], -1))
        
        if self.emb_type == 'pos_enc':
            y = y.view(y.shape[0],-1)

        mode_probabilities = torch.cat([y[:, self.output_dim * i - 1].unsqueeze(1) for i in range(1,self.num_modes+1)], dim=1)
        predictions = torch.cat([y[:, self.output_dim * (i-1) : self.output_dim*i - 1]  for i in  range(1,self.num_modes+1)], dim=1)

        # Normalize the probabilities to sum to 1 for inference.
        ##mode_probabilities = y[:, -self.num_modes:].clone()
        ##predictions = y[:, :-self.num_modes]

        if not self.training:
            mode_probabilities = F.softmax(mode_probabilities, dim=-1)

        return torch.cat((predictions, mode_probabilities), 1)
        

if __name__ == '__main__':

    history_frames = 7
    future_frames = 6
    hidden_dims = 768
    heads = 2

    input_dim = 7*(history_frames)
    output_dim = 2*future_frames + 1

    model = SCOUT_MTP(input_dim=input_dim, hidden_dim=hidden_dims, emb_dim=512, emb_type='emb', output_dim=output_dim, heads=heads,  ew_dims= True,
                   dropout=0.1, bn=False, feat_drop=0., attn_drop=0., att_ew=False, backbone='resnet50', freeze=True)
    
    
    g = dgl.graph(([0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5]))
    e_w = torch.rand(6, 2)
    snorm_n = torch.rand(6, 1)
    snorm_e = torch.rand(6, 1)
    feats = torch.rand(6, history_frames, 7)
    maps = torch.rand(6, 3, 112, 112)
    out = model(g, feats, e_w,snorm_n,snorm_e,  maps)

    #summary(model.feature_extractor, input_size=(1,112,112), device='cpu')
    test_dataset = nuscenes_Dataset(train_val_test='test', rel_types=True, history_frames=history_frames, future_frames=future_frames, local_frame=False) 
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)

    for batch in test_dataloader:
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos, maps, scene = batch
        e_w = batched_graph.edata['w']#.unsqueeze(1)
        out = model(batched_graph, feats,e_w,snorm_n,snorm_e, maps)
        print(out.shape)