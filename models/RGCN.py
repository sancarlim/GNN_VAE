import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from dgl.nn.pytorch import utils

class RelGraphConv(nn.Module):
    r"""
    Description
    -----------
    Relational graph convolution layer.
    Relational graph convolution is introduced in "`Modeling Relational Data with Graph
    Convolutional Networks <https://arxiv.org/abs/1703.06103>`__"
    and can be described as below:
    .. math::
       h_i^{(l+1)} = \sigma(\sum_{r\in\mathcal{R}}
       \sum_{j\in\mathcal{N}^r(i)}\frac{1}{c_{i,r}}W_r^{(l)}h_j^{(l)}+W_0^{(l)}h_i^{(l)})
    where :math:`\mathcal{N}^r(i)` is the neighbor set of node :math:`i` w.r.t. relation
    :math:`r`. :math:`c_{i,r}` is the normalizer equal
    to :math:`|\mathcal{N}^r(i)|`. :math:`\sigma` is an activation function. :math:`W_0`
    is the self-loop weight.
    The basis regularization decomposes :math:`W_r` by:
    .. math::
       W_r^{(l)} = \sum_{b=1}^B a_{rb}^{(l)}V_b^{(l)}
    where :math:`B` is the number of bases, :math:`V_b^{(l)}` are linearly combined
    with coefficients :math:`a_{rb}^{(l)}`.
    The block-diagonal-decomposition regularization decomposes :math:`W_r` into :math:`B`
    number of block diagonal matrices. We refer :math:`B` as the number of bases.
    The block regularization decomposes :math:`W_r` by:
    .. math::
       W_r^{(l)} = \oplus_{b=1}^B Q_{rb}^{(l)}
    where :math:`B` is the number of bases, :math:`Q_{rb}^{(l)}` are block
    bases with shape :math:`R^{(d^{(l+1)}/B)*(d^{l}/B)}`.
    Parameters
    ----------
    in_feat : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feat : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    num_rels : int
        Number of relations. .
    regularizer : str
        Which weight regularizer to use "basis" or "bdd".
        "basis" is short for basis-diagonal-decomposition.
        "bdd" is short for block-diagonal-decomposition.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: ``None``.
    bias : bool, optional
        True if bias is added. Default: ``True``.
    activation : callable, optional
        Activation function. Default: ``None``.
    self_loop : bool, optional
        True to include self loop message. Default: ``True``.
    low_mem : bool, optional
        True to use low memory implementation of relation message passing function. Default: False.
        This option trades speed with memory consumption, and will slowdown the forward/backward.
        Turn it on when you encounter OOM problem during training or evaluation. Default: ``False``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``
    layer_norm: float, optional
        Add layer norm. Default: ``False``
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels,
                 regularizer="basis",
                 num_bases=None,
                 bias=True,
                 activation=None,
                 self_loop=True,
                 low_mem=False,
                 dropout=0.0,
                 layer_norm=False):
        super(RelGraphConv, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.num_bases = num_bases
        if self.num_bases is None or self.num_bases > self.num_rels or self.num_bases <= 0:
            self.num_bases = self.num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.low_mem = low_mem
        self.layer_norm = layer_norm

        if regularizer == "basis":
            # add basis weights
            self.weight = nn.Parameter(th.Tensor(self.num_bases, self.in_feat, self.out_feat))
            if self.num_bases < self.num_rels:
                # linear combination coefficients
                self.w_comp = nn.Parameter(th.Tensor(self.num_rels, self.num_bases))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            if self.num_bases < self.num_rels:
                nn.init.xavier_uniform_(self.w_comp,
                                        gain=nn.init.calculate_gain('relu'))
            # message func
            self.message_func = self.basis_message_func
        elif regularizer == "bdd":
            if in_feat % self.num_bases != 0 or out_feat % self.num_bases != 0:
                raise ValueError(
                    'Feature size must be a multiplier of num_bases (%d).'
                    % self.num_bases
                )
            # add block diagonal weights
            self.submat_in = in_feat // self.num_bases
            self.submat_out = out_feat // self.num_bases

            # assuming in_feat and out_feat are both divisible by num_bases
            self.weight = nn.Parameter(th.Tensor(
                self.num_rels, self.num_bases * self.submat_in * self.submat_out))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            # message func
            self.message_func = self.bdd_message_func
        else:
            raise ValueError("Regularizer must be either 'basis' or 'bdd'")

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # layer norm
        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(out_feat, elementwise_affine=True) #nn.GroupNorm(32, out_feat, affine=True) #

        # weight for self loop
        if self.self_loop:
            #self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            self.loop_weight = nn.Linear(in_feat, out_feat, bias=False)
            nn.init.xavier_uniform_(self.loop_weight.weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def basis_message_func(self, edges):
        """Message function for basis regularizer"""
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            weight = th.matmul(self.w_comp, weight).view(
                self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight

        # calculate msg @ W_r before put msg into edge
        # if src is th.int64 we expect it is an index select
        if edges.src['h'].dtype != th.int64 and self.low_mem:
            etypes = th.unique(edges.data['type'])
            msg = th.empty((edges.src['h'].shape[0], self.out_feat),
                           device=edges.src['h'].device, dtype=torch.float16)
            for etype in etypes:
                loc = edges.data['type'] == etype
                w = weight[etype]
                src = edges.src['h'][loc]* edges.data['w'][loc]
                sub_msg = th.matmul(src, w)
                msg[loc] = sub_msg
        else:
            # put W_r into edges then do msg @ W_r
            msg = utils.bmm_maybe_select(edges.src['h'], weight, edges.data['type'])

        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}

        def att_message_func(self, edges):
            """Message function for attention-based gcn"""
            if self.num_bases < self.num_rels:
                # generate all weights from bases
                weight = self.weight.view(self.num_bases,
                                        self.in_feat * self.out_feat)
                weight = th.matmul(self.w_comp, weight).view(
                    self.num_rels, self.in_feat, self.out_feat)
            else:
                weight = self.weight

            # calculate msg @ W_r before put msg into edge
            # if src is th.int64 we expect it is an index select
            if edges.src['h'].dtype != th.int64 and self.low_mem:
                etypes = th.unique(edges.data['type'])
                msg = th.empty((edges.src['h'].shape[0], self.out_feat),
                            device=edges.src['h'].device, dtype=torch.float32)
                for etype in etypes:
                    loc = edges.data['type'] == etype
                    w = weight[etype]
                    src = edges.src['h'][loc]* edges.data['w'][loc]
                    sub_msg = th.matmul(src, w)
                    msg[loc] = sub_msg
            else:
                # put W_r into edges then do msg @ W_r
                msg = utils.bmm_maybe_select(edges.src['h'], weight, edges.data['type'])

            if 'norm' in edges.data:
                msg = msg * edges.data['norm']
            return {'msg': msg}

    def bdd_message_func(self, edges):
        """Message function for block-diagonal-decomposition regularizer"""
        if edges.src['h'].dtype == th.int64 and len(edges.src['h'].shape) == 1:
            raise TypeError('Block decomposition does not allow integer ID feature.')

        # calculate msg @ W_r before put msg into edge
        if self.low_mem:
            etypes = th.unique(edges.data['type'])
            msg = th.empty((edges.src['h'].shape[0], self.out_feat),
                           device=edges.src['h'].device)
            for etype in etypes:
                loc = edges.data['type'] == etype
                w = self.weight[etype].view(self.num_bases, self.submat_in, self.submat_out)
                src = edges.src['h'][loc].view(-1, self.num_bases, self.submat_in)
                sub_msg = th.einsum('abc,bcd->abd', src, w)
                sub_msg = sub_msg.reshape(-1, self.out_feat)
                msg[loc] = sub_msg
        else:
            weight = self.weight.index_select(0, edges.data['type']).view(
                -1, self.submat_in, self.submat_out)
            node = edges.src['h'].view(-1, 1, self.submat_in)
            msg = th.bmm(node, weight).view(-1, self.out_feat)
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}

    def forward(self, g, feat,e_w, etypes, norm=None):
        """
        Description
        -----------
        Forward computation
        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor
            Input node features. Could be either
                * :math:`(|V|, D)` dense tensor
                * :math:`(|V|,)` int64 vector, representing the categorical values of each
                  node. It then treat the input feature as an one-hot encoding feature.
        etypes : torch.Tensor
            Edge type tensor. Shape: :math:`(|E|,)`
        norm : torch.Tensor
            Optional edge normalizer tensor. Shape: :math:`(|E|, 1)`.
        Returns
        -------
        torch.Tensor
            New node features.
        """
        with g.local_scope():
            g.srcdata['h'] = feat
            g.edata['type'] = etypes
            g.edata['w'] = e_w
            if norm is not None:
                g.edata['norm'] = norm
            if self.self_loop:
                #loop_message = utils.matmul_maybe_select(feat[:g.number_of_dst_nodes()],
                #                                         self.loop_weight)
                loop_message = self.loop_weight(feat)
            # message passing
            g.update_all(self.message_func, fn.sum(msg='msg', out='h'))
            # apply bias and activation
            node_repr = g.dstdata['h']
            if self.layer_norm:
                node_repr = self.layer_norm_weight(node_repr)
            if self.bias:
                node_repr = node_repr + self.h_bias
            if self.self_loop:
                node_repr = node_repr + loop_message
            if self.activation:
                node_repr = self.activation(node_repr)
            node_repr = self.dropout(node_repr)
            return node_repr
'''
class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer

        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # weight bases in equation (3)
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))

        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

    def forward(self, g, h, rel_type, norm):
        with g.local_scope():
            g.ndata['h'] = h
            g.edata['rel_type'] = rel_type
            g.edata['norm'] = norm
            if self.num_bases < self.num_rels:
                # generate all weights from bases (equation (3))
                weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
                weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                            self.in_feat, self.out_feat)
            else:
                weight = self.weight

        
            def message_func(edges):
                w = weight[edges.data['rel_type']]
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                msg = msg * edges.data['norm']
                return {'msg': msg}

            def apply_func(nodes):
                h = nodes.data['h']
                if self.bias:
                    h = h + self.bias
                if self.activation:
                    h = self.activation(h)
                return {'h': h}

            g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)
            return g.ndata['h']
'''


class RGCN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_rels,
                 num_bases=-1, num_hidden_layers=2, embedding=True, bn=True, dropout=0.1):
        super(RGCN, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.embedding = embedding
        self.dropout=dropout
        self.bn=bn

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        self.i2h = self.build_input_layer()
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        # hidden to output
        self.h2o = self.build_output_layer()
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.i2h.weight, gain=gain)
        nn.init.xavier_normal_(self.h2o.weight, gain=gain)

    def build_input_layer(self):
        return nn.Linear(self.in_dim, self.h_dim)

    def build_hidden_layer(self):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, 'basis', self.num_bases,
                         activation=F.relu, self_loop=True, dropout=self.dropout, low_mem=True, layer_norm=self.bn)

    def build_output_layer(self):
        #return RGCNLayer(self.h_dim, self.out_dim, self.num_rels, self.num_bases,
        #                 activation=F.relu
        return nn.Linear(self.h_dim,self.out_dim)

    def forward(self, g, inputs,e_w, rel_type, norm):
        #reshape to have shape (B*V,T*C) [c1,c2,...,c6]
        h = inputs.view(inputs.shape[0],-1)

        # input embedding
        if self.embedding:
            h = self.i2h(h)
        #g.ndata['h'] = h

        for layer in self.layers:
            h = layer(g, h.float(),e_w, rel_type, norm)  #Aplica layer_norm -> relu -> dropout
        #h = g.ndata.pop('h')

        return self.h2o(h)
