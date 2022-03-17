import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

os.environ["DGLBACKEND"] = "pytorch"
import numpy as np
from torchvision.models import resnet18
import efemarai as ef


from NuScenes.nuscenes_Dataset import nuscenes_Dataset, collate_batch
from torch.utils.data import DataLoader

class MLP_Enc(nn.Module):
    "Encoder: MLP that takes GNN output as input and returns mu and log variance of the latent distribution."
    "The stddev of the distribution is treated as the log of the variance of the normal distribution for numerical stability."

    def __init__(self, in_dim, z_dim, dropout=0.2):
        super(MLP_Enc, self).__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.linear = nn.Linear(in_dim, in_dim // 2)
        self.log_var = nn.Linear(in_dim // 2, z_dim)
        self.mu = nn.Linear(in_dim // 2, z_dim)
        self.dropout_l = nn.Dropout(dropout)

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.normal_(self.log_var.weight, 0, sqrt(1.0 / self.sigma.in_dim))
        nn.init.xavier_normal_(self.mu.weight)
        nn.init.kaiming_normal_(self.linear.weight, a=0.01, nonlinearity="leaky_relu")

    def forward(self, h):
        h = self.dropout_l(h)
        h = F.leaky_relu(self.linear(h))
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
        nn.init.kaiming_normal_(self.linear0.weight, a=0.02, nonlinearity="leaky_relu")

    def forward(self, h):
        h = self.dropout(h)
        h = F.leaky_relu(self.linear0(h), negative_slope=0.02)
        h = self.dropout(h)
        y = self.linear1(h)
        return y


class My_GATLayer(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        e_dims,
        relu=True,
        feat_drop=0.0,
        attn_drop=0.0,
        att_ew=False,
        res_weight=True,
        res_connection=True,
    ):
        super(My_GATLayer, self).__init__()
        self.linear_self = nn.Linear(in_feats, out_feats, bias=False)
        self.linear_func = nn.Linear(in_feats, out_feats, bias=False)
        self.att_ew = att_ew
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
        nn.init.kaiming_normal_(self.linear_self.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.linear_func.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(
            self.attention_func.weight, a=0.02, nonlinearity="leaky_relu"
        )

    def edge_attention(self, edges):
        concat_z = torch.cat([edges.src["z"], edges.dst["z"]], dim=-1)

        if self.att_ew:
            concat_z = torch.cat(
                [edges.src["z"], edges.dst["z"], edges.data["w"]], dim=-1
            )

        src_e = self.attention_func(concat_z)
        src_e = F.leaky_relu(src_e)
        return {"e": src_e}

    def message_func(self, edges):
        return {"z": edges.src["z"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        h_s = nodes.data["h_s"]
        a = self.attn_drop_l(F.softmax(nodes.mailbox["e"], dim=1))
        h = h_s + torch.sum(a * nodes.mailbox["z"], dim=1)
        return {"h": h}

    def forward(self, g, h, snorm_n):
        with g.local_scope():
            h_in = h.clone()
            g.ndata["h"] = h
            # feat dropout
            h = self.feat_drop_l(h)
            g.ndata["h_s"] = self.linear_self(h)
            g.ndata["z"] = self.linear_func(h)
            g.apply_edges(self.edge_attention)
            g.update_all(self.message_func, self.reduce_func)
            h = g.ndata["h"]
            # h = h * snorm_n # normalize activation w.r.t. graph node size
            if self.relu:
                h = torch.relu(h)
            if self.res_con:
                h = h_in + h
            return h


class MultiHeadGATLayer(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        e_dims,
        relu=True,
        merge="cat",
        feat_drop=0.0,
        attn_drop=0.0,
        att_ew=False,
        res_weight=True,
        res_connection=True,
    ):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(
                My_GATLayer(
                    in_feats,
                    out_feats,
                    e_dims,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    att_ew=att_ew,
                    res_weight=res_weight,
                    res_connection=res_connection,
                )
            )
        self.merge = merge

    def forward(self, g, h, snorm_n):
        head_outs = [attn_head(g, h, snorm_n) for attn_head in self.heads]
        if self.merge == "cat":
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))


class GAT_VAE(nn.Module):
    def __init__(
        self,
        hidden_dim,
        dropout=0.2,
        feat_drop=0.0,
        attn_drop=0.0,
        heads=1,
        att_ew=False,
        ew_dims=1,
    ):
        super().__init__()
        self.heads = heads
        self.resize_e = nn.ReplicationPad1d(hidden_dim // 2 - 1)

        self.gat_1 = MultiHeadGATLayer(
            hidden_dim,
            hidden_dim,
            e_dims=hidden_dim // 2 * 2,
            res_weight=True,
            res_connection=True,
            num_heads=heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            att_ew=att_ew,
        )
        self.resize_e2 = nn.ReplicationPad1d(hidden_dim * heads // 2 - 1)
        self.gat_2 = MultiHeadGATLayer(
            hidden_dim * heads,
            hidden_dim * heads,
            e_dims=hidden_dim * heads // 2 * 2,
            res_weight=True,
            res_connection=True,
            num_heads=1,
            feat_drop=0.0,
            attn_drop=0.0,
            att_ew=att_ew,
        )

    def forward(self, g, h, e_w, snorm_n):
        e = self.resize_e(torch.unsqueeze(e_w, dim=1)).flatten(start_dim=1)
        g.edata["w"] = e
        h = self.gat_1(g, h, snorm_n)
        if self.heads > 1:
            e = self.resize_e2(torch.unsqueeze(e_w, dim=1)).flatten(start_dim=1)
            g.edata["w"] = e
        h = self.gat_2(g, h, snorm_n)
        return h


class VAE_GNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        z_dim,
        output_dim,
        fc=False,
        dropout=0.2,
        feat_drop=0.0,
        attn_drop=0.0,
        heads=2,
        att_ew=True,
        ew_dims=2,
        backbone="resnet",
        freeze=7,
        bn=False,
        gn=False,
    ):
        super().__init__()
        self.heads = heads
        self.fc = fc
        self.z_dim = z_dim
        self.bn = bn
        self.gn = gn

        ###############
        # Map Encoder #
        ###############
        model_ft = resnet18(pretrained=True)
        modules = list(model_ft.children())[:-3]
        modules.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        self.feature_extractor = torch.nn.Sequential(*modules)
        ct = 0
        for child in self.feature_extractor.children():
            ct += 1
            if ct < freeze:
                for param in child.parameters():
                    param.requires_grad = False
        enc_dims = hidden_dim + output_dim + 256
        dec_dims = z_dim + hidden_dim + 256

        ############################
        # Input Features Embedding #
        ############################
        self.embedding_h = nn.Linear(input_dim, hidden_dim)

        #############
        #  ENCODER  #
        #############
        self.GNN_enc = GAT_VAE(
            enc_dims,
            dropout=dropout,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            heads=heads,
            att_ew=att_ew,
            ew_dims=ew_dims,
        )
        encoder_dims = enc_dims * heads
        self.MLP_encoder = MLP_Enc(encoder_dims + output_dim, z_dim, dropout=dropout)

        #############
        #  DECODER  #
        #############
        self.GNN_decoder = GAT_VAE(
            dec_dims,
            dropout=dropout,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            heads=heads,
            att_ew=False,
            ew_dims=ew_dims,
        )  # If att_ew --> embedding_e_dec
        self.MLP_decoder = MLP_Dec(
            dec_dims * heads + z_dim, dec_dims, output_dim, dropout
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
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

    def inference(self, g, feats, e_w, snorm_n, snorm_e, maps):
        """
        Samples from a normal distribution and decodes conditioned to the GNN outputs.
        """
        # Reshape from (B*V,T,C) to (B*V,T*C)
        feats = feats.contiguous().view(feats.shape[0], -1)

        # Input embedding
        h_emb = self.embedding_h(feats)

        # Map Encoding
        maps_emb = self.feature_extractor(maps)

        # Sample from gaussian distribution
        z_sample = torch.distributions.Normal(
            torch.zeros(
                (feats.shape[0], self.z_dim), dtype=feats.dtype, device=feats.device
            ),
            torch.ones(
                (feats.shape[0], self.z_dim), dtype=feats.dtype, device=feats.device
            ),
        ).sample()

        # DECODE
        h_dec = torch.cat([maps_emb.flatten(start_dim=1), h_emb, z_sample], dim=-1)
        if self.bn:
            h_dec = self.bn_dec(h_dec)
        elif self.gn:
            h_dec = self.gn_dec(h_dec)
        h = self.GNN_decoder(g, h_dec, e_w, snorm_n)
        h = torch.cat([h, z_sample], dim=-1)
        recon_y = self.MLP_decoder(h)
        return recon_y

    def forward(self, g, feats, e_w, snorm_n, snorm_e, gt, maps):
        # Reshape from (B*V,T,C) to (B*V,T*C)
        feats = feats.contiguous().view(feats.shape[0], -1)
        gt = gt.contiguous().view(gt.shape[0], -1)

        # Input embedding
        h_emb = self.embedding_h(feats)

        # Map encoding
        maps_emb = self.feature_extractor(maps)

        #### ENCODE ####
        h = torch.cat([maps_emb.flatten(start_dim=1), h_emb, gt], dim=-1)
        # h = self.linear_cat(h)
        h = self.GNN_enc(g, h, e_w, snorm_n)
        h = torch.cat([h, gt], dim=-1)
        mu, log_var = self.MLP_encoder(h)  # Latent distribution

        #### Sample from the latent distribution ###
        z_sample = self.reparameterize(mu, log_var)

        #### DECODE ####
        h_dec = torch.cat([maps_emb.flatten(start_dim=1), h_emb, z_sample], dim=-1)

        h = self.GNN_decoder(g, h_dec, e_w, snorm_n)
        h = torch.cat([h, z_sample], dim=-1)
        recon_y = self.MLP_decoder(h)
        return recon_y, mu, log_var


if __name__ == "__main__":
    history_frames = 7
    future_frames = 10
    hidden_dims = 200

    input_dim = 7*(history_frames-1)
    output_dim = 2*future_frames + 1

    model = VAE_GNN(input_dim, hidden_dims, 16, output_dim)
    '''
    g = dgl.graph(([0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5]))
    e_w = torch.rand(6, 2)
    snorm_n = torch.rand(6, 1)
    snorm_e = torch.rand(6, 1)
    feats = torch.rand(6, 4, 9)
    gt = torch.rand(6, 12, 2)
    maps = torch.rand(6, 3, 112, 112)
    '''
    test_dataset = nuscenes_Dataset(train_val_test='train', rel_types=True, history_frames=history_frames, future_frames=future_frames) 
    test_dataloader = DataLoader(test_dataset, batch_size=3, shuffle=False, collate_fn=collate_batch)


    for batch in test_dataloader:
        batched_graph, output_masks,snorm_n, snorm_e, feats, labels_pos, maps, scene = batch
        e_w = batched_graph.edata['w']
        ef.add_view(maps, view=ef.View.Image)
        ef.inspect(maps, name='maps')
        #e_w= e_w.unsqueeze(1)
        with ef.scan():
            y = model.inference(batched_graph, feats, e_w,snorm_n,snorm_e, maps)
    '''
    while 1:
        with ef.scan():
            out, mu, log_var = model(g, feats, e_w, snorm_n, snorm_e, gt, maps)

        print(out.shape)
    '''
