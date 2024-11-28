# from torch_geometric.nn.inits import glorot
import sys

import torch.nn as nn
import torch
import math
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


class IntEncoding(nn.Module):
    """Implement the fixed PE function."""

    def __init__(self, d_model, max_len=10000):
        super(IntEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is a single scalar value
        embed = Variable(self.pe[x, :], requires_grad=False)
        return embed


class ScaledDotProductAttention(torch.nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        # self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, mask=None):
        # edge_pri
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        # attn = attn * edge_pri  # point-wise multiply
        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn)  # [n * b, l_q, l_k]
        # attn = self.dropout(attn)  # [n * b, l_v, d]
        # output = torch.bmm(attn, v)
        return attn


class gat_hetero(nn.Module):
    """
    gnn layer
    transformer-like structure
    """

    def __init__(self, in_dim, out_dim, num_types, n_heads, dropout=0.2,
                 use_norm=True, version=None, ps_alpha=0.2):
        super(gat_hetero, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        # self.total_rel = num_types * num_relations * num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.attn = None
        self.version = version

        self.k_w = nn.ModuleList()
        self.q_w = nn.ModuleList()
        self.v_w = nn.ModuleList()
        self.a_w = nn.ModuleList()      # use in aggregation
        self.norms = nn.ModuleList()    # normalization layers

        for t in range(num_types):
            self.k_w.append(nn.Linear(in_dim, out_dim, bias=False))
            self.q_w.append(nn.Linear(in_dim, out_dim, bias=False))
            self.v_w.append(nn.Linear(in_dim, out_dim, bias=False))
            self.a_w.append(nn.Linear(out_dim, out_dim, bias=False))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        # TODO:  what is this?
        self.skip = nn.Parameter(torch.ones(num_types))
        self.drop = nn.Dropout(dropout)

        # content-based attention scores
        self.attn_calculator = ScaledDotProductAttention(temperature=self.sqrt_dk)
        self.softmax_dim1 = torch.nn.Softmax(dim=1)
        self.ps_alpha = ps_alpha

    def set_version(self, version):
        self.version = version

    def forward(self, tgt_n_feat,
                tgt_node_types,
                src_n_feat,
                src_node_types,
                src_node_rate=None,
                src_node_ps=None,
                mask=None):
        """
        Given the embedding of current src tgt nodes at the l-1 layer, tgt node embedding at l-th layer.
            tgt_n_feat:     [BS, dim]               # target node feature (l-1) layer
            tgt_node_types: [BS]                    # target node types
            src_n_feat:     [BS, num_ngb, dim]      # source node feature (l-1) layer
            time_diff:      [BS, num_ngb]           # time interval between tgt and src nodes
            src_node_types: [BS, num_ngb]           # pop of src edge -> measure their influence
            src_node_pop:   [BS, num_ngb]           # node type of src
            mask            [BS, num_ngb]           # # mask of src node list -> 0s are padding neighbors
        Return:
            embedding of tgt node at l-th layer. [N, out_dim]
        """
        data_size = tgt_n_feat.size(dim=0)
        num_ngb = src_n_feat.size(dim=1)  # [N, dim]
        device = tgt_n_feat.device
        res = tgt_n_feat
        '''
            - Get Q K V for attention calculation 
        '''
        tgt_n_feat = torch.unsqueeze(tgt_n_feat, 1)
        # self.n_heads * self.d_k = out_dim
        q = torch.zeros(data_size, 1, self.n_heads * self.d_k).to(torch.double).to(device)
        k = torch.zeros(data_size, num_ngb, self.n_heads * self.d_k).to(torch.double).to(device)
        # TODO: do not use value transfer: ?
        v = torch.zeros(data_size, num_ngb, self.n_heads * self.d_k).to(torch.double).to(device)
        for node_type_tmp in range(self.num_types):
            src_type_ind = (src_node_types.cpu().numpy() == node_type_tmp)  # [bs, n_ngb]
            tgt_type_ind = (tgt_node_types.cpu().numpy() == node_type_tmp)
            if src_type_ind.sum() != 0:  # k and v
                k_n_feat_slice = self.k_w[node_type_tmp](src_n_feat[src_type_ind])
                v_n_feat_slice = self.v_w[node_type_tmp](src_n_feat[src_type_ind])

                # edge-type related transformations
                # k[src_type_ind] = torch.matmul(k_n_feat_slice, self.relation_att[edge_type_i])
                # v[src_type_ind] = torch.matmul(v_n_feat_slice, self.relation_msg[edge_type_i])

                k[src_type_ind] = k_n_feat_slice
                v[src_type_ind] = v_n_feat_slice
            if tgt_type_ind.sum() != 0:
                q[tgt_type_ind] = self.q_w[node_type_tmp](tgt_n_feat[tgt_type_ind])  
                # [b, 1, in_dim] -> [b, 1, out_dim]
        '''
            - multi-headed attention module for content-based attention scores       
        '''
        q = q.view(data_size, 1, self.n_heads, self.d_k)
        k = k.view(data_size, num_ngb, self.n_heads, self.d_k)
        v = v.view(data_size, num_ngb, self.n_heads, self.d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, 1, self.d_k)        # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, num_ngb, self.d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, num_ngb, self.d_k)  # (n*b) x lv x dv
        # org: [data_size, num_ngb, self.n_heads]
        mask_attn = mask.view(data_size, 1, num_ngb)
        attn_content = self.attn_calculator(q, k, mask=mask_attn.repeat(self.n_heads, 1, 1))
        # [n_head * BS, 1, num_ngb]
        '''              
            rating and pop based neighbor importance  
        '''  # node degree based importance weights?
        if self.version == 4:
            # get inversion ps score; ngb with smaller ps should be more importance
            src_node_ps = 1 / src_node_ps
            # 1. normalize the PS score of src neighbors (probability sum=1.0)
            # print(src_node_ps[0:10])
            ps_sum = src_node_ps.sum(axis=1, keepdims=True)
            src_node_ps = src_node_ps/ps_sum
            # print(src_node_ps[0:10])
            ps_attn = src_node_ps
            # sys.exit()
        '''
            - combine the time decay with multiplication 
        '''
        if self.version == 4:
            ps_attn = torch.unsqueeze(ps_attn, dim=1).repeat(self.n_heads, 1, 1)
            self.attn = attn_content * (1 - self.ps_alpha) + ps_attn * self.ps_alpha
        else:
            self.attn = attn_content
        message = torch.bmm(self.attn, v)
        message = message.view(self.n_heads, data_size, 1, self.d_k)
        message = message.permute(1, 2, 0, 3).contiguous().view(data_size, -1)  # [N, out_dim]
        '''
            - aggregation
            1. residual h_l = h_(l-1) + weight_(type) * linear_(type)(message)
            2. layer norm             
        '''
        output = torch.zeros(data_size, self.out_dim).to(torch.double).to(device)
        # fc-layer and weights
        for tgt_n_type in range(self.num_types):
            idx = (tgt_node_types.cpu().numpy() == int(tgt_n_type))
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_w[tgt_n_type](message[idx]))
            # node type-specified aggregate rate.
            beta = torch.sigmoid(self.skip[tgt_n_type])
            if self.use_norm:
                output[idx] = self.norms[tgt_n_type](trans_out * beta + res[idx] * (1 - beta))
            else:
                output[idx] = trans_out * beta + res[idx] * (1 - beta)
        return output
