import copy
# import sys

import torch
# import numpy as np
from torch import nn
from .modules import gat_hetero
import math
from torch.autograd import Variable


criterion_binary = torch.nn.BCELoss()
criterion_binary_el = torch.nn.BCELoss(reduce=False)


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


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        # self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        # x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class MergeLayer_pop(torch.nn.Module):
    """
    Edge prediction layer, TODO: remove
    """
    def __init__(self, dim1, dim2, dim3, dim4, pop='total'):
        super().__init__()
        # self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.pop_num = {"null": 0, "total": 1, "dyn": 1, 'both': 2}
        self.fc1 = torch.nn.Linear(dim1 + dim2 + self.pop_num[pop], dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        # [BS, dim + 1] * 2 or [BS, dim + 1] + [BS, dim] or [BS, dim] * 2
        x = torch.cat([x1, x2], dim=1)
        # x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class DGR(nn.Module):
    def __init__(self, model_config):
        """
        model_config:
            dropout:     [float] dropout rate
            node_num:    [dict] key -> the name of node type, value -> {0: x, 1: x}
            latent_dim:    [int]: 32 64 etc.
            n_gnn_layer: [int] number of GNN layers
            ngh_sampler: [NeighborFinder_HIN] temporal neighbor sampler
            n_heads:     [int] number of heads of the graph attention aggregation layer.
            num_ngb:     [int] number of sampled neighbors

            # TODO: can be further incorporated.
            node_feat_path: [str] path to node features; add initial node features (user and item features)
        """
        super(DGR, self).__init__()
        # model config
        self.model_config = model_config
        self.latent_dim = model_config["latent_dim"]
        self.node_dim = None
        self.n_channels = model_config["n_channels"]
        '''
            node embedding tables and time encoder function
        '''
        all_num_nodes = sum(self.model_config['node_num'].values())  # number of user + number of items
        if model_config["n_channels"] == 2:
            # node feature tables (like and dislike)
            assert self.latent_dim % 2 == 0
            self.node_dim = self.latent_dim // 2
            self.embedding_n = torch.nn.Embedding(num_embeddings=all_num_nodes + 1, embedding_dim=self.node_dim)
        else:  # one channel
            self.node_dim = self.latent_dim
        self.embedding_p = torch.nn.Embedding(num_embeddings=all_num_nodes + 1, embedding_dim=self.node_dim)
        print("Node embedding tables.")
        print(self.embedding_p.weight.shape)
        self.rating_embedding = torch.nn.Embedding(num_embeddings=6, embedding_dim=self.node_dim)
        self.int_encoder = IntEncoding(self.node_dim)  # time encoder
        '''
            model parameters 
        '''
        # dropout function
        self.drop = nn.Dropout(model_config['dropout'])
        # Before message passing, create a feature transform layer for each time of node.
        self.adapt_ws = nn.ModuleList()
        for _ in model_config['node_num']:
            self.adapt_ws.append(nn.Linear(self.node_dim, self.node_dim))
        # Message passing and Aggregating  (GNN layers to pass and aggregate features from neighbors)
        self.gat_layers = torch.nn.ModuleList(
            [gat_hetero(self.node_dim,  # in dim
                        self.node_dim,  # out dim
                        len(model_config['node_num']),  # num of node types
                        model_config['n_heads'],
                        model_config['dropout'],
                        version=model_config['version'],
                        ps_alpha=model_config['ps_alpha'],
                        use_norm=False)                  # use_norm: normalize outputs of the last layer.
             for _ in range(model_config['n_gnn_layer'])])

        # cosine similarity between pos emb and neg emb
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def set_version(self, version):
        for i in range(self.model_config['n_gnn_layer']):
            self.gat_layers[i].set_version(version)

    def node_embed(self, ngb_nodes, curr_layers, l_nid, pos=True):
        """
        ngb_nodes: ngb nodes info of the target nodes
            [target_node_info, first-order ngb info, second-order ngb, ....]
            target_node_info: [ n_id, n_time, n_type, rating_pop] each with the shape of [BS]

            rating_pop will not be used here. -> it will be used in the loss.

            first-order ngb info:
            node_records, t_records, rating_records, n_type_records, pop_records

            each with the shape of: [BS, num_ngb]

            second-order of ngb info: [BS, num_ngb,  num_ngb]

        curr_layers: current layer (indicate the in which gnn layer we are extracting the node embedding)
        l_nid      : target nodes  (return embedding of nodes of the l_nid-th layer in the ngb_nodes list)
        """
        assert (curr_layers >= 0)
        device = ngb_nodes[0][-1].device
        """
            get the target node ids, cut_time and node type info
        """
        # get the current n_id, n_cut_time, n_type in this layer.
        src_idx_l, cut_time_l, idx_types = ngb_nodes[l_nid][0], ngb_nodes[l_nid][1], ngb_nodes[l_nid][2]
        if l_nid > 0:
            src_idx_l, cut_time_l, idx_types = src_idx_l.flatten(), cut_time_l.flatten(), idx_types.flatten()
        batch_size = src_idx_l.size(dim=0)
        """
            lookup node features for the target nodes 
        """
        if self.model_config["n_channels"] == 2:
            if pos:
                src_node_feat_org = self.embedding_p(src_idx_l).to(device)
            else:
                src_node_feat_org = self.embedding_n(src_idx_l).to(device)
        else:  # only has one embedding for all nodes.
            src_node_feat_org = self.embedding_p(src_idx_l).to(device)

        if curr_layers == 0:  # if this is the first GNN layer, return the node feat from lookup table
            # TODO: the adapt_ws if included in the attention module, remove this
            # src_node_feat_trans = torch.zeros(batch_size, self.node_dim).to(torch.double).to(device)
            # for t_id in range(len(self.model_config['node_num'])):   # type_id
            #     idx = (idx_types == int(t_id))
            #     if idx.sum() == 0:
            #         continue
            #     src_node_feat_trans[idx] = torch.tanh(self.adapt_ws[t_id](src_node_feat_org[idx]))
            src_node_feat_trans = src_node_feat_org
            if l_nid == 0:
                # final target node,
                return src_node_feat_trans
            else:
                # rating encoding (trainable) TODO: rating should be 1->5
                rating_embed = self.rating_embedding(ngb_nodes[l_nid][3].long().flatten())
                # [batch_size, node_dime]
                return src_node_feat_trans + rating_embed
        else:
            # the embedding of the target nodes at the l-1 th layer
            src_node_conv_feat = self.node_embed(ngb_nodes,
                                                 curr_layers=curr_layers - 1,
                                                 l_nid=l_nid)
            src_ngh_node = ngb_nodes[l_nid + 1][0].view(batch_size, self.model_config['num_ngb'])
            src_ngh_time = ngb_nodes[l_nid + 1][1].view(batch_size, self.model_config['num_ngb'])
            src_ngh_type = ngb_nodes[l_nid + 1][2].view(batch_size, self.model_config['num_ngb']).long()
            src_ngh_rate = ngb_nodes[l_nid + 1][3].view(batch_size, self.model_config['num_ngb']).long()
            src_ngh_pop = ngb_nodes[l_nid + 1][4].view(batch_size, self.model_config['num_ngb'])
            src_ngh_ps = ngb_nodes[l_nid + 1][5].view(batch_size, self.model_config['num_ngb'])
            # print(src_ngh_ps[0])
            # TODO: node pop for future calculation of PS in the input sequence.
            # src_ngh_pop = ngb_nodes[l_nid + 1][4].view(batch_size, self.model_config['num_ngb']).long()
            # delta time  [BS, num_ngb]
            src_ngh_t_delta = torch.subtract(cut_time_l.view(batch_size, 1), src_ngh_time).to(torch.int).long()
            # feature of ngb nodes at the l-1 layer
            src_ngh_node_conv_feat = self.node_embed(ngb_nodes,
                                                     curr_layers=curr_layers - 1,
                                                     l_nid=l_nid + 1)
            # TODO: add time interval embeddings for the ngb nodes
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size,
                                                       self.model_config['num_ngb'],
                                                       self.node_dim)
            time_delta_embed = self.int_encoder(src_ngh_t_delta)  # [BS, num_ngb, node_dim]
            src_ngh_feat = src_ngh_feat + time_delta_embed
            # Aggregation
            mask = src_ngh_node == 0  # [N， n_ngb]
            gat_het = self.gat_layers[curr_layers - 1]
            local = gat_het(src_node_conv_feat,          # target node feature (l-1)
                            idx_types,                   # target node type
                            src_ngh_feat,                # source node feature (l-1)
                            src_ngh_type,                # node type of src
                            src_node_ps=src_ngh_ps,      # source node
                            src_node_rate=src_ngh_rate,  # pop of src edge -> measure their influence
                            mask=mask)                   # mask of src node list -> 0s are padding neighbors
            return local

    def forward(self, interaction, ngh_pos, ngh_neg):
        # interaction = {"user": x, "item": x, "ts": xx}
        # node_records, t_records, rating_records, n_type_records, pop_records
        user_ngb_p = [[interaction['user'], interaction['ts'], interaction['user_t']]]  # n_id, n_time, n_type
        # interaction['u_rate_pop']
        item_ngb_p = [[interaction['item'], interaction['ts'], interaction['item_t']]]
        # interaction['i_rate_pop']
        if self.n_channels == 2:
            user_ngb_n = copy.deepcopy(user_ngb_p)
            item_ngb_n = copy.deepcopy(user_ngb_p)

        for i in range(self.model_config['n_gnn_layer']):
            user_ngb_p.append(ngh_pos[f'ngb_{i}']['user'])  # id,time,n_type,e_type,pop,pop_d
            item_ngb_p.append(ngh_pos[f'ngb_{i}']['item'])
            if self.n_channels == 2:
                user_ngb_n.append(ngh_neg[f'ngb_{i}']['user'])  # id,time,n_type,e_type,pop,pop_d
                item_ngb_n.append(ngh_neg[f'ngb_{i}']['item'])

        # parameters for the self.node_embed method: ngb_nodes, curr_layers, l_nid
        user_embed_p = self.node_embed(user_ngb_p, curr_layers=self.model_config['n_gnn_layer'], l_nid=0)
        item_embed_p = self.node_embed(item_ngb_p, curr_layers=self.model_config['n_gnn_layer'], l_nid=0)
        if self.n_channels == 2:
            user_embed_n = self.node_embed(user_ngb_n, curr_layers=self.model_config['n_gnn_layer'], l_nid=0)
            item_embed_n = self.node_embed(item_ngb_n, curr_layers=self.model_config['n_gnn_layer'], l_nid=0)

            user_embed = torch.cat((user_embed_p, user_embed_n), 1)   # [B, dim]
            item_embed = torch.cat((item_embed_p, item_embed_n), 1)

            user_embed_p_detached = user_embed_p.detach()
            item_embed_p_detached = item_embed_p.detach()
            cos_user = self.cos(user_embed_p_detached, user_embed_n).mean() + 1
            cos_item = self.cos(item_embed_p_detached, item_embed_n).mean() + 1

            return user_embed, item_embed, cos_user + cos_item
        else:
            return user_embed_p, item_embed_p


class DGR2(nn.Module):
    def __init__(self, model_config):
        """
        model_config:
            dropout:     [float] dropout rate
            node_num:    [dict] key -> the name of node type, value -> {0: x, 1: x}
            latent_dim:    [int]: 32 64 etc.
            n_gnn_layer: [int] number of GNN layers
            ngh_sampler: [NeighborFinder_HIN] temporal neighbor sampler
            n_heads:     [int] number of heads of the graph attention aggregation layer.
            num_ngb:     [int] number of sampled neighbors

            # TODO: can be further incorporated.
            node_feat_path: [str] path to node features; add initial node features (user and item features)
        """
        super(DGR2, self).__init__()
        # model config
        self.model_config = model_config
        self.latent_dim = model_config["latent_dim"]
        self.node_dim = None
        self.n_channels = model_config["n_channels"]
        '''
            node embedding tables and time encoder function
        '''
        all_num_nodes = sum(self.model_config['node_num'].values())  # number of user + number of items
        if model_config["n_channels"] == 2:
            # node feature tables (like and dislike)
            assert self.latent_dim % 2 == 0
            self.node_dim = self.latent_dim // 2
            self.embedding_n = torch.nn.Embedding(num_embeddings=all_num_nodes + 1, embedding_dim=self.node_dim)
        else:  # one channel
            self.node_dim = self.latent_dim
        self.embedding_p = torch.nn.Embedding(num_embeddings=all_num_nodes + 1, embedding_dim=self.node_dim)
        print("Node embedding tables.")
        print(self.embedding_p.weight.shape)
        self.rating_embedding = torch.nn.Embedding(num_embeddings=6, embedding_dim=self.node_dim)
        self.int_encoder = IntEncoding(self.node_dim)  # time encoder
        '''
            model parameters 
        '''
        # dropout function
        self.drop = nn.Dropout(model_config['dropout'])
        # Before message passing, create a feature transform layer for each time of node.
        self.adapt_ws = nn.ModuleList()
        for _ in model_config['node_num']:
            self.adapt_ws.append(nn.Linear(self.node_dim, self.node_dim))
        # Message passing and Aggregating  (GNN layers to pass and aggregate features from neighbors)
        self.gat_layers_p = torch.nn.ModuleList(
            [gat_hetero(self.node_dim,  # in dim
                        self.node_dim,  # out dim
                        len(model_config['node_num']),  # num of node types
                        model_config['n_heads'],
                        model_config['dropout'],
                        use_norm=False)                  # use_norm: normalize outputs of the last layer.
             for _ in range(model_config['n_gnn_layer'])])
        self.gat_layers_n = torch.nn.ModuleList(
            [gat_hetero(self.node_dim,  # in dim
                        self.node_dim,  # out dim
                        len(model_config['node_num']),  # num of node types
                        model_config['n_heads'],
                        model_config['dropout'],
                        use_norm=False)  # use_norm: normalize outputs of the last layer.
             for _ in range(model_config['n_gnn_layer'])])

        # cosine similarity between pos emb and neg emb
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def node_embed(self, ngb_nodes, curr_layers, l_nid, pos=True):
        """
        ngb_nodes: ngb nodes info of the target nodes
            [target_node_info, first-order ngb info, second-order ngb, ....]
            target_node_info: [ n_id, n_time, n_type, rating_pop] each with the shape of [BS]

            rating_pop will not be used here. -> it will be used in the loss.

            first-order ngb info:
            node_records, t_records, rating_records, n_type_records, pop_records

            each with the shape of: [BS, num_ngb]

            second-order of ngb info: [BS, num_ngb,  num_ngb]

        curr_layers: current layer (indicate the in which gnn layer we are extracting the node embedding)
        l_nid      : target nodes  (return embedding of nodes of the l_nid-th layer in the ngb_nodes list)
        """
        assert (curr_layers >= 0)
        device = ngb_nodes[0][-1].device
        """
            get the target node ids, cut_time and node type info
        """
        # get the current n_id, n_cut_time, n_type in this layer.
        src_idx_l, cut_time_l, idx_types = ngb_nodes[l_nid][0], ngb_nodes[l_nid][1], ngb_nodes[l_nid][2]
        if l_nid > 0:
            src_idx_l, cut_time_l, idx_types = src_idx_l.flatten(), cut_time_l.flatten(), idx_types.flatten()
        batch_size = src_idx_l.size(dim=0)
        """
            lookup node features for the target nodes 
        """
        if self.model_config["n_channels"] == 2:
            if pos:
                src_node_feat_org = self.embedding_p(src_idx_l).to(device)
            else:
                src_node_feat_org = self.embedding_n(src_idx_l).to(device)
        else:  # only has one embedding for all nodes.
            src_node_feat_org = self.embedding_p(src_idx_l).to(device)

        if curr_layers == 0:  # if this is the first GNN layer, return the node feat from lookup table
            # TODO: the adapt_ws if included in the attention module, remove this
            # src_node_feat_trans = torch.zeros(batch_size, self.node_dim).to(torch.double).to(device)
            # for t_id in range(len(self.model_config['node_num'])):   # type_id
            #     idx = (idx_types == int(t_id))
            #     if idx.sum() == 0:
            #         continue
            #     src_node_feat_trans[idx] = torch.tanh(self.adapt_ws[t_id](src_node_feat_org[idx]))
            src_node_feat_trans = src_node_feat_org
            if l_nid == 0:
                # final target node on rating info
                return src_node_feat_trans
            else:
                # rating encoding (trainable) TODO: rating should be 1->5
                rating_embed = self.rating_embedding(ngb_nodes[l_nid][3].long().flatten())
                # [batch_size, node_dime]
                return src_node_feat_trans + rating_embed
        else:
            # the embedding of the target nodes at the l-1 th layer
            src_node_conv_feat = self.node_embed(ngb_nodes,
                                                 curr_layers=curr_layers - 1,
                                                 l_nid=l_nid)
            src_ngh_node = ngb_nodes[l_nid + 1][0].view(batch_size, self.model_config['num_ngb'])
            src_ngh_time = ngb_nodes[l_nid + 1][1].view(batch_size, self.model_config['num_ngb'])
            src_ngh_type = ngb_nodes[l_nid + 1][2].view(batch_size, self.model_config['num_ngb']).long()
            src_ngh_rate = ngb_nodes[l_nid + 1][3].view(batch_size, self.model_config['num_ngb']).long()
            # TODO: node pop for future calculation of PS in the input sequence.
            # src_ngh_pop = ngb_nodes[l_nid + 1][4].view(batch_size, self.model_config['num_ngb']).long()
            # delta time  [BS, num_ngb]
            src_ngh_t_delta = torch.subtract(cut_time_l.view(batch_size, 1), src_ngh_time).to(torch.int).long()
            # feature of ngb nodes at the l-1 layer
            src_ngh_node_conv_feat = self.node_embed(ngb_nodes,
                                                     curr_layers=curr_layers - 1,
                                                     l_nid=l_nid + 1)
            # TODO: add time interval embeddings for the ngb nodes
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size,
                                                       self.model_config['num_ngb'],
                                                       self.node_dim)
            time_delta_embed = self.int_encoder(src_ngh_t_delta)  # [BS, num_ngb, node_dim]
            src_ngh_feat = src_ngh_feat + time_delta_embed
            # Aggregation
            mask = src_ngh_node == 0  # [N， n_ngb]
            if pos:
                gat_het = self.gat_layers_p[curr_layers - 1]
            else:
                gat_het = self.gat_layers_n[curr_layers - 1]

            local = gat_het(src_node_conv_feat,          # target node feature (l-1)
                            idx_types,                   # target node type
                            src_ngh_feat,                # source node feature (l-1)
                            src_ngh_type,                # node type of src
                            src_node_rate=src_ngh_rate,  # pop of src edge -> measure their influence
                            mask=mask)                   # mask of src node list -> 0s are padding neighbors

            return local

    def forward(self, interaction, ngh_pos, ngh_neg):
        # interaction = {"user": x, "item": x, "ts": xx}
        # node_records, t_records, rating_records, n_type_records, pop_records
        user_ngb_p = [[interaction['user'], interaction['ts'], interaction['user_t']]]  # n_id, n_time, n_type
        # interaction['u_rate_pop']
        item_ngb_p = [[interaction['item'], interaction['ts'], interaction['item_t']]]
        # interaction['i_rate_pop']
        if self.n_channels == 2:
            user_ngb_n = copy.deepcopy(user_ngb_p)
            item_ngb_n = copy.deepcopy(user_ngb_p)

        for i in range(self.model_config['n_gnn_layer']):
            user_ngb_p.append(ngh_pos[f'ngb_{i}']['user'])  # id,time,n_type,e_type,pop,pop_d
            item_ngb_p.append(ngh_pos[f'ngb_{i}']['item'])
            if self.n_channels == 2:
                user_ngb_n.append(ngh_neg[f'ngb_{i}']['user'])  # id,time,n_type,e_type,pop,pop_d
                item_ngb_n.append(ngh_neg[f'ngb_{i}']['item'])

        # parameters for the self.node_embed method: ngb_nodes, curr_layers, l_nid
        user_embed_p = self.node_embed(user_ngb_p, curr_layers=self.model_config['n_gnn_layer'], l_nid=0)
        item_embed_p = self.node_embed(item_ngb_p, curr_layers=self.model_config['n_gnn_layer'], l_nid=0)
        if self.n_channels == 2:
            user_embed_n = self.node_embed(user_ngb_n, curr_layers=self.model_config['n_gnn_layer'], l_nid=0)
            item_embed_n = self.node_embed(item_ngb_n, curr_layers=self.model_config['n_gnn_layer'], l_nid=0)

            # user_embed = torch.cat((user_embed_p, user_embed_n), 1)   # [B, dim]
            # item_embed = torch.cat((item_embed_p, item_embed_n), 1)

            # user_embed_p_detached = user_embed_p.detach()
            # item_embed_p_detached = item_embed_p.detach()
            cos_user = self.cos(user_embed_p, user_embed_n).mean() + 1
            cos_item = self.cos(item_embed_p, item_embed_n).mean() + 1
            return (user_embed_p, user_embed_n), (item_embed_p, item_embed_n), cos_user + cos_item
        else:
            return user_embed_p, item_embed_p
