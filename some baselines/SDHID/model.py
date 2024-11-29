import torch.nn as nn
import torch.nn.functional as F
import torch
import scipy.sparse as sp
import numpy as np

class MLP(nn.Module):
    def __init__(self, layers, activation=None, dropout=0.1):
        super().__init__()
        self.fc = []
        for idx, (in_dim, out_dim) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc.append(nn.Linear(in_dim, out_dim))
            if idx == len(layers) - 2:
                break
            if activation == 'ReLU':
                self.fc.append(nn.ReLU())
            elif activation == 'SiLU':
                self.fc.append(nn.SiLU())
            elif activation == 'GeLU':
                self.fc.append(nn.GELU())
            else:
                self.fc.append(nn.ReLU())
            if dropout is not None:
                self.fc.append(nn.Dropout(p=dropout))
        self.fc = nn.Sequential(*self.fc)

    def forward(self, x):
        return self.fc(x)

class MultiheadAttention_FFN(nn.Module):
    def __init__(self, dim:int, head_num:int):
        super().__init__()
        assert dim % head_num == 0
        self.W_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.head_num = head_num
        self.FFN = MLP([dim, 2 * dim, dim], 'SiLU')

    def forward(self, seq, attn_mask):
        bs, n, dim = seq.shape
        head_dim = dim // self.head_num
        x = self.W_qkv(seq)
        query, key, value = torch.split(x, [dim, dim, dim], dim=-1)
        query = query.view(bs, n, self.head_num, head_dim)
        key = key.view(bs, n, self.head_num, head_dim)
        value = value.view(bs, n, self.head_num, head_dim).permute(0, 2, 1, 3)

        attn_score = torch.einsum("bnhd,bmhd->bhnm", query, key) / (head_dim ** 0.5)
        attn_score = attn_score + attn_mask.unsqueeze(0).unsqueeze(0)
        attn_score = F.softmax(attn_score, dim=-1)  # [bs, head, n, n]

        output = torch.einsum('bhnm,bhmd->bnhd', attn_score, value)
        output = self.FFN(output.reshape(bs, n, dim))

        return output

class Log2feats(nn.Module):
    def __init__(self, device, dim=64, seq_len=50, blocks=2, head_num=8):
        super().__init__()
        self.num_blocks = blocks
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.emb_dropout = torch.nn.Dropout(p=0.5)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(dim, eps=1e-8)

        for _ in range(self.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = MultiheadAttention_FFN(dim, head_num)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = MLP([dim, dim * 2, dim])
            self.forward_layers.append(new_fwd_layer)

        attention_mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        self.attention_mask = torch.where(attention_mask, -torch.inf, 0)

    def forward(self, log_seqs, timeline_mask):
        seqs = log_seqs
        positions = torch.arange(log_seqs.shape[1]).to(seqs.device)
        seqs += self.pos_emb(positions).unsqueeze(0).repeat(log_seqs.shape[0], 1, 1)
        seqs = self.emb_dropout(seqs)
        seqs *= ~timeline_mask # broadcast in last dim

        for i in range(len(self.attention_layers)):
            # Q = self.attention_layernorms[i](seqs)
            mha_outputs = self.attention_layers[i](seqs, attn_mask=self.attention_mask)
            seqs = seqs + mha_outputs

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

class CapsultNetwork(nn.Module):
    def __init__(self, pre_capsule_num, next_capsule_num, in_dim, out_dim, iter_time):
        super().__init__()
        self.W = nn.Parameter(torch.randn(pre_capsule_num, next_capsule_num, in_dim, out_dim))
        self.iter_time = iter_time

    def squash(self, x):
        s_squared_norm = (x ** 2).sum(-1, keepdim=True)
        scale = s_squared_norm / (1 + s_squared_norm)
        return scale * x / torch.sqrt(s_squared_norm + 1e-8)

    def forward(self, x):
        # input: [item_num, k, dim/k], output: [item_num, dim]
        b = torch.zeros(x.shape[0], *self.W.shape[:2]).to(x.device) # [b, k, K]
        p = torch.einsum('bkd,kKdD->bkKD', x, self.W) # [bkKD]
        for _ in range(self.iter_time):
            c = F.softmax(b, dim=-1)
            q = torch.einsum('bkKD,bkK->bKD', p, c)  # [bKD]
            q = self.squash(q)
            b += torch.einsum('bkKD,bKD->bkK', p, q)
        return q.view(q.shape[0], -1)



class SDHID(nn.Module):
    def __init__(self, uvs, ivs, graph, args):
        super().__init__()
        self.uvs = uvs
        self.ivs = ivs
        self.graph = self.get_norm_adj_mat(graph).to(args.device)
        self.mask_value = ivs
        self.k = args.k
        self.dim = args.dim
        self.device = args.device
        self.layers = args.layers
        self.reg_weight = args.reg_weight
        self.CL_weight = args.CL_weight
        self.dCov_weight = args.dCov_weight
        self.item_embedding = nn.Embedding(ivs, args.dim)
        self.user_seq_emb_layer = Log2feats(args.device, args.dim, args.maxlen, args.block, args.head_num)
        self.capsule_layer = CapsultNetwork(args.k, args.k, args.dim//args.k, args.dim//args.k, args.iter_time)
        self.init_linear = nn.ModuleList()
        for _ in range(self.k):
            self.init_linear.append(nn.Linear(args.dim, args.dim // args.k))
        self.activate = nn.LeakyReLU()
        self.MLP = MLP([args.dim * 2, args.dim, 1], 'ReLU')
        self.loss = nn.MSELoss()
        self.apply(self.xavier_init)

    def xavier_init(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.ModuleList):
            for sub_module in module:
                if isinstance(sub_module, nn.Linear):
                    nn.init.xavier_normal_(sub_module.weight.data)
                    if sub_module.bias is not None:
                        nn.init.constant_(sub_module.bias.data, 0)

    def get_norm_adj_mat(self, graph):
        # build adj matrix
        inter_M = graph
        inter_M_t = graph.transpose()
        A = sp.dok_matrix((graph.shape[0] + graph.shape[1], graph.shape[0] + graph.shape[1]), dtype=np.float32)
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + graph.shape[0]), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + graph.shape[0], inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        sumArr = A.sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D @ A @ D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def forward(self, user=None, seq=None, pos=None, mode='train'):
        # get item embeddings in hypergraph
        channel_item_embedding = []
        multi_interest_seq_embedding = []
        for i in range(self.k):
            item_embedding = self.init_linear[i](self.item_embedding.weight)
            item_embedding = F.normalize(item_embedding, dim=-1, p=2)

            multi_interest_seq_embedding.append(F.pad(item_embedding, (0, 0, 0, 1))[seq])

            u_i_embedding = torch.cat([torch.zeros(self.uvs, item_embedding.shape[-1]).to(self.device), item_embedding], dim=0)
            u_i_embeddings_list = [u_i_embedding]

            for layer_idx in range(self.layers):
                u_i_embedding = torch.sparse.mm(self.graph, u_i_embedding)
                if layer_idx % 2 == 1:
                    u_i_embeddings_list.append(u_i_embedding)

            u_i_embedding = torch.stack(u_i_embeddings_list, dim=1)
            u_i_embedding = torch.mean(u_i_embedding, dim=1)
            _, item_embedding = torch.split(u_i_embedding, [self.uvs, self.ivs])
            channel_item_embedding.append(item_embedding)
        channel_item_embedding = torch.stack(channel_item_embedding, dim=1)

        # capsule network -> final item_emb
        X = self.capsule_layer(channel_item_embedding) # [item_num, dim]
        if mode == 'test':
            return F.pad(X, (0, 0, 0, 1))

        # get session embedding in dual grpah
        u_i_embedding = torch.cat([torch.zeros(self.uvs, X.shape[-1]).to(self.device), X], dim=0)
        u_i_embeddings_list = []

        for layer_idx in range(3):
            u_i_embedding = torch.sparse.mm(self.graph, u_i_embedding)
            if layer_idx % 2 == 0:
                u_i_embeddings_list.append(u_i_embedding)

        u_i_embedding = torch.stack(u_i_embeddings_list, dim=1)
        u_i_embedding = torch.mean(u_i_embedding, dim=1)
        user_embedding, _ = torch.split(u_i_embedding, [self.uvs, self.ivs])
        seq_emb_dual = user_embedding[user]

        # self-attention
        timeline_mask = (seq == self.mask_value).unsqueeze(-1).to(torch.bool)
        X = F.pad(X, (0, 0, 0, 1))
        seq_emb = X[seq]  # [bs, n, d]
        pos_emb = X[pos]
        seq_emb = self.user_seq_emb_layer(seq_emb, timeline_mask) # [bs, n, d]

        # mean-pooling & calc_logits
        sg = torch.cumsum(seq_emb, dim=1)
        sg = sg / ((torch.arange(sg.shape[1]).view(1, -1, 1).repeat(sg.shape[0], 1, 1)).to(sg.device) + 1)
        logits = self.MLP(torch.cat([sg, pos_emb], dim=-1)).squeeze()

        seq_len = (~timeline_mask).sum(dim=1).squeeze().to(torch.long)
        index1 = torch.arange(seq_len.shape[0]).to(seq_len.device)
        return X, logits, sg[index1, seq_len-1], seq_emb_dual, multi_interest_seq_embedding

    def reg_loss(self, *embeddings):
        # return sum([torch.norm(embeddings[i]) / embeddings[i].shape[0] for i in range(len(embeddings))])
        return 1. / 2 * sum([(embeddings[i] ** 2).sum() / embeddings[i].shape[0] for i in range(len(embeddings))])

    def compute_centered_matrix(self, D, timeline_mask=None):
        # timeline_mask = timeline_mask.unsqueeze(-1).unsqueeze(-1)
        # D = D * ~timeline_mask # [B, n, d, d]
        # row_mean = torch.sum(D, dim=-1, keepdim=True) / torch.sum(~timeline_mask, dim=0, keepdim=True) # [B, n, d, 1]
        # col_mean = torch.sum(D, dim=-2, keepdim=True) / torch.sum(~timeline_mask, dim=1, keepdim=True) # [B, n, 1, d]
        # overall_mean = torch.sum(D, dim=(-2, -1), keepdim=True) / torch.sum(~timeline_mask).view(-1, 1, 1) # [B, n, 1, 1]
        row_mean = torch.mean(D, dim=-1, keepdim=True)
        col_mean = torch.mean(D, dim=-2, keepdim=True)
        overall_mean = torch.mean(D, dim=(-2, -1), keepdim=True)

        D_centered = D - row_mean - col_mean + overall_mean
        return D_centered

    def distance_covariance(self, X, Y, timeline_mask):
        X = X.unsqueeze(-1)
        Y = Y.unsqueeze(-1)
        D_X = torch.cdist(X, X, p=2)
        D_Y = torch.cdist(Y, Y, p=2)
        timeline_mask = timeline_mask.unsqueeze(-1).unsqueeze(-1)
        D_X_centered = self.compute_centered_matrix(D_X) # [B, n, d, d]
        D_Y_centered = self.compute_centered_matrix(D_Y) # [B, n, d, d]

        dCov = torch.sqrt(torch.sum(D_X_centered * D_Y_centered * ~timeline_mask) / torch.sum(~timeline_mask) / (D_X.shape[-1]**2))
        return dCov

    def contrastive_loss(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        random_idx = torch.randperm(y.shape[0])
        pos_logits = F.logsigmoid(torch.mul(x, y).sum(dim=-1)).mean()
        neg_logits = F.logsigmoid(1.0 - torch.mul(x, y[random_idx]).sum(dim=-1)).mean()
        return -pos_logits - neg_logits


    def calc_loss(self, user, seq, pos, rating):
        timeline_mask = (seq == self.mask_value).to(torch.bool)
        item_emb, logits, seq_emb, seq_emb_dual, multi_interest_seq_emb = self.forward(user, seq, pos)
        loss = torch.mean(((logits - rating) * ~timeline_mask) ** 2) # [bs, n]
        reg_loss = self.reg_weight * self.reg_loss(item_emb[pos], item_emb[seq.view(-1)])
        dcov_loss = 0
        for i in range(len(multi_interest_seq_emb) - 1):
            for j in range(i + 1, len(multi_interest_seq_emb)):
                dcov_loss += self.distance_covariance(multi_interest_seq_emb[i], multi_interest_seq_emb[j], timeline_mask)
        cl_loss = self.contrastive_loss(seq_emb, seq_emb_dual)
        loss += reg_loss + self.dCov_weight * dcov_loss + self.CL_weight * cl_loss
        return loss

    def predict_emb(self, ):
        with torch.no_grad():
            item_emb = self.forward(mode='test')
            return item_emb

    def predict_logits(self, seq_emb, target_emb, timeline_mask):
        seq_emb = self.user_seq_emb_layer(seq_emb, timeline_mask.unsqueeze(-1))  # [bs, n, d]
        seq_len = (~timeline_mask).sum(dim=-1).to(torch.long)
        index1 = torch.arange(seq_len.shape[0]).to(seq_len.device)
        sg = torch.cumsum(seq_emb, dim=1)
        sg = sg / ((torch.arange(sg.shape[1]).view(1, -1, 1).repeat(sg.shape[0], 1, 1)).to(sg.device) + 1)
        logits = self.MLP(torch.cat([sg[index1, seq_len-1], target_emb], dim=-1)).squeeze()

        return logits

    def rating_pred_loss_level(self, pre_rating, target_rating):
        # get the mean squared error for each level (rating)
        result = {}
        count = {}
        pre_rating, target_rating = pre_rating.cpu().tolist(), target_rating.cpu().tolist()
        for pre_rate, target_rate in zip(pre_rating, target_rating):
            if target_rate not in result:
                result[target_rate] = []
                count[target_rate] = 0
            result[target_rate].append(pre_rate - target_rate)
            count[target_rate] += 1

        keys = list(result.keys())
        keys.sort()
        # mse
        mse_result = {}
        mae_result = {}
        weights = []
        for key in keys:
            result[key] = np.array(result[key])
            mse_result[key] = np.power(result[key], 2).mean()
            mae_result[key] = np.abs(result[key]).mean()
            weights.append(count[key])
        # macro: all classes have equal weights
        macro_mae = np.average(list(mae_result.values()))
        macro_mse = np.average(list(mse_result.values()))

        # weighted: each class has a weight equal to the number of samples in that class
        weighted_mae = np.average(list(mae_result.values()), weights=weights)
        weighted_mse = np.average(list(mse_result.values()), weights=weights)
        return macro_mse, macro_mae, weighted_mse, weighted_mae
