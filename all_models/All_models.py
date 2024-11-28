import torch
import torch.nn as nn
from torch.nn.modules import activation

# import all embedding modules
from .tradition import *
from .seq import *
from .DSRec import *
import torch.nn.functional as F

# from .graph import *


# *** Base output module for rating prediction ***
class rate_predict(nn.Module):

    def __init__(self, dim, layers_size, problem="regression", dropout=0.1):
        super(rate_predict, self).__init__()
        self.dim = dim
        self.dropout = dropout
        self.n_layers = len(layers_size)
        self.problem = problem
        # model
        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(layers_size[:-1], layers_size[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output_r = torch.nn.Linear(in_features=layers_size[-1], out_features=1)  # output layer
        self.logistic = torch.nn.Sigmoid()  # logistic layer for output ratings.

        self.proba_fc = torch.nn.Linear(layers_size[-1], layers_size[-1])
        self.affine_output_c = torch.nn.Linear(in_features=layers_size[-1], out_features=5)  # output layer

        self.softmax = nn.Softmax(dim=1)  # softmax layer
        self.activation_function = torch.nn.ReLU()
        self.dropout_function = torch.nn.Dropout(p=self.dropout)

    def last_output(self, user_embed, item_embed):
        feat = torch.cat([user_embed, item_embed], dim=-1)  # the concat latent vector
        # feat = feat.float()
        for idx, _ in enumerate(range(len(self.fc_layers))):
            feat = self.fc_layers[idx](feat)
            feat = self.dropout_function(feat)
            feat = self.activation_function(feat)
            # vector = torch.nn.BatchNorm1d()(vector)  # batch normalization?
        return feat

    def get_rate(self, feat):
        # feat = self.last_feature(user_embed, item_embed)
        rating = self.affine_output_r(feat)
        rating = torch.squeeze(self.logistic(rating))
        return rating

    def get_probabilities(self, feat):
        feat = feat.detach()
        # feat = self.last_output(user_embed, item_embed)
        feat = self.activation_function(self.proba_fc(feat))
        rating = self.affine_output_c(feat)
        rating_prob = torch.squeeze(self.softmax(rating))
        return rating_prob


class rate_predict_c(nn.Module):

    def __init__(self, dim, layers_size, problem="regression", dropout=0.1):
        super(rate_predict_c, self).__init__()
        self.dim = dim
        self.dropout = dropout
        self.n_layers = len(layers_size)
        self.problem = problem
        # model
        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(layers_size[:-1], layers_size[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output_r = torch.nn.Linear(in_features=layers_size[-1], out_features=1)  # output layer
        self.logistic = torch.nn.Sigmoid()  # logistic layer for output ratings.

        self.proba_fc = torch.nn.Linear(layers_size[-1], layers_size[-1])
        self.affine_output_c = torch.nn.Linear(in_features=layers_size[-1], out_features=5)  # output layer

        self.softmax = nn.Softmax(dim=1)  # softmax layer
        self.activation_function = torch.nn.ReLU()
        self.dropout_function = torch.nn.Dropout(p=self.dropout)
        self.cosine = nn.CosineSimilarity()

    def last_output(self, user_embed, item_embed):
        feat = torch.cat([user_embed, item_embed], dim=-1)  # the concat latent vector
        # feat = feat.float()
        for idx, _ in enumerate(range(len(self.fc_layers))):
            feat = self.fc_layers[idx](feat)
            feat = self.dropout_function(feat)
            feat = self.activation_function(feat)
            # vector = torch.nn.BatchNorm1d()(vector)  # batch normalization?
        return feat

    def get_rate(self, user_emb, item_emb):
        cos_sim_score = torch.squeeze(self.cosine(user_emb, item_emb))
        rating = (cos_sim_score + 1) / 2
        return rating

    def get_probabilities(self, feat):
        feat = feat.detach()
        # feat = self.last_output(user_embed, item_embed)
        feat = self.activation_function(self.proba_fc(feat))
        rating = self.affine_output_c(feat)
        rating_prob = torch.squeeze(self.softmax(rating))
        return rating_prob


# *** Traditional methods ***
class MF_rate(rate_predict):
    def __init__(self,
                 config: dict):
        super(MF_rate, self).__init__(config['latent_dim'],
                                      config['out_layers'],
                                      problem=config['problem'],
                                      dropout=config['dropout'])
        self.num_users = config['n_users']
        self.num_items = config['n_items']
        self.latent_dim = config['latent_dim']
        self.embed_model = MF_embeds(config=config)  # nn.ModuleList()
        self.f = nn.Sigmoid()
        if self.problem == "regression":
            self.mse = nn.MSELoss()
        else:
            self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, user, item, ratings):
        # predicted ratings
        user_emb, item_emb = self.embed_model(user, item)
        feat = self.last_output(user_embed=user_emb, item_embed=item_emb)
        pre_ratings = torch.squeeze(self.get_rate(feat))
        # loss
        reg_loss = (1 / 2) * (user_emb.norm(2).pow(2) +
                              item_emb.norm(2).pow(2)) / float(len(user))
        if self.problem == 'regression':
            loss = self.mse(pre_ratings, ratings)
        else:
            loss = self.cross_entropy(pre_ratings, ratings)  # input, target
        return loss, reg_loss, pre_ratings


class MF_rate_t(nn.Module):
    def __init__(self, config: dict):
        super(MF_rate_t, self).__init__()
        self.num_users = config['n_users']
        self.num_items = config['n_items']
        self.latent_dim = config['latent_dim']
        self.embed_model = MF_embeds(config=config)
        self.f = nn.Sigmoid()
        self.mse = nn.MSELoss()
        # self.bias = nn.Parameter()
        self.user_biases = torch.nn.Embedding(self.num_users + 1, 1)
        self.item_biases = torch.nn.Embedding(self.num_items + 1, 1)

    def forward(self, user, item, ratings):
        # predicted ratings
        user_emb, item_emb = self.embed_model(user, item)
        # [B, dim]
        user_bias = self.user_biases(user)
        item_bias = self.item_biases(item)
        pre_ratings = torch.squeeze((user_emb * item_emb).sum(1))     # [B]
        # print(pre_ratings.size(), user_bias.size(), item_bias.size())
        # sys.exit()
        pre_ratings = pre_ratings + torch.squeeze(user_bias) + torch.squeeze(item_bias)
        pre_ratings = self.f(pre_ratings)  # scale to 0 -> 1
        # loss
        reg_loss = (1 / 2) * (user_emb.norm(2).pow(2) +
                              item_emb.norm(2).pow(2)) / float(len(user))
        loss = self.mse(pre_ratings, ratings)

        return loss, reg_loss, pre_ratings


# *** Sequential methods ***
class SAS_rate(rate_predict):
    def __init__(self,
                 config: dict):
        super(SAS_rate, self).__init__(config['latent_dim'],
                                       config['out_layers'],
                                       problem=config['problem'],
                                       dropout=config['dropout'])
        self.embed_model = SASRec_embeds(user_num=config['n_users'],
                                         item_num=config['n_items'],
                                         args=config['args'])
        self.maxlen = config['args'].maxlen
        self.merger = nn.Linear(config['latent_dim'] * 2, config['latent_dim'])
        if self.problem == 'regression':
            self.mse = nn.MSELoss(reduce=False)
        else:
            self.cross_entropy = nn.CrossEntropyLoss(reduce=False)  # input, target

    def forward(self, user_seq, target_seq, target_rate, user_id, his_rate=None):
        # predicted ratings
        if his_rate is not None:
            his_rate = (his_rate * 5)
        seq_emb, item_emb, user_emb = self.embed_model(user_seq, target_seq, user_id, neg_seqs=his_rate)
        user_emb = torch.unsqueeze(user_emb, 1).repeat(1, self.maxlen, 1)
        final_user = self.merger(torch.cat((seq_emb, user_emb), dim=2))
        # [bs, max_len, latent_dim]  TODO: include user_emb or not?
        feat = self.last_output(user_embed=seq_emb, item_embed=item_emb)
        pre_ratings = torch.squeeze(self.get_rate(feat))
        # loss
        reg_loss = (1 / 2) * (item_emb.norm(2).pow(2) +
                              user_emb.norm(2).pow(2)) / float(len(user_seq))
        # reg_loss = (1 / 2) * (item_emb.norm(2).pow(2)) / float(len(user_seq))
        if self.problem == 'regression':
            loss = self.mse(pre_ratings, target_rate)
        else:
            # [bs, max_len, C]
            pre_ratings, target_rate = pre_ratings, target_rate
            loss = self.cross_entropy(pre_ratings, target_rate)  # input, target
        # loss = self.mse(pre_ratings, target_rate)
        # print(mask.sum())
        # sys.exit(loss)
        # loss = loss * mask
        return loss, reg_loss, pre_ratings


class GRU_rate(rate_predict):
    def __init__(self,
                 config: dict):
        super(GRU_rate, self).__init__(config['latent_dim'],
                                       config['out_layers'],
                                       problem=config['problem'],
                                       dropout=config['dropout'])
        self.embed_model = GRU4Rec_embeds(user_num=config['n_users'],
                                          item_num=config['n_items'],
                                          args=config['args'])
        self.merger = nn.Linear(config['latent_dim'] * 2, config['latent_dim'])
        if self.problem == 'regression':
            self.mse = nn.MSELoss(reduce=False)
        else:
            self.cross_entropy = nn.CrossEntropyLoss(reduce=False)  # input, target

    def forward(self, user_seq, target_seq, target_rate, user_id):
        # predicted ratings
        seq_emb, item_emb, user_emb = self.embed_model(user_seq, target_seq, user_id, neg_seqs=None)
        user_emb = torch.unsqueeze(user_emb, 1).repeat(1, self.maxlen, 1)
        final_user = self.merger(torch.cat((seq_emb, item_emb), dim=2))
        # [bs, max_len, latent_dim]
        feat = self.last_output(user_embed=user_emb, item_embed=item_emb)
        pre_ratings = torch.squeeze(self.get_rate(feat))
        # loss
        reg_loss = (1 / 2) * (item_emb.norm(2).pow(2) +
                              user_emb.norm(2).pow(2)) / float(len(user_seq))
        if self.problem == 'regression':
            loss = self.mse(pre_ratings, target_rate)
        else:
            # [bs, max_len, C]
            pre_ratings, target_rate = pre_ratings, target_rate
            loss = self.cross_entropy(pre_ratings, target_rate)  # input, target
        return loss, reg_loss, pre_ratings


class GNN_rate(rate_predict):

    def __init__(self, config: dict):
        super(GNN_rate, self).__init__(config['latent_dim'],
                                       config['out_layers'],
                                       problem=config['problem'],
                                       dropout=config['dropout'])
        self.GNN_embed = DGR(model_config=config)
        self.mse = nn.MSELoss(reduce=False)
        # TODO: built a layer for classification (get the probability)
        self.ps_alpha = nn.Parameter(torch.ones(1))
        self.cross_entropy = nn.CrossEntropyLoss()  # input, target
        self.clip = config["clip"]
        self.n_channels = config["n_channels"]
        self.max_number = 0
        self.min_number = 10000
        self.version = 0

    def set_version(self, version):
        self.GNN_embed.set_version(version)
        self.version = version

    def get_ps(self, interaction, ngh_pos, ngh_neg):
        # 在预测ps的时候就需要本来是bias的输入？
        target_ratings = interaction["rate"]
        # predicted rating
        if self.n_channels == 2:
            org_version = self.version
            self.set_version(1)   # 只要version不是4，就不会包括ps在aggregation里面
            user_embed, item_embed, cos_loss = self.GNN_embed(interaction, ngh_pos, ngh_neg)
            self.set_version(org_version)
        else:
            org_version = self.version
            self.set_version(1)
            user_embed, item_embed = self.GNN_embed(interaction, ngh_pos, ngh_neg)
            self.set_version(org_version)
        feat = self.last_output(user_embed=user_embed, item_embed=item_embed)
        # Propensity scores
        pred_prob = self.get_probabilities(feat).to(float)
        rate_idx = target_ratings * 5 - 1
        # classification loss
        target_one_hot = F.one_hot(rate_idx.long(), num_classes=5).to(float)
        ps_loss = self.cross_entropy(pred_prob, target_one_hot)
        # TODO: how to select
        pred_prob_target = pred_prob[target_one_hot.to(bool)]  # p(R_{u, i} = r)s
        user_rating_n = interaction["u_rate_pop"]
        item_rating_n = interaction["i_rate_pop"]
        # ps = user_rating_n * item_rating_n * pred_prob_target
        # TODO: To get the final PS: 1. softmax and scale; 2. learnable scale and sigmoid;
        number = user_rating_n * item_rating_n  # [BS]
        number = torch.pow(number, 0.6)  # TODO: to be tuned
        min_n, max_n = number.min(), number.max()
        if min_n < self.min_number:
            self.min_number = min_n
        if max_n > self.max_number:
            self.max_number = max_n
        number -= self.min_number
        number /= self.max_number * 0.2  # normalize (0->5) within this batch
        ps = number * pred_prob_target
        # second method
        # ps = interaction['rate_p'] * pred_prob_target * number
        # ps = self.logistic(user_rating_n * item_rating_n * self.ps_alpha) * pred_prob_target
        # clip
        ps = torch.clamp(ps, min=self.clip)  # , max=2.0
        return ps, ps_loss

    def forward(self, interaction, ngh_pos, ngh_neg):
        # target rating
        target_ratings = interaction["rate"]
        # predicted rating
        if self.n_channels == 2:
            user_embed, item_embed, cos_loss = self.GNN_embed(interaction, ngh_pos, ngh_neg)
        else:
            user_embed, item_embed = self.GNN_embed(interaction, ngh_pos, ngh_neg)

        feat = self.last_output(user_embed=user_embed, item_embed=item_embed)
        pred_ratings = self.get_rate(feat)
        # Propensity scores
        pred_prob = self.get_probabilities(feat).to(float)
        rate_idx = target_ratings * 5 - 1
        # classification loss
        target_one_hot = F.one_hot(rate_idx.long(), num_classes=5).to(float)
        ps_loss = self.cross_entropy(pred_prob, target_one_hot)
        # TODO: how to select
        pred_prob_target = pred_prob[target_one_hot.to(bool)]  # p(R_{u, i} = r)s
        user_rating_n = interaction["u_rate_pop"]
        item_rating_n = interaction["i_rate_pop"]
        # ps = user_rating_n * item_rating_n * pred_prob_target
        # TODO: To get the final PS: 1. softmax and scale; 2. learnable scale and sigmoid;
        number = user_rating_n * item_rating_n   # [BS]
        number = torch.pow(number, 0.6)   # TODO: to be tuned
        min_n, max_n = number.min(), number.max()
        if min_n < self.min_number:
            self.min_number = min_n
        if max_n > self.max_number:
            self.max_number = max_n
        number -= self.min_number
        number /= self.max_number * 0.2      # normalize (0->5) within this batch
        ps = number * pred_prob_target

        # second method
        # ps = interaction['rate_p'] * pred_prob_target * number

        # ps = self.logistic(user_rating_n * item_rating_n * self.ps_alpha) * pred_prob_target
        # clip
        ps = torch.clamp(ps, min=self.clip)     # , max=2.0
        # loss function
        mse_loss = self.mse(target_ratings, pred_ratings)
        # loss = loss / ps
        # mse_loss = torch.mean(loss)
        if self.n_channels == 2:
            return mse_loss, pred_ratings, ps_loss, ps, cos_loss
        else:
            return mse_loss, pred_ratings, ps_loss, ps


class GNN_rate2(rate_predict):

    def __init__(self, config: dict):
        super(GNN_rate2, self).__init__(config['latent_dim'],
                                        config['out_layers'],
                                        problem=config['problem'],
                                        dropout=config['dropout'])
        # TODO before this reconfig the out_layers of the configs.
        self.GNN_embed = DGR2(model_config=config)   # TODO DGR2
        self.mse = nn.MSELoss(reduce=False)
        # TODO: built a layer for classification (get the probability)
        self.ps_alpha = nn.Parameter(torch.ones(1))
        self.cross_entropy = nn.CrossEntropyLoss()  # input, target
        self.clip = config["clip"]
        self.n_channels = 2

    def forward(self, interaction, ngh_pos, ngh_neg):
        # target rating
        target_ratings = interaction["rate"]
        # predicted rating
        user_embed, item_embed, cos_loss = self.GNN_embed(interaction, ngh_pos, ngh_neg)
        # user_embed/item_embed = [user_emb_p, user_emb__n]
        feat1 = self.last_output(user_embed=user_embed[0], item_embed=item_embed[0])
        feat2 = self.last_output(user_embed=user_embed[0], item_embed=item_embed[1])
        feat3 = self.last_output(user_embed=user_embed[1], item_embed=item_embed[0])
        feat4 = self.last_output(user_embed=user_embed[1], item_embed=item_embed[1])

        r_pp = torch.squeeze(self.get_rate(feat1))
        r_nn = 1 - torch.squeeze(self.get_rate(feat4))
        r_pn = 1 - torch.squeeze(self.get_rate(feat2))
        r_np = 1 - torch.squeeze(self.get_rate(feat3))
        pred_ratings = (r_pp + r_nn + r_pn + r_np) / 4  # in range 0-1

        # Propensity scores
        pred_prob1 = self.get_probabilities(feat1).to(float)
        pred_prob2 = self.get_probabilities(feat2).to(float)
        pred_prob3 = self.get_probabilities(feat3).to(float)
        pred_prob4 = self.get_probabilities(feat4).to(float)
        pred_prob = (pred_prob1 + pred_prob2 + pred_prob3 + pred_prob4) / 4
        rate_idx = target_ratings * 5 - 1
        # classification loss
        target_one_hot = F.one_hot(rate_idx.long(), num_classes=5).to(float)
        ps_loss = self.cross_entropy(pred_prob, target_one_hot)
        # TODO: how to select
        pred_prob_target = pred_prob[target_one_hot.to(bool)]  # p(R_{u, i} = r)s
        user_rating_n = interaction["u_rate_pop"]
        item_rating_n = interaction["i_rate_pop"]
        # ps = user_rating_n * item_rating_n * pred_prob_target
        # TODO: To get the final PS: 1. softmax and scale; 2. learnable scale and sigmoid;
        number = user_rating_n * item_rating_n   # [BS]
        number = torch.pow(number, 0.6)   # TODO: to be tuned
        number -= number.min()
        number /= number.max() * 0.2      # normalize (0->5) within this batch
        # ps = number * pred_prob_target
        # second method
        ps = interaction['rate_p'] * pred_prob_target * number

        # ps = self.logistic(user_rating_n * item_rating_n * self.ps_alpha) * pred_prob_target
        # clip
        ps = torch.clamp(ps, min=self.clip, max=1.0)
        # loss function
        mse_loss = self.mse(target_ratings, pred_ratings)
        # loss = loss / ps
        # mse_loss = torch.mean(loss)
        if self.n_channels == 2:
            return mse_loss, pred_ratings, ps_loss, ps, cos_loss
        else:
            return mse_loss, pred_ratings, ps_loss, ps


class GNN_rate3(rate_predict_c):

    def __init__(self, config: dict):
        super(GNN_rate3, self).__init__(config['latent_dim'],
                                        config['out_layers'],
                                        problem=config['problem'],
                                        dropout=config['dropout'])
        # TODO before this reconfig the out_layers of the configs.
        self.GNN_embed = DGR2(model_config=config)   # TODO DGR2
        self.mse = nn.MSELoss(reduce=False)
        # TODO: built a layer for classification (get the probability)
        self.ps_alpha = nn.Parameter(torch.ones(1))
        self.cross_entropy = nn.CrossEntropyLoss()  # input, target
        self.clip = config["clip"]
        self.n_channels = 2

    def forward(self, interaction, ngh_pos, ngh_neg):
        # target rating
        target_ratings = interaction["rate"]
        # predicted rating
        user_embed, item_embed, cos_loss = self.GNN_embed(interaction, ngh_pos, ngh_neg)
        # user_embed/item_embed = [user_emb_p, user_emb__n]
        feat1 = self.last_output(user_embed=user_embed[0], item_embed=item_embed[0])
        feat2 = self.last_output(user_embed=user_embed[0], item_embed=item_embed[1])
        feat3 = self.last_output(user_embed=user_embed[1], item_embed=item_embed[0])
        feat4 = self.last_output(user_embed=user_embed[1], item_embed=item_embed[1])

        r_pp = torch.squeeze(self.get_rate(user_emb=user_embed[0], item_emb=item_embed[0]))
        r_pn = 1 - torch.squeeze(self.get_rate(user_emb=user_embed[0], item_emb=item_embed[1]))
        r_np = 1 - torch.squeeze(self.get_rate(user_emb=user_embed[1], item_emb=item_embed[0]))
        r_nn = 1 - torch.squeeze(self.get_rate(user_emb=user_embed[1], item_emb=item_embed[1]))
        pred_ratings = (r_pp + r_nn + r_pn + r_np) / 4  # in range 0-1

        # Propensity scores
        pred_prob1 = self.get_probabilities(feat1).to(float)
        pred_prob2 = self.get_probabilities(feat2).to(float)
        pred_prob3 = self.get_probabilities(feat3).to(float)
        pred_prob4 = self.get_probabilities(feat4).to(float)
        pred_prob = (pred_prob1 + pred_prob2 + pred_prob3 + pred_prob4) / 4
        rate_idx = target_ratings * 5 - 1
        # classification loss
        target_one_hot = F.one_hot(rate_idx.long(), num_classes=5).to(float)
        ps_loss = self.cross_entropy(pred_prob, target_one_hot)
        # TODO: how to select
        pred_prob_target = pred_prob[target_one_hot.to(bool)]  # p(R_{u, i} = r)s
        user_rating_n = interaction["u_rate_pop"]
        item_rating_n = interaction["i_rate_pop"]
        # ps = user_rating_n * item_rating_n * pred_prob_target
        # TODO: To get the final PS: 1. softmax and scale; 2. learnable scale and sigmoid;
        number = user_rating_n * item_rating_n   # [BS]
        number = torch.pow(number, 0.6)   # TODO: to be tuned
        number -= number.min()
        number /= number.max() * 0.2      # normalize (0->5) within this batch
        # ps = number * pred_prob_target
        # second method
        ps = interaction['rate_p'] * pred_prob_target * number

        # ps = self.logistic(user_rating_n * item_rating_n * self.ps_alpha) * pred_prob_target
        # clip
        ps = torch.clamp(ps, min=self.clip, max=1.0)
        # loss function
        mse_loss = self.mse(target_ratings, pred_ratings)
        # loss = loss / ps
        # mse_loss = torch.mean(loss)
        if self.n_channels == 2:
            return mse_loss, pred_ratings, ps_loss, ps, cos_loss
        else:
            return mse_loss, pred_ratings, ps_loss, ps


class GNN_rate_v4(rate_predict):

    def __init__(self, config: dict):
        super(GNN_rate_v4, self).__init__(config['latent_dim'],
                                          config['out_layers'],
                                          problem=config['problem'],
                                          dropout=config['dropout'])
        self.GNN_embed = DGR(model_config=config)
        self.mse = nn.MSELoss(reduce=False)
        # TODO: built a layer for classification (get the probability)
        self.ps_alpha = nn.Parameter(torch.ones(1))
        self.cross_entropy = nn.CrossEntropyLoss()  # input, target
        self.clip = config["clip"]
        self.n_channels = config["n_channels"]

    def forward(self, interaction, ngh_pos, ngh_neg):
        # target rating
        target_ratings = interaction["rate"]
        # predicted rating
        if self.n_channels == 2:
            user_embed, item_embed, cos_loss = self.GNN_embed(interaction, ngh_pos, ngh_neg)
        else:
            user_embed, item_embed = self.GNN_embed(interaction, ngh_pos, ngh_neg)
        feat = self.last_output(user_embed=user_embed, item_embed=item_embed)
        pred_ratings = self.get_rate(feat)
        # Propensity scores
        pred_prob = self.get_probabilities(feat).to(float)
        rate_idx = target_ratings * 5 - 1
        # classification loss
        target_one_hot = F.one_hot(rate_idx.long(), num_classes=5).to(float)
        ps_loss = self.cross_entropy(pred_prob, target_one_hot)
        # TODO: how to select
        pred_prob_target = pred_prob[target_one_hot.to(bool)]  # p(R_{u, i} = r)s
        user_rating_n = interaction["u_rate_pop"]
        item_rating_n = interaction["i_rate_pop"]
        # ps = user_rating_n * item_rating_n * pred_prob_target
        # TODO: To get the final PS: 1. softmax and scale; 2. learnable scale and sigmoid;
        number = user_rating_n * item_rating_n   # [BS]
        number = torch.pow(number, 0.6)   # TODO: to be tuned
        number -= number.min()
        number /= number.max() * 0.2      # normalize (0->5) within this batch
        # ps = number * pred_prob_target

        # second method
        ps = interaction['rate_p'] * pred_prob_target

        # ps = self.logistic(user_rating_n * item_rating_n * self.ps_alpha) * pred_prob_target
        # clip
        ps = torch.clamp(ps, min=self.clip, max=1.0)
        # loss function
        mse_loss = self.mse(target_ratings, pred_ratings)
        # loss = loss / ps
        # mse_loss = torch.mean(loss)
        if self.n_channels == 2:
            return mse_loss, pred_ratings, ps_loss, ps, cos_loss
        else:
            return mse_loss, pred_ratings, ps_loss, ps