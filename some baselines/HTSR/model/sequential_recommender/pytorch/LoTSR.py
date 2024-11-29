__author__ = "AoRan Zhang"
__email__ = "justzhangaoran@gmail.com"

__all__ = ["HTransRec_l"]


from model.base import AbstractRecommender
from model.MLP import MLP
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from util.pytorch import pairwise_loss, pointwise_loss
from util.pytorch import l2_distance, l2_loss
from util.common import Reduction
from util.pytorch import get_initializer
from data import TimeOrderPairwiseSampler, TimeOrderPointwiseSampler

def cosh_neg_power(tensor): 
    return torch.log(tensor + torch.sqrt(tensor**2 - 1.))

def hyper_distance(u,v):
    d = u.size(1) - 1
    uv = u * v
    uv = torch.cat((-uv.narrow(1, 0, 1), uv.narrow(1, 1, d)), dim=1)
    return cosh_neg_power(-torch.sum(uv, dim=1, keepdim=False))

def hyper_distance_eval(u,v):
    d = u.size(1) - 1
    u = torch.cat((-u.narrow(1, 0, 1), u.narrow(1, 1, d)), dim=1)
    uv = torch.matmul(u, v)
    return cosh_neg_power(-uv)

class Euclidean_to_Hpyerbolic:
    def __init__(self, eps=1e-5, PROJ_EPS=1e-5):
        self.eps = eps
        self.PROJ_EPS = PROJ_EPS
        self.tanh = nn.Tanh()
        self._eps = 1e-10
        self.max_norm = 1e3
    
    def ldot(self, u, v, keepdim=False):
        d = u.size(1) - 1
        uv = u * v
        uv = torch.cat((-uv.narrow(1, 0, 1), uv.narrow(1, 1, d)), dim=1)
        return torch.sum(uv, dim=1, keepdim=keepdim)
    
    def normalize(self, w):
        d = w.size(-1) - 1
        narrowed = w.narrow(-1, 1, d)
        if self.max_norm:
            narrowed = torch.renorm(narrowed.view(-1, d), 2, 0, self.max_norm)
        first = 1 + torch.sum(torch.pow(narrowed, 2), dim=-1, keepdim=True)
        first = torch.sqrt(first)
        return torch.cat((first, narrowed), dim=1)   
    
    def exp_map_normalize(self, v):
        ones = torch.ones_like(v)
        ones[:, 0] = 0
        v = torch.mul(ones,v)
        return self.exp_map_zero(v)
    
    def exp_map_x(self, p, d_p, p_normalize=True):
        ldv = self.ldot(d_p, d_p, keepdim=True)
        nd_p = torch.sqrt(torch.clamp(ldv + self.eps, self._eps))

        t = torch.clamp(nd_p, max=self.max_norm)
        newp = (torch.cosh(t) * p) + (torch.sinh(t) * d_p / nd_p)

        if p_normalize:
            newp = self.normalize(newp)
        return newp
    
    def exp_map_zero(self, v):
        zeros = torch.zeros_like(v)
        zeros[:, 0] = 1
        return self.exp_map_x(zeros, v)
    
    def log_map_zero(self, y, i=-1):
        zeros = torch.zeros_like(y)
        zeros[:, 0] = 1
        return self.log_map_x(zeros, y)
    
    def log_map_x(self, x, y):
        xy = self.ldot(x, y).unsqueeze(-1)
        tmp = torch.sqrt(torch.clamp(xy * xy - 1 + self.eps, self._eps))
        v = cosh_neg_power(-xy) / (tmp) * torch.addcmul(y, xy, x)        
        result = v
        return result

     

class _HTransRec_l(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super(_HTransRec_l, self).__init__()

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, embed_dim)
        self.item_embeddings = nn.Embedding(num_items, embed_dim)
        self.global_transition = Parameter(torch.Tensor(1, embed_dim))

        self.item_biases = nn.Embedding(num_items, 1)

        # weight initialization
        self.reset_parameters()
        self.E2H = Euclidean_to_Hpyerbolic()

    def reset_parameters(self, init_method="uniform"):
        #init = get_initializer(init_method)
        zero_init = get_initializer("zeros")

        zero_init(self.user_embeddings.weight)        
        nn.init.normal_(self.global_transition, 0, 0.01)
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        zero_init(self.item_biases.weight)

    def forward(self, user_ids, last_items, pre_items):
        user_embs = self.user_embeddings(user_ids)
        last_item_embs = self.item_embeddings(last_items)
        pre_item_embs = self.item_embeddings(pre_items)
        pre_item_bias = self.item_biases(pre_items)
        transed_emb = user_embs + self.global_transition + last_item_embs
        
        transed_emb = self.E2H.exp_map_zero(transed_emb)
        pre_item_embs = self.E2H.exp_map_zero(pre_item_embs)
        
        hat_y = -hyper_distance(transed_emb, pre_item_embs) + torch.squeeze(pre_item_bias)
        
        transed_emb = self.E2H.log_map_zero(transed_emb)
        pre_item_embs = self.E2H.log_map_zero(pre_item_embs)
        return hat_y

    def predict(self, user_ids, last_items, pre_items):
        user_embs = self.user_embeddings(user_ids)
        last_item_embs = self.item_embeddings(last_items)
        transed_emb = user_embs + self.global_transition + last_item_embs
        
        transed_emb = self.E2H.exp_map_zero(transed_emb)
        # item = self.E2H.exp_map_zero(self.item_embeddings.weight)
        pre_item_embs = self.item_embeddings(pre_items)
        pre_item_embs = self.E2H.exp_map_zero(pre_item_embs)
        
        # ratings = -hyper_distance_eval(transed_emb, item.T)
        # ratings += torch.squeeze(self.item_biases.weight)
        # return ratings
        return transed_emb, pre_item_embs


class HTransRec_l(AbstractRecommender):
    def __init__(self, config):
        super(HTransRec_l, self).__init__(config)
        self.lr = config["lr"]
        self.reg = config["reg"]
        self.emb_size = config["embedding_size"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]

        self.param_init = config["param_init"]
        self.is_pairwise = config["is_pairwise"]
        self.loss_func = config["loss_func"]

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.user_pos_dict = self.dataset.train_data.to_user_dict(by_time=True)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.htransrec_l = _HTransRec_l(self.num_users, self.num_items, self.emb_size).to(self.device)
        self.htransrec_l.reset_parameters(self.param_init)
        self.optimizer = torch.optim.Adam(self.htransrec_l.parameters(), lr=self.lr)
        self.MLP = MLP([self.emb_size * 2, self.emb_size, 1], 'ReLU').to(self.device)
        self.MLP_opt = torch.optim.Adam(self.MLP.parameters(), lr=self.lr)
        self.MSE_Loss = nn.MSELoss(reduction='mean')

    def train_model(self):
        if self.is_pairwise:
            self._train_pairwise()
        else:
            self._train_pointwise()

    def _train_pairwise(self):
        data_iter = TimeOrderPairwiseSampler(self.dataset.train_data,
                                             len_seqs=1, len_next=1, num_neg=1,
                                             batch_size=self.batch_size,
                                             shuffle=True, drop_last=False)

        # self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            self.htransrec_l.train()
            avg_loss = []
            for bat_users, bat_last_items, bat_pos_items, bat_neg_items, _, _ in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_last_items = torch.from_numpy(bat_last_items).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                yui = self.htransrec_l(bat_users, bat_last_items, bat_pos_items)
                yuj = self.htransrec_l(bat_users, bat_last_items, bat_neg_items)

                loss = pairwise_loss(self.loss_func, yui-yuj, reduction=Reduction.SUM)
                reg_loss = l2_loss(self.htransrec_l.user_embeddings(bat_users),
                                   self.htransrec_l.global_transition,
                                   self.htransrec_l.item_embeddings(bat_last_items),
                                   self.htransrec_l.item_embeddings(bat_pos_items),
                                   self.htransrec_l.item_embeddings(bat_neg_items),
                                   self.htransrec_l.item_biases(bat_pos_items),
                                   self.htransrec_l.item_biases(bat_neg_items)
                                   )
                loss += self.reg * reg_loss
                avg_loss.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f'epoch {epoch}  avg_bpr_loss:{sum(avg_loss) / len(avg_loss)}')
            # result = self.evaluate_model()
            # self.logger.info("epoch %d:\t%s" % (epoch, result))

        self.htransrec_l.eval()
        for epoch in range(self.epochs):
            self.MLP.train()
            avg_loss = []
            for bat_users, bat_last_items, bat_pos_items, bat_neg_items, bat_rating_seqs, bat_next_rating in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_last_items = torch.from_numpy(bat_last_items).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                # bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                user_emb, item_emb = self.htransrec_l.predict(bat_users, bat_last_items, bat_pos_items)
                logits = self.MLP(torch.cat([user_emb, item_emb], dim=-1)).squeeze()
                labels = torch.from_numpy(bat_next_rating).float().to(self.device)
                loss = self.MSE_Loss(logits, labels)
                avg_loss.append(loss.item())
                self.MLP_opt.zero_grad()
                loss.backward()
                self.MLP_opt.step()
            print(f'epoch {epoch}  avg_mse_loss:{sum(avg_loss) / len(avg_loss)}')

        st = 0
        bs = 1024
        all_logits, all_rating = None, None
        while st < len(self.dataset.test_data):
            en = st + bs if bs + st <= len(self.dataset.test_data) else len(self.dataset.test_data)
            user = self.dataset.test_data[st:en, 0]
            item = self.dataset.test_data[st:en, 1]
            rating = self.dataset.test_data[st:en, 2]
            last_item = self.dataset.test_data[st:en, 3]
            user, item, rating, last_item = torch.from_numpy(user).long(), torch.from_numpy(item).long(), torch.from_numpy(rating), torch.from_numpy(last_item).long()
            user, item, rating, last_item = user.to(self.device), item.to(self.device), rating.to(self.device), last_item.to(self.device)
            user_emb, item_emb = self.htransrec_l.predict(user, last_item, item)
            logits = self.MLP(torch.cat([user_emb, item_emb], dim=-1)).squeeze()
            all_logits = logits if all_logits is None else torch.cat([all_logits, logits], dim=0)
            all_rating = rating if all_rating is None else torch.cat([all_rating, rating], dim=0)
            st = en

        macro_mse, macro_mae, weighted_mse, weighted_mae = self.rating_pred_loss_level(all_logits, all_rating)
        print(f'macro_mse:{macro_mse}, macro_mae:{macro_mae}')
        print(f'weighted_mse:{weighted_mse}, weighted_mae:{weighted_mae}')


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


    def _train_pointwise(self):
        data_iter = TimeOrderPointwiseSampler(self.dataset.train_data,
                                              len_seqs=1, len_next=1, num_neg=1,
                                              batch_size=self.batch_size,
                                              shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            self.htransrec_l.train()
            for bat_users, bat_last_items, bat_items, bat_labels in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_last_items = torch.from_numpy(bat_last_items).long().to(self.device)
                bat_items = torch.from_numpy(bat_items).long().to(self.device)
                bat_labels = torch.from_numpy(bat_labels).float().to(self.device)
                yui = self.htransrec_l(bat_users, bat_last_items, bat_items)

                loss = pointwise_loss(self.loss_func, yui, bat_labels, reduction=Reduction.SUM)
                reg_loss = l2_loss(self.htransrec_l.user_embeddings(bat_users),
                                   self.htransrec_l.global_transition,
                                   self.htransrec_l.item_embeddings(bat_last_items),
                                   self.htransrec_l.item_embeddings(bat_items),
                                   self.htransrec_l.item_biases(bat_items)
                                   )
                loss += self.reg * reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def evaluate_model(self):
        self.htransrec_l.eval()
        return self.evaluator.evaluate(self)

    def predict(self, users):
        last_items = [self.user_pos_dict[u][-1] for u in users]
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        last_items = torch.from_numpy(np.asarray(last_items)).long().to(self.device)
        return self.htransrec_l.predict(users, last_items).cpu().detach().numpy()
