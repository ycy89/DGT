__author__ = "AoRan Zhang"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["HTransRec"]


from model.base import AbstractRecommender
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from util.pytorch import pairwise_loss, pointwise_loss
from util.pytorch import l2_distance, l2_loss
from util.common import Reduction
from util.pytorch import get_initializer
from data import TimeOrderPairwiseSampler, TimeOrderPointwiseSampler



def clip_by_norm(x, clip_norm, dim=-1):
    norm = torch.square(x).sum(dim, keepdim=True)
    output = torch.where(norm > clip_norm ** 2, x * clip_norm / (norm + 1e-6), x)
    return output
    
def cosh_neg_power(tensor): 
    return torch.log(tensor + torch.sqrt(tensor**2 - 1.))

def hyper_distance(tensor_x,tensor_y):
    norn_x = torch.sum(torch.square(tensor_x),dim=-1, keepdim=False)
    norn_x = torch.clamp(norn_x,0,1.-1e-6)
    norn_y = torch.sum(torch.square(tensor_y),dim=-1, keepdim=False)
    norn_y = torch.clamp(norn_y,0,1.-1e-6)
    x_y_distance = torch.sum(torch.square(tensor_x - tensor_y),dim=-1, keepdim=False)
    return cosh_neg_power(1. + 2. * (x_y_distance / ((1. - norn_x) * (1. - norn_y))))

def th_atanh(x, EPS):
	values = torch.min(x, torch.Tensor([1.0 - EPS]).cuda())
	return 0.5 * (torch.log(1 + values + EPS) - torch.log(1 - values + EPS))
               
def squared_norm(x, dim=-1, keepdim=True):
    return (x ** 2).sum(dim=dim, keepdim=keepdim)

def th_dot(x, y, keepdim=True):
	return torch.sum(x * y, dim=1, keepdim=keepdim)

def th_norm(x, dim=1):
	return torch.norm(x, 2, dim, keepdim=True)

class Euclidean_to_Hpyerbolic:
    def __init__(self, EPS=1e-5, PROJ_EPS=1e-5):
        self.EPS = EPS
        self.PROJ_EPS = PROJ_EPS
        self.tanh = nn.Tanh()
        
    def normalize(self, x):
        return torch.renorm(x, 2, 0, (1. - self.EPS))
    
    def log_map_zero(self, y):
        diff = y + self.EPS
        norm_diff = th_norm(diff)
        return th_atanh(norm_diff, self.EPS) * (-diff) / norm_diff

    def exp_map_zero(self, v):
        v = v + self.EPS
        norm_v = th_norm(v)
        result = self.tanh(norm_v) * v / (norm_v) 
        return self.normalize(result)
     

class _HTransRec(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super(_HTransRec, self).__init__()

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, embed_dim)
        self.item_embeddings = nn.Embedding(num_items, embed_dim)
        self.global_transition = Parameter(torch.Tensor(1, embed_dim))

        self.item_biases = nn.Embedding(num_items, 1)

        # weight initialization
        self.reset_parameters()
        self.E2H = Euclidean_to_Hpyerbolic()

    def reset_parameters(self, init_method="uniform"):
        init = get_initializer(init_method)
        zero_init = get_initializer("zeros")

        zero_init(self.user_embeddings.weight)
        init(self.global_transition)
        init(self.item_embeddings.weight)
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

    def predict(self, user_ids, last_items):
        user_embs = self.user_embeddings(user_ids)
        last_item_embs = self.item_embeddings(last_items)
        transed_emb = user_embs + self.global_transition + last_item_embs
        
        transed_emb = self.E2H.exp_map_zero(transed_emb)
        item = self.E2H.exp_map_zero(self.item_embeddings.weight)
        ratings = -hyper_distance(transed_emb.unsqueeze(dim=1), item)
        ratings += torch.squeeze(self.item_biases.weight)
        return ratings


class HTransRec(AbstractRecommender):
    def __init__(self, config):
        super(HTransRec, self).__init__(config)
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
        self.htransrec = _HTransRec(self.num_users, self.num_items, self.emb_size).to(self.device)
        self.htransrec.reset_parameters(self.param_init)
        self.optimizer = torch.optim.Adam(self.htransrec.parameters(), lr=self.lr)

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
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            self.htransrec.train()
            for bat_users, bat_last_items, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_last_items = torch.from_numpy(bat_last_items).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                yui = self.htransrec(bat_users, bat_last_items, bat_pos_items)
                yuj = self.htransrec(bat_users, bat_last_items, bat_neg_items)

                loss = pairwise_loss(self.loss_func, yui-yuj, reduction=Reduction.SUM)
                reg_loss = l2_loss(self.htransrec.user_embeddings(bat_users),
                                   self.htransrec.global_transition,
                                   self.htransrec.item_embeddings(bat_last_items),
                                   self.htransrec.item_embeddings(bat_pos_items),
                                   self.htransrec.item_embeddings(bat_neg_items),
                                   self.htransrec.item_biases(bat_pos_items),
                                   self.htransrec.item_biases(bat_neg_items)
                                   )
                loss += self.reg * reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def _train_pointwise(self):
        data_iter = TimeOrderPointwiseSampler(self.dataset.train_data,
                                              len_seqs=1, len_next=1, num_neg=1,
                                              batch_size=self.batch_size,
                                              shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            self.htransrec.train()
            for bat_users, bat_last_items, bat_items, bat_labels in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_last_items = torch.from_numpy(bat_last_items).long().to(self.device)
                bat_items = torch.from_numpy(bat_items).long().to(self.device)
                bat_labels = torch.from_numpy(bat_labels).float().to(self.device)
                yui = self.htransrec(bat_users, bat_last_items, bat_items)

                loss = pointwise_loss(self.loss_func, yui, bat_labels, reduction=Reduction.SUM)
                reg_loss = l2_loss(self.htransrec.user_embeddings(bat_users),
                                   self.htransrec.global_transition,
                                   self.htransrec.item_embeddings(bat_last_items),
                                   self.htransrec.item_embeddings(bat_items),
                                   self.htransrec.item_biases(bat_items)
                                   )
                loss += self.reg * reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def evaluate_model(self):
        self.htransrec.eval()
        return self.evaluator.evaluate(self)

    def predict(self, users):
        last_items = [self.user_pos_dict[u][-1] for u in users]
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        last_items = torch.from_numpy(np.asarray(last_items)).long().to(self.device)
        return self.htransrec.predict(users, last_items).cpu().detach().numpy()
