import os
from data_loader import *
import torch
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader


class Empty_class(object):
    pass


def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, logger):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, logger)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, logger)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, logger):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if torch.cuda.device_count() > 1:  # multi-GPU
            torch.save(model.module.state_dict(), self.path)
        else:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def save_dict_result(filename, dict_data):
    with open(filename, 'w') as fp:
        json.dump(dict_data, fp)


def eval_one_epoch(model_t, data_eval, device, method='pro',
                   show_bar=False, test=False, out_file=None,
                   num_val=1, stage=None, n_channels=1,
                   is_ps_in=False, offset=None):
    # MSE
    res_loss = []
    weight_bs = []
    pbar = tqdm(enumerate(data_eval), disable=show_bar)
    org_ratings = []
    pred_ratings = []
    ps_scores = []
    ps_input = {}
    if test:
        pbar.set_description(f"Test Epoch")
    else:
        pbar.set_description(f"Validation Epoch")
    with torch.no_grad():
        model_t = model_t.eval()
        for ind, data_batch in pbar:
            # put data to device
            if method == "pro":
                interaction, ngb_pos, ngb_neg = data_batch[0], data_batch[1], data_batch[2]
                bs = interaction['user'].size()[0]
                # to device
                for l_key in ngb_pos:  # ngb_0 ngb_1 (k-hop ngb)
                    for e_key in ngb_pos[l_key]:  # pos or neg
                        for i in range(len(ngb_pos[l_key][e_key])):  # 6 info of neighbor nodes
                            ngb_pos[l_key][e_key][i] = ngb_pos[l_key][e_key][i].to(device)
                            if n_channels == 2:
                                ngb_neg[l_key][e_key][i] = ngb_neg[l_key][e_key][i].to(device)
                for key in interaction:
                    interaction[key] = interaction[key].to(device)
                if n_channels == 2:
                    mse_loss, rating, ps_loss, ps, cos_loss = model_t(interaction, ngb_pos, ngb_neg)
                else:
                    mse_loss, rating, ps_loss, ps = model_t(interaction, ngb_pos, ngb_neg)
                pbar.set_postfix(mse_loss=mse_loss.mean().item(), ps_loss=ps_loss.item())

                if is_ps_in:
                    user_b = interaction['user'].cpu().numpy() - offset[0]
                    item_b = interaction['item'].cpu().numpy() - offset[1]
                    ps_in = ps.cpu().detach().numpy()
                    for i in range(len(user_b)):
                        key = str(user_b[i]) + "_" + str(item_b[i])  # TODO add time info
                        ps_input[key] = ps_in[i]

                if stage in ["3", "4"] or stage == "ps_l":  # debias loss ps_l
                    mse_loss = mse_loss / ps
                if stage == "naive":
                    mse_loss = mse_loss / interaction["rate_p"]
                loss = mse_loss.mean()
            else:
                for key in data_batch:
                    data_batch[key] = data_batch[key].to(device)
                bs = data_batch['user'].size()[0]
                if len(data_batch['item'].size()) > 1:
                    # print(data_batch['rating'].size())
                    # print(list(data_batch['rating'].cpu().numpy()))
                    loss, _, rating = model_t(data_batch['user'], data_batch['item'], data_batch['rating'],
                                              data_batch['user_id'], his_rate=data_batch['input_rate'])
                    loss = loss[:, -num_val:]
                    loss = torch.mean(loss)
                    # print(rating.size())
                    # print(rating)
                    # print()
                else:
                    loss, _, rating = model_t(data_batch['user'], data_batch['item'], data_batch['rating'])
                pbar.set_postfix(loss=loss.item())

            res_loss.append(loss.item())
            weight_bs.append(bs)
            if test:   # TODO: record predicted rating for other metrics
                if method == "seq":
                    org_ratings.extend(list(torch.squeeze(data_batch['rating'][:, -num_val:]).cpu().numpy()))
                    pred_ratings.extend(list(torch.squeeze(rating[:, -num_val:]).cpu().numpy()))
                elif method == "trad":  # interaction based method
                    org_ratings.extend(list(data_batch['rating'].cpu().numpy()))
                    pred_ratings.extend(list(rating.cpu().numpy()))
                else:
                    org_ratings.extend(list(interaction['rate'].cpu().numpy()))
                    pred_ratings.extend(list(rating.cpu().numpy()))
                    ps_scores.extend(list(ps.cpu().numpy()))

        loss_final = np.average(res_loss, weights=weight_bs)
        if is_ps_in:
            return loss_final, ps_input
        if test:  # save all predicted ratings and the original ratings for future metrics calculation
            result = {'target': org_ratings, 'predict': pred_ratings}
            if method == "pro":
                result["ps"] = ps_scores
            df = pd.DataFrame(result)
            df.to_csv(out_file)
            # MAE = mae(org_ratings, pred_ratings)
            mse_res, mae_res, macro_res, weighted_res = rating_pred_loss_level(pred_ratings, org_ratings)
            return macro_res, weighted_res, mse_res, mae_res
        else:
            return loss_final


def eval_classification(model_t, data_eval, device, show_bar=False, n_channels=1):
    res_loss = []
    weight_bs = []
    pbar = tqdm(enumerate(data_eval), disable=show_bar)
    org_ratings = []
    pred_ratings = []
    ps_scores = []
    pbar.set_description(f"PS validation Epoch")
    with torch.no_grad():
        model_t = model_t.eval()
        for ind, data_batch in pbar:
            # put data to device

            interaction, ngb_pos, ngb_neg = data_batch[0], data_batch[1], data_batch[2]
            bs = interaction['user'].size()[0]
            # to device
            for l_key in ngb_pos:  # ngb_0 ngb_1 (k-hop ngb)
                for e_key in ngb_pos[l_key]:  # pos or neg
                    for i in range(len(ngb_pos[l_key][e_key])):  # 6 info of neighbor nodes
                        ngb_pos[l_key][e_key][i] = ngb_pos[l_key][e_key][i].to(device)
                        if n_channels == 2:
                            ngb_neg[l_key][e_key][i] = ngb_neg[l_key][e_key][i].to(device)
            for key in interaction:
                interaction[key] = interaction[key].to(device)
            if n_channels == 2:
                mse_loss, rating, ps_loss, ps, cos_loss = model_t(interaction, ngb_pos, ngb_neg)
            else:
                mse_loss, rating, ps_loss, ps = model_t(interaction, ngb_pos, ngb_neg)
            pbar.set_postfix(mse_loss=mse_loss.mean().item(), ps_loss=ps_loss.item())
            loss = ps_loss
            res_loss.append(loss.item())
            weight_bs.append(bs)

            org_ratings.extend(list(interaction['rate'].cpu().numpy()))
            pred_ratings.extend(list(rating.cpu().numpy()))
            ps_scores.extend(list(ps.cpu().numpy()))

        loss_final = np.average(res_loss, weights=weight_bs)
        return loss_final


def mode_configuration(args_tmp):
    if args_tmp.model == "SAS_rate":
        args_sas = Empty_class()
        setattr(args_sas, "device", args_tmp.device)
        setattr(args_sas, "hidden_units", args_tmp.latent_dim)
        setattr(args_sas, "dropout_rate", args_tmp.dropout)
        setattr(args_sas, "num_blocks", 2)   # the default parameters from SASRec
        setattr(args_sas, "num_heads", 1)
        setattr(args_sas, 'maxlen', args_tmp.maxlen)
        model_config = {'latent_dim': args_tmp.latent_dim,
                        'out_layers': args_tmp.out_layers,
                        'n_users': args_tmp.n_user,
                        'dropout': args_tmp.dropout,
                        'n_items': args_tmp.n_item,
                        'args': args_sas,
                        'problem': args_tmp.problem}
        return model_config
    elif args_tmp.model in ["MF_rate", "MF_rate_t"]:
        model_config = {'latent_dim': args_tmp.latent_dim,  # rating prediction parameters (the same for all methods)
                        'out_layers': args_tmp.out_layers,
                        'dropout': args_tmp.dropout,  # number of users and items
                        'n_users': args_tmp.n_user,
                        'n_items': args_tmp.n_item,
                        'device': args_tmp.device,
                        'problem': args_tmp.problem}
        return model_config
    elif "GNN_rate" in args_tmp.model:
        args_target = Empty_class()
        setattr(args_target, "device", args_tmp.device)
        setattr(args_target, "dropout_rate", args_tmp.dropout)
        setattr(args_target, "num_blocks", 2)
        model_config = {'latent_dim': args_tmp.latent_dim,
                        "n_heads": 2,
                        'maxlen': args_tmp.maxlen,
                        'num_ngb': args_tmp.maxlen,
                        'out_layers': args_tmp.out_layers,
                        'n_users': args_tmp.n_user,
                        'dropout': args_tmp.dropout,
                        'n_items': args_tmp.n_item,
                        'args': args_target,
                        'problem': args_tmp.problem,
                        'n_gnn_layer': args_tmp.n_layers,
                        'node_num': {0: args_tmp.n_user, 1: args_tmp.n_item},
                        "clip": args_tmp.clip,
                        "version": args_tmp.version,
                        "n_channels": args_tmp.n_channels,
                        "ps_alpha": args_tmp.ps_alpha}
        return model_config
    else:
        print("---- haven't finish config this methods. working on it ...")


def get_pad_mask(seq, pad_index, device):
    mask = seq == pad_index
    mask = (1 - mask.to(int)).to(torch.float32)  # 0, 1
    return mask.to(device)


def rating_pred_loss_level(pre_rating, target_rating):
    # get the mean squared error for each level (rating)
    # print(type(pre_rating), type(target_rating))
    # print(pre_rating[0:10])
    # print(target_rating[0:10])
    result = {}
    count = {}
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
    return mse_result, mae_result, (macro_mse, macro_mae), (weighted_mse, weighted_mae)


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_loader(args_t, stage='train', ps=None):
    data_path = os.path.join(args_t.data_path, f'amazon-{args_t.dataset}')
    filename = os.path.join(data_path, "user_seq2.csv")
    file_pop = os.path.join(data_path, "node_pop.pickle")
    rate_prob_file = os.path.join(data_path, "rate_dict.pickle")
    num_node = {"user": args_t.n_user, "item": args_t.n_item}
    data_ = get_data_hin_sim(filename, num_node)
    adj, data_train, data_valid, data_test = data_.get_data()
    if stage == "train":
        ngb_sampler = NeighborFinder_HIN(adj_list=adj['train'],
                                         pop_file=file_pop,
                                         uniform=True,
                                         offset=args_t.offset,
                                         num_ngb=args_t.maxlen)
        result_loader = DHIN_loader(input_data=data_train,
                                    ngh_finder=ngb_sampler,
                                    num_layer=args_t.n_layers,
                                    pop_file=file_pop,
                                    offset=args_t.offset,
                                    n_channels=args_t.n_channels,
                                    rate_prob=rate_prob_file)
    else:
        # graph sampler (full)
        ngb_sampler = NeighborFinder_HIN(adj_list=adj['full'],
                                         pop_file=file_pop,
                                         uniform=True,
                                         offset=args_t.offset,
                                         num_ngb=args_t.maxlen)
        if stage == "valid":
            result_loader = DHIN_loader(input_data=data_valid,
                                        ngh_finder=ngb_sampler,
                                        num_layer=args_t.n_layers,
                                        pop_file=file_pop,
                                        offset=args_t.offset,
                                        n_channels=args_t.n_channels,
                                        rate_prob=rate_prob_file)
        else:
            result_loader = DHIN_loader(input_data=data_test,
                                        ngh_finder=ngb_sampler,
                                        num_layer=args_t.n_layers,
                                        pop_file=file_pop,
                                        offset=args_t.offset,
                                        n_channels=args_t.n_channels,
                                        rate_prob=rate_prob_file)
    if ps is not None:
        result_loader.set_ps(ps)
    if stage == 'train':
        return DataLoader(result_loader, batch_size=args_t.bs, num_workers=6, pin_memory=True, shuffle=True, drop_last=False)
    else:
        return DataLoader(result_loader, batch_size=args_t.bs, num_workers=6, pin_memory=True, shuffle=False, drop_last=False)
