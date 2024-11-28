from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import logging
import random
import torch
import shutil
import sys
import os
import time

from data_loader import *
from argparser import arg_parser_single
import all_models
import utils


def get_model(model_name):
    # model_str = f"Baselines.{model_name}"
    method = getattr(all_models, model_name)
    return method


def get_logger(path_dir, mode="train"):
    logging.basicConfig(level=logging.INFO)
    logger_t = logging.getLogger()
    logger_t.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'{path_dir}/{mode}_info_{time.time()}.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger_t.addHandler(fh)
    logger_t.addHandler(ch)
    logger_t.info(args)
    return logger_t


def show_parameters(model_tmp, logger_t):
    logger_t.info("======Model parameters")
    for para_name, params in model_tmp.named_parameters():
        logger_t.info(f"{para_name}, {params.requires_grad}, {params.data.size()}, {params.dtype}")


def get_loader(args_t, stage='train'):
    data_path = os.path.join(args_t.data_path, f'amazon-{args_t.dataset}')
    loader_func = None
    if os.path.isdir(data_path):
        if args_t.method == 'seq':
            # loader_func = seq_loader
            result_loader = seq_loader(os.path.join(data_path, "user_seq2.csv"),
                                       num_val=args_t.num_val,
                                       max_len=args_t.maxlen,
                                       mode=stage)
        elif args_t.method == 'trad':
            # loader_func = interaction_loader
            result_loader = interaction_loader(os.path.join(data_path, "user_seq2.csv"),
                                               num_val=args_t.num_val,
                                               mode=stage)
        else:
            result_loader = None
    else:
        print(f"{args_t.dataset} not found")
        sys.exit()
    if stage == 'train':
        return DataLoader(result_loader, batch_size=args.bs, num_workers=6, pin_memory=True, shuffle=True)
    else:
        return DataLoader(result_loader, batch_size=args.bs, num_workers=6, pin_memory=True, shuffle=False)


if __name__ == "__main__":
    # arguments set up
    args = arg_parser_single()
    # setup seeds for reproducibility
    utils.set_seed(args.n_run + 2022)
    # Output setup
    out_main = f"{args.result_path}/{args.dataset}/{args.method}_{args.prefix}_{args.n_run}"
    out_path = f"{out_main}/checkpoint/model.pt"
    if args.train == "train" and os.path.isdir(out_main):
        shutil.rmtree(out_main)
    if not os.path.isdir(out_main):
        os.mkdir(out_main)
        os.mkdir(f"{out_main}/checkpoint")
    logger_file = f"{args.result_path}/{args.dataset}/{args.method}_{args.prefix}_{args.n_run}"
    logger = get_logger(logger_file, args.train)  # "train", "continue", "test
    # Model setup
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config = utils.mode_configuration(args)
    model = get_model(args.model)(model_config)
    model.to(torch.double)
    model = model.to(args.device)
    show_parameters(model, logger)
    # Training steps
    if args.train != 'test':
        # get training data
        train_loader = get_loader(args, stage='train')
        valid_loader = get_loader(args, stage='valid')
        # continue training ?
        if args.train == 'continue':
            # load model checkpoint from disk
            model.load_state_dict(torch.load(out_path, map_location=args.device))
            # model = model.to(args.device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        # early stopping
        early_stop = utils.EarlyStopping(patience=args.patient, verbose=True, path=out_path)
        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        # results
        train_loss = []
        valid_loss = []
        # training steps
        if "MF_rate_t" in args.prefix:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

        for epoch in range(1, args.epochs + 1):
            model = model.train()
            # train epoch
            train_loss_epoch = []
            pbar = tqdm(enumerate(train_loader), disable=args.bar)
            pbar.set_description(f"Training Epoch {epoch}")
            for ind, data_batch in pbar:
                # optimizer.zero_grad()
                for param in model.parameters():
                    param.grad = None
                if args.method == "pro":
                    interaction, ngb_pos, ngb_neg = data_batch[0], data_batch[1], data_batch[2]
                    # to device
                    for l_key in ngb_pos:  # ngb_0 ngb_1 (k-hop ngb)
                        for e_key in ngb_pos[l_key]:  # pos or neg
                            for i in range(len(ngb_pos[l_key][e_key])):  # 6 info of neighbor nodes
                                ngb_pos[l_key][e_key][i] = ngb_pos[l_key][e_key][i].to(args.device)
                                ngb_neg[l_key][e_key][i] = ngb_neg[l_key][e_key][i].to(args.device)
                    for key in interaction:
                        interaction[key] = interaction[key].to(args.device)
                    loss, _, ps_loss, _ = model(interaction, ngb_pos, ngb_neg)
                    final_loss = loss + ps_loss
                    pbar.set_postfix(mse_loss=loss.item(), ps_loss=ps_loss.item())
                else:
                    for key in data_batch:
                        data_batch[key] = data_batch[key].to(args.device)
                    if args.method == "seq":  # seq input
                        if "no_rate" in args.prefix:
                            loss, reg_loss, _ = model(data_batch['user'], data_batch['item'],
                                                      data_batch['rating'], data_batch['user_id'],
                                                      his_rate=None)
                        else:
                            loss, reg_loss, _ = model(data_batch['user'], data_batch['item'],
                                                      data_batch['rating'], data_batch['user_id'],
                                                      his_rate=data_batch['input_rate'])
                        mask = utils.get_pad_mask(data_batch['user'], 0, args.device)
                        loss = torch.sum(torch.mul(loss, mask)) / torch.sum(mask)
                    elif args.method == "trad":  # interaction input
                        loss, reg_loss, _ = model(data_batch['user'], data_batch['item'], data_batch['rating'])
                    else:
                        print("wrong method, haven't implemented yet")
                        sys.exit()
                    final_loss = loss + reg_loss * args.reg_alpha  # TODO weight_decay and reg_alpha ?
                    pbar.set_postfix(loss=loss.item(), reg_loss=reg_loss.item())
                # final_loss = loss
                final_loss.backward()
                optimizer.step()
                train_loss_epoch.append(loss.item())
            if "MF_rate_t" in args.prefix:
                scheduler.step()

            train_loss.append(np.mean(train_loss_epoch))
            # validation epoch
            val_loss = utils.eval_one_epoch(model, valid_loader, args.device, show_bar=True,
                                            test=False, num_val=args.num_val, method=args.method)
            valid_loss.append(val_loss)
            # log info
            cur_lr = utils.get_lr(optimizer)
            logger.info(f"Epoch {epoch}: train mse: {train_loss[-1]}, validation mse: {val_loss}, lr: {cur_lr}")
            # early stopping
            early_stop(val_loss, model, logger)
            if early_stop.early_stop:
                logger.info(f'Early stopping at {epoch} epoch...')
                break
    # load the best model
    model.load_state_dict(torch.load(out_path, map_location=args.device))
    model = model.to(args.device)
    # test data
    test_loader = get_loader(args, stage='test')
    # run test
    test_file = f"{out_main}/test_org_{time.time()}.csv"
    macro_res, weighted_res, mse_res, mae_res = utils.eval_one_epoch(model, test_loader, args.device,
                                                                     show_bar=True, test=True, method=args.method,
                                                                     out_file=test_file, num_val=args.num_val)
    logger.info(f"test macro mse: {macro_res[0]}, macro mae: {macro_res[1]}")
    logger.info(f"test weighted mse: {weighted_res[0]}, weighted mae: {weighted_res[1]}")

    with open(test_file.replace(".csv", ".txt"), 'w') as fout:
        fout.write(f"test macro mse and mae: {macro_res[0]}\t{macro_res[1]}\n")
        fout.write(f"test weighted mse and mae : {weighted_res[0]}\t{weighted_res[1]}\n")
        fout.write(f"test org mse: {mse_res}\n")
        fout.write(f"test org mae: {mae_res}\n")

