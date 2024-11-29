from torch.utils.data import DataLoader
import torch.nn as nn
import logging
import torch
import shutil
import os
import time

from data_loader import *
from argparser import arg_parser_single
import all_models
import utils
from train_pro import train_round, test_model, train_ps


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
    filename = os.path.join(data_path, "user_seq.csv")
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
    if stage == 'train':
        return DataLoader(result_loader, batch_size=args_t.bs, num_workers=6, pin_memory=True, shuffle=True)
    else:
        return DataLoader(result_loader, batch_size=args_t.bs, num_workers=6, pin_memory=True, shuffle=False)


if __name__ == "__main__":
    # arguments set up
    args = arg_parser_single()
    # setup seeds for reproducibility
    utils.set_seed(args.n_run + 2022)
    # Output setup
    out_main = f"{args.result_path}/{args.dataset}/{args.method}_{args.prefix}"
    if args.version == 3:  # focus on different clip rate
        out_main += f"_clip{args.clip}"
    elif args.version == 5: # focus on different ps_weight in input seq
        out_main += f"_ps{args.ps_alpha}"
    elif args.version == 4:  # both clip and ps input
        out_main += f"_clip{args.clip}_ps{args.ps_alpha}"
    else:
        pass
    out_main += f"_{args.n_run}"
    out_path = f"{out_main}/checkpoint/model.pt"
    if args.train == "train" and os.path.isdir(out_main):
        shutil.rmtree(out_main)
    if not os.path.isdir(out_main):
        os.makedirs(out_main, exist_ok=True)
        os.makedirs(f"{out_main}/checkpoint", exist_ok=True)
    # logger_file = f"{args.result_path}/{args.dataset}/{args.method}_{args.prefix}_{args.n_run}"
    logger = get_logger(out_main, args.train)  # "train", "continue", "test
    # Model setup
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config = utils.mode_configuration(args)
    model = get_model(args.model)(model_config)
    if args.version in [3, 4, 5]:  # v3 needs PS on loss (based on V2)
        print("load previous model")
    #     org_path = out_path.replace("V3", "V2")
    #     print(org_path)
    #     model.load_state_dict(torch.load(org_path, map_location=args.device))
    # if args.version == 4 or args.version == 5:  # v4 needs PS on input sequence (based on V3)
        org_path = out_path.split("V")[0] + "V2" + "_" + str(args.n_run)
        org_path += "/checkpoint/model.pt"
        pretrained_dict = torch.load(org_path, map_location=args.device)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

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
            logger.info("Continue training ... ")
            model.load_state_dict(torch.load(out_path, map_location=args.device))
        if torch.cuda.device_count() > 1:
            logger.info("Train on multiple GPUs ... ")
            model = nn.DataParallel(model)
        # early stopping
        early_stop = utils.EarlyStopping(patience=args.patient, verbose=True, path=out_path)
        # pre-train the rating prediction model

        if args.version < 3:  # V1 or V2
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
            logger.info(f"Train version V{args.version} of model")
            train_round(args, model, early_stop, optimizer, logger, stage="pre1")
            test_model(model, out_path, args, out_main, logger, stage=str(args.version))
        else:   # TODO: V3 and V4 to be implemented and tested
            # pre-train the PS model
            logger.info(f"Train version V{args.version} of model")
            if args.version == 3:  # V3 needs to train the PS model
                # PS-loss is used;
                logging.info("Pre-train the PS model")
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
                train_ps(args, model, early_stop, optimizer, logger)
                # This test is to verify that the training of PS module does not change the pre-trained base model
                test_model(model, out_path, args, out_main, logger, stage="ps_pre")
                # Tune with PS loss
                # new early stop and optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr / 10, weight_decay=args.wd)
                early_stop = utils.EarlyStopping(patience=args.patient, verbose=True, path=out_path)
                train_round(args, model, early_stop, optimizer, logger, stage="ps_l")
                test_model(model, out_path, args, out_main, logger, stage=str(args.version))
            elif args.version == 4:   # load pre-trained base and PS models for the final version
                # Final version;
                # Fix the ps_dict in the tune stage
                logging.info("Pre-train the PS model")
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
                model.set_version(3)  # avoid ps in GNN aggregation step
                ps_dict = train_ps(args, model, early_stop, optimizer, logger, get_ps=True)
                # This test is to verify that the training of PS module does not change the pre-trained base model
                test_model(model, out_path, args, out_main, logger, stage="ps_pre")

                early_stop = utils.EarlyStopping(patience=args.patient, verbose=True, path=out_path)
                logger.info("Fine tune with ps input and ps loss")
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr / 10, weight_decay=args.wd)
                ps_input = train_round(args, model, early_stop, optimizer, logger, stage="ps_l",
                                       is_ps_in=True, ps_input=ps_dict)
                test_model(model, out_path, args, out_main, logger, stage=str(args.version), ps=ps_input)
                # remove model to save space; comment out if needed
                os.remove(out_path)
            elif args.version == 5:  # only apply ps score to input sequence not the loss function.
                #
                logging.info("Pre-train the PS model")
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
                model.set_version(3)  # avoid ps in GNN aggregation step
                ps_dict = train_ps(args, model, early_stop, optimizer, logger, get_ps=True)
                # This test is to verify that the training of PS module does not change the pre-trained base model
                test_model(model, out_path, args, out_main, logger, stage="ps_pre")
                logger.info("Fine tune with ps input and ps loss")
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr / 10, weight_decay=args.wd)
                early_stop = utils.EarlyStopping(patience=args.patient, verbose=True, path=out_path)
                # use norm mse loss: pre1
                ps_input = train_round(args, model, early_stop, optimizer, logger, stage="pre1",
                                       is_ps_in=True, ps_input=ps_dict)
                test_model(model, out_path, args, out_main, logger, stage=f"{args.version}", ps=ps_input)
                # remove model to save space; comment out if needed
                os.remove(out_path)
            elif args.version == 10:   # naive PS score
                logger.info("Naive PS module")
                org_path = out_path.replace("V10", "V2")
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
                model.load_state_dict(torch.load(org_path, map_location=args.device))
                train_round(args, model, early_stop, optimizer, logger, stage="naive")
                test_model(model, out_path, args, out_main, logger, stage=str(args.version))
            else:
                print("Wrong model version!")
                sys.exit()
    else:  # just run test
        test_model(model, out_path, args, out_main, logger, stage="N")


# TODO:
#  1. in the training of the classification probability, can further add negative sample to have 6 classes.
#  2. (optional) do not use the PS weighted aggregation for the feature for prediction the classification probability.

