import argparse
import os

import numpy as np


def arg_parser_single():
    parser = argparse.ArgumentParser('train entrance')
    # ===== Data related parameters =================
    parser.add_argument('--main_path', type=str, default='./')
    # parser.add_argument('--result_path', type=str, default='/ceph/11329/chain/DHIN_Debias/result')
    parser.add_argument('--dataset', type=str, default='beauty', help='movie, cd, book')
    parser.add_argument('--n_run', type=int, default=3, help='repeat times of the same setting')
    parser.add_argument('--method', type=str, default='seq', help='model types: seq, trad, graph, debias')
    parser.add_argument('--model', type=str, default='MF_rate',
                        help='model name, must be predefined. all models are defined in Baselines')
    parser.add_argument('--problem', type=str, default="regression",
                        help='regression problem or classification problem')

    # ===== Training parameters =================
    parser.add_argument('--bs', type=int, default=512, help='batch_size')
    parser.add_argument('--train', type=str, default='train', help='train, continue, test')
    parser.add_argument('--prefix', type=str, default='v1', help='model prefix')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--reg_alpha', type=float, default=0.0005, help='reg loss for involved embeds')
    parser.add_argument('--wd', type=float, default=0, help='weight decay fir optimizer')
    parser.add_argument('--early_stop', type=int, default=5, help='early stop (default: 0 ->do not early stop)')
    parser.add_argument('--gpu', type=int, default=1, help='number of GPUs (default: 1)')
    # ===== Model parameters =====
    parser.add_argument('--latent_dim', type=int, default=64, help='size of embedding')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout rate for prediction module(default: 0.1)')
    parser.add_argument('--num_val', type=int, default=1,
                        help='the number of valid and testing samples (default: 1)')
    # for seq model
    parser.add_argument('--maxlen', type=int, default=50,
                        help='maximum seq length (default: 50)')
    parser.add_argument('--clip', type=float, default=0.05)
    parser.add_argument('--n_channels', type=int, default=2)
    parser.add_argument('--n_layers', type=int, default=1,
                        help='number of GNN layers (default: 1)')
    parser.add_argument('--version', type=int, default=1,
                        help='model version for proposed ablation study (default: 1)')

    # ===== Evaluation parameters =====
    parser.add_argument('--patient', type=int, default=5,
                        help='early stop patient')
    parser.add_argument('--bar', type=str, default="False",
                        help='show bar plot of tqdm')
    parser.add_argument('--ps_alpha', type=float, default=0.2,
                        help='ps alpha')
    # parser.add_argument('--device', type=str, default='cuda:0')
    n_users = {"cd": 26193 + 1, "movie": 40593 + 1, "book": 218106 + 1,
               "sport": 93233 + 1, "beauty": 5925 + 1, "music": 7471 + 1}
    n_items = {"cd": 63592 + 1, "movie": 49848 + 1, "book": 365641 + 1,
               "sport": 195398 + 1, "beauty": 5398 + 1, "music": 32905 + 1}
    args = parser.parse_args()
    args.n_user = n_users[args.dataset]
    args.n_item = n_items[args.dataset]
    args.bar = not eval(args.bar)
    args.out_layers = np.array([2 * args.latent_dim, 64, 32])
    # args.out_layers = np.array([2, 4, 2, 1]) * args.latent_dim
    args.out_layers = list(args.out_layers)
    if args.model == 'GNN_rate2' or args.model == 'GNN_rate3':
        args.out_layers[0] = args.out_layers[0] // 2
    args.reg_alpha /= (args.latent_dim // 16)
    MAIN_P = args.main_path
    args.data_path = os.path.join(MAIN_P, "datasets")
    args.result_path = os.path.join(MAIN_P, "result")
    # my proposed
    args.offset = [0, args.n_user]
    return args
