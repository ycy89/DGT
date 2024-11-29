import pandas as pd
import ast
import numpy as np
import torch.utils.data as Data
from scipy.sparse import coo_matrix
import torch

class SASTestDataset(Data.Dataset):
    def __init__(self, table, rate_table, item_vocab_size, maxlen):
        super().__init__()
        self.table = table
        self.rate_table = rate_table
        self.users = list(table.keys())
        self.users.sort()
        self.item_vocab_size = item_vocab_size
        self.maxlen = maxlen

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        seq = self.item_vocab_size * torch.ones([self.maxlen], dtype=torch.long)
        size = min(len(self.table[self.users[idx]]) - 1, self.maxlen)
        seq[:size] = torch.tensor(self.table[self.users[idx]][-size - 1:-1])
        rate = torch.tensor(self.rate_table[self.users[idx]])
        return torch.tensor(self.users[idx]), seq, torch.tensor(self.table[self.users[idx]][-1]), rate

class SASDataset(Data.Dataset):
    def __init__(self, table, rate_table, item_vocab_size, maxlen):
        super().__init__()
        self.table = table
        self.rate_table = rate_table
        self.user_vocab_size = len(table)
        self.users = list(table.keys())
        self.users.sort()
        self.item_vocab_size = item_vocab_size
        self.maxlen = maxlen
        self.trainset = set()

    def __len__(self):
        return self.user_vocab_size

    def __getitem__(self, idx):
        seq = self.item_vocab_size * torch.ones([self.maxlen], dtype=torch.long)
        pos = self.item_vocab_size * torch.ones([self.maxlen], dtype=torch.long)
        rate = self.item_vocab_size * torch.ones([self.maxlen], dtype=torch.float)

        # ts = set(self.table[self.users[idx]])
        size = min(len(self.table[self.users[idx]]) - 1, self.maxlen)

        seq[:size] = torch.tensor(self.table[self.users[idx]][-size - 1:-1])
        pos[:size] = torch.tensor(self.table[self.users[idx]][-size:])
        rate[:size] = torch.tensor(self.rate_table[self.users[idx]][-size:])
        # sec = (0, self.item_vocab_size)
        # sample = np.random.randint(sec[0], sec[1])
        # while sample in ts or (sample not in self.trainset):
        #     sample = np.random.randint(sec[0], sec[1])
        # neg = torch.tensor(sample)
        return torch.tensor(self.users[idx]), seq, pos, rate

def get_data(args):
    file = 'dataset/amazon_' + args.dataset + '/user_seq.csv'
    df = pd.read_csv(file)
    train_users, train_items, train_ratings, train_times = [], [], [], []
    valid_users, valid_items, valid_ratings, valid_times = [], [], [], []
    test_users, test_items, test_ratings, test_times = [], [], [], []
    train_item_table, valid_item_table, test_item_table = {}, {}, {}
    train_rate_table, valid_rate_table, test_rate_table = {}, {}, {}

    for i in range(len(df)):
        user = df['reviewerID'][i] - 1
        item_seq = ast.literal_eval(df['asin'][i])
        item_seq = [item - 2 for item in item_seq]
        rate_seq = ast.literal_eval(df['overall'][i])
        rate_seq = [rate / 5.0 for rate in rate_seq]

        train_users.extend([user for _ in range(len(item_seq) - 2)])
        valid_users.append(user)
        test_users.append(user)

        train_items.extend(item_seq[:-2])
        valid_items.append(item_seq[-2])
        test_items.append(item_seq[-1])

        train_item_table[user] = item_seq[:-2]
        valid_item_table[user] = item_seq[:-1]
        test_item_table[user] = item_seq

        train_rate_table[user] = rate_seq[:-2]
        valid_rate_table[user] = rate_seq[-2]
        test_rate_table[user] = rate_seq[-1]


    uvs = max(train_users + valid_users + test_users) + 1
    ivs = max(train_items + valid_items + test_items) + 1

    _row = np.array(train_users)
    _col = np.array(train_items)
    _data = np.array([1] * len(train_users))

    train_ui_graph = coo_matrix((_data, (_row, _col)), shape=(uvs, ivs), dtype=np.float32)

    trainloader = Data.DataLoader(SASDataset(train_item_table, train_rate_table, ivs, args.maxlen),
                                  batch_size=args.train_batch_size,
                                  shuffle=True, num_workers=2)

    testloader = Data.DataLoader(SASTestDataset(test_item_table, test_rate_table, ivs, args.maxlen),
                                 batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader, uvs, ivs, train_ui_graph

