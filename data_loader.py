import pickle
import sys

import torch.utils.data as data
import numpy as np
import pandas as pd


def seq_padding(item_seq, rate_seq, length_enc):
    item_seq = list(item_seq)
    rate_seq = list(rate_seq)
    if len(item_seq) >= length_enc:
        enc_in = item_seq[-length_enc:]
        rate_seq = rate_seq[-length_enc:]
    else:
        enc_in = [0] * (length_enc - len(item_seq)) + item_seq
        rate_seq = [0] * (length_enc - len(rate_seq)) + rate_seq

    input_seq = [0] + enc_in[0:-1]
    history_rate = [0] + rate_seq[0:-1]

    output_seq = [0] + enc_in[1:]
    target_rate = [0] + rate_seq[1:]
    return np.array(input_seq, dtype=int), np.array(history_rate, dtype=float), \
        np.array(output_seq, dtype=int), np.array(target_rate, dtype=float)


class seq_loader(data.Dataset):
    """
    Data loader for none-sequential models
    """
    def __init__(self, filename, max_len=30, num_val=1, mode="train"):
        # results data
        self.user_data = []  # index as user id, each element is a list of lists (item, time, rating)
        self.max_len = max_len
        self.mode = mode
        self.num_val_test = num_val
        # load data
        df = pd.read_csv(filename)
        # uid, item_seq_train, time_seq_train, rate_train; item_seq_valid, time_seq_valid, rate_valid;
        # item_seq_test, time_seq_test, rate_test
        for index, row in df.iterrows():  # TODO: expand sequence get short sequence as training sample?
            # self.user_data.append([eval(row['asin']), eval(row['overall']), int(row['reviewerID'])])
            user_id = int(row['reviewerID'])
            item_l = eval(row['asin'])
            rate_l = eval(row['overall'])
            if mode == "train":
                start_idx = 0
                item_l = item_l[: -self.num_val_test * 2]
                rate_l = rate_l[: -self.num_val_test * 2]
                while True:
                    if start_idx + max_len <= len(item_l):
                        self.user_data.append([item_l[start_idx: start_idx + max_len],
                                               rate_l[start_idx: start_idx + max_len],
                                               user_id])
                        start_idx += 1
                    else:
                        self.user_data.append([item_l, rate_l, user_id])
                        break
            else:  # validation and test
                self.user_data.append([item_l, rate_l, user_id])

    def __getitem__(self, index):
        item_seq, rate_seq = self.user_data[index][0], self.user_data[index][1]
        user_id = self.user_data[index][2]
        # rating normalization
        rate_seq = [val / 5.0 for val in rate_seq]

        # padding
        # item_seq = seq_padding(item_seq, rate_seq, self.max_len)
        # if self.mode == "train":
        #     pass
        #     # item_seq = item_seq[:-self.num_val_test * 2]
        #     # rate_seq = rate_seq[:-self.num_val_test * 2]
        #     # time_seq = time_seq[:-self.num_val_test * 2]
        # elif
        if self.mode == "valid":
            item_seq = item_seq[:-self.num_val_test]  # remove the last one (which is the test item)
            rate_seq = rate_seq[:-self.num_val_test]
        else:  # test -> get the full sequence; train ->already cut in the data loading process
            pass
        # sequence input formulation and sequence padding
        input_seq, input_rate, out_seq, out_rate = seq_padding(item_seq, rate_seq, self.max_len)
        # TODO ignores input_rate for the original sequence rec models. (They did not consider this)
        return {"user": input_seq, "item": out_seq, "rating": out_rate,
                "user_id": user_id, "input_rate": input_rate}

    def __len__(self):
        return len(self.user_data)


class interaction_loader(data.Dataset):
    """inputs are user behavior sequence file"""
    def __init__(self, filename, num_val=1, mode='train'):
        # results data
        self.user_data = []  # (uid, item, time, rate)
        self.mode = mode
        self.num_val_test = num_val
        # load data
        df = pd.read_csv(filename)
        for index, row in df.iterrows():
            uid = int(row['reviewerID'])
            item_seq = eval(row['asin'])
            time_seq = eval(row['unixReviewTime'])
            rating_seq = eval(row['overall'])
            num = len(item_seq)
            if self.mode == "train":
                start, end = 0,  num - self.num_val_test * 2
            elif self.mode == "valid":
                start, end = num - self.num_val_test * 2, num - self.num_val_test
            else:
                start, end = num - self.num_val_test, num

            for i in range(start, end):
                self.user_data.append([uid, item_seq[i], time_seq[i], rating_seq[i]])

    def __getitem__(self, index):
        uid, item, time, rate = self.user_data[index]
        # rating normalization
        rate = rate / 5.0
        return {"user": uid, "item": item, "time": time, "rating": rate}

    def __len__(self):
        return len(self.user_data)


# My temporal GNN data split and loader
class get_data_hin_sim:   # simplified version of data get_data_hin from the DHIN project
    def __init__(self, filename,
                 num_node=None,
                 num_eval=1):
        """
        @param filename:  csv file path
        @param num_node: {"user": 1, "item":2}
        """
        # decide headers
        self.num_nodes = num_node
        self.filename = filename
        self.num_eval = num_eval
        self.headers = ['reviewerID', 'asin', 'overall', 'unixReviewTime']

    def get_data(self):
        """
        @return:
            head-edge ['author_pid', 'pid_venue', 'pid_ref']
            src: head1 -> name1 -> code1
            tgt: head2 -> name2 -> code2
            adj['full'][code1][code2] = [[], []]  -> list len = number of code1 set; value -> code2 value
            adj['train'][code1][code2]
        """
        adj = {'train': {}, 'full': {}}
        # init adj
        user_num = self.num_nodes['user']
        item_num = self.num_nodes['item']
        # here 0-> user; 1->item
        adj['train'][0] = {1: [[] for _ in range(user_num + 1)]}   # every user has a list of item history
        adj['train'][1] = {0: [[] for _ in range(item_num + 1)]}   # every item has a list of user history
        # undirected
        adj['full'][0] = {1: [[] for _ in range(user_num + 1)]}
        adj['full'][1] = {0: [[] for _ in range(item_num + 1)]}

        # get org data
        df = pd.read_csv(self.filename)
        data_tmp = {}
        for head in self.headers:
            data_tmp[head] = []
            column_data = eval(f"df.{head}.to_numpy()")   # list of list
            if head != "reviewerID":  # list data
                for val in column_data:
                    data_tmp[head].append(eval(val))
                data_tmp[head] = np.array(data_tmp[head], dtype=object)
            else:
                data_tmp[head] = column_data  # scalar data (user ID)

        # data split
        # get train, valid and test data
        data_train = {}
        data_valid = {}
        data_test = {}
        for key in data_tmp:
            if key == "reviewerID":
                data_train[key] = data_tmp[key]
                data_valid[key] = data_tmp[key]
                data_test[key] = data_tmp[key]
            else:
                data_train[key] = []
                data_valid[key] = []
                data_test[key] = []
                for i in range(len(data_tmp[key])):
                    data_train[key].append(data_tmp[key][i][:-self.num_eval * 2])
                    data_valid[key].append(data_tmp[key][i][-self.num_eval * 2: -self.num_eval])
                    data_test[key].append(data_tmp[key][i][-self.num_eval:])

        # get train adj data (for neighbor sampling)
        src_t = 0
        tgt_t = 1
        for tmp_id in range(len(data_train['overall'])):  # for each paper/movie
            # user seq
            src_node = data_train['reviewerID'][tmp_id]  # user ID
            tgt_nodes = data_train['asin'][tmp_id]       # list of items
            time_tmp = data_train['unixReviewTime'][tmp_id]     # list of interaction time
            rating_tmp = data_train['overall'][tmp_id]          # list of rating

            for ind, tgt_node in enumerate(tgt_nodes):
                adj['train'][src_t][tgt_t][src_node].append((tgt_node, time_tmp[ind], rating_tmp[ind]))
                # undirected graph: name -> movie
                adj['train'][tgt_t][src_t][tgt_node].append((src_node, time_tmp[ind], rating_tmp[ind]))

        # get full graph data in all time
        for tmp_id in range(len(data_tmp['overall'])):
            # user seq
            src_node = data_tmp['reviewerID'][tmp_id]  # user ID
            tgt_nodes = data_tmp['asin'][tmp_id]  # list of items
            time_tmp = data_tmp['unixReviewTime'][tmp_id]  # list of interaction time
            rating_tmp = data_tmp['overall'][tmp_id]  # list of rating

            for ind, tgt_node in enumerate(tgt_nodes):
                adj['full'][src_t][tgt_t][src_node].append((tgt_node, time_tmp[ind], rating_tmp[ind]))
                # undirected graph: name -> movie
                adj['full'][tgt_t][src_t][tgt_node].append((src_node, time_tmp[ind], rating_tmp[ind]))
        # TODO: sort the tgt_t to src_t according to time.
        return adj, data_train, data_valid, data_test


# Neighbor samplers
def binary_search(lis, num):
    if lis[-1] < num:     # all values in the list is smaller than num
        return len(lis) - 1, False
    if lis[0] > num:
        return -1, False  # all values in the list is bigger than num
    # search
    left = 0
    right = len(lis) - 1
    while left <= right:
        mid = (left + right) // 2
        if num < lis[mid]:
            right = mid - 1
        elif num > lis[mid]:
            left = mid + 1
        else:  # found
            return mid, True
    if lis[right] < num:
        return right, False
    if lis[left] < num:
        return left, False


class NeighborFinder_HIN:
    def __init__(self,
                 adj_list,
                 pop_file=None,
                 uniform=False,
                 offset=None,
                 num_ngb=50):
        """
        Params
        ------
        offset : [int]: [0, num_user]
        """
        self.num_ngb = num_ngb   # number of neighbors to be sampled
        self.num_n_t = 2         # user and item
        self.uniform = uniform   # sample uniformly?
        self.adj_list = None
        self.update_adj(adj_list)

        # read out the dynamic node popularity(degree)-> for node propensity score
        self.node_pop_dict = pickle.load(open(pop_file, 'rb'))
        # self.node_pop_dict = json.load(open(pop_file, 'r'))  # {"user": [{"time":degree}], "item": [{"time":degree}]}
        self.node_id_offset = offset
        self.ps = None

    def update_pop(self, pop_file):
        self.node_pop_dict = pickle.load(open(pop_file, 'rb'))

    def set_ps(self, ps):
        self.ps = ps
        self.adj_list[0][1]['ps'] = []   # user to item
        self.adj_list[1][0]['ps'] = []
        # user -> item
        for i in range(len(self.adj_list[0][1]['nid'])):
            u_id = i
            if len(self.adj_list[0][1]['nid'][u_id]) > 0:
                ps_list = []
                for j in range(len(self.adj_list[0][1]['nid'][u_id])):
                    i_id = self.adj_list[0][1]['nid'][u_id][j]
                    t_id = self.adj_list[0][1]['ts'][u_id][j]
                    key = str(u_id) + '_' + str(i_id) + '_' + str(t_id)
                    if key not in ps:  # only test item, but does not affect the input seq
                        ps_list.append(1.0)
                    else:
                        ps_list.append(ps[key])
                self.adj_list[0][1]['ps'].append(ps_list)
            else:
                self.adj_list[0][1]['ps'].append([])
        # item -> user
        for i in range(len(self.adj_list[1][0]['nid'])):
            i_id = i
            if len(self.adj_list[1][0]['nid'][i_id]) > 0:
                ps_list = []
                for j in range(len(self.adj_list[1][0]['nid'][i_id])):
                    u_id = self.adj_list[1][0]['nid'][i_id][j]
                    t_id = self.adj_list[1][0]['ts'][i_id][j]
                    key = str(u_id) + '_' + str(i_id) + '_' + str(t_id)
                    if key not in ps:
                        ps_list.append(1.0)
                    else:
                        ps_list.append(ps[key])
                self.adj_list[1][0]['ps'].append(ps_list)
            else:
                self.adj_list[1][0]['ps'].append([])

    def update_adj(self, adj):
        # adj already undirected if required.
        self.adj_list = {}
        for src_t in adj:
            self.adj_list[src_t] = {}
            for tgt_t in adj[src_t]:
                self.adj_list[src_t][tgt_t] = {'nid': [], 'ts': [], 'rating': []}
                for neighbor_arr in adj[src_t][tgt_t]:
                    # (src_node, time_tmp[ind], rating_tmp[ind])
                    if len(neighbor_arr) > 0:  # all neighbors
                        # sort according to time, already done.
                        neighbor_arr = sorted(neighbor_arr, key=lambda x: x[1])
                        self.adj_list[src_t][tgt_t]['nid'].append([x[0] for x in neighbor_arr])
                        self.adj_list[src_t][tgt_t]['ts'].append([x[1] for x in neighbor_arr])
                        self.adj_list[src_t][tgt_t]['rating'].append([x[2] for x in neighbor_arr])
                    else:
                        self.adj_list[src_t][tgt_t]['nid'].append([])
                        self.adj_list[src_t][tgt_t]['ts'].append([])
                        self.adj_list[src_t][tgt_t]['rating'].append([])

    def find_before(self, src_node_id, src_node_type, cut_time, pos=None):
        """
        Given a node, and the current time, find all the neighbors of the node before cut_time.
        Params
        ------
            src_node_id:    int, node id. 输入带有offset
            src_node_type:  int, encoded node type
            cut_time:       float, cut time
        Return:
            输出node id 带有offset
        """
        # goal
        idx = []
        ts = []
        rating = []
        n_type = []
        n_pop = []
        ps_result = []
        # remove node id offset
        if src_node_id == 0:
            result = {'nid': [],
                      'ts': [],
                      'rating': [],
                      'n_type': [],
                      'n_pop': [],
                      'ps_score': []}
            return result
        src_n_id_trans = src_node_id - self.node_id_offset[src_node_type]
        # print(src_n_id_trans)
        for tgt_t in self.adj_list[src_node_type]:
            try:
                # all neighbors
                ngb_idx_tmp = self.adj_list[src_node_type][tgt_t]['nid'][src_n_id_trans]
                ngb_ts_tmp = self.adj_list[src_node_type][tgt_t]['ts'][src_n_id_trans]
                ngb_rating_tmp = self.adj_list[src_node_type][tgt_t]['rating'][src_n_id_trans]
                if self.ps is not None:
                    ngb_ps_tmp = self.adj_list[src_node_type][tgt_t]['ps'][src_n_id_trans]
                    # print(len(ngb_ps_tmp), print(len(ngb_ts_tmp)))
                else:
                    ngb_ps_tmp = np.ones(len(ngb_rating_tmp))
                # all pos/neg neighbors
                if pos != -1:
                    if pos:  # True for positive neighbors
                        sel_index = np.array(ngb_rating_tmp) > 3
                    else:    # False for negative neighbors
                        sel_index = np.array(ngb_rating_tmp) <= 3
                else:        # select all neighbors
                    sel_index = np.array(ngb_rating_tmp) > 0

                ngb_idx_tmp = np.array(ngb_idx_tmp)[sel_index]
                ngb_ts_tmp = np.array(ngb_ts_tmp)[sel_index]
                ngb_rating_tmp = np.array(ngb_rating_tmp)[sel_index]
                ngb_ps_tmp = np.array(ngb_ps_tmp)[sel_index]
            except:
                print(src_node_id, src_node_type, cut_time)
                print(type(src_node_id), type(src_node_type), type(cut_time))
                print(len(ngb_ps_tmp), print(len(ngb_ts_tmp)))
                sys.exit()
            # no neighbors for this node
            if len(ngb_idx_tmp) == 0 or len(ngb_ts_tmp) == 0:
                continue
            # binary search for neighbors before given timestamp
            
            # print(ngb_idx_tmp, ngb_ts_tmp, ngb_rating_tmp)
            
            if ngb_ts_tmp[0] > cut_time: 
                ngb_ts_tmp = []
            elif ngb_ts_tmp[-1] < cut_time:
                pass
            else:
                left = 0
                right = len(ngb_idx_tmp) - 1
                while left + 1 < right:
                    mid = (left + right) // 2
                    curr_t = ngb_ts_tmp[mid]
                    if curr_t < cut_time:  # <=
                        left = mid
                    else:
                        right = mid
                if ngb_ts_tmp[right] < cut_time:
                    ngb_idx_tmp = ngb_idx_tmp[:right + 1]
                    ngb_ts_tmp = ngb_ts_tmp[:right + 1]
                    ngb_rating_tmp = ngb_rating_tmp[:right + 1]
                    ngb_ps_tmp = ngb_ps_tmp[:right + 1]
                else:
                    ngb_idx_tmp = ngb_idx_tmp[:right]
                    ngb_ts_tmp = ngb_ts_tmp[:right]
                    ngb_rating_tmp = ngb_rating_tmp[:right]
                    ngb_ps_tmp = ngb_ps_tmp[:right]
            
                # print(left, right)
            
            # no neighbors for this node before time "cut_time"
            if len(ngb_ts_tmp) == 0:
                continue

            # time_current = ngb_ts_tmp[-1]       #
            length = len(ngb_idx_tmp)
            idx.extend([val + self.node_id_offset[tgt_t] for val in ngb_idx_tmp])  # node id offset
            ts.extend(ngb_ts_tmp)
            rating.extend(ngb_rating_tmp)
            ps_result.extend(ngb_ps_tmp)
            n_type.extend([tgt_t] * length)

            # 每个邻接点的node popularity ()
            ngb_pop_tmp = []
            # time unrelated node pop (propensity distribution)
            node_dict = {0: "user", 1: "item"}
            for ind, ngb_id in enumerate(ngb_idx_tmp):
                rate = int(ngb_rating_tmp[ind]) - 1
                ngb_pop_tmp.append(self.node_pop_dict[node_dict[tgt_t]][ngb_id][rate])

            # for ind, ngb_id in enumerate(ngb_idx_tmp):  # time related
            #     time_current = ngb_ts_tmp[ind]
            #     time_list, pop_list = self.node_pop_dict[tgt_t][ngb_id]
            #     index, found = binary_search(time_list, time_current)
            #     if index == -1:
            #         ngb_pop_tmp.append(0)
            #     else:
            #         ngb_pop_tmp.append(sum(pop_list[: index + 1]))  # 0-> index

            n_pop.extend(ngb_pop_tmp)
        result = {'nid': idx,
                  'ts': [int(val) for val in ts],
                  'rating': [int(val) for val in rating],
                  'n_type': n_type,
                  'n_pop': n_pop,
                  'ps_score': ps_result}
        return result

    def get_temporal_neighbor(self,
                              src_type_l=None,
                              src_idx_l=None,
                              cut_time_l=None,
                              num_neighbors=None,
                              pos=None):
        if num_neighbors is None:
            if pos:
                num_neighbors = 30
            else:
                num_neighbors = 50
        if type(src_type_l) == int:
            src_type_l = [src_type_l]
            src_idx_l = [src_idx_l]
            cut_time_l = [cut_time_l]

        assert (len(src_idx_l) == len(cut_time_l) and len(src_type_l) == len(cut_time_l))
        num_src = len(src_idx_l)
        # placeholders, padding with zeros.
        result = {
            "nid": np.zeros((num_src, num_neighbors)).astype(np.int32),
            "ts": np.zeros((num_src, num_neighbors)).astype(np.int32),
            "rating": np.zeros((num_src, num_neighbors)).astype(np.float32),
            'n_type': np.zeros((num_src, num_neighbors)).astype(np.int16),
            "n_pop": np.zeros((num_src, num_neighbors)).astype(np.int16),
            "ps_score": np.ones((num_src, num_neighbors)).astype(np.float32)
        }

        for i, (src_idx, src_type, cut_time) in enumerate(zip(src_idx_l, src_type_l, cut_time_l)):  # para
            # print(src_idx, src_type, cut_time)
            src_ngb_dict = self.find_before(src_idx, src_type, cut_time, pos=pos)
            # TODO: try the following strategies
            #  1. sample most recent
            #  2. time importance sampling
            #  3. sample each type of node separately.  (notebook test_loader.ipynb)
            total_ngb = len(src_ngb_dict['nid'])  # total number of neighbor
            if total_ngb == 0:  # no neighbors
                continue  # next node
            # 1. random sample with replace
            # sampled_idx = np.random.randint(0, total_ngb, num_neighbors)
            # 2. select latest from the neighbor sequence,
            sampled_idx = np.arange(total_ngb)
            while len(sampled_idx) < num_neighbors:        #
                sampled_idx = np.concatenate((sampled_idx, np.arange(total_ngb)), axis=0)
            sampled_idx = sampled_idx[-num_neighbors:]
            for key in result:
                result[key][i, :] = np.array(src_ngb_dict[key])[sampled_idx]
        return result

    def find_k_hop(self, k, src_idx_l, src_type_l, cut_time_l, pos=None):
        """
        Sampling the k-hop sub graph
        pos; True, find the positive neighbors (edge/rating >3)
        pos; True, find the negative neighbors (edge/rating <=3)
        """
        # 1-hop neighbors
        result_first = self.get_temporal_neighbor(
            src_type_l=src_type_l, src_idx_l=src_idx_l,
            cut_time_l=cut_time_l, num_neighbors=self.num_ngb, pos=pos)

        node_records = [result_first['nid']]
        t_records = [result_first['ts']]
        rating_records = [result_first['rating']]
        n_type_records = [result_first['n_type']]
        pop_records = [result_first['n_pop']]
        ps_score = [result_first['ps_score']]

        for _ in range(k - 1):
            ngn_node_est, ngh_t_est, ngb_type_est = node_records[-1], t_records[-1], n_type_records[-1]

            # [N, n_ngb, n_ngb,...] k-1 n_ngb
            orig_shape = ngn_node_est.shape
            ngn_node_est, ngh_t_est, ngb_type_est = ngn_node_est.flatten(), ngh_t_est.flatten(), ngb_type_est.flatten()
            # print(ngn_node_est.shape, ngh_t_est.shape, ngb_type_est.shape)
            result_tmp = self.get_temporal_neighbor(
                src_type_l=ngb_type_est, src_idx_l=ngn_node_est,
                cut_time_l=ngh_t_est, num_neighbors=self.num_ngb, pos=pos)

            node_records.append(result_tmp['nid'].reshape(*orig_shape, self.num_ngb))
            t_records.append(result_tmp['ts'].reshape(*orig_shape, self.num_ngb))
            rating_records.append(result_tmp['rating'].reshape(*orig_shape, self.num_ngb))
            n_type_records.append(result_tmp['n_type'].reshape(*orig_shape, self.num_ngb))
            pop_records.append(result_tmp['n_pop'].reshape(*orig_shape, self.num_ngb))
            ps_score.append(result_tmp['ps_score'].reshape(*orig_shape, self.num_ngb))

        return node_records, t_records, n_type_records, rating_records, pop_records, ps_score


# dataloader
class DHIN_loader(data.Dataset):
    def __init__(self, input_data, ngh_finder, num_layer, pop_file, offset, n_channels,
                 rate_prob=None):
        """
        :param input_data: dict with keys: ['reviewerID', 'asin', 'overall', 'unixReviewTime']
        :param ngh_finder:
        """
        self.user_data = []    # (uid, item, rate, time)
        self.load_user_data(input_data)
        self.ngh_finder = ngh_finder
        self.num_layer = num_layer
        self.node_pop_dict = pickle.load(open(pop_file, 'rb'))
        self.offset = offset
        self.n_channels = n_channels
        self.rate_prob = pickle.load(open(rate_prob, 'rb'))
        self.ps = None

        # scale
        for key in self.rate_prob:
            self.rate_prob[key] = np.power(self.rate_prob[key], 0.5)

        all_rating = sum(list(self.rate_prob.values()))
        for key in self.rate_prob:
            self.rate_prob[key] /= all_rating

        # print("rating probability")
        # print(self.rate_prob)

        # {"user": [{"time":degree}], "item": [{"time":degree}]}

    def set_ps(self, ps):
        self.ps = ps
        self.ngh_finder.set_ps(ps)

    def set_ngh_finder(self, ngh_finder):
        self.ngh_finder = ngh_finder

    def load_user_data(self, input_data):
        for i in range(len(input_data['asin'])):  # for each user
            user_id = input_data['reviewerID'][i]
            item_list = input_data['asin'][i]
            rating_list = input_data['overall'][i]
            time_list = input_data['unixReviewTime'][i]
            for ind, item_id in enumerate(item_list):
                sample = (user_id, item_id, rating_list[ind] / 5.0, time_list[ind])
                self.user_data.append(sample)
        # print(f"number of samples: {len(self.user_data)}")

    def get_node_degree(self, src_type, src_id, rate):
        # time irrelevant node propensity distribution
        rating_degree = self.node_pop_dict[src_type][src_id][rate]
        return rating_degree

    def __getitem__(self, index):
        uid_l, item_l, rate_l, time_l = self.user_data[index]
        user_pop = self.get_node_degree("user", uid_l, int(rate_l * 5) - 1)
        item_pop = self.get_node_degree("item", item_l, int(rate_l * 5) - 1)
        uid_l += self.offset[0]
        item_l += self.offset[1]
        interaction_dict = {"user": uid_l,
                            "user_t": 0,
                            "item": item_l,
                            "item_t": 1,
                            "rate": rate_l, "ts": time_l,
                            'rate_p': self.rate_prob[int(rate_l * 5)],
                            'u_rate_pop': np.power(user_pop, 0.75),
                            'i_rate_pop': np.power(item_pop, 0.75)}
        # rating normalization
        # node_id, ts, nid_type, rating, node_pop
        if self.n_channels == 2:
            ngb_pos = {}
            src_ngb = self.ngh_finder.find_k_hop(self.num_layer, uid_l, 0, time_l, pos=True)   # user pos ngb
            tgt_ngb = self.ngh_finder.find_k_hop(self.num_layer, item_l, 1, time_l, pos=True)  # item pos ngb
            for ind in range(self.num_layer):
                ngb_pos[f'ngb_{ind}'] = {'user': [np.squeeze(src_ngb[idx][ind]) for idx in range(len(src_ngb))],
                                         'item': [np.squeeze(tgt_ngb[idx][ind]) for idx in range(len(tgt_ngb))]}
            ngb_neg = {}
            src_ngb_neg = self.ngh_finder.find_k_hop(self.num_layer, uid_l, 0, time_l, pos=False)  # user neg ngb
            tgt_ngb_neg = self.ngh_finder.find_k_hop(self.num_layer, item_l, 1, time_l, pos=False)  # item neg ngb
            for ind in range(self.num_layer):
                ngb_neg[f'ngb_{ind}'] = {'user': [np.squeeze(src_ngb_neg[idx][ind]) for idx in range(len(src_ngb_neg))],
                                         'item': [np.squeeze(tgt_ngb_neg[idx][ind]) for idx in range(len(tgt_ngb_neg))]}

            return interaction_dict, ngb_pos, ngb_neg
        else:
            ngb_pos = {}
            src_ngb = self.ngh_finder.find_k_hop(self.num_layer, uid_l, 0, time_l, pos=-1)  # user pos ngb
            tgt_ngb = self.ngh_finder.find_k_hop(self.num_layer, item_l, 1, time_l, pos=-1)  # item pos ngb
            for ind in range(self.num_layer):
                ngb_pos[f'ngb_{ind}'] = {'user': [np.squeeze(src_ngb[idx][ind]) for idx in range(len(src_ngb))],
                                         'item': [np.squeeze(tgt_ngb[idx][ind]) for idx in range(len(tgt_ngb))]}
            return interaction_dict, ngb_pos, 0

    def __len__(self):
        return len(self.user_data)
