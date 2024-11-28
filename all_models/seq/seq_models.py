from torch import nn
import torch

from .SASRec import SASRec


class SASRec_embeds(SASRec):

    def forward(self, user_seq, log_seqs, user_id=None, neg_seqs=None):
        # neg_seq -> actually the history rating sequence
        if neg_seqs is not None:
            user_feats = self.log2feats_rate(user_seq, neg_seqs)
        else:
            user_feats = self.log2feats(user_seq)    # user history to emb
        if user_id is None:
            user_embeds = user_feats
        else:
            user_embeds = self.user_emb(user_id)    # user id to user emb
        item_embeds = self.item_emb(log_seqs)       # item id to item emb
        return user_feats, item_embeds, user_embeds


class GRU4Rec_embeds(nn.Module):

    def __init__(self, user_num, item_num, args):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        # init user and item embeddings
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        # sequence embedding module
        self.GRU = nn.GRU(args.embedding_dim, args.hidden_size, args.num_layers,
                          dropout=args.dropout_hidden)
        self.h2o = nn.Linear(args.hidden_size, args.output_size)  # hidden state to output size.
        self.emb_dropout = nn.Dropout(args.drop)

    def his2feats(self, log_seqs):
        # log_seqs: bs, L
        # look up
        seqs = self.item_emb(log_seqs)
        # seqs *= self.item_emb.embedding_dim ** 0.5
        seqs = self.emb_dropout(seqs)
        timeline_mask = log_seqs == 0
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim
        # [BS, L, dim] -> L,BS, dim
        gru_in = torch.permute(seqs, (1, 0, 2))
        output, _ = self.GRU(gru_in)  # TODO: [L, BS, H]
        output = torch.permute(output, (1, 0, 2))
        # [BS, L, dim]
        return output

    def forward(self, user_seq, log_seqs, user_id=None):
        user_feats = self.his2feats(user_seq)       # user history to emb
        if user_id is None:
            user_embeds = user_feats
        else:
            user_embeds = self.user_embe(user_id)   # user id to user emb
        item_embeds = self.item_emb(log_seqs)       # item id to item emb
        return user_feats, item_embeds, user_embeds
