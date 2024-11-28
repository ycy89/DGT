import torch


class MF_embeds(torch.nn.Module):
    def __init__(self, config: dict):
        super(MF_embeds, self).__init__()
        self.num_users = config['n_users']
        self.num_items = config['n_items']
        self.latent_dim = config['latent_dim']
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users + 1, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items + 1, embedding_dim=self.latent_dim)

        self.embedding_user.weight.data.uniform_(-1, 1)
        self.embedding_item.weight.data.uniform_(-1, 1)

    def forward(self, user, item):
        user_emb = self.embedding_user(user)
        item_emb = self.embedding_item(item)

        return user_emb, item_emb
