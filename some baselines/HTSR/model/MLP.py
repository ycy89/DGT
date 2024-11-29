import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layers:list[int], activation:str):
        super().__init__()
        network = []
        for n, (indim, outdim) in enumerate(zip(layers[:-1], layers[1:])):
            network.append(nn.Linear(indim, outdim))
            if n < len(layers) - 2:
                if activation == 'SiLU':
                    network.append(nn.SiLU())
                elif activation == 'Sigmoid':
                    network.append(nn.Sigmoid())
                else:
                    network.append(nn.ReLU())
        self.network = nn.Sequential(*network)

    def forward(self, x):
        output = self.network(x)
        return output

