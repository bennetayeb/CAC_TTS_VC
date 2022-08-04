import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, n_input, n_latent, layer_sizes):
        super(Encoder, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        self.layer_sizes = [n_input] + layer_sizes + [n_latent]
        self.encoder_layers = []

        for idx in range(len(self.layer_sizes) - 1):
            fc1 = nn.Linear(self.layer_sizes[idx], self.layer_sizes[idx + 1])
            nn.init.xavier_uniform_(fc1.weight)
            self.encoder_layers.append(fc1)
            bn1 = nn.BatchNorm1d(self.layer_sizes[idx + 1])
            self.encoder_layers.append(bn1)
            act1 = nn.PReLU()
            self.encoder_layers.append(act1)

        self.encoder = nn.Sequential(*self.encoder_layers)

    def forward(self, x):
        return(self.encoder(x))

