import torch
import torch.nn as nn
import torch.nn.functional as F


hyper = {
    "nEpochs":100,
    "dimRNA":2386,
    "dimATAC":2386,
    "layer_sizes":[1024, 512, 256],
    "nz":128,
    "batchSize":512,
    "lr":1e-3,
    "add_hinge":True,
    "lamb_hinge":10,
    "lamb_match":1,
    "lamb_nn":1.5,
    "lamb_kl":1e-9,
    "lamb_anc":1e-9,
    "clip_grad":0.1,
    "checkpoint_path": './checkpoint/best_model_dcca_hinge.pt',
    "device": "cuda" if torch.cuda.is_available() else "cpu",

}

class FC_VAE(nn.Module):
    def __init__(self, n_input, nz, layer_sizes=hyper["layer_sizes"]):
        super(FC_VAE, self).__init__()
        self.n_input = n_input
        self.nz = nz
        self.layer_sizes = layer_sizes

        self.encoder_layers = []

        self.encoder_layers.append(nn.Linear(n_input, self.layer_sizes[0]))
        self.encoder_layers.append(nn.LeakyReLU(inplace=True))
        self.encoder_layers.append(nn.BatchNorm1d(self.layer_sizes[0]))

        for layer_idx in range(len(layer_sizes)-1):
            if layer_idx == len(layer_sizes) - 2:
                self.encoder_layers.append(nn.Linear(self.layer_sizes[layer_idx], self.layer_sizes[layer_idx+1]))
                self.encoder_layers.append(nn.LeakyReLU(inplace=True))
            else:
                self.encoder_layers.append(nn.Linear(self.layer_sizes[layer_idx], self.layer_sizes[layer_idx+1]))
                self.encoder_layers.append(nn.BatchNorm1d(self.layer_sizes[layer_idx+1]))
                self.encoder_layers.append(nn.LeakyReLU(inplace=True))

        self.encoder = nn.Sequential(
            *self.encoder_layers
        )
        self.fc1 = nn.Linear(self.layer_sizes[-1], nz)
        self.fc2 = nn.Linear(self.layer_sizes[-1], nz)

        self.decoder_layers = []
        self.decoder_layers.append(nn.Linear(nz, self.layer_sizes[-1]))
        self.decoder_layers.append(nn.BatchNorm1d(self.layer_sizes[-1]))
        self.decoder_layers.append(nn.LeakyReLU(inplace=True))

        for layer_idx in range(len(self.layer_sizes)-1, 0, -1):
            self.decoder_layers.append(nn.Linear(self.layer_sizes[layer_idx], self.layer_sizes[layer_idx-1]))
            self.decoder_layers.append(nn.BatchNorm1d(self.layer_sizes[layer_idx-1]))
            self.decoder_layers.append(nn.LeakyReLU(inplace=True))

        self.decoder_layers.append(nn.Linear(self.layer_sizes[0], self.n_input))

        self.decoder = nn.Sequential(
            *self.decoder_layers
        )
    def encode(self, x):
        h = self.encoder(x)
        return self.fc1(h), self.fc2(h)

    def reparametrize(self, mu, logvar):
        #calculate std from log(var)
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, z, mu, logvar

    def get_latent_var(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z
    
    def generate(self, z):
        return self.decode(z)

