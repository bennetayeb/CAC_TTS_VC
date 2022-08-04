import torch
import torch.nn as nn

class CONV_AU(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # b, 16, 1768, 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # b,  32, 884, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),  # b, 64, 878, 1
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # b, 32, 884, 7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1,  output_padding=1),  # b, 16, 1768, 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # b, 1, 3536, 28
            nn.Sigmoid()
        )
    def get_latent(self, x):
        x = self.encoder(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x