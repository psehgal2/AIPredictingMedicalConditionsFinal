import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__()
        self.activation = nn.GELU
        sizes = {
            0: [in_channels, hidden_dim],
            1: [hidden_dim, 2 * hidden_dim],
            2: [2 * hidden_dim, 2 * hidden_dim]
        }
        block = lambda i: nn.Sequential( # 2H x 2W -> H x W
            nn.Conv2d(sizes[i][0], sizes[i][1], kernel_size=3, padding=1, stride=2), 
            self.activation(),
            nn.Conv2d(sizes[i][1], sizes[i][1], kernel_size=3, padding=1),
            self.activation(),
        )
        n_blocks = 5
        self.net = nn.Sequential(
            # 1 x 256 x 256 -> 2h x 8 x 8
            block(0), block(1), block(2), block(2), block(2), 
            # 2h x 8 x 8 -> 1 x latent_dim
            nn.Flatten(), 
            nn.Linear(2 * hidden_dim * (256 // (2 ** n_blocks)) ** 2, latent_dim)
        )
   
    def forward(self, x):
        x = self.net(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__()
        self.activation = nn.GELU
        
        # 1 x latent_dim -> 2 x 8 x 8 
        n_blocks = 5
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * hidden_dim * (256 // (2 ** n_blocks)) ** 2),
            self.activation()
        )

        sizes = {
            0: [2 * hidden_dim, 2 * hidden_dim], # 2 x 8 x 8 -> 2 x 16 x 16
            1: [2 * hidden_dim, 2 * hidden_dim], # 2 x 16 x 16 -> 2 x 32 x 32
            2: [2 * hidden_dim, 2 * hidden_dim], # 2 x 32 x 32 -> 2 x 64 x 64
            3: [2 * hidden_dim, hidden_dim], # 2 x 64 x 64 -> 1 x 128 x 128
        }
        block = lambda i: nn.Sequential( # H x W -> 2H x 2W
            nn.ConvTranspose2d(sizes[i][0], sizes[i][1], kernel_size=3, output_padding=1, padding=1, stride=2),
            self.activation(),
            nn.Conv2d(sizes[i][1], sizes[i][1], kernel_size=3, padding=1),
            self.activation(),
        )

        # 2 x 8 x 8 -> 1 x 256 x 256
        self.net = nn.Sequential(
            # 2 x 8 x 8 -> 1 x 128 x 128
            block(0), block(1), block(2), block(3),
            # 1 x 128 x 128 -> 1 x 256 x 256
            nn.ConvTranspose2d(hidden_dim, in_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 8, 8)
        x = self.net(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dim, latent_dim)
        self.decoder = Decoder(in_channels, hidden_dim, latent_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        r = self.decoder(z)
        return r
