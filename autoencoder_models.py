import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributions as td
import bpdb


class Flatten(nn.Module):
    def forward(self, input):
        return input.flatten(start_dim=1, end_dim=-1)


class UnFlatten(nn.Module):
    def __init__(self, size):
        super(UnFlatten, self).__init__()
        self.size = size

    def forward(self, input):
        return input.view(input.size(0), self.size, 1, 1)


class GenericUnflatten(nn.Module):
    def __init__(self, shape):
        super(GenericUnflatten, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(input.shape[0], *self.shape)


class ArgMax(nn.Module):
    def forward(self, input):
        return torch.argmax(input, 1)


def MultivariateNormalDiag(loc, scale_diag):
    if loc.dim() < 1:
        raise ValueError("loc must be at least one-dimensional.")
    return td.independent.Independent(td.normal.Normal(loc, scale_diag), 1)


class VanillaAutoenc(nn.Module):
    """For 64x64 images"""
    def __init__(self, n_channels=3, z_dim=512, categorical=True):
        super(VanillaAutoenc, self).__init__()
        self.encoder = nn.Sequential(
            # Input batchsize x n_channels x 64 x 64
            nn.Conv2d(n_channels, n_channels, kernel_size=4, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels),
            nn.Conv2d(n_channels, n_channels, kernel_size=4, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels),
            nn.Conv2d(n_channels, n_channels * 2, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 2),
            nn.Conv2d(n_channels * 2, n_channels * 4, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 4),
            nn.Conv2d(n_channels * 4, n_channels * 8, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 8),
            nn.Conv2d(n_channels * 8, n_channels * 8, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 8),
            nn.Conv2d(n_channels * 8, z_dim, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(z_dim),
            Flatten(),
            nn.Linear(z_dim*4, z_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(z_dim),
            nn.Linear(z_dim, z_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(z_dim),
        )

        decoder_layers = [
            nn.Linear(z_dim, z_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(z_dim),
            nn.Linear(z_dim, z_dim * 4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(z_dim * 4),
            GenericUnflatten((z_dim, 2, 2)),
            nn.ConvTranspose2d(z_dim, n_channels * 8, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 8),
            nn.ConvTranspose2d(n_channels * 8, n_channels * 8, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 8),
            nn.ConvTranspose2d(n_channels * 8, n_channels * 4, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 4),
            nn.ConvTranspose2d(n_channels * 4, n_channels * 2, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 2),
            nn.ConvTranspose2d(n_channels * 2, n_channels, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels),
            nn.Conv2d(n_channels, n_channels, kernel_size=4, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels),
            nn.Conv2d(n_channels, n_channels, kernel_size=4, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels),
        ]

        if categorical:
            # Applies softmax to every channel
            # This only makes sense if using one hot encoding
            decoder_layers.append(nn.Softmax(dim=1))
        else:
            decoder_layers.append(nn.Sigmoid)

        self.decoder = nn.Sequential(
            *decoder_layers
        )

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)

def make_distribution(latent_dist_params):
    """Returns the distribution from the encoder's output."""
    latent_size = latent_dist_params.size(-1)//2
    params = torch.split(latent_dist_params, latent_size, dim=-1)
    mean, log_var = params[0], params[1]
    std = log_var.exp().pow(0.5)
    q_z = MultivariateNormalDiag(loc=mean, scale_diag=std)
    return q_z


def kl_loss(q_z, p_z=None):
  """Takes KL-divergence between posterior (q_z) and prior (p_z)."""
  if p_z is None:
      p_z = MultivariateNormalDiag(torch.zeros_like(q_z.mean),
          torch.ones_like(q_z.stddev))
  return td.kl_divergence(q_z, p_z).sum()


class VAE(nn.Module):
    """For 64x64 images"""
    def __init__(self, n_channels=3, z_dim=512, categorical=True):
        super(VAE, self).__init__()
        # Multiply z_dim * 2 because our latent space is parametrized by a mean
        # and log_std for each dimension of the latent space
        encoder_z_dim = z_dim*2
        self.encoder = nn.Sequential(
            # Input batchsize x n_channels x 64 x 64
            nn.Conv2d(n_channels, n_channels, kernel_size=4, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels),
            nn.Conv2d(n_channels, n_channels, kernel_size=4, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels),
            nn.Conv2d(n_channels, n_channels * 2, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 2),
            nn.Conv2d(n_channels * 2, n_channels * 4, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 4),
            nn.Conv2d(n_channels * 4, n_channels * 8, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 8),
            nn.Conv2d(n_channels * 8, n_channels * 8, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 8),
            nn.Conv2d(n_channels * 8, encoder_z_dim, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(encoder_z_dim),
            Flatten(),
            nn.Linear(encoder_z_dim*4, encoder_z_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(encoder_z_dim),
            nn.Linear(encoder_z_dim, encoder_z_dim),
            #nn.LeakyReLU(),
            #nn.BatchNorm1d(z_dim),
        )

        decoder_layers = [
            nn.Linear(z_dim, z_dim, ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(z_dim),
            nn.Linear(z_dim, z_dim * 4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(z_dim * 4),
            GenericUnflatten((z_dim, 2, 2)),
            nn.ConvTranspose2d(z_dim, n_channels * 8, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 8),
            nn.ConvTranspose2d(n_channels * 8, n_channels * 8, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 8),
            nn.ConvTranspose2d(n_channels * 8, n_channels * 4, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 4),
            nn.ConvTranspose2d(n_channels * 4, n_channels * 2, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 2),
            nn.ConvTranspose2d(n_channels * 2, n_channels, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels),
            nn.Conv2d(n_channels, n_channels, kernel_size=4, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels),
            nn.Conv2d(n_channels, n_channels, kernel_size=4, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels),
        ]

        if categorical:
            # Applies softmax to every channel
            # This only makes sense if using one hot encoding
            decoder_layers.append(nn.Softmax(dim=1))
        else:
            decoder_layers.append(nn.Sigmoid)

        self.decoder = nn.Sequential(
            *decoder_layers
        )

    def forward(self, x):
      latent_dist_params = self.encoder(x)
      q_z = make_distribution(latent_dist_params)
      z = q_z.sample()
      return self.decoder(z)




class VanillaDisc(nn.Module):
    """Attempts to determine if a sample from the latent dim is random
    gaussian, or generated by the generator"""

    def __init__(self, nz, extra_layers=2) -> None:
        super().__init__()
        layers = []
        for _ in range(extra_layers):
            layers.append(
            nn.Linear(nz, nz))
            layers.append(nn.LeakyReLU())
            layers.append(nn.BatchNorm1d(nz))

        self.main = nn.Sequential(*layers, 
                nn.Linear(nz, nz // 2),
                nn.LeakyReLU(),
                nn.BatchNorm1d(nz//2),
                nn.Linear(nz//2, nz//4),
                nn.LeakyReLU(),
                nn.BatchNorm1d(nz//4),
                nn.Linear(nz//4, 1),
                nn.Sigmoid()
                )


    def forward(self, input):
        return self.main(input)

