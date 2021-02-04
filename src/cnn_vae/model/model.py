import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn_vae.model.layers.decoder import ConvTransposeDecoderNetwork
from cnn_vae.prob_utils import get_normal_distr_from_params
from cnn_vae.model.layers.encoder import EncoderNetwork

_STD_CLAMP_MIN = -10
_STD_CLAMP_MAX = 3


class Model(nn.Module):

    def __init__(self,
                 encoder_params: dict,
                 decoder_params: dict,
                 stochastic_dim: int,
                 device: str,
                 ):
        super().__init__()
        self.device = device
        self.stochastic_dim = stochastic_dim
        self.encoder_network = EncoderNetwork(**encoder_params)
        self.decoder_network = ConvTransposeDecoderNetwork(stochastic_dim, **decoder_params)
        self.mu_projector = nn.Linear(encoder_params['fc2_hidden'], stochastic_dim)
        self.sigma_projector = nn.Linear(encoder_params['fc2_hidden'], stochastic_dim)
        self.to(device)

    def get_prior_z_distr_params(self, samples_count: int):
        mu_prior = torch.zeros([samples_count, self.stochastic_dim]).to(self.device)
        sigma_prior = torch.ones_like(mu_prior).to(self.device)
        return mu_prior, sigma_prior

    @staticmethod
    def sample_from_posterior(mu, sigma):
        distr = get_normal_distr_from_params(mu, sigma)
        return distr.rsample()

    def sample_prior_z(self, samples_count: int = 1):
        mu_prior, sigma_prior = self.get_prior_z_distr_params(samples_count)
        distr = get_normal_distr_from_params(mu_prior, sigma_prior)
        return distr.sample()

    def forward(self, x):
        x = self.encoder_network(x)

        mu = self.mu_projector(x)
        log_sigma = self.sigma_projector(x)
        sigma = F.softplus(torch.clamp(log_sigma, _STD_CLAMP_MIN, _STD_CLAMP_MAX))

        z = self.sample_from_posterior(mu, sigma)
        output = self.decoder_network(z)

        return output, mu, F.softplus(log_sigma)



