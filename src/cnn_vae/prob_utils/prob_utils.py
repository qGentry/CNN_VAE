import torch
from torch.distributions import Normal, Independent


def get_normal_distr_from_params(mu, sigma):
    base_normal = Normal(mu, sigma)
    mvn = Independent(base_normal, 1)
    return mvn


def calc_diag_mvn_kl_loss(mu1, sigma1, mu2, sigma2):
    distr1 = get_normal_distr_from_params(mu1, sigma1)
    distr2 = get_normal_distr_from_params(mu2, sigma2)
    kl_div = torch.distributions.kl.kl_divergence(distr1, distr2)
    return kl_div


def calc_loglikelihood(inputs: torch.Tensor, outputs: torch.Tensor, sigma_prior: float):
    predicted = outputs.flatten(1)
    true_mu = inputs.flatten(1)
    base_normal = Normal(
        true_mu,
        torch.tensor(torch.ones_like(true_mu) * sigma_prior, dtype=torch.float32)
    )
    mvn = Independent(base_normal, 1)
    llh = mvn.log_prob(predicted)
    return llh
