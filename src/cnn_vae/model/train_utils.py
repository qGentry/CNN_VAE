from catalyst import dl

from cnn_vae.prob_utils import calc_diag_mvn_kl_loss, calc_loglikelihood


class Runner(dl.Runner):

    def __init__(self, prior_y_sigma: float = 1):
        super().__init__()
        self.prior_y_sigma = prior_y_sigma

    def predict_batch(self, batch, **kwargs):
        pass

    def _handle_batch(self, batch):
        images, _ = batch
        reconstructed_images, mu, sigma = self.model(images)

        prior_mu, prior_sigma = self.model.get_prior_z_distr_params(images.shape[0])

        kl_div_loss = calc_diag_mvn_kl_loss(mu, sigma, prior_mu, prior_sigma).mean()
        reconstruction_loss = -calc_loglikelihood(images, reconstructed_images, self.prior_y_sigma).mean()

        total_loss = reconstruction_loss

        self.batch_metrics.update(
            {
                "ELBO": total_loss.item(),
                "kl_div_loss": kl_div_loss.item(),
                "reconstruction_loss": reconstruction_loss.item(),
            }
        )

        if self.is_train_loader:
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
