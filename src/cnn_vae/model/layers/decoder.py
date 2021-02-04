import torch
import torch.nn as nn

from cnn_vae.model.nn_utils import make_linear_layer, make_upconv_layer


class ConvTransposeDecoderNetwork(nn.Module):

    def __init__(
            self,
            stochastic_dim: int,
            fc1_hidden: int,
            fc2_hidden: int,
            fc3_hidden: int,
            dropout_p: float,
    ):
        super().__init__()

        self.linears = nn.Sequential(
            make_linear_layer(stochastic_dim, fc1_hidden, dropout_p),
            make_linear_layer(fc1_hidden, fc2_hidden, dropout_p),
            make_linear_layer(fc2_hidden, fc3_hidden, dropout_p),
        )

        self.upconvs = nn.Sequential(
            make_upconv_layer(2048, 1024, kernel_size=2, stride=2, dropout_p=dropout_p),
            make_upconv_layer(1024, 512, kernel_size=3, stride=2, dropout_p=dropout_p),
            make_upconv_layer(512, 256, kernel_size=5, stride=2, dropout_p=dropout_p),
            make_upconv_layer(256, 256, kernel_size=3, stride=2, dropout_p=dropout_p),
            make_upconv_layer(256, 256, kernel_size=3, stride=2, dropout_p=dropout_p),
            make_upconv_layer(256, 128, kernel_size=3, stride=2, dropout_p=dropout_p),
            make_upconv_layer(128, 64, kernel_size=3, stride=2, dropout_p=dropout_p),
            make_upconv_layer(64, 3, kernel_size=2, stride=1, dropout_p=dropout_p),
        )

    def forward(self, x: torch.Tensor):
        x = self.linears(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.upconvs(x)
        return x
