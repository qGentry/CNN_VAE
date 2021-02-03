import torch.nn as nn
import torchvision

from cnn_vae.model.nn_utils import make_linear_layer


class EncoderNetwork(nn.Module):

    def __init__(self,
                 fc1_hidden: int,
                 fc2_hidden: int,
                 dropout_p: float,
                 resnet_type: str = 'resnet50'
                 ):
        super().__init__()
        backbone = getattr(torchvision.models, resnet_type)(pretrained=True)
        modules = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        self.linears = [
            make_linear_layer(backbone.fc.in_features, fc1_hidden, dropout_p),
            make_linear_layer(fc1_hidden, fc2_hidden, dropout_p),
        ]

    def forward(self, x):
        x = self.backbone(x)
        x = x.squeeze()
        for layer in self.linears:
            x = layer(x)
        return x
