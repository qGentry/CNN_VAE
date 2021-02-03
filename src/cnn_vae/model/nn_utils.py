import torch.nn as nn


def make_linear_layer(input_dim, output_dim, dropout_p):
    result = nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(),
        nn.Dropout(p=dropout_p),
    )
    return result


def make_upconv_layer(in_channel, out_channels, kernel_size, stride, dropout_p):
    result = nn.Sequential(
        nn.ConvTranspose2d(in_channel, out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Dropout2d(dropout_p),
    )
    return result
