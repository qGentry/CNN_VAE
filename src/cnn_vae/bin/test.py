from cnn_vae.model.model import Model
from cnn_vae.dataset.dataset import get_dataloaders

dataloaders = get_dataloaders("data", batch_size=16)

batch = next(iter(dataloaders['test']))

encoder_params = {
    "fc1_hidden": 1024,
    "fc2_hidden": 512,
    "dropout_p": 0.3,
}

decoder_params = {
    "fc1_hidden": 512,
    "fc2_hidden": 1024,
    "fc3_hidden": 2048,
    "dropout_p": 0.3,
}

model = Model(
    encoder_params,
    decoder_params,
    stochastic_dim=256,
)

input_images = batch[0]

output, mu, sigma = model(input_images)
